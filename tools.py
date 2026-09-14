import json
from datetime import date, datetime
from pathlib import Path

import duckdb
from langsmith import traceable

from models import RepairPlan, RepairResult
from pipeline import DATABASE_PATH, LOG_PATH, write_log
from validation import EXPECTED_COLUMNS, validate_table


@traceable(run_type="tool", name="Read Pipeline Logs")
def read_pipeline_logs(log_path: str = str(LOG_PATH), limit: int = 20) -> list[dict]:
    """Return the most recent structured pipeline log records."""
    path = Path(log_path)
    if not path.exists():
        return []

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return records[-max(1, min(limit, 100)) :]


@traceable(run_type="tool", name="Inspect Affected Table")
def inspect_table(
    database_path: str = str(DATABASE_PATH),
    table_name: str = "raw_orders",
    limit: int = 5,
) -> dict:
    """Return a table's schema, row count, and sample records."""
    table = _quote_identifier(table_name)

    with duckdb.connect(database_path, read_only=True) as connection:
        schema_rows = connection.execute(f"DESCRIBE {table}").fetchall()
        row_count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        sample = _fetch_dicts(
            connection.execute(f"SELECT * FROM {table} LIMIT ?", [max(1, min(limit, 50))])
        )

    return {
        "table": table_name,
        "row_count": row_count,
        "schema": {row[0]: row[1] for row in schema_rows},
        "sample_rows": sample,
    }


@traceable(run_type="tool", name="Run Quality Checks")
def get_quality_results(
    database_path: str = str(DATABASE_PATH),
    table_name: str = "raw_orders",
) -> dict:
    """Rerun the deterministic quality checks on an affected table."""
    with duckdb.connect(database_path, read_only=True) as connection:
        return validate_table(connection, table_name).model_dump()


@traceable(run_type="tool", name="Find Problem Rows")
def find_problem_rows(
    database_path: str = str(DATABASE_PATH),
    table_name: str = "raw_orders",
    limit: int = 20,
) -> dict:
    """Return rows involved in duplicate-ID or missing-value failures."""
    table = _quote_identifier(table_name)
    limit = max(1, min(limit, 100))

    with duckdb.connect(database_path, read_only=True) as connection:
        columns = [row[0] for row in connection.execute(f"DESCRIBE {table}").fetchall()]

        duplicate_rows = []
        if "order_id" in columns:
            duplicate_rows = _fetch_dicts(
                connection.execute(
                    f"""
                    SELECT *
                    FROM {table}
                    WHERE order_id IN (
                        SELECT order_id
                        FROM {table}
                        GROUP BY order_id
                        HAVING COUNT(*) > 1
                    )
                    ORDER BY order_id
                    LIMIT ?
                    """,
                    [limit],
                )
            )

        available_columns = [column for column in EXPECTED_COLUMNS if column in columns]
        missing_rows = []

        if available_columns:
            conditions = " OR ".join(
                f'{_quote_identifier(column)} IS NULL' for column in available_columns
            )
            missing_rows = _fetch_dicts(
                connection.execute(
                    f"SELECT * FROM {table} WHERE {conditions} LIMIT ?",
                    [limit],
                )
            )

    return {
        "duplicate_rows": duplicate_rows,
        "rows_with_missing_values": missing_rows,
    }


@traceable(run_type="tool", name="Apply Approved Repair")
def apply_repair(
    database_path: str,
    table_name: str,
    repair_plan: dict,
    approved: bool,
) -> RepairResult:
    """Validate and execute one allowlisted repair inside a transaction."""
    plan = RepairPlan.model_validate(repair_plan)

    if not approved:
        raise PermissionError("A repair cannot run without human approval")

    if plan.action == "manual_review":
        return RepairResult(
            success=False,
            action=plan.action,
            rows_affected=0,
            message="Manual review requested; no automated repair was applied.",
        )

    table = _quote_identifier(table_name)
    write_log("repair_started", action=plan.action, parameters=plan.parameters)

    try:
        with duckdb.connect(database_path) as connection:
            connection.execute("BEGIN TRANSACTION")
            rows_affected = _execute_operation(
                connection,
                table,
                plan.action,
                plan.parameters,
            )
            connection.execute("COMMIT")
    except Exception as error:
        write_log("repair_failed", action=plan.action, error=str(error))
        return RepairResult(
            success=False,
            action=plan.action,
            rows_affected=0,
            message=str(error),
        )

    write_log("repair_completed", action=plan.action, rows_affected=rows_affected)
    return RepairResult(
        success=True,
        action=plan.action,
        rows_affected=rows_affected,
        message="Approved repair completed.",
    )


def _execute_operation(connection, table: str, action: str, parameters: dict) -> int:
    if action == "rename_column":
        _require_parameters(parameters, {"old_name", "new_name"})
        old_name = parameters["old_name"]
        new_name = parameters["new_name"]

        if not isinstance(old_name, str) or not isinstance(new_name, str):
            raise ValueError("Column names must be strings")

        columns = [row[0] for row in connection.execute(f"DESCRIBE {table}").fetchall()]
        missing = set(EXPECTED_COLUMNS) - set(columns)
        unexpected = set(columns) - set(EXPECTED_COLUMNS)

        if old_name not in unexpected or new_name not in missing:
            raise ValueError(
                "A rename must map an unexpected column to a missing expected column"
            )

        connection.execute(
            f"ALTER TABLE {table} RENAME COLUMN "
            f"{_quote_identifier(old_name)} TO {_quote_identifier(new_name)}"
        )
        return 0

    if action == "remove_duplicates":
        _require_parameters(parameters, {"key_column", "keep"})
        key_column = parameters["key_column"]
        keep = parameters["keep"]

        if key_column != "order_id" or keep != "first":
            raise ValueError("Duplicates can only be removed by order_id, keeping first")

        column = _quote_identifier(key_column)
        rows_affected = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT ROW_NUMBER() OVER (PARTITION BY {column}) AS row_number
                FROM {table}
            )
            WHERE row_number > 1
            """
        ).fetchone()[0]

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT * EXCLUDE (_repair_row_number)
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY {column}
                ) AS _repair_row_number
                FROM {table}
            )
            WHERE _repair_row_number = 1
            """
        )
        return rows_affected

    if action == "fill_missing_values":
        _require_parameters(parameters, {"column", "value"})
        column_name = parameters["column"]
        value = parameters["value"]
        _require_existing_columns(connection, table, [column_name])

        if value is None:
            raise ValueError("The replacement value cannot be null")

        column = _quote_identifier(column_name)
        rows_affected = connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL"
        ).fetchone()[0]
        connection.execute(
            f"UPDATE {table} SET {column} = ? WHERE {column} IS NULL",
            [value],
        )
        return rows_affected

    if action == "remove_incomplete_rows":
        _require_parameters(parameters, {"columns"})
        columns = parameters["columns"]

        if not isinstance(columns, list) or not columns:
            raise ValueError("columns must be a non-empty list")

        _require_existing_columns(connection, table, columns)
        condition = " OR ".join(
            f"{_quote_identifier(column)} IS NULL" for column in columns
        )
        rows_affected = connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {condition}"
        ).fetchone()[0]
        connection.execute(f"DELETE FROM {table} WHERE {condition}")
        return rows_affected

    raise ValueError(f"Unsupported repair action: {action}")


def _require_parameters(parameters: dict, expected: set[str]) -> None:
    if set(parameters) != expected:
        raise ValueError(f"Expected parameters: {', '.join(sorted(expected))}")


def _require_existing_columns(connection, table: str, columns: list[str]) -> None:
    if not all(isinstance(column, str) for column in columns):
        raise ValueError("Column names must be strings")

    available = {
        row[0] for row in connection.execute(f"DESCRIBE {table}").fetchall()
    }
    invalid = set(columns) - set(EXPECTED_COLUMNS)
    missing = set(columns) - available

    if invalid or missing:
        raise ValueError("Repair references an invalid or missing column")


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _fetch_dicts(cursor: duckdb.DuckDBPyConnection) -> list[dict]:
    columns = [description[0] for description in cursor.description]
    return [
        {column: _serialise(value) for column, value in zip(columns, row)}
        for row in cursor.fetchall()
    ]


def _serialise(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value
