from pathlib import Path

import duckdb

from models import QualityCheck, ValidationResult


EXPECTED_COLUMNS = [
    "order_id",
    "customer_id",
    "order_date",
    "product",
    "quantity",
    "unit_price",
    "status",
]


def validate_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str = "raw_orders",
) -> ValidationResult:
    """Run deterministic quality checks against a DuckDB table."""
    table = f'"{table_name}"'
    columns = [row[0] for row in connection.execute(f"DESCRIBE {table}").fetchall()]

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in columns]
    unexpected_columns = [column for column in columns if column not in EXPECTED_COLUMNS]

    schema_check = QualityCheck(
        passed=not missing_columns and not unexpected_columns,
        details={
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
        },
    )

    if "order_id" in columns:
        duplicate_rows = connection.execute(
            f"""
            SELECT order_id, COUNT(*) AS occurrences
            FROM {table}
            GROUP BY order_id
            HAVING COUNT(*) > 1
            ORDER BY order_id
            """
        ).fetchall()
        duplicate_count = sum(occurrences - 1 for _, occurrences in duplicate_rows)
        duplicate_ids = [order_id for order_id, _ in duplicate_rows]
    else:
        duplicate_count = None
        duplicate_ids = []

    duplicate_check = QualityCheck(
        passed=duplicate_count == 0,
        details={"count": duplicate_count, "order_ids": duplicate_ids},
    )

    available_columns = [column for column in EXPECTED_COLUMNS if column in columns]
    null_counts = {}

    if available_columns:
        expressions = [
            f'COUNT(*) FILTER (WHERE "{column}" IS NULL)'
            for column in available_columns
        ]
        counts = connection.execute(
            f"SELECT {', '.join(expressions)} FROM {table}"
        ).fetchone()
        null_counts = dict(zip(available_columns, counts))

    missing_value_check = QualityCheck(
        passed=all(count == 0 for count in null_counts.values()),
        details={
            "count": sum(null_counts.values()),
            "columns": {
                column: count for column, count in null_counts.items() if count
            },
        },
    )

    checks = {
        "schema": schema_check,
        "duplicates": duplicate_check,
        "missing_values": missing_value_check,
    }

    return ValidationResult(
        passed=all(check.passed for check in checks.values()),
        row_count=connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
        checks=checks,
    )


def validate_file(data_path: Path) -> ValidationResult:
    """Load a CSV into memory and return its quality-check results."""
    with duckdb.connect() as connection:
        connection.execute(
            """
            CREATE TABLE raw_orders AS
            SELECT * FROM read_csv_auto(?, header = true)
            """,
            [str(data_path)],
        )
        return validate_table(connection)


if __name__ == "__main__":
    files = [Path("data/orders.csv"), *sorted(Path("data/incidents").glob("*.csv"))]

    for file_path in files:
        result = validate_file(file_path)
        status = "PASS" if result.passed else "FAIL"
        print(f"\n{status}: {file_path}")
        print(result.model_dump())