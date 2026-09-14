import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from models import ValidationResult
from validation import validate_table


DATA_PATH = Path("data/orders.csv")
DATABASE_PATH = Path("orders.duckdb")
LOG_PATH = Path("pipeline.log")


def write_log(event: str, **details) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **details,
    }
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


class DataQualityError(Exception):
    def __init__(self, result: ValidationResult):
        self.result = result
        failed_checks = [
            name for name, check in result.checks.items() if not check.passed
        ]
        super().__init__(f"Data quality checks failed: {', '.join(failed_checks)}")


def run_pipeline(
    data_path: Path = DATA_PATH,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Load raw order data and create a small processed orders table."""
    write_log("pipeline_started", source=str(data_path), database=str(database_path))

    if not data_path.exists():
        write_log("pipeline_failed", error=f"Order data not found: {data_path}")
        raise FileNotFoundError(f"Order data not found: {data_path}")

    try:
        with duckdb.connect(str(database_path)) as connection:
            connection.execute("DROP TABLE IF EXISTS orders")

            connection.execute(
                """
                CREATE OR REPLACE TABLE raw_orders AS
                SELECT * FROM read_csv_auto(?, header = true)
                """,
                [str(data_path)],
            )

            validation_result = validate_table(connection)
            if not validation_result.passed:
                write_log(
                    "validation_failed",
                    result=validation_result.model_dump(),
                )
                raise DataQualityError(validation_result)

            write_log("validation_passed", row_count=validation_result.row_count)

            connection.execute(
                """
                CREATE OR REPLACE TABLE orders AS
                SELECT
                    order_id,
                    customer_id,
                    CAST(order_date AS DATE) AS order_date,
                    product,
                    quantity,
                    unit_price,
                    ROUND(quantity * unit_price, 2) AS total_amount,
                    status
                FROM raw_orders
                """
            )

            row_count = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

    except DataQualityError:
        raise
    except Exception as error:
        write_log("pipeline_failed", error=str(error))
        raise

    write_log("pipeline_completed", row_count=row_count)
    return row_count


def preview_orders(database_path: Path = DATABASE_PATH) -> list[tuple]:
    """Return a few processed rows so the pipeline result is easy to inspect."""
    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT order_id, customer_id, product, total_amount, status
            FROM orders
            ORDER BY order_id
            LIMIT 5
            """
        ).fetchall()