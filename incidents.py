import csv
from pathlib import Path


SOURCE_PATH = Path("data/orders.csv")
INCIDENTS_DIR = Path("data/incidents")
SCENARIOS = ("schema_drift", "duplicates", "missing_values")


def read_orders(source_path: Path = SOURCE_PATH) -> tuple[list[str], list[dict]]:
    with source_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def create_incident(scenario: str) -> Path:
    """Create one faulty CSV from the clean orders dataset."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    fieldnames, orders = read_orders()

    if scenario == "schema_drift":
        fieldnames[fieldnames.index("customer_id")] = "customerId"
        for order in orders:
            order["customerId"] = order.pop("customer_id")

    elif scenario == "duplicates":
        orders.extend([orders[2].copy(), orders[6].copy()])

    elif scenario == "missing_values":
        orders[4]["customer_id"] = ""
        orders[7]["unit_price"] = ""

    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INCIDENTS_DIR / f"{scenario}.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(orders)

    return output_path


if __name__ == "__main__":
    for scenario_name in SCENARIOS:
        path = create_incident(scenario_name)
        print(f"Created {scenario_name}: {path}")