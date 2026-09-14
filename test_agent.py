import os
from uuid import uuid4

os.environ["LANGSMITH_TRACING"] = "false"

import duckdb
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agent
from models import Diagnosis, RepairPlan


@pytest.fixture(autouse=True)
def isolate_test_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class FakeModel:
    """Return fixed Pydantic responses instead of calling an LLM."""

    def __init__(self, diagnosis: Diagnosis, repair_plan: RepairPlan):
        self.diagnosis = diagnosis
        self.repair_plan = repair_plan

    def with_structured_output(self, schema, **kwargs):
        response = self.diagnosis if schema is Diagnosis else self.repair_plan

        class StructuredResponse:
            def invoke(self, prompt):
                return response

        return StructuredResponse()


def make_diagnosis(incident_type: str) -> Diagnosis:
    return Diagnosis(
        incident_type=incident_type,
        probable_cause=f"Test diagnosis: {incident_type}",
        evidence=["Deterministic test evidence"],
        confidence=1.0,
    )


def make_plan(action: str, parameters: dict) -> RepairPlan:
    return RepairPlan(
        action=action,
        description=f"Test repair: {action}",
        parameters=parameters,
        risk="low",
    )


def create_database(tmp_path, scenario: str) -> str:
    database_path = tmp_path / f"{scenario}.duckdb"
    customer_column = "customerId" if scenario == "schema_drift" else "customer_id"
    rows = [
        [1, "C1", "2026-09-01", "Mouse", 1, 10.0, "completed"],
        [2, "C2", "2026-09-02", "Keyboard", 1, 20.0, "completed"],
        [3, "C3", "2026-09-03", "Headset", 1, 30.0, "pending"],
    ]

    if scenario == "duplicates":
        rows.append(rows[0].copy())
    elif scenario == "missing_values":
        rows[0][1] = None
        rows[1][5] = None
    elif scenario == "single_missing_value":
        rows[0][6] = None

    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            f"""
            CREATE TABLE raw_orders (
                order_id INTEGER,
                {customer_column} VARCHAR,
                order_date DATE,
                product VARCHAR,
                quantity INTEGER,
                unit_price DOUBLE,
                status VARCHAR
            )
            """
        )
        connection.executemany(
            "INSERT INTO raw_orders VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    return str(database_path)


def start_graph(monkeypatch, database_path, diagnosis, repair_plan):
    monkeypatch.setattr(agent, "model", FakeModel(diagnosis, repair_plan))
    graph = agent.builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid4())}}
    paused = graph.invoke(
        {"database_path": database_path, "table_name": "raw_orders"},
        config=config,
    )
    return graph, config, paused


def test_graph_pauses_for_approval(tmp_path, monkeypatch):
    database_path = create_database(tmp_path, "duplicates")
    graph, config, paused = start_graph(
        monkeypatch,
        database_path,
        make_diagnosis("duplicates"),
        make_plan("remove_duplicates", {"key_column": "order_id", "keep": "first"}),
    )

    assert paused["__interrupt__"]
    assert graph.get_state(config).next == ("request_approval",)


def test_rejected_repair_does_not_change_data(tmp_path, monkeypatch):
    database_path = create_database(tmp_path, "duplicates")
    graph, config, _ = start_graph(
        monkeypatch,
        database_path,
        make_diagnosis("duplicates"),
        make_plan("remove_duplicates", {"key_column": "order_id", "keep": "first"}),
    )

    result = graph.invoke(Command(resume={"approved": False}), config=config)

    with duckdb.connect(database_path, read_only=True) as connection:
        row_count = connection.execute("SELECT COUNT(*) FROM raw_orders").fetchone()[0]

    assert result["approval"]["approved"] is False
    assert "repair_result" not in result
    assert row_count == 4


@pytest.mark.parametrize(
    ("scenario", "incident_type", "action", "parameters", "rows_affected"),
    [
        (
            "schema_drift",
            "schema_drift",
            "rename_column",
            {"old_name": "customerId", "new_name": "customer_id"},
            0,
        ),
        (
            "duplicates",
            "duplicates",
            "remove_duplicates",
            {"key_column": "order_id", "keep": "first"},
            1,
        ),
        (
            "missing_values",
            "missing_values",
            "remove_incomplete_rows",
            {"columns": ["customer_id", "unit_price"]},
            2,
        ),
        (
            "single_missing_value",
            "missing_values",
            "fill_missing_values",
            {"column": "status", "value": "unknown"},
            1,
        ),
    ],
)
def test_successful_recovery(
    tmp_path,
    monkeypatch,
    scenario,
    incident_type,
    action,
    parameters,
    rows_affected,
):
    database_path = create_database(tmp_path, scenario)
    graph, config, _ = start_graph(
        monkeypatch,
        database_path,
        make_diagnosis(incident_type),
        make_plan(action, parameters),
    )

    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["repair_result"]["success"] is True
    assert result["repair_result"]["rows_affected"] == rows_affected
    assert result["recovery_validation"]["passed"] is True
    assert result["recovered"] is True


def test_manual_review_does_not_change_data(tmp_path, monkeypatch):
    database_path = create_database(tmp_path, "duplicates")
    graph, config, _ = start_graph(
        monkeypatch,
        database_path,
        make_diagnosis("duplicates"),
        make_plan("manual_review", {}),
    )

    result = graph.invoke(Command(resume={"approved": True}), config=config)

    with duckdb.connect(database_path, read_only=True) as connection:
        row_count = connection.execute("SELECT COUNT(*) FROM raw_orders").fetchone()[0]

    assert result["repair_result"]["success"] is False
    assert result["recovered"] is False
    assert row_count == 4


def test_invalid_repair_is_blocked(tmp_path, monkeypatch):
    database_path = create_database(tmp_path, "duplicates")
    graph, config, _ = start_graph(
        monkeypatch,
        database_path,
        make_diagnosis("duplicates"),
        make_plan(
            "remove_duplicates",
            {"key_column": "customer_id", "keep": "first"},
        ),
    )

    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["repair_result"]["success"] is False
    assert result["recovered"] is False
    assert result["recovery_validation"]["checks"]["duplicates"]["passed"] is False


def test_wrong_diagnosis_is_caught_by_validation(tmp_path, monkeypatch):
    database_path = create_database(tmp_path, "missing_values")
    graph, config, _ = start_graph(
        monkeypatch,
        database_path,
        make_diagnosis("duplicates"),
        make_plan("remove_duplicates", {"key_column": "order_id", "keep": "first"}),
    )

    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["repair_result"]["success"] is True
    assert result["repair_result"]["rows_affected"] == 0
    assert result["recovery_validation"]["checks"]["missing_values"]["passed"] is False
    assert result["recovered"] is False