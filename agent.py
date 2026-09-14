import json
import os
import sqlite3
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from models import ApprovalDecision, Diagnosis, RepairPlan
from prompts import DIAGNOSIS_PROMPT, REPAIR_PROMPT
from tools import (
    apply_repair,
    find_problem_rows,
    get_quality_results,
    inspect_table,
    read_pipeline_logs,
)


load_dotenv()
model = None
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints.sqlite"


class IncidentState(TypedDict, total=False):
    database_path: str
    table_name: str
    logs: list[dict]
    table_info: dict
    quality_results: dict
    problem_rows: dict
    diagnosis: dict
    repair_plan: dict
    approval: dict
    repair_result: dict
    recovery_validation: dict
    recovered: bool


def get_model():
    global model
    if model is None:
        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0,
        )
    return model


def investigate(state: IncidentState) -> dict:
    database_path = state["database_path"]
    table_name = state.get("table_name", "raw_orders")

    return {
        "logs": read_pipeline_logs(limit=10),
        "table_info": inspect_table(database_path, table_name),
        "quality_results": get_quality_results(database_path, table_name),
        "problem_rows": find_problem_rows(database_path, table_name),
    }


def diagnose(state: IncidentState) -> dict:
    evidence = {
        "logs": state["logs"],
        "table_info": state["table_info"],
        "quality_results": state["quality_results"],
        "problem_rows": state["problem_rows"],
    }
    prompt = DIAGNOSIS_PROMPT.format(evidence=json.dumps(evidence, indent=2))
    diagnosis = get_model().with_structured_output(
        Diagnosis,
        method="function_calling",
    ).invoke(prompt)
    return {"diagnosis": diagnosis.model_dump()}


def plan_repair(state: IncidentState) -> dict:
    prompt = REPAIR_PROMPT.format(
        diagnosis=json.dumps(state["diagnosis"], indent=2),
        quality_results=json.dumps(state["quality_results"], indent=2),
        problem_rows=json.dumps(state["problem_rows"], indent=2),
    )
    repair_plan = get_model().with_structured_output(
        RepairPlan,
        method="function_calling",
    ).invoke(prompt)
    return {"repair_plan": repair_plan.model_dump()}


def request_approval(state: IncidentState) -> dict:
    decision = interrupt(
        {
            "question": "Do you approve this repair plan?",
            "repair_plan": state["repair_plan"],
        }
    )
    approval = ApprovalDecision.model_validate(decision)
    return {"approval": approval.model_dump()}


def route_after_approval(
    state: IncidentState,
) -> Literal["execute_repair", "end"]:
    return "execute_repair" if state["approval"]["approved"] else "end"


def execute_repair(state: IncidentState) -> dict:
    result = apply_repair(
        database_path=state["database_path"],
        table_name=state["table_name"],
        repair_plan=state["repair_plan"],
        approved=state["approval"]["approved"],
    )
    return {"repair_result": result.model_dump()}


def validate_recovery(state: IncidentState) -> dict:
    validation = get_quality_results(
        database_path=state["database_path"],
        table_name=state["table_name"],
    )
    recovered = state["repair_result"]["success"] and validation["passed"]
    return {
        "recovery_validation": validation,
        "recovered": recovered,
    }


builder = StateGraph(IncidentState)
builder.add_node("investigate", investigate)
builder.add_node("diagnose", diagnose)
builder.add_node("plan_repair", plan_repair)
builder.add_node("request_approval", request_approval)
builder.add_node("execute_repair", execute_repair)
builder.add_node("validate_recovery", validate_recovery)

builder.add_edge(START, "investigate")
builder.add_edge("investigate", "diagnose")
builder.add_edge("diagnose", "plan_repair")
builder.add_edge("plan_repair", "request_approval")
builder.add_conditional_edges(
    "request_approval",
    route_after_approval,
    {"execute_repair": "execute_repair", "end": END},
)
builder.add_edge("execute_repair", "validate_recovery")
builder.add_edge("validate_recovery", END)

# Agent Server provides persistence when this graph is opened in Studio.
studio_graph = builder.compile()

# The command-line application keeps its checkpoints in a local SQLite file.
connection = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
checkpointer = SqliteSaver(connection)
graph = builder.compile(checkpointer=checkpointer)
