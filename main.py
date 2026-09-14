import argparse
import json
from uuid import uuid4

from dotenv import load_dotenv
from incidents import SCENARIOS, create_incident
from langchain_core.tracers.langchain import wait_for_all_tracers
from models import ApprovalDecision
from pipeline import DATABASE_PATH, DataQualityError, preview_orders, run_pipeline


load_dotenv()


def run_clean_pipeline() -> None:
    row_count = run_pipeline()

    print(f"Pipeline completed: {row_count} orders loaded into {DATABASE_PATH}")
    print("\nPreview:")

    for order in preview_orders():
        print(order)


def show_plan(state: dict) -> None:
    print("\nDiagnosis:")
    print(json.dumps(state["diagnosis"], indent=2))
    print("\nProposed repair:")
    print(json.dumps(state["repair_plan"], indent=2))


def graph_config(
    thread_id: str,
    run_name: str,
    incident_type: str | None = None,
) -> dict:
    metadata = {"thread_id": thread_id}
    tags = ["pipeline-recovery"]

    if incident_type:
        metadata["incident_type"] = incident_type
        tags.append(incident_type)

    return {
        "configurable": {"thread_id": thread_id},
        "run_name": run_name,
        "tags": tags,
        "metadata": metadata,
    }


def request_decision(thread_id: str, incident_type: str | None = None) -> None:
    from agent import graph
    from langgraph.types import Command

    answer = input("\nDo you approve this repair plan? [y/N/later]: ").strip().lower()

    if answer in {"later", "l"}:
        print(f"\nIncident paused. Resume with: python main.py --resume {thread_id}")
        return

    decision = ApprovalDecision(approved=answer in {"y", "yes"})
    config = graph_config(thread_id, "Resume Incident Recovery", incident_type)
    result = graph.invoke(Command(resume=decision.model_dump()), config=config)

    if result["approval"]["approved"]:
        print("\nRepair result:")
        print(json.dumps(result["repair_result"], indent=2))
        print("\nRecovery validation:")
        print(json.dumps(result["recovery_validation"], indent=2))

        if result["recovered"]:
            print("\nRecovery successful: all quality checks passed.")
        else:
            print("\nRecovery incomplete: quality checks still fail.")
    else:
        print("\nRepair rejected. No changes were made.")


def run_incident(scenario: str, thread_id: str | None = None) -> None:
    data_path = create_incident(scenario)

    try:
        run_pipeline(data_path)
    except DataQualityError as error:
        print(f"Pipeline failed: {error}")

        from agent import graph
        thread_id = thread_id or str(uuid4())
        config = graph_config(thread_id, "Investigate Pipeline Incident", scenario)
        result = graph.invoke(
            {"database_path": str(DATABASE_PATH), "table_name": "raw_orders"},
            config=config,
            durability="sync",
        )

        print(f"\nIncident thread: {thread_id}")
        show_plan(result)
        request_decision(thread_id, scenario)


def resume_incident(thread_id: str) -> None:
    from agent import graph

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    if not snapshot.values:
        print(f"No saved incident found for thread: {thread_id}")
        return

    if not snapshot.next:
        print(f"Incident {thread_id} is already complete.")
        return

    print(f"Restored incident thread: {thread_id}")
    show_plan(snapshot.values)
    incident_type = snapshot.values["diagnosis"]["incident_type"]
    request_decision(thread_id, incident_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--incident", choices=SCENARIOS)
    mode.add_argument("--resume", metavar="THREAD_ID")
    parser.add_argument("--thread-id")
    args = parser.parse_args()

    if args.resume:
        resume_incident(args.resume)
    elif args.incident:
        run_incident(args.incident, args.thread_id)
    else:
        run_clean_pipeline()


if __name__ == "__main__":
    try:
        main()
    finally:
        wait_for_all_tracers()
