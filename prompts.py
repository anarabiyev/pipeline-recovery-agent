DIAGNOSIS_PROMPT = """
You are investigating a failed orders data pipeline.

Use only the supplied logs, table information, quality results, and problem rows.
Identify the most likely incident type and explain the supporting evidence.
If the evidence is insufficient or contradictory, return incident_type="unknown".

Investigation evidence:
{evidence}
"""


REPAIR_PROMPT = """
You are preparing a safe repair plan for a failed orders data pipeline.

Choose only one supported action:
- rename_column: parameters old_name and new_name
- remove_duplicates: parameters key_column and keep
- fill_missing_values: parameters column and value
- remove_incomplete_rows: parameters columns
- manual_review: use when a safe repair cannot be inferred

Prefer manual_review when the evidence does not justify changing data.
Do not generate SQL or invent unsupported actions.

Diagnosis:
{diagnosis}

Quality results:
{quality_results}

Problem rows:
{problem_rows}
"""