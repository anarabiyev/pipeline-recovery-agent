from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


IncidentType = Literal[
    "schema_drift",
    "duplicates",
    "missing_values",
    "multiple_issues",
    "unknown",
]

RepairAction = Literal[
    "rename_column",
    "remove_duplicates",
    "fill_missing_values",
    "remove_incomplete_rows",
    "manual_review",
]

ParameterValue = str | int | float | bool | None | list[str]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QualityCheck(StrictModel):
    passed: bool
    details: dict[str, object] = Field(default_factory=dict)


class ValidationResult(StrictModel):
    passed: bool
    row_count: int = Field(ge=0)
    checks: dict[str, QualityCheck]


class Diagnosis(StrictModel):
    incident_type: IncidentType
    probable_cause: str
    evidence: list[str]
    confidence: float = Field(ge=0, le=1)


class RepairPlan(StrictModel):
    action: RepairAction
    description: str
    parameters: dict[str, ParameterValue]
    risk: Literal["low", "medium", "high"]


class ApprovalDecision(StrictModel):
    approved: bool
    comment: str | None = None


class RepairResult(StrictModel):
    success: bool
    action: RepairAction
    rows_affected: int = Field(ge=0)
    message: str