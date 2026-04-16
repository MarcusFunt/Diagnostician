from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class FactCategory(StrEnum):
    DEMOGRAPHICS = "demographics"
    CHIEF_COMPLAINT = "chief_complaint"
    TIMELINE = "timeline"
    SYMPTOM = "symptom"
    PAST_MEDICAL_HISTORY = "past_medical_history"
    MEDICATION = "medication"
    SOCIAL_HISTORY = "social_history"
    VITAL = "vital"
    PHYSICAL_EXAM = "physical_exam"
    LAB = "lab"
    IMAGING = "imaging"
    PATHOLOGY = "pathology"
    PROCEDURE = "procedure"
    MICROBIOLOGY = "microbiology"
    DIAGNOSIS = "diagnosis"
    DIFFERENTIAL_TAG = "differential_tag"
    TEACHING_POINT = "teaching_point"
    HINT = "hint"


class ProvenanceKind(StrEnum):
    SOURCE_EXACT = "source_grounded_exact"
    SOURCE_PARAPHRASE = "source_grounded_paraphrase"
    SYNTHETIC_CONNECTIVE = "approved_synthetic_safe_connective_detail"
    FORBIDDEN_SYNTHETIC = "forbidden_synthetic_diagnostic_fact"


class DisplayBlockType(StrEnum):
    NARRATIVE = "narrative"
    PATIENT_DIALOGUE = "patient_dialogue"
    ATTENDING_COMMENT = "attending_comment"
    LAB_RESULT = "lab_result"
    IMAGING_REPORT = "imaging_report"
    PATHOLOGY_REPORT = "pathology_report"
    WARNING = "warning"
    HINT = "hint"
    SYSTEM_STATUS = "system_status"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class ActionType(StrEnum):
    ASK_PATIENT_QUESTION = "ask_patient_question"
    REQUEST_EXAM_DETAIL = "request_exam_detail"
    ORDER_LAB = "order_lab"
    ORDER_IMAGING = "order_imaging"
    REQUEST_PATHOLOGY_DETAIL = "request_pathology_detail"
    SUBMIT_DIFFERENTIAL = "submit_differential"
    REQUEST_HINT = "request_hint"


class RunStatus(StrEnum):
    ACTIVE = "active"
    COMPLETE = "complete"
    ABANDONED = "abandoned"


class ValidationStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    FALLBACK_USED = "fallback_used"


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    path: str
    title: str
    source_type: Literal["json", "markdown", "pdf", "text", "multicare"] = "json"
    deidentified: bool = True
    citation: str | None = None
    raw_text: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    source_document_id: UUID
    kind: ProvenanceKind
    locator: str | None = None
    quote: str | None = None
    note: str | None = None


class CaseFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    case_id: UUID
    category: FactCategory
    label: str
    value: str
    provenance_ids: list[UUID] = Field(default_factory=list)
    spoiler: bool = False
    initially_visible: bool = False
    reveal_actions: list[ActionType] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def search_text(self) -> str:
        return " ".join([self.category.value, self.label, self.value, *self.tags]).lower()


class RevealPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    case_id: UUID
    initial_fact_ids: list[UUID] = Field(default_factory=list)
    diagnosis_locked: bool = True
    max_facts_per_turn: int = Field(default=3, ge=1, le=10)
    action_category_map: dict[ActionType, list[FactCategory]] = Field(default_factory=dict)


class TruthCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    title: str
    source_document_ids: list[UUID] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    demographics: dict[str, Any] = Field(default_factory=dict)
    chief_complaint: str
    final_diagnosis: str
    diagnosis_aliases: list[str] = Field(default_factory=list)
    difficulty: str = "prototype"
    specialty: str | None = None
    tags: list[str] = Field(default_factory=list)
    facts: list[CaseFact] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    reveal_policy: RevealPolicy | None = None
    teaching_points: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("diagnosis_aliases")
    @classmethod
    def include_final_diagnosis_alias(cls, value: list[str], info: Any) -> list[str]:
        final_diagnosis = info.data.get("final_diagnosis")
        aliases = {alias.strip() for alias in value if alias.strip()}
        if final_diagnosis:
            aliases.add(str(final_diagnosis).strip())
        return sorted(aliases)

    @computed_field
    @property
    def approved_for_play(self) -> bool:
        return self.review_status == ReviewStatus.APPROVED


class VisibleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[CaseFact] = Field(default_factory=list)


class DisplayBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    type: DisplayBlockType
    title: str
    body: str
    fact_ids: list[UUID] = Field(default_factory=list)
    provenance_ids: list[UUID] = Field(default_factory=list)
    severity: Severity = Severity.INFO


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    case_id: UUID
    status: RunStatus = RunStatus.ACTIVE
    stage: str = "opening"
    visible_fact_ids: list[UUID] = Field(default_factory=list)
    ordered_tests: list[str] = Field(default_factory=list)
    requested_clues: list[str] = Field(default_factory=list)
    submitted_differentials: list[str] = Field(default_factory=list)
    hint_count: int = 0
    turn_count: int = 0
    score: int | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class PlayerTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    player_text: str = ""
    target: str | None = None
    selected_evidence_ids: list[UUID] = Field(default_factory=list)
    client_timestamp: datetime | None = None


class DiagnosisSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis: str
    rationale: str = ""
    client_timestamp: datetime | None = None


class SoftAuditScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spoiler_risk: float = Field(default=0.0, ge=0, le=1)
    contradiction_risk: float = Field(default=0.0, ge=0, le=1)
    plausibility: float = Field(default=1.0, ge=0, le=1)
    tone_fit: float = Field(default=1.0, ge=0, le=1)
    notes: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    status: ValidationStatus
    hard_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    allowed_fact_ids: list[UUID] = Field(default_factory=list)
    revealed_fact_ids: list[UUID] = Field(default_factory=list)
    soft_audit: SoftAuditScore = Field(default_factory=SoftAuditScore)
    fallback_used: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class TurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_state: RunState
    display_blocks: list[DisplayBlock]
    newly_revealed_facts: list[CaseFact] = Field(default_factory=list)
    visible_evidence: VisibleEvidence
    validation: ValidationReport


class ScoreSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct: bool
    final_score: int = Field(ge=0, le=100)
    diagnosis_points: int = Field(ge=0)
    differential_points: int = Field(ge=0)
    efficiency_penalty: int = Field(ge=0)
    testing_penalty: int = Field(ge=0)
    hint_penalty: int = Field(ge=0)
    dangerous_miss_penalty: int = Field(ge=0)
    rationale: list[str] = Field(default_factory=list)


class CaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    case_id: UUID
    diagnosis: str
    player_score: ScoreSummary
    key_findings: list[CaseFact]
    teaching_points: list[str]
    provenance: list[Provenance]
    turn_timeline: list[DisplayBlock]


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: UUID | None = None
    specialty: str | None = None
    difficulty: str | None = None


class IngestionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    source_document_id: UUID
    case_id: UUID | None = None
    accepted: bool
    playable: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    embedding_fallback_fact_ids: list[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class RunSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_state: RunState
    visible_evidence: VisibleEvidence
    display_blocks: list[DisplayBlock] = Field(default_factory=list)
