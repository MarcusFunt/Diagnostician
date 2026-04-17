from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID, uuid4

from diagnostician.core.windows_platform import disable_slow_wmi_platform_probe

disable_slow_wmi_platform_probe()

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from diagnostician.core.schemas import (
    CaseReview,
    DisplayBlock,
    ReviewStatus,
    RunState,
    ScoreSummary,
    SourceDocument,
    TruthCase,
    TurnResponse,
    ValidationReport,
)
from diagnostician.db.models import (
    CaseFactEmbeddingRow,
    CaseFactRow,
    CaseRow,
    ReviewRow,
    RunRow,
    ScoreRow,
    SourceDocumentRow,
    TurnRow,
    ValidationLogRow,
)


def dump_model(model):
    return model.model_dump(mode="json")


class GameStore(Protocol):
    def save_source_document(self, source: SourceDocument) -> SourceDocument: ...

    def save_truth_case(
        self, truth_case: TruthCase, embeddings: dict[UUID, list[float]] | None = None
    ) -> TruthCase: ...

    def list_approved_cases(
        self, specialty: str | None = None, difficulty: str | None = None
    ) -> list[TruthCase]: ...

    def get_case(self, case_id: UUID) -> TruthCase: ...

    def save_run(self, run_state: RunState) -> RunState: ...

    def get_run(self, run_id: UUID) -> RunState: ...

    def append_turn(
        self, run_id: UUID, request_payload: dict, response: TurnResponse
    ) -> UUID: ...

    def list_turn_blocks(self, run_id: UUID) -> list[DisplayBlock]: ...

    def log_validation(
        self, run_id: UUID, report: ValidationReport, turn_id: UUID | None = None
    ) -> None: ...

    def save_score(self, run_id: UUID, score: ScoreSummary) -> None: ...

    def save_review(self, review: CaseReview) -> CaseReview: ...

    def get_review(self, run_id: UUID) -> CaseReview | None: ...


@dataclass
class InMemoryGameStore:
    source_documents: dict[UUID, SourceDocument] = field(default_factory=dict)
    cases: dict[UUID, TruthCase] = field(default_factory=dict)
    embeddings: dict[UUID, list[float]] = field(default_factory=dict)
    runs: dict[UUID, RunState] = field(default_factory=dict)
    turn_payloads: dict[UUID, list[dict]] = field(default_factory=dict)
    validation_logs: dict[UUID, list[ValidationReport]] = field(default_factory=dict)
    scores: dict[UUID, ScoreSummary] = field(default_factory=dict)
    reviews: dict[UUID, CaseReview] = field(default_factory=dict)

    def save_source_document(self, source: SourceDocument) -> SourceDocument:
        self.source_documents[source.id] = source
        return source

    def save_truth_case(
        self, truth_case: TruthCase, embeddings: dict[UUID, list[float]] | None = None
    ) -> TruthCase:
        self.cases[truth_case.id] = truth_case
        if embeddings:
            self.embeddings.update(embeddings)
        return truth_case

    def list_approved_cases(
        self, specialty: str | None = None, difficulty: str | None = None
    ) -> list[TruthCase]:
        cases = [
            case
            for case in self.cases.values()
            if case.review_status == ReviewStatus.APPROVED
            and (specialty is None or case.specialty == specialty)
            and (difficulty is None or case.difficulty == difficulty)
        ]
        return sorted(cases, key=lambda case: case.created_at)

    def get_case(self, case_id: UUID) -> TruthCase:
        return self.cases[case_id]

    def save_run(self, run_state: RunState) -> RunState:
        self.runs[run_state.id] = run_state
        return run_state

    def get_run(self, run_id: UUID) -> RunState:
        return self.runs[run_id]

    def append_turn(
        self, run_id: UUID, request_payload: dict, response: TurnResponse
    ) -> UUID:
        turn_id = uuid4()
        self.turn_payloads.setdefault(run_id, []).append(
            {
                "id": str(turn_id),
                "request": request_payload,
                "response": dump_model(response),
            }
        )
        return turn_id

    def list_turn_blocks(self, run_id: UUID) -> list[DisplayBlock]:
        blocks: list[DisplayBlock] = []
        for payload in self.turn_payloads.get(run_id, []):
            blocks.extend(
                DisplayBlock.model_validate(block)
                for block in payload["response"].get("display_blocks", [])
            )
        return blocks

    def log_validation(
        self, run_id: UUID, report: ValidationReport, turn_id: UUID | None = None
    ) -> None:
        self.validation_logs.setdefault(run_id, []).append(report)

    def save_score(self, run_id: UUID, score: ScoreSummary) -> None:
        self.scores[run_id] = score

    def save_review(self, review: CaseReview) -> CaseReview:
        self.reviews[review.run_id] = review
        return review

    def get_review(self, run_id: UUID) -> CaseReview | None:
        return self.reviews.get(run_id)


class SqlAlchemyGameStore:
    def __init__(self, session: Session):
        self.session = session

    def save_source_document(self, source: SourceDocument) -> SourceDocument:
        row = SourceDocumentRow(
            id=source.id,
            path=source.path,
            title=source.title,
            source_type=source.source_type,
            deidentified=source.deidentified,
            citation=source.citation,
            raw_text=source.raw_text,
            payload=dump_model(source),
        )
        self.session.merge(row)
        self.session.commit()
        return source

    def save_truth_case(
        self, truth_case: TruthCase, embeddings: dict[UUID, list[float]] | None = None
    ) -> TruthCase:
        existing = self.session.get(CaseRow, truth_case.id)
        if existing is not None:
            self.session.execute(delete(CaseFactRow).where(CaseFactRow.case_id == truth_case.id))
            self.session.delete(existing)
            self.session.flush()

        row = CaseRow(
            id=truth_case.id,
            title=truth_case.title,
            review_status=truth_case.review_status.value,
            difficulty=truth_case.difficulty,
            specialty=truth_case.specialty,
            final_diagnosis=truth_case.final_diagnosis,
            truth_payload=dump_model(truth_case),
        )
        self.session.add(row)
        self.session.flush()

        embeddings = embeddings or {}
        for fact in truth_case.facts:
            fact_row = CaseFactRow(
                id=fact.id,
                case_id=truth_case.id,
                category=fact.category.value,
                label=fact.label,
                value=fact.value,
                spoiler=fact.spoiler,
                initially_visible=fact.initially_visible,
                payload=dump_model(fact),
            )
            self.session.add(fact_row)
            if fact.id in embeddings:
                self.session.add(
                    CaseFactEmbeddingRow(
                        fact_id=fact.id,
                        embedding_model="nomic-embed-text",
                        embedding=embeddings[fact.id],
                    )
                )

        self.session.commit()
        return truth_case

    def list_approved_cases(
        self, specialty: str | None = None, difficulty: str | None = None
    ) -> list[TruthCase]:
        stmt = select(CaseRow).where(CaseRow.review_status == ReviewStatus.APPROVED.value)
        if specialty is not None:
            stmt = stmt.where(CaseRow.specialty == specialty)
        if difficulty is not None:
            stmt = stmt.where(CaseRow.difficulty == difficulty)
        rows = self.session.scalars(stmt.order_by(CaseRow.created_at)).all()
        return [TruthCase.model_validate(row.truth_payload) for row in rows]

    def get_case(self, case_id: UUID) -> TruthCase:
        row = self.session.get(CaseRow, case_id)
        if row is None:
            raise KeyError(f"Case not found: {case_id}")
        return TruthCase.model_validate(row.truth_payload)

    def save_run(self, run_state: RunState) -> RunState:
        row = RunRow(
            id=run_state.id,
            case_id=run_state.case_id,
            status=run_state.status.value,
            stage=run_state.stage,
            payload=dump_model(run_state),
            score=run_state.score,
        )
        self.session.merge(row)
        self.session.commit()
        return run_state

    def get_run(self, run_id: UUID) -> RunState:
        row = self.session.get(RunRow, run_id)
        if row is None:
            raise KeyError(f"Run not found: {run_id}")
        return RunState.model_validate(row.payload)

    def append_turn(
        self, run_id: UUID, request_payload: dict, response: TurnResponse
    ) -> UUID:
        turn_count = len(self.turn_payloads(run_id)) + 1
        row = TurnRow(
            run_id=run_id,
            turn_index=turn_count,
            action_type=request_payload.get("action_type", "unknown"),
            request_payload=request_payload,
            response_payload=dump_model(response),
        )
        self.session.add(row)
        self.session.flush()
        self.session.commit()
        return row.id

    def turn_payloads(self, run_id: UUID) -> Sequence[TurnRow]:
        stmt = select(TurnRow).where(TurnRow.run_id == run_id).order_by(TurnRow.turn_index)
        return self.session.scalars(stmt).all()

    def list_turn_blocks(self, run_id: UUID) -> list[DisplayBlock]:
        blocks: list[DisplayBlock] = []
        for row in self.turn_payloads(run_id):
            blocks.extend(
                DisplayBlock.model_validate(block)
                for block in row.response_payload.get("display_blocks", [])
            )
        return blocks

    def log_validation(
        self, run_id: UUID, report: ValidationReport, turn_id: UUID | None = None
    ) -> None:
        self.session.add(
            ValidationLogRow(run_id=run_id, turn_id=turn_id, payload=dump_model(report))
        )
        self.session.commit()

    def save_score(self, run_id: UUID, score: ScoreSummary) -> None:
        self.session.merge(ScoreRow(run_id=run_id, payload=dump_model(score)))
        self.session.commit()

    def save_review(self, review: CaseReview) -> CaseReview:
        self.session.merge(ReviewRow(run_id=review.run_id, payload=dump_model(review)))
        self.session.commit()
        return review

    def get_review(self, run_id: UUID) -> CaseReview | None:
        stmt = select(ReviewRow).where(ReviewRow.run_id == run_id)
        row = self.session.scalars(stmt).first()
        if row is None:
            return None
        return CaseReview.model_validate(row.payload)
