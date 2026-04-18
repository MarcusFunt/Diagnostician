from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import random
from typing import Protocol
from uuid import UUID, uuid4

from diagnostician.core.windows_platform import disable_slow_wmi_platform_probe

disable_slow_wmi_platform_probe()

from sqlalchemy import delete, select
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from diagnostician.core.schemas import (
    CaseSummary,
    CaseReview,
    DisplayBlock,
    ReasoningStep,
    ReviewStatus,
    RunCreateRequest,
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
    return model.model_dump(mode="json", exclude_computed_fields=True)


class GameStore(Protocol):
    def save_source_document(self, source: SourceDocument) -> SourceDocument: ...

    def save_truth_case(
        self, truth_case: TruthCase, embeddings: dict[UUID, list[float]] | None = None
    ) -> TruthCase: ...

    def list_approved_cases(
        self, specialty: str | None = None, difficulty: str | None = None
    ) -> list[TruthCase]: ...

    def list_case_summaries(
        self,
        *,
        specialty: str | None = None,
        difficulty: str | None = None,
        q: str | None = None,
        limit: int = 24,
        cursor: str | None = None,
    ) -> tuple[list[CaseSummary], str | None, int]: ...

    def select_approved_case(self, request: RunCreateRequest) -> TruthCase | None: ...

    def get_case(self, case_id: UUID) -> TruthCase: ...

    def save_run(self, run_state: RunState) -> RunState: ...

    def get_run(self, run_id: UUID) -> RunState: ...

    def append_turn(
        self, run_id: UUID, request_payload: dict, response: TurnResponse
    ) -> UUID: ...

    def list_turn_blocks(self, run_id: UUID) -> list[DisplayBlock]: ...

    def list_turn_steps(self, run_id: UUID) -> list[ReasoningStep]: ...

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

    def list_case_summaries(
        self,
        *,
        specialty: str | None = None,
        difficulty: str | None = None,
        q: str | None = None,
        limit: int = 24,
        cursor: str | None = None,
    ) -> tuple[list[CaseSummary], str | None, int]:
        offset = _parse_cursor(cursor)
        limit = _bounded_limit(limit)
        approved = self.list_approved_cases(specialty=specialty, difficulty=difficulty)
        if q:
            query = q.casefold()
            approved = [
                case
                for case in approved
                if query in case.title.casefold()
                or query in case.chief_complaint.casefold()
                or any(query in tag.casefold() for tag in case.tags)
            ]
        total = len(approved)
        page = approved[offset : offset + limit]
        next_offset = offset + len(page)
        next_cursor = str(next_offset) if next_offset < total else None
        return [_case_summary(case) for case in page], next_cursor, total

    def select_approved_case(self, request: RunCreateRequest) -> TruthCase | None:
        if request.case_id is not None:
            truth_case = self.get_case(request.case_id)
            return truth_case if truth_case.approved_for_play else None

        cases = self.list_approved_cases(
            specialty=request.specialty,
            difficulty=request.difficulty,
        )
        if request.exclude_case_ids:
            excluded = set(request.exclude_case_ids)
            filtered = [case for case in cases if case.id not in excluded]
            if filtered:
                cases = filtered
        if not cases:
            return None
        if request.randomize:
            return random.choice(cases)
        return cases[0]

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

    def list_turn_steps(self, run_id: UUID) -> list[ReasoningStep]:
        return [
            _turn_step_from_payload(index, payload.get("request", {}), payload.get("response", {}))
            for index, payload in enumerate(self.turn_payloads.get(run_id, []), start=1)
        ]

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
        self.session.merge(row)
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

    def list_case_summaries(
        self,
        *,
        specialty: str | None = None,
        difficulty: str | None = None,
        q: str | None = None,
        limit: int = 24,
        cursor: str | None = None,
    ) -> tuple[list[CaseSummary], str | None, int]:
        offset = _parse_cursor(cursor)
        limit = _bounded_limit(limit)
        stmt = _approved_case_row_stmt(specialty=specialty, difficulty=difficulty, q=q)
        total = self.session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = self.session.scalars(stmt.order_by(CaseRow.created_at).offset(offset).limit(limit)).all()
        next_offset = offset + len(rows)
        next_cursor = str(next_offset) if next_offset < total else None
        return [_case_summary(TruthCase.model_validate(row.truth_payload)) for row in rows], next_cursor, total

    def select_approved_case(self, request: RunCreateRequest) -> TruthCase | None:
        if request.case_id is not None:
            truth_case = self.get_case(request.case_id)
            return truth_case if truth_case.approved_for_play else None

        stmt = _approved_case_row_stmt(
            specialty=request.specialty,
            difficulty=request.difficulty,
            q=None,
        )
        if request.exclude_case_ids:
            stmt = stmt.where(CaseRow.id.notin_(request.exclude_case_ids))
            if self.session.scalar(select(func.count()).select_from(stmt.subquery())) == 0:
                stmt = _approved_case_row_stmt(
                    specialty=request.specialty,
                    difficulty=request.difficulty,
                    q=None,
                )
        stmt = stmt.order_by(func.random() if request.randomize else CaseRow.created_at).limit(1)
        row = self.session.scalars(stmt).first()
        if row is None:
            return None
        return TruthCase.model_validate(row.truth_payload)

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

    def list_turn_steps(self, run_id: UUID) -> list[ReasoningStep]:
        return [
            _turn_step_from_payload(index, row.request_payload, row.response_payload)
            for index, row in enumerate(self.turn_payloads(run_id), start=1)
        ]

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


def _approved_case_row_stmt(
    *,
    specialty: str | None = None,
    difficulty: str | None = None,
    q: str | None = None,
):
    stmt = select(CaseRow).where(CaseRow.review_status == ReviewStatus.APPROVED.value)
    if specialty is not None:
        stmt = stmt.where(CaseRow.specialty == specialty)
    if difficulty is not None:
        stmt = stmt.where(CaseRow.difficulty == difficulty)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(
                CaseRow.title.ilike(like),
                CaseRow.specialty.ilike(like),
                CaseRow.difficulty.ilike(like),
            )
        )
    return stmt


def _case_summary(case: TruthCase) -> CaseSummary:
    diagnosis_terms = {_normalize_tag(term) for term in [case.final_diagnosis, *case.diagnosis_aliases]}
    safe_tags = [tag for tag in case.tags if _normalize_tag(tag) not in diagnosis_terms]
    return CaseSummary(
        id=case.id,
        title=case.title,
        chief_complaint=case.chief_complaint,
        difficulty=case.difficulty,
        specialty=case.specialty,
        tags=safe_tags,
        curation_notes=case.curation_notes,
        created_at=case.created_at,
    )


def _turn_step_from_payload(index: int, request_payload: dict, response_payload: dict) -> ReasoningStep:
    request = request_payload.get("request") if request_payload.get("action_type") == "submit_diagnosis" else request_payload
    request = request if isinstance(request, dict) else {}
    blocks = response_payload.get("display_blocks", [])
    revealed = response_payload.get("newly_revealed_facts", [])
    return ReasoningStep(
        turn_index=index,
        action_type=str(request_payload.get("action_type") or request.get("action_type") or "unknown"),
        target=request.get("target") if isinstance(request.get("target"), str) else None,
        player_text=str(request.get("player_text") or request.get("diagnosis") or ""),
        response_titles=[
            str(block.get("title"))
            for block in blocks
            if isinstance(block, dict) and block.get("title")
        ],
        revealed_fact_labels=[
            str(fact.get("label"))
            for fact in revealed
            if isinstance(fact, dict) and fact.get("label")
        ],
    )


def _parse_cursor(cursor: str | None) -> int:
    if cursor is None or cursor == "":
        return 0
    try:
        return max(0, int(cursor))
    except ValueError:
        return 0


def _bounded_limit(limit: int) -> int:
    return min(100, max(1, limit))


def _normalize_tag(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())
