from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from diagnostician.core.schemas import (
    ActionType,
    CaseFact,
    CaseReview,
    DiagnosisSubmission,
    DisplayBlock,
    DisplayBlockType,
    FactCategory,
    PlayerTurnRequest,
    RunCreateRequest,
    RunSnapshot,
    RunState,
    RunStatus,
    Severity,
    TruthCase,
    TurnResponse,
    ValidationReport,
    ValidationStatus,
    VisibleEvidence,
)
from diagnostician.llm.ollama_client import OllamaClient
from diagnostician.services.scoring import score_run
from diagnostician.services.store import GameStore, dump_model
from diagnostician.services.validation import validate_display_blocks

try:  # pragma: no cover - exercised when langgraph is installed
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - local environments can run deterministic path
    END = None
    StateGraph = None


@dataclass
class DiagnosticWorkflow:
    store: GameStore
    llm_client: OllamaClient | None = None

    def __post_init__(self) -> None:
        self.llm_client = self.llm_client or OllamaClient()
        self.graphs = build_workflow_graphs()

    def create_run(self, request: RunCreateRequest) -> TurnResponse:
        truth_case = self._select_case(request)
        policy = _require_policy(truth_case)
        initial_fact_ids = policy.initial_fact_ids
        run_state = RunState(case_id=truth_case.id, visible_fact_ids=initial_fact_ids, stage="history")
        visible_facts = _facts_by_ids(truth_case, initial_fact_ids)
        blocks = [
            DisplayBlock(
                type=DisplayBlockType.NARRATIVE,
                title="Case Presentation",
                body=_opening_text(truth_case, visible_facts),
                fact_ids=initial_fact_ids,
                provenance_ids=_provenance_ids(visible_facts),
            )
        ]
        validation = validate_display_blocks(truth_case, run_state, blocks, initial_fact_ids)
        if validation.status == ValidationStatus.FAIL:
            blocks, validation = _safe_fallback(
                truth_case,
                run_state,
                initial_fact_ids,
                title="Case Presentation",
                block_type=DisplayBlockType.NARRATIVE,
            )

        self.store.save_run(run_state)
        response = TurnResponse(
            run_state=run_state,
            display_blocks=blocks,
            newly_revealed_facts=visible_facts,
            visible_evidence=VisibleEvidence(facts=visible_facts),
            validation=validation,
        )
        turn_id = self.store.append_turn(
            run_state.id, {"action_type": "start_run", "request": dump_model(request)}, response
        )
        self.store.log_validation(run_state.id, validation, turn_id)
        return response

    def get_snapshot(self, run_id: UUID) -> RunSnapshot:
        run_state = self.store.get_run(run_id)
        truth_case = self.store.get_case(run_state.case_id)
        visible = _facts_by_ids(truth_case, run_state.visible_fact_ids)
        return RunSnapshot(
            run_state=run_state,
            visible_evidence=VisibleEvidence(facts=visible),
            display_blocks=self.store.list_turn_blocks(run_id),
        )

    def handle_turn(self, run_id: UUID, request: PlayerTurnRequest) -> TurnResponse:
        run_state = self.store.get_run(run_id)
        truth_case = self.store.get_case(run_state.case_id)
        if run_state.status != RunStatus.ACTIVE:
            return self._completed_run_response(truth_case, run_state)

        updated_state = run_state.model_copy(deep=True)
        updated_state.turn_count += 1
        updated_state.stage = _stage_for_action(request.action_type)
        _apply_action_state_updates(updated_state, request)

        allowed_new_facts = self._allowed_new_facts(truth_case, run_state, request)
        already_visible = _facts_by_ids(truth_case, run_state.visible_fact_ids)
        allowed_fact_ids = [*run_state.visible_fact_ids, *(fact.id for fact in allowed_new_facts)]
        blocks = self._generate_turn_blocks(truth_case, run_state, request, allowed_new_facts)
        validation = validate_display_blocks(truth_case, run_state, blocks, allowed_fact_ids)

        if validation.status == ValidationStatus.FAIL:
            blocks, validation = _safe_fallback(
                truth_case,
                run_state,
                run_state.visible_fact_ids,
                title="No New Information",
                block_type=DisplayBlockType.SYSTEM_STATUS,
            )
            allowed_new_facts = []
        else:
            updated_state.visible_fact_ids = _unique_ids(
                [*run_state.visible_fact_ids, *validation.revealed_fact_ids]
            )

        self.store.save_run(updated_state)
        visible = _facts_by_ids(truth_case, updated_state.visible_fact_ids)
        newly_revealed = [fact for fact in allowed_new_facts if fact.id in validation.revealed_fact_ids]
        response = TurnResponse(
            run_state=updated_state,
            display_blocks=blocks,
            newly_revealed_facts=newly_revealed,
            visible_evidence=VisibleEvidence(facts=visible),
            validation=validation,
        )
        turn_id = self.store.append_turn(run_id, dump_model(request), response)
        self.store.log_validation(run_id, validation, turn_id)
        return response

    def submit_diagnosis(self, run_id: UUID, submission: DiagnosisSubmission) -> CaseReview:
        run_state = self.store.get_run(run_id)
        truth_case = self.store.get_case(run_state.case_id)
        if run_state.status == RunStatus.COMPLETE:
            existing = self.store.get_review(run_id)
            if existing is not None:
                return existing

        completed_state = run_state.model_copy(deep=True)
        completed_state.status = RunStatus.COMPLETE
        completed_state.stage = "review"
        score = score_run(truth_case, completed_state, submission.diagnosis)
        completed_state.score = score.final_score

        spoiler_facts = [fact for fact in truth_case.facts if fact.spoiler]
        completed_state.visible_fact_ids = _unique_ids(
            [*completed_state.visible_fact_ids, *(fact.id for fact in spoiler_facts)]
        )
        self.store.save_run(completed_state)
        self.store.save_score(run_id, score)

        diagnosis_block = DisplayBlock(
            type=DisplayBlockType.ATTENDING_COMMENT,
            title="Final Diagnosis",
            body=f"The source-grounded final diagnosis is {truth_case.final_diagnosis}.",
            fact_ids=[fact.id for fact in spoiler_facts],
            provenance_ids=_provenance_ids(spoiler_facts),
            severity=Severity.SUCCESS if score.correct else Severity.WARNING,
        )
        validation = validate_display_blocks(
            truth_case,
            completed_state,
            [diagnosis_block],
            completed_state.visible_fact_ids,
        )
        response = TurnResponse(
            run_state=completed_state,
            display_blocks=[diagnosis_block],
            newly_revealed_facts=spoiler_facts,
            visible_evidence=VisibleEvidence(facts=_facts_by_ids(truth_case, completed_state.visible_fact_ids)),
            validation=validation,
        )
        turn_id = self.store.append_turn(
            run_id,
            {"action_type": "submit_diagnosis", "request": dump_model(submission)},
            response,
        )
        self.store.log_validation(run_id, validation, turn_id)

        review = CaseReview(
            run_id=run_id,
            case_id=truth_case.id,
            diagnosis=truth_case.final_diagnosis,
            player_score=score,
            key_findings=_key_findings(truth_case),
            teaching_points=truth_case.teaching_points,
            provenance=truth_case.provenance,
            turn_timeline=self.store.list_turn_blocks(run_id),
        )
        return self.store.save_review(review)

    def get_review(self, run_id: UUID) -> CaseReview:
        review = self.store.get_review(run_id)
        if review is None:
            raise KeyError(f"Review not found for run {run_id}")
        return review

    def _select_case(self, request: RunCreateRequest) -> TruthCase:
        if request.case_id is not None:
            truth_case = self.store.get_case(request.case_id)
            if not truth_case.approved_for_play:
                raise ValueError("Requested case is not approved for play.")
            return truth_case

        cases = self.store.list_approved_cases(
            specialty=request.specialty,
            difficulty=request.difficulty,
        )
        if not cases:
            raise ValueError("No approved playable cases are available.")
        return cases[0]

    def _allowed_new_facts(
        self, truth_case: TruthCase, run_state: RunState, request: PlayerTurnRequest
    ) -> list[CaseFact]:
        policy = _require_policy(truth_case)
        categories = set(policy.action_category_map.get(request.action_type, []))
        visible = set(run_state.visible_fact_ids)
        candidates = [
            fact
            for fact in truth_case.facts
            if fact.id not in visible
            and fact.category in categories
            and not fact.spoiler
            and (not fact.reveal_actions or request.action_type in fact.reveal_actions)
        ]
        ranked = sorted(
            candidates,
            key=lambda fact: _match_score(fact, request),
            reverse=True,
        )
        ranked = [fact for fact in ranked if _match_score(fact, request) > 0] or ranked
        return ranked[: policy.max_facts_per_turn]

    def _generate_turn_blocks(
        self,
        truth_case: TruthCase,
        run_state: RunState,
        request: PlayerTurnRequest,
        facts: list[CaseFact],
    ) -> list[DisplayBlock]:
        if not facts:
            return [
                DisplayBlock(
                    type=DisplayBlockType.SYSTEM_STATUS,
                    title="No New Information",
                    body=_no_new_information_text(request),
                    severity=Severity.INFO,
                )
            ]

        fallback_body = _facts_to_response_text(request, facts)
        generated = self.llm_client.generate(
            prompt=_turn_prompt(truth_case, run_state, request, facts),
            system=(
                "You rephrase only the provided visible medical facts for a diagnostic game. "
                "Do not add diagnoses or facts that are not listed."
            ),
        )
        body = generated.text.strip() if generated.text.strip() else fallback_body
        block_type = _block_type_for_action(request.action_type, facts)
        return [
            DisplayBlock(
                type=block_type,
                title=_title_for_action(request.action_type),
                body=body,
                fact_ids=[fact.id for fact in facts],
                provenance_ids=_provenance_ids(facts),
                severity=Severity.INFO,
            )
        ]

    def _completed_run_response(self, truth_case: TruthCase, run_state: RunState) -> TurnResponse:
        block = DisplayBlock(
            type=DisplayBlockType.SYSTEM_STATUS,
            title="Run Complete",
            body="This run is complete. Open the case review to inspect the source-grounded explanation.",
            severity=Severity.WARNING,
        )
        validation = validate_display_blocks(truth_case, run_state, [block], run_state.visible_fact_ids)
        return TurnResponse(
            run_state=run_state,
            display_blocks=[block],
            newly_revealed_facts=[],
            visible_evidence=VisibleEvidence(facts=_facts_by_ids(truth_case, run_state.visible_fact_ids)),
            validation=validation,
        )


def build_workflow_graphs() -> dict[str, object]:
    if StateGraph is None:
        return {}

    def passthrough(state: dict) -> dict:
        return state

    graphs: dict[str, object] = {}
    for name, nodes in {
        "start_run": ["load_case", "initialize_state", "generate_opening", "validate", "persist"],
        "turn": ["classify_action", "compute_allowed_facts", "generate_response", "validate", "persist"],
        "diagnosis": ["score", "generate_review", "persist"],
    }.items():
        graph = StateGraph(dict)
        previous = None
        for node in nodes:
            graph.add_node(node, passthrough)
            if previous is None:
                graph.set_entry_point(node)
            else:
                graph.add_edge(previous, node)
            previous = node
        graph.add_edge(previous, END)
        graphs[name] = graph.compile()
    return graphs


def _require_policy(truth_case: TruthCase):
    if truth_case.reveal_policy is None:
        raise ValueError("Truth case is missing reveal policy.")
    return truth_case.reveal_policy


def _opening_text(truth_case: TruthCase, visible_facts: list[CaseFact]) -> str:
    demographics = ", ".join(f"{key}: {value}" for key, value in truth_case.demographics.items())
    facts = " ".join(f"{fact.label}: {fact.value}" for fact in visible_facts)
    prefix = f"{demographics}. " if demographics else ""
    return f"{prefix}Chief concern: {truth_case.chief_complaint}. {facts}".strip()


def _facts_by_ids(truth_case: TruthCase, fact_ids: list[UUID]) -> list[CaseFact]:
    fact_map = {fact.id: fact for fact in truth_case.facts}
    return [fact_map[fact_id] for fact_id in fact_ids if fact_id in fact_map]


def _provenance_ids(facts: list[CaseFact]) -> list[UUID]:
    return _unique_ids([prov_id for fact in facts for prov_id in fact.provenance_ids])


def _unique_ids(ids: list[UUID]) -> list[UUID]:
    seen: set[UUID] = set()
    ordered: list[UUID] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _safe_fallback(
    truth_case: TruthCase,
    run_state: RunState,
    allowed_fact_ids: list[UUID],
    title: str,
    block_type: DisplayBlockType,
) -> tuple[list[DisplayBlock], ValidationReport]:
    facts = [fact for fact in _facts_by_ids(truth_case, allowed_fact_ids) if not fact.spoiler]
    body = _facts_to_plain_text(facts) if facts else "No additional source-grounded information is available for that request."
    blocks = [
        DisplayBlock(
            type=block_type,
            title=title,
            body=body,
            fact_ids=[fact.id for fact in facts],
            provenance_ids=_provenance_ids(facts),
            severity=Severity.INFO,
        )
    ]
    report = validate_display_blocks(truth_case, run_state, blocks, allowed_fact_ids)
    return blocks, report.model_copy(
        update={"status": ValidationStatus.FALLBACK_USED, "fallback_used": True}
    )


def _facts_to_plain_text(facts: list[CaseFact]) -> str:
    return " ".join(f"{fact.label}: {fact.value}" for fact in facts)


def _apply_action_state_updates(run_state: RunState, request: PlayerTurnRequest) -> None:
    if request.action_type in {ActionType.ORDER_LAB, ActionType.ORDER_IMAGING, ActionType.REQUEST_PATHOLOGY_DETAIL}:
        ordered = request.target or request.player_text or request.action_type.value
        run_state.ordered_tests.append(ordered)
    if request.action_type == ActionType.REQUEST_HINT:
        run_state.hint_count += 1
    if request.action_type == ActionType.SUBMIT_DIFFERENTIAL and request.player_text:
        run_state.submitted_differentials.extend(_split_differential(request.player_text))
    if request.player_text:
        run_state.requested_clues.append(request.player_text)


def _split_differential(text: str) -> list[str]:
    normalized = text.replace("\n", ",").replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _stage_for_action(action_type: ActionType) -> str:
    if action_type == ActionType.SUBMIT_DIFFERENTIAL:
        return "differential"
    if action_type == ActionType.REQUEST_HINT:
        return "hint"
    return "investigation"


def _match_score(fact: CaseFact, request: PlayerTurnRequest) -> int:
    query = f"{request.target or ''} {request.player_text or ''}".casefold()
    score = 1
    if fact.label.casefold() in query:
        score += 5
    for tag in fact.tags:
        if tag.casefold() in query:
            score += 3
    if fact.category.value.replace("_", " ") in query:
        score += 2
    return score


def _facts_to_response_text(request: PlayerTurnRequest, facts: list[CaseFact]) -> str:
    if request.action_type == ActionType.ASK_PATIENT_QUESTION:
        return " ".join(f"{fact.value}" for fact in facts)
    return _facts_to_plain_text(facts)


def _turn_prompt(
    truth_case: TruthCase,
    run_state: RunState,
    request: PlayerTurnRequest,
    facts: list[CaseFact],
) -> str:
    fact_lines = "\n".join(f"- {fact.label}: {fact.value}" for fact in facts)
    return (
        f"Case title: {truth_case.title}\n"
        f"Run stage: {run_state.stage}\n"
        f"Player action: {request.action_type.value}\n"
        f"Player text: {request.player_text}\n"
        f"Allowed facts:\n{fact_lines}\n"
        "Return a concise in-world response using only those facts."
    )


def _block_type_for_action(action_type: ActionType, facts: list[CaseFact]) -> DisplayBlockType:
    categories = {fact.category for fact in facts}
    if FactCategory.LAB in categories or FactCategory.MICROBIOLOGY in categories:
        return DisplayBlockType.LAB_RESULT
    if FactCategory.IMAGING in categories:
        return DisplayBlockType.IMAGING_REPORT
    if FactCategory.PATHOLOGY in categories:
        return DisplayBlockType.PATHOLOGY_REPORT
    if action_type == ActionType.ASK_PATIENT_QUESTION:
        return DisplayBlockType.PATIENT_DIALOGUE
    if action_type == ActionType.REQUEST_HINT:
        return DisplayBlockType.HINT
    return DisplayBlockType.ATTENDING_COMMENT


def _title_for_action(action_type: ActionType) -> str:
    return {
        ActionType.ASK_PATIENT_QUESTION: "Patient Response",
        ActionType.REQUEST_EXAM_DETAIL: "Exam Findings",
        ActionType.ORDER_LAB: "Lab Results",
        ActionType.ORDER_IMAGING: "Imaging Report",
        ActionType.REQUEST_PATHOLOGY_DETAIL: "Pathology Detail",
        ActionType.SUBMIT_DIFFERENTIAL: "Attending Feedback",
        ActionType.REQUEST_HINT: "Hint",
    }[action_type]


def _no_new_information_text(request: PlayerTurnRequest) -> str:
    if request.action_type == ActionType.SUBMIT_DIFFERENTIAL:
        return "Your differential has been recorded. No new case facts were revealed."
    return "No additional source-grounded information is available for that request yet."


def _key_findings(truth_case: TruthCase) -> list[CaseFact]:
    priority = {
        FactCategory.IMAGING,
        FactCategory.LAB,
        FactCategory.PHYSICAL_EXAM,
        FactCategory.SYMPTOM,
        FactCategory.DIAGNOSIS,
    }
    findings = [fact for fact in truth_case.facts if fact.category in priority]
    return findings[:8]
