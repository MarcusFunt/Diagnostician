from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
from typing import Any
from uuid import UUID

from diagnostician.core.config import get_settings
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
    SoftAuditScore,
    TruthCase,
    TurnResponse,
    ValidationReport,
    ValidationStatus,
    VisibleEvidence,
    utcnow,
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
        self.graphs = build_workflow_graphs(self)

    def create_run(self, request: RunCreateRequest) -> TurnResponse:
        if "start_run" not in self.graphs:
            return self._create_run_without_graph(request)
        state = self.graphs["start_run"].invoke({"request": request, "attempt": 0})
        return state["response"]

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
        if "turn" not in self.graphs:
            return self._handle_turn_without_graph(run_id, request, truth_case, run_state)
        state = self.graphs["turn"].invoke(
            {
                "run_id": run_id,
                "request": request,
                "run_state": run_state,
                "truth_case": truth_case,
                "attempt": 0,
            }
        )
        return state["response"]

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

    def start_load_case(self, state: dict[str, Any]) -> dict[str, Any]:
        truth_case = self._select_case(state["request"])
        return _state_with(state, truth_case=truth_case)

    def start_initialize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        truth_case = state["truth_case"]
        policy = _require_policy(truth_case)
        initial_fact_ids = policy.initial_fact_ids
        visible_facts = _facts_by_ids(truth_case, initial_fact_ids)
        run_state = RunState(
            case_id=truth_case.id,
            visible_fact_ids=initial_fact_ids,
            stage="history",
        )
        return _state_with(
            state,
            run_state=run_state,
            visible_facts=visible_facts,
            allowed_fact_ids=initial_fact_ids,
            review_feedback=[],
        )

    def start_generate_story(self, state: dict[str, Any]) -> dict[str, Any]:
        truth_case = state["truth_case"]
        run_state = state["run_state"]
        visible_facts = state["visible_facts"]
        fallback_body = _opening_text(truth_case, visible_facts)
        generated = self._generate_case_text(
            prompt=_story_prompt(
                truth_case,
                run_state,
                visible_facts,
                state.get("review_feedback", []),
            ),
            system=_case_generator_system_prompt(),
        )
        body = generated.text.strip() if generated.text.strip() else fallback_body
        blocks = [
            DisplayBlock(
                type=DisplayBlockType.NARRATIVE,
                title="Case Presentation",
                body=body,
                fact_ids=[fact.id for fact in visible_facts],
                provenance_ids=_provenance_ids(visible_facts),
                severity=Severity.INFO,
            )
        ]
        return _state_with(state, blocks=blocks, generation=generated)

    def start_audit_story(self, state: dict[str, Any]) -> dict[str, Any]:
        audit = self._audit_blocks(
            truth_case=state["truth_case"],
            run_state=state["run_state"],
            blocks=state["blocks"],
            allowed_fact_ids=state["allowed_fact_ids"],
            purpose="opening case story",
        )
        return _state_with(state, medical_audit=audit)

    def start_validate_story(self, state: dict[str, Any]) -> dict[str, Any]:
        validation = validate_display_blocks(
            state["truth_case"],
            state["run_state"],
            state["blocks"],
            state["allowed_fact_ids"],
        )
        validation = _apply_medical_audit(validation, state["medical_audit"])
        return _state_with(state, validation=validation, review_feedback=_validation_feedback(validation))

    def start_prepare_retry(self, state: dict[str, Any]) -> dict[str, Any]:
        return _state_with(state, attempt=state.get("attempt", 0) + 1)

    def start_use_fallback(self, state: dict[str, Any]) -> dict[str, Any]:
        truth_case = state["truth_case"]
        run_state = state["run_state"]
        blocks, validation = _safe_fallback(
            truth_case,
            run_state,
            state["allowed_fact_ids"],
            title="Case Presentation",
            block_type=DisplayBlockType.NARRATIVE,
        )
        return _state_with(state, blocks=blocks, validation=validation)

    def start_persist(self, state: dict[str, Any]) -> dict[str, Any]:
        truth_case = state["truth_case"]
        run_state = state["run_state"].model_copy(deep=True)
        validation = state["validation"]
        blocks = state["blocks"]
        visible_facts = state["visible_facts"]
        run_state.case_story = blocks[0].body if blocks else ""
        run_state.story_fact_ids = _unique_ids(state["allowed_fact_ids"])
        run_state.run_summary = _build_run_summary(run_state, visible_facts, blocks)
        run_state.updated_at = utcnow()

        self.store.save_run(run_state)
        response = TurnResponse(
            run_state=run_state,
            display_blocks=blocks,
            newly_revealed_facts=visible_facts,
            visible_evidence=VisibleEvidence(facts=visible_facts),
            validation=validation,
        )
        turn_id = self.store.append_turn(
            run_state.id,
            {"action_type": "start_run", "request": dump_model(state["request"])},
            response,
        )
        self.store.log_validation(run_state.id, validation, turn_id)
        return _state_with(state, run_state=run_state, response=response)

    def turn_prepare_state(self, state: dict[str, Any]) -> dict[str, Any]:
        run_state = state["run_state"]
        request = state["request"]
        updated_state = run_state.model_copy(deep=True)
        updated_state.turn_count += 1
        updated_state.stage = _stage_for_action(request.action_type)
        updated_state.updated_at = utcnow()
        _apply_action_state_updates(updated_state, request)
        prior_blocks = self.store.list_turn_blocks(state["run_id"])
        return _state_with(state, updated_state=updated_state, prior_blocks=prior_blocks, review_feedback=[])

    def turn_compute_allowed_facts(self, state: dict[str, Any]) -> dict[str, Any]:
        truth_case = state["truth_case"]
        run_state = state["run_state"]
        request = state["request"]
        allowed_new_facts = self._allowed_new_facts(truth_case, run_state, request)
        allowed_fact_ids = [*run_state.visible_fact_ids, *(fact.id for fact in allowed_new_facts)]
        return _state_with(state, allowed_new_facts=allowed_new_facts, allowed_fact_ids=allowed_fact_ids)

    def turn_generate_response(self, state: dict[str, Any]) -> dict[str, Any]:
        blocks = self._generate_turn_blocks(
            truth_case=state["truth_case"],
            run_state=state["updated_state"],
            request=state["request"],
            facts=state["allowed_new_facts"],
            prior_blocks=state["prior_blocks"],
            feedback=state.get("review_feedback", []),
        )
        return _state_with(state, blocks=blocks)

    def turn_audit_response(self, state: dict[str, Any]) -> dict[str, Any]:
        if not state["allowed_new_facts"]:
            return _state_with(state, medical_audit=_approved_audit("no new generated medical facts"))
        audit = self._audit_blocks(
            truth_case=state["truth_case"],
            run_state=state["updated_state"],
            blocks=state["blocks"],
            allowed_fact_ids=state["allowed_fact_ids"],
            purpose="interactive case turn",
        )
        return _state_with(state, medical_audit=audit)

    def turn_validate_response(self, state: dict[str, Any]) -> dict[str, Any]:
        validation = validate_display_blocks(
            state["truth_case"],
            state["updated_state"],
            state["blocks"],
            state["allowed_fact_ids"],
        )
        validation = _apply_medical_audit(validation, state["medical_audit"])
        return _state_with(state, validation=validation, review_feedback=_validation_feedback(validation))

    def turn_prepare_retry(self, state: dict[str, Any]) -> dict[str, Any]:
        return _state_with(state, attempt=state.get("attempt", 0) + 1)

    def turn_use_fallback(self, state: dict[str, Any]) -> dict[str, Any]:
        blocks, validation = _safe_fallback(
            state["truth_case"],
            state["updated_state"],
            state["run_state"].visible_fact_ids,
            title="No New Information",
            block_type=DisplayBlockType.SYSTEM_STATUS,
        )
        return _state_with(state, blocks=blocks, validation=validation, allowed_new_facts=[])

    def turn_persist(self, state: dict[str, Any]) -> dict[str, Any]:
        truth_case = state["truth_case"]
        updated_state = state["updated_state"].model_copy(deep=True)
        validation = state["validation"]
        if validation.status != ValidationStatus.FALLBACK_USED:
            updated_state.visible_fact_ids = _unique_ids(
                [*state["run_state"].visible_fact_ids, *validation.revealed_fact_ids]
            )
        visible = _facts_by_ids(truth_case, updated_state.visible_fact_ids)
        updated_state.run_summary = _build_run_summary(
            updated_state,
            visible,
            state["blocks"],
            request=state["request"],
        )
        updated_state.updated_at = utcnow()

        self.store.save_run(updated_state)
        newly_revealed = [
            fact for fact in state["allowed_new_facts"] if fact.id in validation.revealed_fact_ids
        ]
        response = TurnResponse(
            run_state=updated_state,
            display_blocks=state["blocks"],
            newly_revealed_facts=newly_revealed,
            visible_evidence=VisibleEvidence(facts=visible),
            validation=validation,
        )
        turn_id = self.store.append_turn(state["run_id"], dump_model(state["request"]), response)
        self.store.log_validation(state["run_id"], validation, turn_id)
        return _state_with(state, updated_state=updated_state, response=response)

    def _create_run_without_graph(self, request: RunCreateRequest) -> TurnResponse:
        truth_case = self._select_case(request)
        policy = _require_policy(truth_case)
        initial_fact_ids = policy.initial_fact_ids
        run_state = RunState(case_id=truth_case.id, visible_fact_ids=initial_fact_ids, stage="history")
        visible_facts = _facts_by_ids(truth_case, initial_fact_ids)
        blocks, validation = _safe_fallback(
            truth_case,
            run_state,
            initial_fact_ids,
            title="Case Presentation",
            block_type=DisplayBlockType.NARRATIVE,
        )
        run_state.case_story = blocks[0].body
        run_state.story_fact_ids = initial_fact_ids
        run_state.run_summary = _build_run_summary(run_state, visible_facts, blocks)
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

    def _handle_turn_without_graph(
        self,
        run_id: UUID,
        request: PlayerTurnRequest,
        truth_case: TruthCase,
        run_state: RunState,
    ) -> TurnResponse:
        updated_state = run_state.model_copy(deep=True)
        updated_state.turn_count += 1
        updated_state.stage = _stage_for_action(request.action_type)
        updated_state.updated_at = utcnow()
        _apply_action_state_updates(updated_state, request)
        allowed_new_facts = self._allowed_new_facts(truth_case, run_state, request)
        allowed_fact_ids = [*run_state.visible_fact_ids, *(fact.id for fact in allowed_new_facts)]
        blocks = self._generate_turn_blocks(
            truth_case,
            updated_state,
            request,
            allowed_new_facts,
            self.store.list_turn_blocks(run_id),
            [],
        )
        validation = validate_display_blocks(truth_case, updated_state, blocks, allowed_fact_ids)
        if validation.status == ValidationStatus.FAIL:
            blocks, validation = _safe_fallback(
                truth_case,
                updated_state,
                run_state.visible_fact_ids,
                title="No New Information",
                block_type=DisplayBlockType.SYSTEM_STATUS,
            )
            allowed_new_facts = []
        else:
            updated_state.visible_fact_ids = _unique_ids(
                [*run_state.visible_fact_ids, *validation.revealed_fact_ids]
            )
        visible = _facts_by_ids(truth_case, updated_state.visible_fact_ids)
        updated_state.run_summary = _build_run_summary(updated_state, visible, blocks, request=request)
        self.store.save_run(updated_state)
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
        if request.exclude_case_ids:
            excluded = set(request.exclude_case_ids)
            filtered = [case for case in cases if case.id not in excluded]
            if filtered:
                cases = filtered
        if not cases:
            raise ValueError("No approved playable cases are available.")
        if request.randomize:
            return random.choice(cases)
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
        matched = [fact for fact in ranked if _match_score(fact, request) > 0]
        if not matched and request.action_type == ActionType.REQUEST_HINT:
            matched = ranked
        return matched[: policy.max_facts_per_turn]

    def _generate_turn_blocks(
        self,
        truth_case: TruthCase,
        run_state: RunState,
        request: PlayerTurnRequest,
        facts: list[CaseFact],
        prior_blocks: list[DisplayBlock],
        feedback: list[str],
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
        generated = self._generate_case_text(
            prompt=_turn_prompt(truth_case, run_state, request, facts, prior_blocks, feedback),
            system=_case_generator_system_prompt(),
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

    def _generate_case_text(self, prompt: str, system: str) -> Any:
        settings = getattr(self.llm_client, "settings", None) or get_settings()
        try:
            return self.llm_client.generate(
                prompt=_disable_qwen_thinking(prompt, settings.case_generator_model),
                system=system,
                model=settings.case_generator_model,
                options={"temperature": 0.35, "top_p": 0.9, "num_predict": 180},
            )
        except TypeError:
            return self.llm_client.generate(prompt=prompt, system=system)

    def _audit_blocks(
        self,
        *,
        truth_case: TruthCase,
        run_state: RunState,
        blocks: list[DisplayBlock],
        allowed_fact_ids: list[UUID],
        purpose: str,
    ) -> dict[str, Any]:
        settings = getattr(self.llm_client, "settings", None) or get_settings()
        if not settings.medical_check_enabled:
            return _approved_audit("medical audit disabled for limited hardware; deterministic validation still applied")
        prompt = _medical_audit_prompt(truth_case, run_state, blocks, allowed_fact_ids, purpose)
        system = (
            "You are a medical consistency and spoiler-safety auditor for an educational "
            "diagnostic game. Return only the requested structured audit. Do not repeat the prompt."
        )
        try:
            med42_plain_text = isinstance(self.llm_client, OllamaClient) and "med42" in settings.medical_check_model.lower()
            if med42_plain_text:
                generated = self.llm_client.generate(
                    prompt=prompt,
                    system=system,
                    model=settings.medical_check_model,
                    options={"temperature": 0.0, "num_predict": 128},
                )
            elif hasattr(self.llm_client, "generate_json"):
                generated = self.llm_client.generate_json(
                    prompt=prompt,
                    system=system,
                    model=settings.medical_check_model,
                    options={"temperature": 0.0, "num_predict": 128},
                )
            else:
                generated = self.llm_client.generate(
                    prompt=prompt,
                    system=system,
                    model=settings.medical_check_model,
                    format="json",
                    options={"temperature": 0.0, "num_predict": 128},
                )
        except TypeError:
            generated = self.llm_client.generate(prompt=prompt, system=system)

        if getattr(generated, "fallback_used", False) or not getattr(generated, "text", "").strip():
            return _approved_audit("medical audit unavailable; deterministic validation still applied")

        data = _extract_audit_data(generated.text)
        if data is None:
            return {
                **_approved_audit("medical audit returned unparsable JSON"),
                "approved": False,
                "notes": ["medical audit returned unparsable JSON"],
            }
        return _normalize_medical_audit(data)

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


def build_workflow_graphs(workflow: DiagnosticWorkflow | None = None) -> dict[str, object]:
    if StateGraph is None or END is None or workflow is None:
        return {}

    def route_validation(state: dict[str, Any]) -> str:
        validation = state["validation"]
        if validation.status == ValidationStatus.PASS:
            return "persist"
        settings = getattr(workflow.llm_client, "settings", None) or get_settings()
        if state.get("attempt", 0) < settings.generation_repair_attempts:
            return "retry"
        return "fallback"

    start_graph = StateGraph(dict)
    start_graph.add_node("load_case", workflow.start_load_case)
    start_graph.add_node("initialize_state", workflow.start_initialize_state)
    start_graph.add_node("generate_story", workflow.start_generate_story)
    start_graph.add_node("audit_story", workflow.start_audit_story)
    start_graph.add_node("validate_story", workflow.start_validate_story)
    start_graph.add_node("prepare_retry", workflow.start_prepare_retry)
    start_graph.add_node("use_fallback", workflow.start_use_fallback)
    start_graph.add_node("persist", workflow.start_persist)
    start_graph.set_entry_point("load_case")
    start_graph.add_edge("load_case", "initialize_state")
    start_graph.add_edge("initialize_state", "generate_story")
    start_graph.add_edge("generate_story", "audit_story")
    start_graph.add_edge("audit_story", "validate_story")
    start_graph.add_conditional_edges(
        "validate_story",
        route_validation,
        {"persist": "persist", "retry": "prepare_retry", "fallback": "use_fallback"},
    )
    start_graph.add_edge("prepare_retry", "generate_story")
    start_graph.add_edge("use_fallback", "persist")
    start_graph.add_edge("persist", END)

    turn_graph = StateGraph(dict)
    turn_graph.add_node("prepare_state", workflow.turn_prepare_state)
    turn_graph.add_node("compute_allowed_facts", workflow.turn_compute_allowed_facts)
    turn_graph.add_node("generate_response", workflow.turn_generate_response)
    turn_graph.add_node("audit_response", workflow.turn_audit_response)
    turn_graph.add_node("validate_response", workflow.turn_validate_response)
    turn_graph.add_node("prepare_retry", workflow.turn_prepare_retry)
    turn_graph.add_node("use_fallback", workflow.turn_use_fallback)
    turn_graph.add_node("persist", workflow.turn_persist)
    turn_graph.set_entry_point("prepare_state")
    turn_graph.add_edge("prepare_state", "compute_allowed_facts")
    turn_graph.add_edge("compute_allowed_facts", "generate_response")
    turn_graph.add_edge("generate_response", "audit_response")
    turn_graph.add_edge("audit_response", "validate_response")
    turn_graph.add_conditional_edges(
        "validate_response",
        route_validation,
        {"persist": "persist", "retry": "prepare_retry", "fallback": "use_fallback"},
    )
    turn_graph.add_edge("prepare_retry", "generate_response")
    turn_graph.add_edge("use_fallback", "persist")
    turn_graph.add_edge("persist", END)

    return {"start_run": start_graph.compile(), "turn": turn_graph.compile()}


def _state_with(state: dict[str, Any], **updates: Any) -> dict[str, Any]:
    next_state = dict(state)
    next_state.update(updates)
    return next_state


def _dump_prompt(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), default=str)


def _case_generator_system_prompt() -> str:
    return (
        "You write concise, in-world responses for a diagnostic reasoning game. "
        "Use only visible or explicitly allowed facts. Do not infer, name, hint at, "
        "or disclose the final diagnosis. Return prose only, with no markdown, JSON, "
        "tool output, or hidden reasoning."
    )


def _disable_qwen_thinking(prompt: str, model: str) -> str:
    normalized_model = model.lower()
    if (
        normalized_model.startswith("qwen3")
        and "instruct" not in normalized_model
        and not prompt.lstrip().startswith("/no_think")
    ):
        return f"/no_think\n{prompt}"
    return prompt


def _story_prompt(
    truth_case: TruthCase,
    run_state: RunState,
    visible_facts: list[CaseFact],
    feedback: list[str],
) -> str:
    prompt = {
        "task": (
            "Write the opening story for this run. Present the patient naturally at the bedside. "
            "Do not reveal tests, imaging, diagnoses, or hidden findings that are not initially visible."
        ),
        "case_metadata": _safe_case_metadata(truth_case),
        "run_summary": run_state.run_summary,
        "visible_facts": [_fact_payload(fact) for fact in visible_facts],
        "safe_case_packet": _safe_case_packet(truth_case, [fact.id for fact in visible_facts]),
        "repair_feedback": feedback,
        "output_rules": [
            "Return 2-4 sentences under 90 words.",
            "Do not mention the final diagnosis or diagnosis aliases.",
            "Do not include JSON, markdown bullets, or chain-of-thought.",
        ],
    }
    return _dump_prompt(prompt)


def _turn_prompt(
    truth_case: TruthCase,
    run_state: RunState,
    request: PlayerTurnRequest,
    facts: list[CaseFact],
    prior_blocks: list[DisplayBlock],
    feedback: list[str],
) -> str:
    allowed_ids = _unique_ids([*run_state.visible_fact_ids, *(fact.id for fact in facts)])
    prompt = {
        "task": (
            "Generate the next response in this diagnostic case. Answer the user action with only "
            "the allowed new facts and already-visible context."
        ),
        "case_story": run_state.case_story,
        "user_action": {
            "action_type": request.action_type.value,
            "target": request.target,
            "player_text": request.player_text,
        },
        "case_metadata": _safe_case_metadata(truth_case),
        "run_summary": run_state.run_summary,
        "prior_accepted_blocks": [_block_payload(block) for block in prior_blocks[-8:]],
        "allowed_new_facts": [_fact_payload(fact) for fact in facts],
        "safe_case_packet": _safe_case_packet(truth_case, allowed_ids),
        "repair_feedback": feedback,
        "output_rules": [
            "Return one concise in-world response under 80 words.",
            "Use allowed facts only; do not add unlisted findings.",
            "Do not disclose the final diagnosis or hidden future facts.",
            "Do not include JSON, markdown tables, or chain-of-thought.",
        ],
    }
    return _dump_prompt(prompt)


def _safe_case_metadata(truth_case: TruthCase) -> dict[str, Any]:
    diagnosis_terms = {_normalize_tag(term) for term in [truth_case.final_diagnosis, *truth_case.diagnosis_aliases]}
    return {
        "case_id": str(truth_case.id),
        "title": truth_case.title,
        "demographics": truth_case.demographics,
        "chief_complaint": truth_case.chief_complaint,
        "difficulty": truth_case.difficulty,
        "specialty": truth_case.specialty,
        "tags": [tag for tag in truth_case.tags if _normalize_tag(tag) not in diagnosis_terms],
    }


def _safe_case_packet(truth_case: TruthCase, full_value_fact_ids: list[UUID]) -> dict[str, Any]:
    full_value_ids = set(full_value_fact_ids)
    facts = []
    for fact in truth_case.facts:
        if fact.spoiler:
            continue
        payload = {
            "category": fact.category.value,
            "label": fact.label,
            "initially_visible": fact.initially_visible,
            "value": fact.value if fact.id in full_value_ids else "<hidden until requested>",
        }
        if fact.id in full_value_ids:
            payload["tags"] = fact.tags
        facts.append(payload)
    return {"non_spoiler_facts": facts}


def _fact_payload(fact: CaseFact) -> dict[str, Any]:
    return {
        "id": str(fact.id),
        "category": fact.category.value,
        "label": fact.label,
        "value": fact.value,
        "tags": fact.tags,
    }


def _block_payload(block: DisplayBlock) -> dict[str, Any]:
    return {
        "type": block.type.value,
        "title": block.title,
        "body": block.body,
        "fact_ids": [str(fact_id) for fact_id in block.fact_ids],
    }


def _medical_audit_prompt(
    truth_case: TruthCase,
    run_state: RunState,
    blocks: list[DisplayBlock],
    allowed_fact_ids: list[UUID],
    purpose: str,
) -> str:
    allowed_set = set(allowed_fact_ids)
    facts = "\n".join(
        _audit_fact_line(index, fact, allowed_set)
        for index, fact in enumerate(truth_case.facts, start=1)
    )
    candidates = "\n".join(
        f"- {block.title}: {_clip_text(block.body, 700)}" for block in blocks
    )
    allowed_facts = "\n".join(
        f"- {fact.label}: {fact.value}" for fact in truth_case.facts if fact.id in allowed_set
    )
    aliases = ", ".join(truth_case.diagnosis_aliases)
    return (
        "Audit this diagnostic-game output for medical consistency and spoiler safety.\n"
        "Return exactly one JSON object with keys: approved, contradiction_risk, spoiler_risk, "
        "plausibility, unsupported_claims, contradictions, notes.\n"
        "Example: {\"approved\":true,\"contradiction_risk\":0,\"spoiler_risk\":0,\"plausibility\":1,"
        "\"unsupported_claims\":[],\"contradictions\":[],\"notes\":[]}\n"
        "Use numbers from 0 to 1. Do not repeat the case text. Do not add wrapper keys.\n"
        "Judge only Candidate output. The full fetched case includes hidden answers for you and is not itself a spoiler.\n"
        "Allowed facts are safe to reveal even when diagnostically suggestive.\n"
        "Set spoiler_risk to 0 when every candidate medical detail is in Allowed facts.\n"
        "Spoiler means Candidate output names the final diagnosis/alias or reveals a hidden fact value, not that it contains clues.\n"
        "Reject if the candidate contradicts the full case, reveals final diagnosis or aliases, "
        "reveals a hidden fact, or adds unsupported medical claims.\n\n"
        f"Purpose: {purpose}\n"
        f"Run stage: {run_state.stage}; turn: {run_state.turn_count}\n"
        "Allowed facts visible to player:\n"
        f"{allowed_facts}\n\n"
        "Full fetched case for auditor only:\n"
        f"Title: {truth_case.title}\n"
        f"Chief complaint: {truth_case.chief_complaint}\n"
        f"Demographics: {_dump_prompt(truth_case.demographics)}\n"
        f"Final diagnosis hidden from player: {truth_case.final_diagnosis}\n"
        f"Diagnosis aliases hidden from player: {aliases}\n"
        "Facts:\n"
        f"{facts}\n\n"
        "Candidate output:\n"
        f"{candidates}\n\n"
        "JSON response:"
    )


def _audit_fact_line(index: int, fact: CaseFact, allowed_fact_ids: set[UUID]) -> str:
    flags = []
    if fact.id in allowed_fact_ids:
        flags.append("allowed")
    if fact.spoiler:
        flags.append("spoiler")
    flag_text = ",".join(flags) if flags else "hidden"
    return f"- F{index} [{flag_text}] {fact.category.value}/{fact.label}: {fact.value}"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_audit_data(text: str) -> dict[str, Any] | None:
    return _extract_json_object(text) or _extract_key_value_audit(text)


def _extract_key_value_audit(text: str) -> dict[str, Any] | None:
    allowed_keys = {
        "approved",
        "contradiction_risk",
        "spoiler_risk",
        "plausibility",
        "unsupported_claims",
        "contradictions",
        "notes",
    }
    parsed: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().strip(",")
        if not line:
            continue
        match = re.match(r'^"?([A-Za-z_]+)"?\s*(?::|=|\s)\s*(.+)$', line)
        if not match:
            continue
        key = match.group(1)
        if key not in allowed_keys:
            continue
        parsed[key] = _parse_audit_scalar(match.group(2).strip().strip(","))
    if {"approved", "contradiction_risk", "spoiler_risk", "plausibility"} <= parsed.keys():
        parsed.setdefault("unsupported_claims", [])
        parsed.setdefault("contradictions", [])
        parsed.setdefault("notes", [])
        return parsed
    return None


def _parse_audit_scalar(value: str) -> Any:
    cleaned = value.strip().strip('"')
    lowered = cleaned.lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    if cleaned in {"[]", "[ ]"}:
        return []
    if cleaned == "0":
        return []
    if cleaned.startswith("["):
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, list) else cleaned
        except json.JSONDecodeError:
            return [cleaned.strip("[] ")] if cleaned.strip("[] ") else []
    try:
        return float(cleaned)
    except ValueError:
        return [cleaned] if len(cleaned) > 1 and " " in cleaned else cleaned


def _normalize_medical_audit(data: dict[str, Any]) -> dict[str, Any]:
    approved = bool(data.get("approved", False))
    default_risk = 0.0 if approved else 1.0
    default_plausibility = 1.0 if approved else 0.0
    return {
        "approved": approved,
        "contradiction_risk": _risk_float(data.get("contradiction_risk"), default=default_risk),
        "spoiler_risk": _risk_float(data.get("spoiler_risk"), default=default_risk),
        "plausibility": _risk_float(data.get("plausibility"), default=default_plausibility),
        "unsupported_claims": _string_list(data.get("unsupported_claims")),
        "contradictions": _string_list(data.get("contradictions")),
        "notes": _string_list(data.get("notes")),
    }


def _approved_audit(note: str) -> dict[str, Any]:
    return {
        "approved": True,
        "contradiction_risk": 0.0,
        "spoiler_risk": 0.0,
        "plausibility": 1.0,
        "unsupported_claims": [],
        "contradictions": [],
        "notes": [note],
    }


def _risk_float(value: Any, *, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _apply_medical_audit(report: ValidationReport, audit: dict[str, Any]) -> ValidationReport:
    hard_errors = list(report.hard_errors)
    warnings = list(report.warnings)
    contradiction_risk = _risk_float(audit.get("contradiction_risk"), default=1.0)
    spoiler_risk = _risk_float(audit.get("spoiler_risk"), default=1.0)
    plausibility = _risk_float(audit.get("plausibility"), default=0.0)
    notes = [*report.soft_audit.notes, *_string_list(audit.get("notes"))]
    unsupported_claims = _string_list(audit.get("unsupported_claims"))
    contradictions = _string_list(audit.get("contradictions"))

    if unsupported_claims:
        warnings.extend(f"Medical audit unsupported claim: {claim}" for claim in unsupported_claims)
    if contradictions:
        warnings.extend(f"Medical audit contradiction: {item}" for item in contradictions)

    audit_failed = (
        not bool(audit.get("approved", False))
        or spoiler_risk >= 0.25
        or contradiction_risk >= 0.40
        or plausibility < 0.70
    )
    if audit_failed:
        hard_errors.append(
            "Medical audit rejected generated output "
            f"(spoiler_risk={spoiler_risk:.2f}, contradiction_risk={contradiction_risk:.2f}, "
            f"plausibility={plausibility:.2f})."
        )

    soft_audit = SoftAuditScore(
        spoiler_risk=max(report.soft_audit.spoiler_risk, spoiler_risk),
        contradiction_risk=max(report.soft_audit.contradiction_risk, contradiction_risk),
        plausibility=min(report.soft_audit.plausibility, plausibility),
        tone_fit=report.soft_audit.tone_fit,
        notes=notes,
    )
    return report.model_copy(
        update={
            "status": ValidationStatus.FAIL if hard_errors else report.status,
            "hard_errors": hard_errors,
            "warnings": warnings,
            "soft_audit": soft_audit,
        }
    )


def _validation_feedback(report: ValidationReport) -> list[str]:
    return [*report.hard_errors, *report.warnings][:8]


def _build_run_summary(
    run_state: RunState,
    visible_facts: list[CaseFact],
    latest_blocks: list[DisplayBlock],
    request: PlayerTurnRequest | None = None,
) -> str:
    parts = [f"Stage: {run_state.stage}. Turn count: {run_state.turn_count}."]
    if request is not None:
        request_text = request.target or request.player_text or request.action_type.value
        parts.append(f"Latest player action: {request.action_type.value} - {_clip_text(request_text, 160)}.")
    if run_state.ordered_tests:
        parts.append("Ordered tests: " + "; ".join(_clip_text(item, 80) for item in run_state.ordered_tests[-6:]) + ".")
    if run_state.submitted_differentials:
        parts.append(
            "Submitted differentials: "
            + "; ".join(_clip_text(item, 80) for item in run_state.submitted_differentials[-8:])
            + "."
        )
    if visible_facts:
        fact_text = "; ".join(
            f"{fact.label}: {_clip_text(fact.value, 160)}" for fact in visible_facts[-12:]
        )
        parts.append(f"Visible findings: {fact_text}.")
    if latest_blocks:
        update_text = " ".join(_clip_text(block.body, 180) for block in latest_blocks[-2:])
        parts.append(f"Latest accepted output: {update_text}")
    return " ".join(parts)


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
    query_tokens = _tokens(query)
    if not query_tokens:
        return 1

    label_tokens = _tokens(fact.label)
    value_tokens = _tokens(fact.value)
    tag_tokens = {token for tag in fact.tags for token in _tokens(tag)}
    category_tokens = _tokens(fact.category.value.replace("_", " "))
    category_tokens |= {f"{token}s" for token in category_tokens}
    fact_tokens = label_tokens | value_tokens | tag_tokens | category_tokens
    overlap = query_tokens & fact_tokens

    score = 1
    if fact.label.casefold() in query:
        score += 8
    for tag in fact.tags:
        if tag.casefold() in query:
            score += 6
    score += len(overlap) * 2
    if query_tokens & (label_tokens | tag_tokens):
        score += 4
    if fact.category.value.replace("_", " ") in query:
        score += 1
    return score if overlap or score > 2 else 0


def _tokens(text: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "at",
        "can",
        "could",
        "do",
        "for",
        "have",
        "is",
        "me",
        "of",
        "on",
        "or",
        "please",
        "request",
        "show",
        "tell",
        "the",
        "there",
        "to",
        "what",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold().replace("-", " "))
        if len(token) > 1 and token not in stop_words
    }


def _facts_to_response_text(request: PlayerTurnRequest, facts: list[CaseFact]) -> str:
    if request.action_type == ActionType.ASK_PATIENT_QUESTION:
        return " ".join(f"{fact.value}" for fact in facts)
    return _facts_to_plain_text(facts)


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


def _normalize_tag(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
