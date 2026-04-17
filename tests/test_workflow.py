from diagnostician.core.schemas import (
    ActionType,
    DisplayBlock,
    DisplayBlockType,
    DiagnosisSubmission,
    PlayerTurnRequest,
    RunCreateRequest,
    RunStatus,
    Severity,
    ValidationStatus,
)
from diagnostician.services.workflows import DiagnosticWorkflow
from diagnostician.services.validation import validate_display_blocks

from tests.helpers import FakeLLMClient, demo_store, populated_store


def test_run_turns_reveal_allowed_facts_without_diagnosis_leakage():
    store = populated_store()
    workflow = DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())

    created = workflow.create_run(RunCreateRequest())
    assert created.run_state.status == RunStatus.ACTIVE
    assert created.newly_revealed_facts
    assert "pulmonary embolism" not in _response_text(created)

    vitals = workflow.handle_turn(
        created.run_state.id,
        PlayerTurnRequest(action_type=ActionType.REQUEST_EXAM_DETAIL, player_text="What are the vitals?"),
    )
    assert vitals.validation.status == ValidationStatus.PASS
    assert any("Heart rate" in fact.value for fact in vitals.newly_revealed_facts)
    assert "pulmonary embolism" not in _response_text(vitals)

    imaging = workflow.handle_turn(
        created.run_state.id,
        PlayerTurnRequest(action_type=ActionType.ORDER_IMAGING, target="CT pulmonary angiography"),
    )
    assert imaging.validation.status == ValidationStatus.PASS
    assert any("filling defects" in fact.value for fact in imaging.newly_revealed_facts)
    assert "pulmonary embolism" not in _response_text(imaging)


def test_case_selection_supports_direct_filters_and_replay_avoidance():
    store = demo_store()
    workflow = DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())
    cases = store.list_approved_cases()
    assert len(cases) == 8

    selected = cases[-1]
    direct = workflow.create_run(RunCreateRequest(case_id=selected.id))
    assert direct.run_state.case_id == selected.id

    first_intro = workflow.create_run(RunCreateRequest(difficulty="intro", randomize=False))
    assert store.get_case(first_intro.run_state.case_id).difficulty == "intro"

    excluded = [case.id for case in cases[:-1]]
    replay_aware = workflow.create_run(RunCreateRequest(exclude_case_ids=excluded, randomize=False))
    assert replay_aware.run_state.case_id == cases[-1].id


def test_reveal_logic_requires_relevant_target_match():
    store = populated_store()
    workflow = DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())
    created = workflow.create_run(RunCreateRequest())

    chest_xray = workflow.handle_turn(
        created.run_state.id,
        PlayerTurnRequest(action_type=ActionType.ORDER_IMAGING, target="chest radiograph"),
    )
    labels = {fact.label for fact in chest_xray.newly_revealed_facts}
    assert "Chest radiograph" in labels
    assert "CT pulmonary angiography" not in labels

    ct = workflow.handle_turn(
        created.run_state.id,
        PlayerTurnRequest(action_type=ActionType.ORDER_IMAGING, target="CT pulmonary angiography"),
    )
    assert any(fact.label == "CT pulmonary angiography" for fact in ct.newly_revealed_facts)


def test_validation_blocks_hidden_fact_text_leakage():
    store = populated_store()
    truth_case = store.list_approved_cases()[0]
    run_state = DiagnosticWorkflow(store=store, llm_client=FakeLLMClient()).create_run(RunCreateRequest()).run_state
    hidden_fact = next(fact for fact in truth_case.facts if fact.label == "D-dimer")
    block = DisplayBlock(
        type=DisplayBlockType.LAB_RESULT,
        title="Leaky Lab",
        body=hidden_fact.value,
        fact_ids=[],
        severity=Severity.INFO,
    )

    report = validate_display_blocks(truth_case, run_state, [block], run_state.visible_fact_ids)

    assert report.status == ValidationStatus.FAIL
    assert any("hidden fact" in error for error in report.hard_errors)


def test_submit_diagnosis_scores_and_creates_review():
    store = populated_store()
    workflow = DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())
    created = workflow.create_run(RunCreateRequest())

    workflow.handle_turn(
        created.run_state.id,
        PlayerTurnRequest(
            action_type=ActionType.SUBMIT_DIFFERENTIAL,
            player_text="pulmonary embolism, pneumonia",
        ),
    )
    review = workflow.submit_diagnosis(
        created.run_state.id,
        DiagnosisSubmission(diagnosis="acute pulmonary embolism", rationale="CTA filling defects"),
    )

    assert review.player_score.correct is True
    assert review.player_score.final_score > 70
    assert review.diagnosis == "Pulmonary embolism"
    assert any("Final Diagnosis" == block.title for block in review.turn_timeline)
    assert workflow.get_snapshot(created.run_state.id).run_state.status == RunStatus.COMPLETE


def test_completed_run_does_not_accept_more_turn_progression():
    store = populated_store()
    workflow = DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())
    created = workflow.create_run(RunCreateRequest())
    review = workflow.submit_diagnosis(
        created.run_state.id,
        DiagnosisSubmission(diagnosis="Pulmonary embolism", rationale="CTA filling defects"),
    )

    response = workflow.handle_turn(
        review.run_id,
        PlayerTurnRequest(action_type=ActionType.ORDER_LAB, target="D-dimer"),
    )

    assert response.run_state.status == RunStatus.COMPLETE
    assert response.newly_revealed_facts == []
    assert response.display_blocks[0].title == "Run Complete"


def _response_text(response) -> str:
    return "\n".join(block.body for block in response.display_blocks).casefold()
