from diagnostician.core.schemas import (
    ActionType,
    DiagnosisSubmission,
    PlayerTurnRequest,
    RunCreateRequest,
    RunStatus,
    ValidationStatus,
)
from diagnostician.services.workflows import DiagnosticWorkflow

from tests.helpers import FakeLLMClient, populated_store


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


def _response_text(response) -> str:
    return "\n".join(block.body for block in response.display_blocks).casefold()
