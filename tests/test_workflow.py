import pytest

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
from diagnostician.core.config import Settings
from diagnostician.ingestion.parser import LocalCaseIngestor
from diagnostician.services.store import InMemoryGameStore
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


def test_start_run_generates_and_persists_per_run_story():
    store = populated_store()
    story = "A 43-year-old woman arrives with abrupt pleuritic chest pain and dyspnea."
    workflow = DiagnosticWorkflow(store=store, llm_client=FakeLLMClient(generation_responses=[story]))

    created = workflow.create_run(RunCreateRequest())
    saved_state = store.get_run(created.run_state.id)

    assert created.display_blocks[0].body == story
    assert saved_state.case_story == story
    assert saved_state.story_fact_ids == created.run_state.visible_fact_ids
    assert "Visible findings" in saved_state.run_summary


def test_turn_generation_reads_story_user_prompt_case_context_and_summary():
    store = populated_store()
    llm = FakeLLMClient(
        generation_responses=[
            "A 43-year-old woman arrives with abrupt pleuritic chest pain and dyspnea.",
            "Heart rate is 118/min with mild tachypnea and oxygen saturation of 93% on room air.",
        ]
    )
    workflow = DiagnosticWorkflow(store=store, llm_client=llm)
    created = workflow.create_run(RunCreateRequest())

    response = workflow.handle_turn(
        created.run_state.id,
        PlayerTurnRequest(action_type=ActionType.REQUEST_EXAM_DETAIL, player_text="What are the vitals?"),
    )

    turn_prompt = llm.generate_calls[1]["prompt"]
    assert created.run_state.case_story in turn_prompt
    assert "What are the vitals?" in turn_prompt
    assert "safe_case_packet" in turn_prompt
    assert "run_summary" in turn_prompt
    assert "Heart rate" in response.run_state.run_summary


def test_medical_audit_rejection_retries_then_uses_safe_fallback():
    store = populated_store()
    rejected = {
        "approved": False,
        "contradiction_risk": 0.8,
        "spoiler_risk": 0,
        "plausibility": 0.5,
        "unsupported_claims": ["unsupported opening detail"],
        "contradictions": ["contradicts source"],
        "notes": ["reject"],
    }
    llm = FakeLLMClient(
        generation_responses=[
            "The patient says a non-source detail.",
            "The patient says a second non-source detail.",
            "The patient says a third non-source detail.",
        ],
        audit_responses=[rejected, rejected, rejected],
    )
    workflow = DiagnosticWorkflow(store=store, llm_client=llm)

    created = workflow.create_run(RunCreateRequest())

    assert created.validation.status == ValidationStatus.FALLBACK_USED
    assert created.validation.fallback_used is True
    assert len(llm.generate_calls) == 3
    assert "non-source detail" not in _response_text(created)


def test_medical_audit_can_be_disabled_for_limited_hardware():
    store = populated_store()
    llm = FakeLLMClient(
        generation_responses=["A source-grounded opening story."],
        audit_responses=[
            {
                "approved": False,
                "contradiction_risk": 1,
                "spoiler_risk": 1,
                "plausibility": 0,
                "unsupported_claims": ["should not be called"],
                "contradictions": [],
                "notes": [],
            }
        ],
        settings=Settings(medical_check_enabled=False),
    )
    workflow = DiagnosticWorkflow(store=store, llm_client=llm)

    created = workflow.create_run(RunCreateRequest())

    assert created.validation.status == ValidationStatus.PASS
    assert llm.audit_calls == []
    assert any("disabled for limited hardware" in note for note in created.validation.soft_audit.notes)


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


def test_automatically_parsed_multicare_case_is_playable(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "cases.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "article_id": "PMC222222",
                "cases": [
                    {
                        "age": 64,
                        "case_id": "PMC222222_01",
                        "case_text": (
                            "A 64-year-old male presented with fever and productive cough. "
                            "Chest radiograph showed right lower-lobe consolidation. "
                            "White blood cell count was elevated. "
                            "Sputum culture confirmed Streptococcus pneumoniae pneumonia."
                        ),
                        "gender": "Male",
                    }
                ],
            }
        ]
    )
    pq.write_table(table, source)
    result = next(LocalCaseIngestor(llm_client=FakeLLMClient()).ingest_path_many(source))
    assert result.truth_case is not None

    store = InMemoryGameStore()
    store.save_source_document(result.source_document)
    store.save_truth_case(result.truth_case, result.embeddings)
    workflow = DiagnosticWorkflow(store=store, llm_client=FakeLLMClient())

    created = workflow.create_run(RunCreateRequest(case_id=result.truth_case.id))
    assert created.run_state.status == RunStatus.ACTIVE
    assert "pneumonia" not in _response_text(created)

    imaging = workflow.handle_turn(
        created.run_state.id,
        PlayerTurnRequest(action_type=ActionType.ORDER_IMAGING, target="chest radiograph"),
    )
    assert imaging.validation.status == ValidationStatus.PASS
    review = workflow.submit_diagnosis(
        created.run_state.id,
        DiagnosisSubmission(diagnosis="Streptococcus pneumoniae pneumonia", rationale="Culture and radiograph"),
    )
    assert review.player_score.correct is True


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


def test_validation_blocks_short_diagnosis_acronym_leakage():
    store = populated_store()
    truth_case = store.list_approved_cases()[0]
    run_state = DiagnosticWorkflow(store=store, llm_client=FakeLLMClient()).create_run(RunCreateRequest()).run_state
    block = DisplayBlock(
        type=DisplayBlockType.ATTENDING_COMMENT,
        title="Leaky Hint",
        body="This presentation should make you think of PE.",
        fact_ids=[],
        severity=Severity.INFO,
    )

    report = validate_display_blocks(truth_case, run_state, [block], run_state.visible_fact_ids)

    assert report.status == ValidationStatus.FAIL
    assert any("final diagnosis" in error for error in report.hard_errors)


def test_validation_blocks_generation_artifacts():
    store = populated_store()
    truth_case = store.list_approved_cases()[0]
    run_state = DiagnosticWorkflow(store=store, llm_client=FakeLLMClient()).create_run(RunCreateRequest()).run_state
    block = DisplayBlock(
        type=DisplayBlockType.ATTENDING_COMMENT,
        title="Wrapped Output",
        body='{"body": "The patient has chest pain."}',
        fact_ids=[],
        severity=Severity.INFO,
    )

    report = validate_display_blocks(truth_case, run_state, [block], run_state.visible_fact_ids)

    assert report.status == ValidationStatus.FAIL
    assert any("structured wrapper" in error for error in report.hard_errors)


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
