import pytest

from diagnostician.core.schemas import FactCategory, ReviewStatus
from diagnostician.ingestion.parser import LocalCaseIngestor, load_cases_from_parquet

from tests.helpers import DEMO_CASES, FakeLLMClient, SAMPLE_CASE


def test_json_ingestion_produces_playable_case_with_embeddings():
    result = LocalCaseIngestor(llm_client=FakeLLMClient()).ingest_path(SAMPLE_CASE)

    assert result.report.accepted is True
    assert result.report.playable is True
    assert result.truth_case is not None
    assert result.truth_case.approved_for_play is True
    assert result.truth_case.reveal_policy is not None
    assert result.truth_case.reveal_policy.initial_fact_ids
    assert len(result.embeddings) == len(result.truth_case.facts)
    assert all(len(vector) == 768 for vector in result.embeddings.values())
    assert any(fact.category == FactCategory.DIAGNOSIS and fact.spoiler for fact in result.truth_case.facts)


def test_all_demo_cases_are_playable_and_spoiler_locked():
    assert len(DEMO_CASES) == 10
    ingestor = LocalCaseIngestor(llm_client=FakeLLMClient())

    for case_path in DEMO_CASES:
        result = ingestor.ingest_path(case_path)

        assert result.report.accepted is True, case_path.name
        assert result.report.playable is True, case_path.name
        assert result.truth_case is not None
        assert result.truth_case.reveal_policy is not None
        assert result.truth_case.reveal_policy.initial_fact_ids
        assert result.truth_case.curation_notes
        assert len([fact for fact in result.truth_case.facts if not fact.spoiler]) >= 11

        diagnosis_facts = [fact for fact in result.truth_case.facts if fact.category == FactCategory.DIAGNOSIS]
        assert diagnosis_facts, case_path.name
        assert all(fact.spoiler for fact in diagnosis_facts), case_path.name
        assert all(fact.provenance_ids for fact in result.truth_case.facts), case_path.name

        visible_text = " ".join(
            fact.value
            for fact in result.truth_case.facts
            if fact.initially_visible or (not fact.spoiler and fact.category != FactCategory.DIAGNOSIS)
        ).casefold()
        for alias in result.truth_case.diagnosis_aliases:
            normalized = alias.casefold()
            if len(normalized) >= 4:
                assert normalized not in visible_text, f"{case_path.name} leaked {alias}"


def test_demo_library_contains_richer_mvp_action_coverage():
    ingestor = LocalCaseIngestor(llm_client=FakeLLMClient())
    categories_by_case = {
        case_path.name: {fact.category for fact in ingestor.ingest_path(case_path).truth_case.facts}
        for case_path in DEMO_CASES
    }

    assert any(FactCategory.ECG in categories for categories in categories_by_case.values())
    assert any(FactCategory.TREATMENT in categories for categories in categories_by_case.values())
    assert any(FactCategory.CONSULT in categories for categories in categories_by_case.values())
    assert any(FactCategory.OBSERVATION in categories for categories in categories_by_case.values())


def test_markdown_ingestion_is_captured_but_not_playable(tmp_path):
    source = tmp_path / "case.md"
    source.write_text("# Draft Case\n\nUnstructured notes.", encoding="utf-8")

    result = LocalCaseIngestor(llm_client=FakeLLMClient()).ingest_path(source)

    assert result.source_document.raw_text == "# Draft Case\n\nUnstructured notes."
    assert result.truth_case is None
    assert result.report.accepted is False
    assert result.report.playable is False


def test_multicare_parquet_ingestion_produces_approved_playable_case(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "cases.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "article_id": "PMC123456",
                "cases": [
                    {
                        "age": 42,
                        "case_id": "PMC123456_01",
                        "case_text": (
                            "A 42-year-old female presented with pleuritic chest pain. "
                            "ECG showed diffuse ST elevation. "
                            "CRP was elevated. "
                            "Workup led to diagnosis of acute pericarditis."
                        ),
                        "gender": "Female",
                    }
                ],
            }
        ]
    )
    pq.write_table(table, source)

    results = list(LocalCaseIngestor(llm_client=FakeLLMClient()).ingest_path_many(source))

    assert len(results) == 1
    result = results[0]
    assert result.source_document.source_type == "multicare"
    assert "pleuritic chest pain" in result.source_document.raw_text
    assert result.report.accepted is True
    assert result.report.playable is True
    assert result.truth_case is not None
    assert result.truth_case.review_status == ReviewStatus.APPROVED
    assert result.truth_case.final_diagnosis == "acute pericarditis"
    assert result.truth_case.approved_for_play is True
    assert result.truth_case.demographics == {"age": 42, "sex": "female"}
    assert result.truth_case.reveal_policy is not None
    assert result.truth_case.reveal_policy.initial_fact_ids
    assert len(result.embeddings) == len(result.truth_case.facts)
    assert any(fact.category == FactCategory.DEMOGRAPHICS for fact in result.truth_case.facts)
    assert any(fact.category == FactCategory.SYMPTOM for fact in result.truth_case.facts)
    diagnosis_facts = [fact for fact in result.truth_case.facts if fact.category == FactCategory.DIAGNOSIS]
    assert diagnosis_facts and all(fact.spoiler for fact in diagnosis_facts)
    opening_text = " ".join(
        fact.value for fact in result.truth_case.facts if fact.initially_visible
    ).casefold()
    assert "acute pericarditis" not in opening_text


def test_multicare_parquet_ingestion_skips_rows_without_parseable_diagnosis(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "cases.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "article_id": "PMC123456",
                "cases": [
                    {
                        "age": 42,
                        "case_id": "PMC123456_01",
                        "case_text": "A 42-year-old female presented with chest pain. Workup continued.",
                        "gender": "Female",
                    }
                ],
            }
        ]
    )
    pq.write_table(table, source)

    results = list(LocalCaseIngestor(llm_client=FakeLLMClient()).ingest_path_many(source))

    assert len(results) == 1
    result = results[0]
    assert result.report.accepted is False
    assert result.report.playable is False
    assert result.truth_case is None
    assert any("parseable final diagnosis" in error for error in result.report.errors)


def test_load_cases_from_parquet_supports_limit_and_offset(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "cases.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "article_id": "PMC111111",
                "cases": [
                    {
                        "age": 31,
                        "case_id": "PMC111111_01",
                        "case_text": (
                            "A 31-year-old male presented with episodic palpitations. "
                            "Telemetry showed narrow-complex tachycardia. "
                            "He was diagnosed with supraventricular tachycardia."
                        ),
                        "gender": "Male",
                    },
                    {
                        "age": 58,
                        "case_id": "PMC111111_02",
                        "case_text": (
                            "A 58-year-old female presented with progressive dyspnea. "
                            "Chest CT showed bilateral hilar lymphadenopathy. "
                            "Biopsy was diagnostic of sarcoidosis."
                        ),
                        "gender": "Female",
                    },
                ],
            }
        ]
    )
    pq.write_table(table, source)

    results = list(load_cases_from_parquet(source, limit=1, offset=1))

    assert len(results) == 1
    result = results[0]
    assert result.report.accepted is True
    assert result.report.playable is True
    assert result.truth_case is not None
    assert "58-year-old female" in result.truth_case.title
    assert result.truth_case.demographics == {"age": 58, "sex": "female"}
    assert result.truth_case.final_diagnosis == "sarcoidosis"
