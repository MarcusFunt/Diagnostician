import pytest

from diagnostician.core.schemas import FactCategory, ReviewStatus
from diagnostician.ingestion.parser import LocalCaseIngestor

from tests.helpers import FakeLLMClient, SAMPLE_CASE


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


def test_markdown_ingestion_is_captured_but_not_playable(tmp_path):
    source = tmp_path / "case.md"
    source.write_text("# Draft Case\n\nUnstructured notes.", encoding="utf-8")

    result = LocalCaseIngestor(llm_client=FakeLLMClient()).ingest_path(source)

    assert result.source_document.raw_text == "# Draft Case\n\nUnstructured notes."
    assert result.truth_case is None
    assert result.report.accepted is False
    assert result.report.playable is False


def test_multicare_parquet_ingestion_produces_review_draft(tmp_path):
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
    assert result.source_document.source_type == "multicare"
    assert result.source_document.raw_text == "A 42-year-old female presented with chest pain. Workup continued."
    assert result.report.accepted is True
    assert result.report.playable is False
    assert result.embeddings == {}
    assert result.truth_case is not None
    assert result.truth_case.review_status == ReviewStatus.NEEDS_REVIEW
    assert result.truth_case.final_diagnosis == ""
    assert result.truth_case.demographics == {"age": 42, "sex": "female"}
    assert result.truth_case.reveal_policy is not None
    assert result.truth_case.reveal_policy.initial_fact_ids
    assert any(fact.category == FactCategory.DEMOGRAPHICS for fact in result.truth_case.facts)
    assert any(fact.category == FactCategory.TIMELINE for fact in result.truth_case.facts)
