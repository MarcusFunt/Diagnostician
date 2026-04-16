from diagnostician.core.schemas import FactCategory
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
