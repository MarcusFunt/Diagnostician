from pathlib import Path

from diagnostician.ingestion.parser import LocalCaseIngestor
from diagnostician.llm.ollama_client import EmbeddingResult, GenerationResult
from diagnostician.services.store import InMemoryGameStore


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CASE = ROOT / "cases" / "source" / "prototype_pe_case.json"
DEMO_CASES = sorted((ROOT / "cases" / "source").glob("*.json"))


class FakeLLMClient:
    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        return GenerationResult(text="", model="fake", fallback_used=True)

    def embed(self, text: str) -> EmbeddingResult:
        value = (sum(ord(char) for char in text) % 100) / 100
        return EmbeddingResult(vector=[value] * 768, model="fake-embedding")

    def health(self) -> dict:
        return {"ok": True, "models": [{"name": "fake"}]}


def populated_store(case_paths=None) -> InMemoryGameStore:
    store = InMemoryGameStore()
    ingestor = LocalCaseIngestor(llm_client=FakeLLMClient())
    for case_path in case_paths or [SAMPLE_CASE]:
        result = ingestor.ingest_path(case_path)
        store.save_source_document(result.source_document)
        assert result.truth_case is not None
        store.save_truth_case(result.truth_case, result.embeddings)
    return store


def demo_store() -> InMemoryGameStore:
    return populated_store(DEMO_CASES)
