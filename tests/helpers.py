from pathlib import Path

from diagnostician.ingestion.parser import LocalCaseIngestor
from diagnostician.llm.ollama_client import EmbeddingResult, GenerationResult
from diagnostician.services.store import InMemoryGameStore


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CASE = ROOT / "cases" / "source" / "prototype_pe_case.json"


class FakeLLMClient:
    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        return GenerationResult(text="", model="fake", fallback_used=True)

    def embed(self, text: str) -> EmbeddingResult:
        value = (sum(ord(char) for char in text) % 100) / 100
        return EmbeddingResult(vector=[value] * 768, model="fake-embedding")

    def health(self) -> dict:
        return {"ok": True, "models": [{"name": "fake"}]}


def populated_store() -> InMemoryGameStore:
    store = InMemoryGameStore()
    ingestor = LocalCaseIngestor(llm_client=FakeLLMClient())
    result = ingestor.ingest_path(SAMPLE_CASE)
    store.save_source_document(result.source_document)
    assert result.truth_case is not None
    store.save_truth_case(result.truth_case, result.embeddings)
    return store
