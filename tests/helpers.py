from pathlib import Path
import json

from diagnostician.ingestion.parser import LocalCaseIngestor
from diagnostician.core.config import Settings
from diagnostician.llm.ollama_client import EmbeddingResult, GenerationResult
from diagnostician.services.store import InMemoryGameStore


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CASE = ROOT / "cases" / "source" / "prototype_pe_case.json"
DEMO_CASES = sorted((ROOT / "cases" / "source").glob("*.json"))


class FakeLLMClient:
    def __init__(self, generation_responses=None, audit_responses=None, settings: Settings | None = None):
        self.settings = settings or Settings(medical_check_enabled=True)
        self.generation_responses = list(generation_responses or [])
        self.audit_responses = list(audit_responses or [])
        self.generate_calls = []
        self.audit_calls = []

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> GenerationResult:
        self.generate_calls.append({"prompt": prompt, "system": system, "kwargs": kwargs})
        model = kwargs.get("model", "fake")
        if self.generation_responses:
            return GenerationResult(text=self.generation_responses.pop(0), model=model)
        return GenerationResult(text="", model=model, fallback_used=True)

    def generate_json(self, prompt: str, system: str | None = None, **kwargs) -> GenerationResult:
        self.audit_calls.append({"prompt": prompt, "system": system, "kwargs": kwargs})
        model = kwargs.get("model", "fake-audit")
        if self.audit_responses:
            payload = self.audit_responses.pop(0)
        else:
            payload = {
                "approved": True,
                "contradiction_risk": 0,
                "spoiler_risk": 0,
                "plausibility": 1,
                "unsupported_claims": [],
                "contradictions": [],
                "notes": [],
            }
        return GenerationResult(text=json.dumps(payload), model=model)

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
