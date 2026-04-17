from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import httpx

from diagnostician.core.config import Settings, get_settings


@dataclass
class GenerationResult:
    text: str
    model: str
    fallback_used: bool = False
    error: str | None = None


@dataclass
class EmbeddingResult:
    vector: list[float]
    model: str
    fallback_used: bool = False
    error: str | None = None


class OllamaClient:
    """Small Ollama HTTP adapter with deterministic development fallbacks."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def health(self) -> dict[str, Any]:
        try:
            response = httpx.get(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/tags",
                timeout=self.settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            return {"ok": True, "models": response.json().get("models", [])}
        except Exception as exc:  # pragma: no cover - environment dependent
            return {"ok": False, "error": str(exc)}

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        *,
        model: str | None = None,
        format: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> GenerationResult:
        selected_model = model or self.settings.generation_model
        payload: dict[str, Any] = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if format is not None:
            payload["format"] = format
        if options:
            payload["options"] = options
        resolved_keep_alive = keep_alive if keep_alive is not None else self.settings.ollama_keep_alive
        if resolved_keep_alive:
            payload["keep_alive"] = resolved_keep_alive
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=self.settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            return GenerationResult(
                text=response.json().get("response", "").strip(),
                model=selected_model,
            )
        except Exception as exc:
            if self.settings.require_ollama:
                raise
            return GenerationResult(
                text="",
                model=selected_model,
                fallback_used=True,
                error=str(exc),
            )

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> GenerationResult:
        return self.generate(
            prompt,
            system=system,
            model=model,
            format="json",
            options=options,
            keep_alive=keep_alive,
        )

    def embed(self, text: str) -> EmbeddingResult:
        payload = {"model": self.settings.embedding_model, "prompt": text}
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/embeddings",
                json=payload,
                timeout=self.settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return EmbeddingResult(vector=_fit_dimensions(embedding, self.settings.embedding_dimensions), model=self.settings.embedding_model)
        except Exception as exc:
            if self.settings.require_ollama:
                raise
            return EmbeddingResult(
                vector=deterministic_embedding(text, self.settings.embedding_dimensions),
                model=f"{self.settings.embedding_model}:deterministic-fallback",
                fallback_used=True,
                error=str(exc),
            )


def deterministic_embedding(text: str, dimensions: int) -> list[float]:
    """Create a stable non-medical embedding for offline tests/dev fallback."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        chunk = hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in chunk)
        counter += 1
    vector = values[:dimensions]
    norm = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / norm for item in vector]


def _fit_dimensions(vector: list[float], dimensions: int) -> list[float]:
    if len(vector) == dimensions:
        return vector
    if len(vector) > dimensions:
        return vector[:dimensions]
    return vector + [0.0] * (dimensions - len(vector))
