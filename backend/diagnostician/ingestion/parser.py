from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from diagnostician.core.schemas import (
    ActionType,
    CaseFact,
    FactCategory,
    IngestionReport,
    Provenance,
    ProvenanceKind,
    RevealPolicy,
    ReviewStatus,
    SourceDocument,
    TruthCase,
)
from diagnostician.llm.ollama_client import OllamaClient


@dataclass
class IngestionResult:
    source_document: SourceDocument
    truth_case: TruthCase | None
    report: IngestionReport
    embeddings: dict[UUID, list[float]]


DEFAULT_ACTION_CATEGORY_MAP: dict[ActionType, list[FactCategory]] = {
    ActionType.ASK_PATIENT_QUESTION: [
        FactCategory.CHIEF_COMPLAINT,
        FactCategory.TIMELINE,
        FactCategory.SYMPTOM,
        FactCategory.PAST_MEDICAL_HISTORY,
        FactCategory.MEDICATION,
        FactCategory.SOCIAL_HISTORY,
    ],
    ActionType.REQUEST_EXAM_DETAIL: [
        FactCategory.VITAL,
        FactCategory.PHYSICAL_EXAM,
    ],
    ActionType.ORDER_LAB: [
        FactCategory.LAB,
        FactCategory.MICROBIOLOGY,
    ],
    ActionType.ORDER_IMAGING: [
        FactCategory.IMAGING,
    ],
    ActionType.REQUEST_PATHOLOGY_DETAIL: [
        FactCategory.PATHOLOGY,
        FactCategory.PROCEDURE,
    ],
    ActionType.SUBMIT_DIFFERENTIAL: [
        FactCategory.DIFFERENTIAL_TAG,
    ],
    ActionType.REQUEST_HINT: [
        FactCategory.HINT,
        FactCategory.TEACHING_POINT,
    ],
}


class LocalCaseIngestor:
    """Ingest local de-identified case source files."""

    def __init__(self, llm_client: OllamaClient | None = None):
        self.llm_client = llm_client or OllamaClient()

    def ingest_path(self, path: str | Path) -> IngestionResult:
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        suffix = source_path.suffix.lower()
        if suffix == ".json":
            return self._ingest_json(source_path)
        if suffix in {".md", ".markdown", ".txt"}:
            return self._ingest_text(source_path, source_type="markdown" if suffix != ".txt" else "text")
        if suffix == ".pdf":
            return self._ingest_pdf(source_path)
        raise ValueError(f"Unsupported source type: {source_path.suffix}")

    def _ingest_json(self, source_path: Path) -> IngestionResult:
        data = json.loads(source_path.read_text(encoding="utf-8"))
        source = _source_from_json(source_path, data)
        errors: list[str] = []
        warnings: list[str] = []

        try:
            truth_case = _truth_case_from_json(data, source)
            _validate_truth_case(truth_case, errors, warnings)
        except Exception as exc:
            truth_case = None
            errors.append(str(exc))

        embeddings: dict[UUID, list[float]] = {}
        fallback_fact_ids: list[UUID] = []
        if truth_case is not None and not errors:
            for fact in truth_case.facts:
                result = self.llm_client.embed(fact.search_text)
                embeddings[fact.id] = result.vector
                if result.fallback_used:
                    fallback_fact_ids.append(fact.id)
            if fallback_fact_ids:
                warnings.append(
                    "Ollama embedding call failed for one or more facts; deterministic fallback embeddings were used."
                )

        report = IngestionReport(
            source_document_id=source.id,
            case_id=truth_case.id if truth_case else None,
            accepted=truth_case is not None and not errors,
            playable=truth_case.approved_for_play if truth_case and not errors else False,
            errors=errors,
            warnings=warnings,
            embedding_fallback_fact_ids=fallback_fact_ids,
        )
        return IngestionResult(source, truth_case, report, embeddings)

    def _ingest_text(self, source_path: Path, source_type: str) -> IngestionResult:
        raw_text = source_path.read_text(encoding="utf-8")
        title = _first_heading(raw_text) or source_path.stem.replace("_", " ").title()
        source = SourceDocument(
            path=str(source_path),
            title=title,
            source_type=source_type,  # type: ignore[arg-type]
            deidentified=True,
            raw_text=raw_text,
        )
        report = IngestionReport(
            source_document_id=source.id,
            accepted=False,
            playable=False,
            errors=[
                "Markdown/text source was captured but not normalized. Convert it to the canonical JSON case format for playability."
            ],
        )
        return IngestionResult(source, None, report, {})

    def _ingest_pdf(self, source_path: Path) -> IngestionResult:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(source_path))
            raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raw_text = None
            error = f"PDF text extraction failed: {exc}"
        else:
            error = "PDF source was captured but not normalized. Convert it to the canonical JSON case format for playability."

        source = SourceDocument(
            path=str(source_path),
            title=source_path.stem.replace("_", " ").title(),
            source_type="pdf",
            deidentified=True,
            raw_text=raw_text,
        )
        report = IngestionReport(
            source_document_id=source.id,
            accepted=False,
            playable=False,
            errors=[error],
        )
        return IngestionResult(source, None, report, {})


def _source_from_json(source_path: Path, data: dict[str, Any]) -> SourceDocument:
    source_data = data.get("source", {})
    return SourceDocument(
        path=str(source_path),
        title=source_data.get("title") or data.get("title") or source_path.stem.replace("_", " ").title(),
        source_type="json",
        deidentified=source_data.get("deidentified", True),
        citation=source_data.get("citation"),
        raw_text=source_data.get("raw_text"),
    )


def _truth_case_from_json(data: dict[str, Any], source: SourceDocument) -> TruthCase:
    case_data = data.get("case", data)
    case_id = UUID(str(case_data["id"])) if case_data.get("id") else uuid4()
    provenance: list[Provenance] = []
    facts: list[CaseFact] = []

    for item in case_data.get("facts", []):
        provenance_ids: list[UUID] = []
        for prov_item in item.get("provenance", []):
            prov = Provenance(
                source_document_id=source.id,
                kind=ProvenanceKind(prov_item.get("kind", ProvenanceKind.SOURCE_PARAPHRASE)),
                locator=prov_item.get("locator"),
                quote=prov_item.get("quote"),
                note=prov_item.get("note"),
            )
            provenance.append(prov)
            provenance_ids.append(prov.id)

        fact = CaseFact(
            id=UUID(str(item["id"])) if item.get("id") else uuid4(),
            case_id=case_id,
            category=FactCategory(item["category"]),
            label=item["label"],
            value=item["value"],
            provenance_ids=provenance_ids,
            spoiler=item.get("spoiler", False),
            initially_visible=item.get("initially_visible", False),
            reveal_actions=[ActionType(action) for action in item.get("reveal_actions", [])],
            tags=item.get("tags", []),
        )
        facts.append(fact)

    initial_fact_ids = [fact.id for fact in facts if fact.initially_visible]
    reveal_policy = RevealPolicy(
        case_id=case_id,
        initial_fact_ids=initial_fact_ids,
        action_category_map=DEFAULT_ACTION_CATEGORY_MAP,
        max_facts_per_turn=case_data.get("max_facts_per_turn", 3),
    )

    return TruthCase(
        id=case_id,
        title=case_data["title"],
        source_document_ids=[source.id],
        review_status=ReviewStatus(case_data.get("review_status", ReviewStatus.NEEDS_REVIEW)),
        demographics=case_data.get("demographics", {}),
        chief_complaint=case_data["chief_complaint"],
        final_diagnosis=case_data["final_diagnosis"],
        diagnosis_aliases=case_data.get("diagnosis_aliases", []),
        difficulty=case_data.get("difficulty", "prototype"),
        specialty=case_data.get("specialty"),
        tags=case_data.get("tags", []),
        facts=facts,
        provenance=provenance,
        reveal_policy=reveal_policy,
        teaching_points=case_data.get("teaching_points", []),
    )


def _validate_truth_case(truth_case: TruthCase, errors: list[str], warnings: list[str]) -> None:
    if not truth_case.source_document_ids:
        errors.append("Case has no source document.")
    if not truth_case.facts:
        errors.append("Case has no structured facts.")
    if not truth_case.final_diagnosis.strip():
        errors.append("Case is missing a final diagnosis.")
    if truth_case.review_status != ReviewStatus.APPROVED:
        warnings.append("Case ingested but is not playable until review_status is approved.")

    provenance_ids = {item.id for item in truth_case.provenance}
    for fact in truth_case.facts:
        if fact.category != FactCategory.DIAGNOSIS and not fact.provenance_ids:
            errors.append(f"Fact '{fact.label}' has no provenance.")
        unknown_ids = [item for item in fact.provenance_ids if item not in provenance_ids]
        if unknown_ids:
            errors.append(f"Fact '{fact.label}' references unknown provenance IDs.")
        if fact.category == FactCategory.DIAGNOSIS and not fact.spoiler:
            errors.append("Diagnosis fact must be marked as spoiler-critical.")

    if not truth_case.reveal_policy or not truth_case.reveal_policy.initial_fact_ids:
        errors.append("Case must define at least one initially visible fact.")


def _first_heading(raw_text: str) -> str | None:
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return None
