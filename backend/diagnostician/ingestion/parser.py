from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

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


@dataclass(frozen=True)
class MultiCareCaseRecord:
    article_id: str
    case_id: str
    case_text: str
    age: int | None
    gender: str | None


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
        if suffix == ".parquet":
            results = list(self.ingest_path_many(source_path, limit=2))
            if not results:
                raise ValueError(f"No MultiCaRe case records found in {source_path}")
            if len(results) > 1:
                raise ValueError(
                    "MultiCaRe Parquet files can contain many case records. "
                    "Use ingest_path_many() or the ingestion CLI to process them."
                )
            return results[0]
        raise ValueError(f"Unsupported source type: {source_path.suffix}")

    def ingest_path_many(
        self, path: str | Path, *, limit: int | None = None, offset: int = 0
    ) -> Iterator[IngestionResult]:
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if limit is not None and limit < 0:
            raise ValueError("limit must be greater than or equal to 0")
        if offset < 0:
            raise ValueError("offset must be greater than or equal to 0")

        if source_path.suffix.lower() == ".parquet":
            yield from self._ingest_multicare_parquet(source_path, limit=limit, offset=offset)
            return

        if offset == 0 and (limit is None or limit > 0):
            yield self.ingest_path(source_path)

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

    def _ingest_multicare_parquet(
        self, source_path: Path, *, limit: int | None = None, offset: int = 0
    ) -> Iterator[IngestionResult]:
        for record in _iter_multicare_case_records(source_path, limit=limit, offset=offset):
            source = _source_from_multicare_record(source_path, record)
            errors: list[str] = []
            warnings: list[str] = [
                "MultiCaRe source was imported as a review draft. Add a reviewed final diagnosis and structured facts before approving it for play."
            ]

            try:
                truth_case = _truth_case_from_multicare_record(record, source)
                _validate_truth_case(truth_case, errors, warnings)
            except Exception as exc:
                truth_case = None
                errors.append(str(exc))

            report = IngestionReport(
                source_document_id=source.id,
                case_id=truth_case.id if truth_case else None,
                accepted=truth_case is not None and not errors,
                playable=False,
                errors=errors,
                warnings=warnings,
            )
            yield IngestionResult(source, truth_case if not errors else None, report, {})


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


def _source_from_multicare_record(source_path: Path, record: MultiCareCaseRecord) -> SourceDocument:
    return SourceDocument(
        id=_stable_uuid("multicare-source", record.case_id),
        path=f"{source_path}#{record.case_id}",
        title=f"MultiCaRe case {record.case_id}",
        source_type="multicare",
        deidentified=True,
        citation=f"MultiCaRe case database; PubMed Central article {record.article_id}",
        raw_text=record.case_text,
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


def _truth_case_from_multicare_record(record: MultiCareCaseRecord, source: SourceDocument) -> TruthCase:
    case_id = _stable_uuid("multicare-case", record.case_id)
    provenance = [
        Provenance(
            id=_stable_uuid("multicare-provenance", record.case_id, "narrative"),
            source_document_id=source.id,
            kind=ProvenanceKind.SOURCE_EXACT,
            locator=f"{record.article_id}/{record.case_id}",
            quote=_clip_text(record.case_text, 500),
            note="MultiCaRe case_text field.",
        )
    ]
    provenance_ids = [provenance[0].id]

    facts: list[CaseFact] = []
    demographics = _multicare_demographics(record)
    demographics_text = _multicare_demographics_text(record)
    if demographics_text:
        facts.append(
            CaseFact(
                id=_stable_uuid("multicare-fact", record.case_id, "demographics"),
                case_id=case_id,
                category=FactCategory.DEMOGRAPHICS,
                label="Demographics",
                value=demographics_text,
                provenance_ids=provenance_ids,
                initially_visible=True,
                reveal_actions=[ActionType.ASK_PATIENT_QUESTION],
                tags=["multicare", record.case_id, record.article_id],
            )
        )

    facts.append(
        CaseFact(
            id=_stable_uuid("multicare-fact", record.case_id, "narrative"),
            case_id=case_id,
            category=FactCategory.TIMELINE,
            label="Clinical narrative",
            value=record.case_text,
            provenance_ids=provenance_ids,
            initially_visible=not facts,
            reveal_actions=[ActionType.ASK_PATIENT_QUESTION],
            tags=["multicare", "source_narrative", record.case_id, record.article_id],
        )
    )

    reveal_policy = RevealPolicy(
        case_id=case_id,
        initial_fact_ids=[fact.id for fact in facts if fact.initially_visible],
        action_category_map=DEFAULT_ACTION_CATEGORY_MAP,
        max_facts_per_turn=3,
    )

    return TruthCase(
        id=case_id,
        title=f"MultiCaRe case {record.case_id}",
        source_document_ids=[source.id],
        review_status=ReviewStatus.NEEDS_REVIEW,
        demographics=demographics,
        chief_complaint=_opening_summary(record.case_text),
        final_diagnosis="",
        diagnosis_aliases=[],
        difficulty="multicare-draft",
        specialty=None,
        tags=["multicare", record.article_id, record.case_id],
        facts=facts,
        provenance=provenance,
        reveal_policy=reveal_policy,
        teaching_points=[],
    )


def _validate_truth_case(truth_case: TruthCase, errors: list[str], warnings: list[str]) -> None:
    if not truth_case.source_document_ids:
        errors.append("Case has no source document.")
    if not truth_case.facts:
        errors.append("Case has no structured facts.")
    if not truth_case.final_diagnosis.strip():
        if truth_case.review_status == ReviewStatus.APPROVED:
            errors.append("Case is missing a final diagnosis.")
        else:
            warnings.append("Case is missing a final diagnosis; it must be reviewed before play.")
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


def _iter_multicare_case_records(
    source_path: Path, *, limit: int | None = None, offset: int = 0
) -> Iterator[MultiCareCaseRecord]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "MultiCaRe Parquet ingestion requires pyarrow. Install it with `pip install -e .[multicare]`."
        ) from exc

    parquet_file = pq.ParquetFile(source_path)
    columns = set(parquet_file.schema_arrow.names)
    if {"case_id", "case_text"}.issubset(columns):
        mode = "flat"
        read_columns = [column for column in ("article_id", "case_id", "case_text", "age", "gender") if column in columns]
    elif {"article_id", "cases"}.issubset(columns):
        mode = "nested"
        read_columns = ["article_id", "cases"]
    else:
        raise ValueError(
            "Parquet file does not match the expected MultiCaRe schema. "
            "Expected flat case_id/case_text columns or nested article_id/cases columns."
        )

    seen = 0
    emitted = 0
    for batch in parquet_file.iter_batches(batch_size=128, columns=read_columns):
        for row in batch.to_pylist():
            for record in _records_from_multicare_row(row, mode):
                if seen < offset:
                    seen += 1
                    continue
                if limit is not None and emitted >= limit:
                    return
                seen += 1
                emitted += 1
                yield record


def _records_from_multicare_row(row: dict[str, Any], mode: str) -> Iterator[MultiCareCaseRecord]:
    if mode == "flat":
        record = _record_from_multicare_case(row, row.get("article_id"))
        if record is not None:
            yield record
        return

    article_id = _clean_scalar(row.get("article_id"))
    cases = row.get("cases") or []
    if isinstance(cases, dict):
        cases = [cases]
    for case_item in cases:
        if not isinstance(case_item, dict):
            continue
        record = _record_from_multicare_case(case_item, article_id)
        if record is not None:
            yield record


def _record_from_multicare_case(case_item: dict[str, Any], article_id: Any) -> MultiCareCaseRecord | None:
    case_id = _clean_scalar(case_item.get("case_id"))
    case_text = _normalize_source_text(case_item.get("case_text"))
    if not case_id or not case_text:
        return None

    resolved_article_id = _clean_scalar(article_id) or case_id.split("_", 1)[0]
    return MultiCareCaseRecord(
        article_id=resolved_article_id,
        case_id=case_id,
        case_text=case_text,
        age=_coerce_int(case_item.get("age")),
        gender=_clean_scalar(case_item.get("gender")) or None,
    )


def _multicare_demographics(record: MultiCareCaseRecord) -> dict[str, Any]:
    demographics: dict[str, Any] = {}
    if record.age is not None:
        demographics["age"] = record.age
    if record.gender:
        demographics["sex"] = record.gender.casefold()
    return demographics


def _multicare_demographics_text(record: MultiCareCaseRecord) -> str:
    parts: list[str] = []
    if record.age is not None:
        parts.append(f"{record.age}-year-old")
    if record.gender:
        parts.append(record.gender.casefold())
    if not parts:
        return ""
    return " ".join(parts) + " patient."


def _opening_summary(text: str) -> str:
    sentence = text.split(". ", 1)[0].strip()
    if len(sentence) < 40 and len(text) > len(sentence):
        sentence = text[:240].strip()
    return _clip_text(sentence, 240)


def _normalize_source_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _stable_uuid(*parts: object) -> UUID:
    return uuid5(NAMESPACE_URL, ":".join(str(part) for part in parts))
