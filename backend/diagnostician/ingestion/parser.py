from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

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


def load_cases_from_parquet(
    path: str | Path = "cases.parquet",
    *,
    limit: int | None = None,
    offset: int = 0,
    batch_size: int = 128,
) -> Iterator[IngestionResult]:
    """Load MultiCaRe cases from a Parquet file without persisting them."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a .parquet file, got: {source_path}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")
    if offset < 0:
        raise ValueError("offset must be greater than or equal to 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    for record in _iter_multicare_case_records(source_path, limit=limit, offset=offset, batch_size=batch_size):
        yield _ingestion_result_from_multicare_record(source_path, record)


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
    ActionType.ORDER_ECG: [
        FactCategory.ECG,
    ],
    ActionType.ORDER_IMAGING: [
        FactCategory.IMAGING,
    ],
    ActionType.REQUEST_PATHOLOGY_DETAIL: [
        FactCategory.PATHOLOGY,
        FactCategory.PROCEDURE,
    ],
    ActionType.GIVE_TREATMENT: [
        FactCategory.TREATMENT,
        FactCategory.MEDICATION,
    ],
    ActionType.REQUEST_CONSULT: [
        FactCategory.CONSULT,
    ],
    ActionType.OBSERVE_PATIENT: [
        FactCategory.OBSERVATION,
        FactCategory.VITAL,
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

    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        *,
        generate_embeddings: bool = True,
        use_llm_extraction: bool = False,
    ):
        self.llm_client = llm_client or OllamaClient()
        self.generate_embeddings = generate_embeddings
        self.use_llm_extraction = use_llm_extraction

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
        self,
        path: str | Path,
        *,
        limit: int | None = None,
        offset: int = 0,
        batch_size: int = 128,
    ) -> Iterator[IngestionResult]:
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if limit is not None and limit < 0:
            raise ValueError("limit must be greater than or equal to 0")
        if offset < 0:
            raise ValueError("offset must be greater than or equal to 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if source_path.suffix.lower() == ".parquet":
            yield from self._ingest_multicare_parquet(
                source_path,
                limit=limit,
                offset=offset,
                batch_size=batch_size,
            )
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
        if truth_case is not None and not errors and self.generate_embeddings:
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
        self,
        source_path: Path,
        *,
        limit: int | None = None,
        offset: int = 0,
        batch_size: int = 128,
    ) -> Iterator[IngestionResult]:
        for record in _iter_multicare_case_records(
            source_path,
            limit=limit,
            offset=offset,
            batch_size=batch_size,
        ):
            yield _ingestion_result_from_multicare_record(
                source_path,
                record,
                llm_client=self.llm_client,
                generate_embeddings=self.generate_embeddings,
                use_llm_extraction=self.use_llm_extraction,
            )


def _source_from_json(source_path: Path, data: dict[str, Any]) -> SourceDocument:
    source_data = data.get("source", {})
    return SourceDocument(
        id=UUID(str(source_data["id"])) if source_data.get("id") else _stable_uuid("json-source", source_path.as_posix()),
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
    case_id = UUID(str(case_data["id"])) if case_data.get("id") else _stable_uuid("json-case", source.path, case_data["title"])
    provenance: list[Provenance] = []
    facts: list[CaseFact] = []

    for index, item in enumerate(case_data.get("facts", [])):
        provenance_ids: list[UUID] = []
        fact_id = (
            UUID(str(item["id"]))
            if item.get("id")
            else _stable_uuid("json-fact", case_id, index, item["category"], item["label"])
        )
        for prov_index, prov_item in enumerate(item.get("provenance", [])):
            prov = Provenance(
                id=(
                    UUID(str(prov_item["id"]))
                    if prov_item.get("id")
                    else _stable_uuid("json-provenance", fact_id, prov_index)
                ),
                source_document_id=source.id,
                kind=ProvenanceKind(prov_item.get("kind", ProvenanceKind.SOURCE_PARAPHRASE)),
                locator=prov_item.get("locator"),
                quote=prov_item.get("quote"),
                note=prov_item.get("note"),
            )
            provenance.append(prov)
            provenance_ids.append(prov.id)

        fact = CaseFact(
            id=fact_id,
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
        curation_notes=case_data.get("curation_notes", ["Locally reviewed playable case."]),
    )


def _truth_case_from_multicare_record(
    record: MultiCareCaseRecord,
    source: SourceDocument,
    *,
    llm_client: OllamaClient | None = None,
    use_llm_extraction: bool = False,
) -> TruthCase:
    case_id = _stable_uuid("multicare-case", record.case_id)
    extracted = _extract_multicare_case(record, llm_client=llm_client, use_llm_extraction=use_llm_extraction)
    if extracted.final_diagnosis is None:
        raise ValueError("MultiCaRe case does not contain a parseable final diagnosis.")

    provenance: list[Provenance] = [
        Provenance(
            id=_stable_uuid("multicare-provenance", record.case_id, "narrative"),
            source_document_id=source.id,
            kind=ProvenanceKind.SOURCE_EXACT,
            locator=f"{record.article_id}/{record.case_id}",
            quote=_clip_text(record.case_text, 500),
            note="MultiCaRe case_text field.",
        )
    ]
    narrative_provenance_ids = [provenance[0].id]

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
                provenance_ids=narrative_provenance_ids,
                initially_visible=True,
                reveal_actions=[ActionType.ASK_PATIENT_QUESTION],
                tags=["multicare", record.case_id, record.article_id],
            )
        )

    for index, item in enumerate(extracted.fact_items):
        prov = Provenance(
            id=_stable_uuid("multicare-provenance", record.case_id, "fact", index, item.quote),
            source_document_id=source.id,
            kind=ProvenanceKind.SOURCE_EXACT,
            locator=f"{record.article_id}/{record.case_id}/sentence-{index + 1}",
            quote=_clip_text(item.quote, 500),
            note="Automatically extracted from MultiCaRe case_text.",
        )
        provenance.append(prov)
        facts.append(
            CaseFact(
                id=_stable_uuid("multicare-fact", record.case_id, item.category.value, index, item.label),
                case_id=case_id,
                category=item.category,
                label=item.label,
                value=item.value,
                provenance_ids=[prov.id],
                initially_visible=item.initially_visible,
                reveal_actions=_actions_for_category(item.category),
                tags=["multicare", record.case_id, record.article_id, *item.tags],
            )
        )

    diagnosis_prov = Provenance(
        id=_stable_uuid("multicare-provenance", record.case_id, "diagnosis"),
        source_document_id=source.id,
        kind=ProvenanceKind.SOURCE_EXACT,
        locator=f"{record.article_id}/{record.case_id}/diagnosis",
        quote=_clip_text(extracted.diagnosis_quote or extracted.final_diagnosis, 500),
        note="Automatically extracted final diagnosis.",
    )
    provenance.append(diagnosis_prov)
    facts.append(
        CaseFact(
            id=_stable_uuid("multicare-fact", record.case_id, "diagnosis"),
            case_id=case_id,
            category=FactCategory.DIAGNOSIS,
            label="Final diagnosis",
            value=f"Final diagnosis: {extracted.final_diagnosis}.",
            provenance_ids=[diagnosis_prov.id],
            spoiler=True,
            tags=["diagnosis", "multicare"],
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
        title=extracted.title or f"MultiCaRe case {record.case_id}",
        source_document_ids=[source.id],
        review_status=ReviewStatus.APPROVED,
        demographics=demographics,
        chief_complaint=extracted.chief_complaint,
        final_diagnosis=extracted.final_diagnosis,
        diagnosis_aliases=extracted.diagnosis_aliases,
        difficulty="multicare-auto",
        specialty=extracted.specialty,
        tags=["multicare", record.article_id, record.case_id, *extracted.tags],
        facts=facts,
        provenance=provenance,
        reveal_policy=reveal_policy,
        teaching_points=extracted.teaching_points,
        curation_notes=[
            "Automatically structured from MultiCaRe; human review is recommended before using as curated educational content."
        ],
    )


def _ingestion_result_from_multicare_record(
    source_path: Path,
    record: MultiCareCaseRecord,
    *,
    llm_client: OllamaClient | None = None,
    generate_embeddings: bool = False,
    use_llm_extraction: bool = False,
) -> IngestionResult:
    source = _source_from_multicare_record(source_path, record)
    errors: list[str] = []
    warnings: list[str] = []

    try:
        truth_case = _truth_case_from_multicare_record(
            record,
            source,
            llm_client=llm_client,
            use_llm_extraction=use_llm_extraction,
        )
        _validate_truth_case(truth_case, errors, warnings)
        _validate_multicare_playability(truth_case, errors)
    except Exception as exc:
        truth_case = None
        errors.append(str(exc))

    embeddings: dict[UUID, list[float]] = {}
    fallback_fact_ids: list[UUID] = []
    if truth_case is not None and not errors and generate_embeddings and llm_client is not None:
        for fact in truth_case.facts:
            result = llm_client.embed(fact.search_text)
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
    return IngestionResult(source, truth_case if not errors else None, report, embeddings)


@dataclass(frozen=True)
class _ExtractedFactItem:
    category: FactCategory
    label: str
    value: str
    quote: str
    initially_visible: bool
    tags: list[str]


@dataclass(frozen=True)
class _ExtractedMultiCareCase:
    title: str | None
    chief_complaint: str
    final_diagnosis: str | None
    diagnosis_aliases: list[str]
    diagnosis_quote: str | None
    specialty: str | None
    tags: list[str]
    teaching_points: list[str]
    fact_items: list[_ExtractedFactItem]


def _extract_multicare_case(
    record: MultiCareCaseRecord,
    *,
    llm_client: OllamaClient | None = None,
    use_llm_extraction: bool = False,
) -> _ExtractedMultiCareCase:
    deterministic = _extract_multicare_case_deterministic(record)
    if deterministic.final_diagnosis or not use_llm_extraction or llm_client is None:
        return deterministic

    llm_extract = _extract_multicare_case_with_llm(record, llm_client)
    if llm_extract.final_diagnosis:
        return llm_extract
    return deterministic


def _extract_multicare_case_deterministic(record: MultiCareCaseRecord) -> _ExtractedMultiCareCase:
    sentences = _clinical_sentences(record.case_text)
    diagnosis, diagnosis_quote = _extract_diagnosis(record.case_text, sentences)
    aliases = _diagnosis_aliases(diagnosis)
    safe_sentences = [sentence for sentence in sentences if not _sentence_mentions_alias(sentence, aliases)]
    opening = _first_clinical_sentence(safe_sentences) or _opening_summary(record.case_text)
    chief_complaint = _chief_complaint_from_sentence(opening)
    fact_items = _extract_fact_items(record, safe_sentences, aliases)
    specialty = _infer_specialty(record.case_text, diagnosis)
    tags = _case_tags(record.case_text, diagnosis, specialty)
    teaching_points = _teaching_points(diagnosis, fact_items)
    title = _title_for_multicare(record, chief_complaint)
    return _ExtractedMultiCareCase(
        title=title,
        chief_complaint=chief_complaint,
        final_diagnosis=diagnosis,
        diagnosis_aliases=aliases,
        diagnosis_quote=diagnosis_quote,
        specialty=specialty,
        tags=tags,
        teaching_points=teaching_points,
        fact_items=fact_items,
    )


def _extract_multicare_case_with_llm(
    record: MultiCareCaseRecord,
    llm_client: OllamaClient,
) -> _ExtractedMultiCareCase:
    prompt = {
        "task": (
            "Extract a concise final diagnosis and 6-12 source-grounded playable diagnostic facts "
            "from this de-identified medical case. Return JSON only."
        ),
        "schema": {
            "final_diagnosis": "short diagnosis string",
            "chief_complaint": "opening presentation without the final diagnosis",
            "facts": [
                {
                    "category": "symptom|physical_exam|lab|ecg|imaging|pathology|procedure|microbiology|past_medical_history|medication|treatment|social_history|timeline|observation",
                    "label": "short label",
                    "value": "fact text without final diagnosis wording",
                    "quote": "source sentence supporting the fact",
                }
            ],
        },
        "case_text": _clip_text(record.case_text, 6000),
    }
    result = llm_client.generate_json(json.dumps(prompt), system="You extract structured medical case data.")
    parsed = _extract_json_object(result.text)
    if not parsed:
        return _extract_multicare_case_deterministic(record)

    diagnosis = _clean_diagnosis(str(parsed.get("final_diagnosis", "")))
    aliases = _diagnosis_aliases(diagnosis)
    facts: list[_ExtractedFactItem] = []
    for index, item in enumerate(parsed.get("facts", []) if isinstance(parsed.get("facts"), list) else []):
        if not isinstance(item, dict):
            continue
        category = _coerce_fact_category(str(item.get("category", ""))) or FactCategory.TIMELINE
        value = _clean_fact_value(str(item.get("value", "")))
        quote = _clean_fact_value(str(item.get("quote", value)))
        if not value or _sentence_mentions_alias(value, aliases):
            continue
        facts.append(
            _ExtractedFactItem(
                category=category,
                label=_clean_label(str(item.get("label") or _label_for_category(category, index))),
                value=value,
                quote=quote,
                initially_visible=index == 0,
                tags=_tags_for_fact(category, value),
            )
        )

    if not facts:
        return _extract_multicare_case_deterministic(record)

    chief_complaint = _clean_fact_value(str(parsed.get("chief_complaint") or facts[0].value))
    if _sentence_mentions_alias(chief_complaint, aliases):
        chief_complaint = _chief_complaint_from_sentence(facts[0].value)
    specialty = _infer_specialty(record.case_text, diagnosis)
    return _ExtractedMultiCareCase(
        title=_title_for_multicare(record, chief_complaint),
        chief_complaint=chief_complaint,
        final_diagnosis=diagnosis or None,
        diagnosis_aliases=aliases,
        diagnosis_quote=diagnosis,
        specialty=specialty,
        tags=_case_tags(record.case_text, diagnosis, specialty),
        teaching_points=_teaching_points(diagnosis, facts),
        fact_items=facts[:30],
    )


def _extract_diagnosis(text: str, sentences: list[str]) -> tuple[str | None, str | None]:
    patterns = [
        r"(?:diagnostic|pathognomonic)\s+of\s+(?P<dx>[^.;\n]+)",
        r"(?:led|lead|leading)\s+to\s+(?:a\s+)?diagnosis\s+of\s+(?P<dx>[^.;\n]+)",
        r"diagnosis\s+(?:was|is)\s+(?:of\s+)?(?P<dx>[^.;\n]+)",
        r"diagnosed\s+(?:with|as having|as|to have)\s+(?P<dx>[^.;\n]+)",
        r"found\s+to\s+have\s+(?P<dx>[^.;\n]+)",
        r"confirmed\s+(?:the\s+diagnosis\s+of\s+)?(?P<dx>[^.;\n]+)",
        r"consistent\s+with\s+(?P<dx>[^.;\n]+)",
        r"(?:showed|revealed|demonstrated)\s+(?:a|an|the)?\s*(?P<dx>[^.;\n]+)",
        r"\bhad\s+(?:a|an|the)?\s*(?P<dx>(?:long\s+)?SCAD\b[^.;\n]*)",
    ]
    search_space = list(reversed(sentences)) or [text]
    for pattern in patterns:
        for sentence in search_space:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if not match:
                continue
            diagnosis = _clean_diagnosis(match.group("dx"))
            if _looks_like_diagnosis(diagnosis):
                return diagnosis, sentence
    return None, None


def _clean_diagnosis(value: str) -> str:
    cleaned = _clean_fact_value(value)
    acronym = _known_diagnosis_acronym(cleaned)
    if acronym:
        return acronym
    cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.split(
        r"\s+(?:harboring|with|due to|secondary to|after|following|which|that|and|but)\s+",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = cleaned.split(",", 1)[0]
    cleaned = cleaned.strip(" ,;:-.")
    return cleaned


def _looks_like_diagnosis(value: str) -> bool:
    if not value:
        return False
    words = value.split()
    if len(words) > 10:
        return False
    normalized = _normalize_text(value)
    if normalized.split(" ", 1)[0] in {"normal", "unremarkable", "negative", "no", "without", "recovery"}:
        return False
    too_generic = {
        "diagnosis",
        "disease",
        "condition",
        "malignancy",
        "infection",
        "tumor",
        "mass",
        "unknown",
    }
    return normalized not in too_generic and len(normalized) >= 4


def _known_diagnosis_acronym(value: str) -> str | None:
    acronyms = {
        "ARDS",
        "DKA",
        "HHS",
        "MI",
        "NSTEMI",
        "PE",
        "SCAD",
        "SLE",
        "STEMI",
        "TTP",
    }
    for token in re.findall(r"\b[A-Z]{2,6}\b", value):
        if token in acronyms:
            return token
    return None


def _diagnosis_aliases(diagnosis: str | None) -> list[str]:
    if not diagnosis:
        return []
    aliases = {diagnosis.strip()}
    for prefix in ("primary ", "acute ", "benign ", "malignant "):
        if diagnosis.casefold().startswith(prefix):
            aliases.add(diagnosis[len(prefix) :].strip())
    acronym = "".join(word[0] for word in re.findall(r"[A-Za-z]+", diagnosis) if word[:1].isupper())
    if 2 <= len(acronym) <= 5:
        aliases.add(acronym)
    return sorted(alias for alias in aliases if alias)


def _extract_fact_items(
    record: MultiCareCaseRecord,
    sentences: list[str],
    aliases: list[str],
) -> list[_ExtractedFactItem]:
    items: list[_ExtractedFactItem] = []
    seen_values: set[str] = set()
    for sentence in sentences:
        if _sentence_mentions_alias(sentence, aliases):
            continue
        category = _classify_sentence(sentence)
        if category is None:
            continue
        value = _clean_fact_value(sentence)
        if not value:
            continue
        normalized = _normalize_text(value)
        if normalized in seen_values:
            continue
        seen_values.add(normalized)
        items.append(
            _ExtractedFactItem(
                category=category,
                label=_label_for_sentence(category, value, len(items)),
                value=value,
                quote=value,
                initially_visible=False,
                tags=_tags_for_fact(category, value),
            )
        )
        if len(items) >= 30:
            break

    if items:
        first_index = next(
            (index for index, item in enumerate(items) if item.category in {FactCategory.CHIEF_COMPLAINT, FactCategory.TIMELINE, FactCategory.SYMPTOM}),
            0,
        )
        first = items[first_index]
        items[first_index] = _ExtractedFactItem(
            category=first.category if first.category != FactCategory.DIAGNOSIS else FactCategory.TIMELINE,
            label=first.label,
            value=first.value,
            quote=first.quote,
            initially_visible=True,
            tags=first.tags,
        )
    return items


def _classify_sentence(sentence: str) -> FactCategory | None:
    text = sentence.casefold()
    if any(token in text for token in ("ecg", "ekg", "electrocardiogram")):
        return FactCategory.ECG
    if any(token in text for token in ("x-ray", "xray", "radiograph", "ct ", " mri", "ultrasound", "angiograph", "pet scan", "echocardiograph", "scan revealed", "imaging")):
        return FactCategory.IMAGING
    if any(token in text for token in ("histolog", "patholog", "biopsy", "immunohist", "stain", "gross examination", "microscopic", "cd31", "cd34")):
        return FactCategory.PATHOLOGY
    if any(token in text for token in ("culture", "pcr", "microbiolog", "gram stain", "viral", "bacterial", "fungal")):
        return FactCategory.MICROBIOLOGY
    if any(token in text for token in ("cbc", "hemoglobin", "platelet", "creatinine", "sodium", "potassium", "esr", "crp", "hba1c", "panel", "laboratory", "blood test", "serum", "urine")):
        return FactCategory.LAB
    if any(token in text for token in ("physical examination", "examination", "exam ", "pupils", "visual acuity", "blood pressure", "heart rate", "respiratory rate", "temperature", "oxygen saturation", "intraocular pressure")):
        if any(token in text for token in ("blood pressure", "heart rate", "respiratory rate", "temperature", "oxygen saturation")):
            return FactCategory.VITAL
        return FactCategory.PHYSICAL_EXAM
    if any(token in text for token in ("underwent", "resection", "surgery", "operation", "procedure", "laparotomy", "endoscopy", "catheter", "bypass", "photocoagulation")):
        return FactCategory.PROCEDURE
    if any(token in text for token in ("treated with", "started on", "received", "therapy", "administered", "fluid", "antibiotic", "insulin", "transfusion")):
        return FactCategory.TREATMENT
    if any(token in text for token in ("taking", "medication")):
        return FactCategory.MEDICATION
    if any(token in text for token in ("consult", "consultation", "referred to", "admitted under")):
        return FactCategory.CONSULT
    if any(token in text for token in ("observed", "observation", "repeat vitals", "serial", "reassessed")):
        return FactCategory.OBSERVATION
    if any(token in text for token in ("smoker", "smoking", "alcohol", "pack-year", "occupation", "family history")):
        return FactCategory.SOCIAL_HISTORY
    if any(token in text for token in ("history of", "prior", "previous", "past medical", "comorbid", "coronary artery disease", "diabetes", "hypertension")):
        return FactCategory.PAST_MEDICAL_HISTORY
    if any(token in text for token in ("presented with", "complained of", "reported", "symptoms", "pain", "fever", "vomiting", "nausea", "cough", "dyspnea", "headache", "vision loss", "swelling")):
        return FactCategory.SYMPTOM
    if len(sentence.split()) >= 7:
        return FactCategory.TIMELINE
    return None


def _label_for_sentence(category: FactCategory, value: str, index: int) -> str:
    if category == FactCategory.SYMPTOM:
        return "Presenting symptoms" if index == 0 else "Symptom detail"
    return _label_for_category(category, index)


def _label_for_category(category: FactCategory, index: int) -> str:
    labels = {
        FactCategory.CHIEF_COMPLAINT: "Opening concern",
        FactCategory.TIMELINE: "Clinical timeline",
        FactCategory.SYMPTOM: "Symptom detail",
        FactCategory.PAST_MEDICAL_HISTORY: "Past medical history",
        FactCategory.MEDICATION: "Medication or treatment",
        FactCategory.SOCIAL_HISTORY: "Social history",
        FactCategory.VITAL: "Vitals",
        FactCategory.PHYSICAL_EXAM: "Physical examination",
        FactCategory.LAB: "Laboratory findings",
        FactCategory.ECG: "ECG",
        FactCategory.IMAGING: "Imaging findings",
        FactCategory.PATHOLOGY: "Pathology findings",
        FactCategory.PROCEDURE: "Procedure detail",
        FactCategory.MICROBIOLOGY: "Microbiology findings",
        FactCategory.TREATMENT: "Treatment response",
        FactCategory.CONSULT: "Consult note",
        FactCategory.OBSERVATION: "Observation update",
    }
    base = labels.get(category, category.value.replace("_", " ").title())
    return base if index == 0 else f"{base} {index + 1}"


def _actions_for_category(category: FactCategory) -> list[ActionType]:
    return [
        action
        for action, categories in DEFAULT_ACTION_CATEGORY_MAP.items()
        if category in categories
    ] or [ActionType.ASK_PATIENT_QUESTION]


def _validate_multicare_playability(truth_case: TruthCase, errors: list[str]) -> None:
    non_spoiler = [fact for fact in truth_case.facts if not fact.spoiler]
    if len(non_spoiler) < 3:
        errors.append("Automatically parsed MultiCaRe case has too few non-spoiler facts for play.")
    if not any(fact.initially_visible and not fact.spoiler for fact in truth_case.facts):
        errors.append("Automatically parsed MultiCaRe case has no safe opening fact.")
    if not any(fact.category == FactCategory.DIAGNOSIS and fact.spoiler for fact in truth_case.facts):
        errors.append("Automatically parsed MultiCaRe case has no spoiler-locked diagnosis fact.")
    initial_text = " ".join(fact.value for fact in truth_case.facts if fact.initially_visible).casefold()
    for alias in truth_case.diagnosis_aliases:
        normalized = _normalize_text(alias)
        if normalized and _normalized_contains_term(_normalize_text(initial_text), normalized):
            errors.append("Automatically parsed MultiCaRe case leaks the diagnosis in opening facts.")


def _clinical_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"\b(Fig|Figs|Dr|Mr|Mrs|Ms)\.", r"\1", normalized)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [_clean_fact_value(part) for part in parts if len(_clean_fact_value(part).split()) >= 4]


def _first_clinical_sentence(sentences: list[str]) -> str | None:
    return sentences[0] if sentences else None


def _chief_complaint_from_sentence(sentence: str) -> str:
    cleaned = _clean_fact_value(sentence)
    match = re.search(r"presented\s+with\s+(?P<complaint>[^.;]+)", cleaned, flags=re.IGNORECASE)
    if match:
        return _clip_text(match.group("complaint").strip(" ,."), 180)
    return _clip_text(cleaned, 180)


def _title_for_multicare(record: MultiCareCaseRecord, chief_complaint: str) -> str:
    prefix = _multicare_demographics_text(record).removesuffix(".")
    if prefix:
        return _clip_text(f"{prefix} with {chief_complaint}".replace(" patient with ", " with "), 120)
    return _clip_text(chief_complaint, 120)


def _clean_fact_value(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    cleaned = re.sub(r"\s*\(Fig(?:ure)?\.?\s*[^)]*\)", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _clean_label(value: str) -> str:
    cleaned = _clean_fact_value(value).strip(" .:")
    return _clip_text(cleaned or "Clinical finding", 80)


def _sentence_mentions_alias(sentence: str, aliases: list[str]) -> bool:
    normalized = _normalize_text(sentence)
    for alias in aliases:
        alias_norm = _normalize_text(alias)
        if alias_norm and _normalized_contains_term(normalized, alias_norm):
            return True
    return False


def _tags_for_fact(category: FactCategory, value: str) -> list[str]:
    text = value.casefold()
    tags = {category.value}
    keyword_tags = {
        "ct": ("ct ", "computed tomography", "angiography"),
        "mri": ("mri", "magnetic resonance"),
        "ultrasound": ("ultrasound", "sonograph"),
        "xray": ("x-ray", "xray", "radiograph"),
        "biopsy": ("biopsy", "histolog", "patholog"),
        "lab": ("cbc", "serum", "laboratory", "blood"),
        "surgery": ("surgery", "resection", "operation", "laparotomy"),
    }
    for tag, needles in keyword_tags.items():
        if any(needle in text for needle in needles):
            tags.add(tag)
    return sorted(tags)


def _case_tags(text: str, diagnosis: str | None, specialty: str | None) -> list[str]:
    tags = {"multicare"}
    if specialty:
        tags.add(specialty)
    lowered = text.casefold()
    for tag, needles in {
        "imaging": ("ct ", "mri", "ultrasound", "radiograph", "x-ray"),
        "pathology": ("biopsy", "histolog", "patholog"),
        "oncology": ("carcinoma", "cancer", "tumor", "malignan"),
        "surgery": ("surgery", "resection", "operation"),
        "infectious disease": ("infection", "culture", "antibiotic"),
    }.items():
        if any(needle in lowered for needle in needles):
            tags.add(tag)
    if diagnosis:
        tags.add(diagnosis)
    return sorted(tags)


def _infer_specialty(text: str, diagnosis: str | None) -> str | None:
    combined = f"{text} {diagnosis or ''}".casefold()
    mapping = [
        ("ophthalmology", ("retina", "visual acuity", "eye", "ocular", "glaucoma", "cataract")),
        ("oncology", ("carcinoma", "cancer", "tumor", "malignan", "metast")),
        ("pulmonology", ("lung", "pulmonary", "bronch", "pleural")),
        ("cardiology", ("myocard", "cardiac", "aortic", "coronary", "heart")),
        ("gastroenterology", ("abdominal", "colon", "gastric", "liver", "pancrea", "bowel")),
        ("neurology", ("seizure", "stroke", "brain", "neurolog", "headache")),
        ("infectious disease", ("infection", "sepsis", "bacterial", "viral", "fungal")),
        ("rheumatology", ("vasculitis", "arthritis", "autoimmune", "lupus")),
        ("endocrinology", ("thyroid", "diabetes", "adrenal", "pituitary")),
        ("nephrology", ("renal", "kidney", "creatinine")),
        ("obstetrics and gynecology", ("pregnan", "uter", "ovarian", "placenta")),
        ("dermatology", ("skin", "rash", "dermat")),
        ("hematology", ("lymphoma", "leukemia", "anemia", "platelet")),
    ]
    for specialty, needles in mapping:
        if any(needle in combined for needle in needles):
            return specialty
    return "general medicine"


def _teaching_points(diagnosis: str | None, facts: list[_ExtractedFactItem]) -> list[str]:
    points = []
    if diagnosis:
        points.append(f"The final diagnosis was {diagnosis}; review which source findings supported it.")
    categories = sorted({fact.category.value.replace("_", " ") for fact in facts})
    if categories:
        points.append("Key source categories included " + ", ".join(categories[:5]) + ".")
    points.append("Re-check the sequence of revealed history, examination, tests, and procedures against the source narrative.")
    return points


def _coerce_fact_category(value: str) -> FactCategory | None:
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    try:
        return FactCategory(normalized)
    except ValueError:
        return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _normalized_contains_term(text: str, term: str) -> bool:
    if len(term) <= 3:
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


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
    source_path: Path,
    *,
    limit: int | None = None,
    offset: int = 0,
    batch_size: int = 128,
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
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
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
