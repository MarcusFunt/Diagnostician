from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from diagnostician.core.schemas import (
    DisplayBlock,
    FactCategory,
    RunState,
    RunStatus,
    SoftAuditScore,
    TruthCase,
    ValidationReport,
    ValidationStatus,
)


def validate_display_blocks(
    truth_case: TruthCase,
    run_state: RunState,
    blocks: list[DisplayBlock],
    allowed_fact_ids: Iterable[UUID],
) -> ValidationReport:
    allowed = set(allowed_fact_ids)
    hard_errors: list[str] = []
    warnings: list[str] = []
    revealed: set[UUID] = set()
    fact_by_id = {fact.id: fact for fact in truth_case.facts}

    for block in blocks:
        for fact_id in block.fact_ids:
            if fact_id not in allowed:
                hard_errors.append(f"Block '{block.title}' revealed disallowed fact {fact_id}.")
                continue
            fact = fact_by_id.get(fact_id)
            if fact is None:
                hard_errors.append(f"Block '{block.title}' referenced unknown fact {fact_id}.")
                continue
            if fact.spoiler and run_state.status != RunStatus.COMPLETE:
                hard_errors.append(f"Block '{block.title}' revealed spoiler-critical fact {fact_id}.")
            if fact.category != FactCategory.DIAGNOSIS and not fact.provenance_ids:
                hard_errors.append(f"Fact '{fact.label}' lacks provenance.")
            revealed.add(fact_id)

        if run_state.status != RunStatus.COMPLETE and _contains_diagnosis_alias(
            block.body, truth_case.diagnosis_aliases
        ):
            hard_errors.append(f"Block '{block.title}' appears to leak the final diagnosis.")

    soft_audit = deterministic_soft_audit(truth_case, blocks)
    if soft_audit.spoiler_risk > 0.5:
        warnings.append("Soft audit detected elevated spoiler risk.")

    return ValidationReport(
        status=ValidationStatus.FAIL if hard_errors else ValidationStatus.PASS,
        hard_errors=hard_errors,
        warnings=warnings,
        allowed_fact_ids=list(allowed),
        revealed_fact_ids=sorted(revealed, key=str),
        soft_audit=soft_audit,
    )


def deterministic_soft_audit(truth_case: TruthCase, blocks: list[DisplayBlock]) -> SoftAuditScore:
    joined = "\n".join(block.body for block in blocks)
    spoiler_risk = 1.0 if _contains_diagnosis_alias(joined, truth_case.diagnosis_aliases) else 0.0
    notes = ["diagnosis alias present"] if spoiler_risk else []
    return SoftAuditScore(
        spoiler_risk=spoiler_risk,
        contradiction_risk=0.0,
        plausibility=1.0,
        tone_fit=1.0,
        notes=notes,
    )


def _contains_diagnosis_alias(text: str, aliases: Iterable[str]) -> bool:
    normalized_text = _normalize(text)
    for alias in aliases:
        normalized_alias = _normalize(alias)
        if len(normalized_alias) >= 4 and normalized_alias in normalized_text:
            return True
    return False


def _normalize(text: str) -> str:
    return " ".join(text.casefold().replace("-", " ").split())
