from __future__ import annotations

from diagnostician.core.schemas import CaseFact, FactCategory, RunState, ScoreSummary, TruthCase


IMPORTANT_DANGEROUS_DIAGNOSES = {
    "acute appendicitis",
    "aortic dissection",
    "diabetic ketoacidosis",
    "dka",
    "ectopic pregnancy",
    "gi bleed",
    "gastrointestinal bleed",
    "hyperkalemia",
    "meningitis",
    "myocardial infarction",
    "pulmonary embolism",
    "sepsis",
    "stroke",
    "thyroid storm",
    "upper gastrointestinal bleed",
}


def score_run(
    truth_case: TruthCase,
    run_state: RunState,
    diagnosis: str,
    rationale: str = "",
) -> ScoreSummary:
    normalized_submission = _normalize(diagnosis)
    normalized_aliases = {_normalize(alias) for alias in truth_case.diagnosis_aliases}
    correct = normalized_submission in normalized_aliases or any(
        alias in normalized_submission for alias in normalized_aliases if len(alias) > 4
    )

    diagnosis_points = 55 if correct else 0
    differential_points = _score_differentials(truth_case, run_state)
    rationale_points = _score_rationale(truth_case, run_state, rationale)
    efficiency_penalty = max(0, run_state.turn_count - 8) * 2
    testing_penalty = max(0, len(run_state.ordered_tests) - 4) * 4
    hint_penalty = run_state.hint_count * 10
    dangerous_miss_penalty = _dangerous_miss_penalty(truth_case, run_state)
    missed_key_findings_penalty = _missed_key_findings_penalty(truth_case, run_state)

    raw_score = (
        diagnosis_points
        + differential_points
        + rationale_points
        + 20
        - efficiency_penalty
        - testing_penalty
        - hint_penalty
        - dangerous_miss_penalty
        - missed_key_findings_penalty
    )
    final_score = max(0, min(100, raw_score))
    rationale = [
        "Final diagnosis matched an accepted alias." if correct else "Final diagnosis did not match an accepted alias.",
        f"Rationale earned {rationale_points} points from revealed key findings.",
        f"{run_state.turn_count} turns used.",
        f"{len(run_state.ordered_tests)} tests ordered.",
        f"{run_state.hint_count} hints used.",
    ]
    if missed_key_findings_penalty:
        rationale.append(f"Missed key findings penalty: {missed_key_findings_penalty}.")

    return ScoreSummary(
        correct=correct,
        final_score=final_score,
        diagnosis_points=diagnosis_points,
        differential_points=differential_points,
        efficiency_penalty=efficiency_penalty,
        testing_penalty=testing_penalty,
        hint_penalty=hint_penalty,
        dangerous_miss_penalty=dangerous_miss_penalty,
        rationale_points=rationale_points,
        missed_key_findings_penalty=missed_key_findings_penalty,
        rationale=rationale,
    )


def _score_differentials(truth_case: TruthCase, run_state: RunState) -> int:
    submitted = {_normalize(item) for item in run_state.submitted_differentials}
    case_tags = {_normalize(item) for item in truth_case.tags}
    aliases = {_normalize(item) for item in truth_case.diagnosis_aliases}
    if submitted & aliases:
        return 15
    if submitted & case_tags:
        return 10
    if submitted:
        return 5
    return 0


def _dangerous_miss_penalty(truth_case: TruthCase, run_state: RunState) -> int:
    case_terms = _dangerous_case_terms(truth_case)
    if not case_terms:
        return 0
    submitted = " ".join(_normalize(item) for item in run_state.submitted_differentials)
    considered = any(term in submitted for term in case_terms)
    return 0 if considered or not run_state.submitted_differentials else 10


def _dangerous_case_terms(truth_case: TruthCase) -> set[str]:
    aliases = {_normalize(item) for item in truth_case.diagnosis_aliases}
    tags = {_normalize(item) for item in truth_case.tags}
    dangerous = {_normalize(item) for item in IMPORTANT_DANGEROUS_DIAGNOSES}
    return {term for term in aliases | tags if term in dangerous}


def _score_rationale(truth_case: TruthCase, run_state: RunState, rationale: str) -> int:
    if not rationale.strip():
        return 0
    normalized = _normalize(rationale)
    visible = set(run_state.visible_fact_ids)
    matches = 0
    for fact in _key_findings(truth_case):
        if fact.id not in visible:
            continue
        if _fact_mentioned(fact, normalized):
            matches += 1
    return min(10, matches * 4)


def _missed_key_findings_penalty(truth_case: TruthCase, run_state: RunState) -> int:
    visible = set(run_state.visible_fact_ids)
    missed = [
        fact
        for fact in _key_findings(truth_case)
        if not fact.spoiler and fact.id not in visible
    ]
    return min(12, len(missed) * 2)


def _key_findings(truth_case: TruthCase) -> list[CaseFact]:
    priority = {
        FactCategory.ECG,
        FactCategory.IMAGING,
        FactCategory.LAB,
        FactCategory.MICROBIOLOGY,
        FactCategory.PATHOLOGY,
        FactCategory.PROCEDURE,
        FactCategory.PHYSICAL_EXAM,
        FactCategory.VITAL,
        FactCategory.SYMPTOM,
        FactCategory.TREATMENT,
        FactCategory.DIAGNOSIS,
    }
    return [fact for fact in truth_case.facts if fact.category in priority][:10]


def _fact_mentioned(fact: CaseFact, normalized_rationale: str) -> bool:
    candidates = [fact.label, *fact.tags]
    candidates.extend(token for token in _normalize(fact.value).split() if len(token) >= 5)
    return any(_normalize(candidate) in normalized_rationale for candidate in candidates if candidate.strip())


def _normalize(text: str) -> str:
    return " ".join(text.casefold().replace("-", " ").split())
