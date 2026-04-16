from __future__ import annotations

from diagnostician.core.schemas import RunState, ScoreSummary, TruthCase


IMPORTANT_DANGEROUS_DIAGNOSES = {
    "pulmonary embolism",
    "myocardial infarction",
    "sepsis",
    "aortic dissection",
    "stroke",
    "ectopic pregnancy",
}


def score_run(truth_case: TruthCase, run_state: RunState, diagnosis: str) -> ScoreSummary:
    normalized_submission = _normalize(diagnosis)
    normalized_aliases = {_normalize(alias) for alias in truth_case.diagnosis_aliases}
    correct = normalized_submission in normalized_aliases or any(
        alias in normalized_submission for alias in normalized_aliases if len(alias) > 4
    )

    diagnosis_points = 55 if correct else 0
    differential_points = _score_differentials(truth_case, run_state)
    efficiency_penalty = max(0, run_state.turn_count - 8) * 2
    testing_penalty = max(0, len(run_state.ordered_tests) - 4) * 4
    hint_penalty = run_state.hint_count * 10
    dangerous_miss_penalty = _dangerous_miss_penalty(run_state)

    raw_score = (
        diagnosis_points
        + differential_points
        + 20
        - efficiency_penalty
        - testing_penalty
        - hint_penalty
        - dangerous_miss_penalty
    )
    final_score = max(0, min(100, raw_score))
    rationale = [
        "Final diagnosis matched an accepted alias." if correct else "Final diagnosis did not match an accepted alias.",
        f"{run_state.turn_count} turns used.",
        f"{len(run_state.ordered_tests)} tests ordered.",
        f"{run_state.hint_count} hints used.",
    ]

    return ScoreSummary(
        correct=correct,
        final_score=final_score,
        diagnosis_points=diagnosis_points,
        differential_points=differential_points,
        efficiency_penalty=efficiency_penalty,
        testing_penalty=testing_penalty,
        hint_penalty=hint_penalty,
        dangerous_miss_penalty=dangerous_miss_penalty,
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


def _dangerous_miss_penalty(run_state: RunState) -> int:
    submitted = " ".join(_normalize(item) for item in run_state.submitted_differentials)
    considered = any(item in submitted for item in IMPORTANT_DANGEROUS_DIAGNOSES)
    return 0 if considered or not run_state.submitted_differentials else 5


def _normalize(text: str) -> str:
    return " ".join(text.casefold().replace("-", " ").split())
