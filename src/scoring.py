"""
@file scoring.py
@description Response-level aggregation logic mapping unit verifications to response evaluations
@module src/scoring
"""

from typing import List, Literal
from src.models import ResponseEvaluationResult, VerificationUnitResult


def aggregate_response(
    sample_id: str,
    setting: str,
    verifier_name: str,
    granularity: Literal["sentence", "claim", "triplet"],
    unit_results: List[VerificationUnitResult],
    rule: Literal["strict", "soft"] = "strict",
) -> ResponseEvaluationResult:
    """Aggregate fine-grained unit verdicts into an overall response evaluation.

    Strict Rule:
      - If ANY unit is CONTRADICTION -> response is CONTRADICTION (Hallucinated = True).
      - Else if ALL units are ENTAILMENT -> response is ENTAILMENT (Hallucinated = False).
      - Else -> response is NEUTRAL (Hallucinated = False).

    Soft Rule:
      - Majority vote across unit labels.

    Args:
        sample_id (str): Sample identifier.
        setting (str): Benchmark context condition.
        verifier_name (str): Verifier engine name.
        granularity (Literal["sentence", "claim", "triplet"]): Granularity level.
        unit_results (List[VerificationUnitResult]): List of unit verification outcomes.
        rule (Literal["strict", "soft"]): Aggregation rule strategy.

    Returns:
        ResponseEvaluationResult: Consolidated response judgment and factual score.
    """
    if not unit_results:
        # Edge case: No units extracted -> Neutral
        return ResponseEvaluationResult(
            sample_id=sample_id,
            setting=setting,
            verifier=verifier_name,
            granularity=granularity,
            unit_results=[],
            response_label="Neutral",
            is_hallucinated=False,
            contradiction_count=0,
            neutral_count=0,
            entailment_count=0,
            score=0.5,
        )

    contra_count = sum(1 for u in unit_results if u.label == "Contradiction")
    neutral_count = sum(1 for u in unit_results if u.label == "Neutral")
    entail_count = sum(1 for u in unit_results if u.label == "Entailment")
    total = len(unit_results)

    if rule == "strict":
        if contra_count > 0:
            resp_label: Literal["Entailment", "Neutral", "Contradiction"] = "Contradiction"
            is_hallucinated = True
        elif entail_count == total:
            resp_label = "Entailment"
            is_hallucinated = False
        else:
            resp_label = "Neutral"
            is_hallucinated = False
    else:  # soft / majority
        if contra_count >= entail_count and contra_count >= neutral_count:
            resp_label = "Contradiction"
            is_hallucinated = True
        elif entail_count >= neutral_count:
            resp_label = "Entailment"
            is_hallucinated = False
        else:
            resp_label = "Neutral"
            is_hallucinated = False

    # Normalized score: 1.0 (all entailed) to 0.0 (all contradicted)
    factual_score = round(max(0.0, (entail_count - contra_count + total) / (2.0 * total)), 4)

    return ResponseEvaluationResult(
        sample_id=sample_id,
        setting=setting,
        verifier=verifier_name,
        granularity=granularity,
        unit_results=unit_results,
        response_label=resp_label,
        is_hallucinated=is_hallucinated,
        contradiction_count=contra_count,
        neutral_count=neutral_count,
        entailment_count=entail_count,
        score=factual_score,
    )
