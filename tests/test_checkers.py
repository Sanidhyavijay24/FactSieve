"""
@file test_checkers.py
@description Unit tests for Checkers and Scoring Aggregation
@module tests/test_checkers
"""

import pytest
from src.llm_checker import LLMJudgeChecker
from src.models import ClaimTriplet, ExtractedClaim, ExtractedSentence, VerificationUnitResult
from src.scoring import aggregate_response
from src.semantic_checker import SemanticSimilarityChecker


def test_scoring_aggregation_strict_contradiction():
    """Verify strict rule: Any contradiction yields response-level Contradiction."""
    units = [
        VerificationUnitResult(
            unit_id="u1",
            sample_id="s1",
            unit_type="triplet",
            unit_text="Eiffel Tower is in London.",
            verifier="nli_deberta",
            label="Contradiction",
        ),
        VerificationUnitResult(
            unit_id="u2",
            sample_id="s1",
            unit_type="triplet",
            unit_text="Eiffel Tower completed in 1889.",
            verifier="nli_deberta",
            label="Entailment",
        ),
    ]

    res = aggregate_response(
        sample_id="s1",
        setting="accurate_context",
        verifier_name="nli_deberta",
        granularity="triplet",
        unit_results=units,
        rule="strict",
    )

    assert res.response_label == "Contradiction"
    assert res.is_hallucinated is True
    assert res.contradiction_count == 1
    assert res.entailment_count == 1


def test_scoring_aggregation_strict_all_entailment():
    """Verify strict rule: All entailment units yield response-level Entailment."""
    units = [
        VerificationUnitResult(
            unit_id="u1",
            sample_id="s2",
            unit_type="triplet",
            unit_text="Paris is in France.",
            verifier="nli_deberta",
            label="Entailment",
        ),
        VerificationUnitResult(
            unit_id="u2",
            sample_id="s2",
            unit_type="triplet",
            unit_text="Paris is the capital.",
            verifier="nli_deberta",
            label="Entailment",
        ),
    ]

    res = aggregate_response(
        sample_id="s2",
        setting="accurate_context",
        verifier_name="nli_deberta",
        granularity="triplet",
        unit_results=units,
        rule="strict",
    )

    assert res.response_label == "Entailment"
    assert res.is_hallucinated is False
    assert res.contradiction_count == 0
    assert res.entailment_count == 2
    assert res.score == 1.0


def test_llm_judge_parser():
    """Verify LLM Judge text output parsing."""
    checker = LLMJudgeChecker()
    assert checker._parse_judge_output("CONTRADICTION") == "Contradiction"
    assert checker._parse_judge_output("The claim is ENTAILMENT based on facts.") == "Entailment"
    assert checker._parse_judge_output("NEUTRAL") == "Neutral"
    assert checker._parse_judge_output("Uncertain") == "Neutral"


def test_semantic_similarity_checker_basic():
    """Verify semantic similarity checker computes valid cosine similarities."""
    checker = SemanticSimilarityChecker(threshold=0.70)
    sim_high = checker.compute_similarity("The Eiffel Tower is in Paris.", "Paris is home to the Eiffel Tower.")
    sim_low = checker.compute_similarity("The Eiffel Tower is in Paris.", "Kangaroos live in Australia.")

    assert sim_high > sim_low
    assert 0.0 <= sim_high <= 1.0
