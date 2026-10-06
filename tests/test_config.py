"""
@file test_config.py
@description Unit tests for configuration loader and domain models
@module tests/test_config
"""

import pytest
from pydantic import ValidationError
from src.config import load_config, AppConfig
from src.models import (
    BenchmarkSample,
    ClaimTriplet,
    ExtractedClaim,
    ExtractedSentence,
    VerificationUnitResult,
    ResponseEvaluationResult,
)


def test_load_config_defaults():
    """Verify that default configuration loads successfully without errors."""
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.project.name == "factsieve"
    assert cfg.llm.temperature == 0.0
    assert cfg.verifiers.nli.batch_size > 0
    assert cfg.scoring.aggregation_rule in ["strict", "soft"]


def test_claim_triplet_verbalization():
    """Verify automatic verbalization of claim triplets."""
    triplet = ClaimTriplet(
        sample_id="s1",
        triplet_index=0,
        subject="Eiffel Tower",
        relation="is located in",
        object="Paris",
    )
    assert triplet.get_verbalized() == "Eiffel Tower is located in Paris."

    custom_triplet = ClaimTriplet(
        sample_id="s2",
        triplet_index=0,
        subject="Eiffel Tower",
        relation="completed in",
        object="1889",
        verbalized_claim="The Eiffel Tower was completed in 1889.",
    )
    assert custom_triplet.get_verbalized() == "The Eiffel Tower was completed in 1889."


def test_strict_models_reject_extra_fields():
    """Ensure Pydantic models reject undeclared extra fields."""
    with pytest.raises(ValidationError):
        BenchmarkSample(
            id="s1",
            question="Where is Paris?",
            reference="France",
            setting="accurate_context",
            unexpected_field="disallowed",
        )


def test_benchmark_sample_creation():
    """Verify valid creation of a benchmark sample."""
    sample = BenchmarkSample(
        id="sample_001",
        question="What is the capital of France?",
        context=["Paris is the capital and most populous city of France."],
        reference="Paris",
        setting="accurate_context",
        response="Paris is the capital.",
    )
    assert sample.id == "sample_001"
    assert len(sample.context) == 1
    assert sample.setting == "accurate_context"
