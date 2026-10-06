"""
@file test_evaluation.py
@description Unit tests for metrics calculation and evaluation summaries
@module tests/test_evaluation
"""

import tempfile
from pathlib import Path
import pytest
from src.evaluation import Evaluator
from src.models import BenchmarkSample, ResponseEvaluationResult


def test_evaluator_metrics_calculation():
    """Verify standard accuracy, macro-f1, and hallucination detection metrics."""
    gt_samples = [
        BenchmarkSample(
            id="s1",
            question="Q1",
            reference="Ref1",
            setting="accurate_context",
            ground_truth_label="Entailment",
        ),
        BenchmarkSample(
            id="s2",
            question="Q2",
            reference="Ref2",
            setting="accurate_context",
            ground_truth_label="Contradiction",
        ),
        BenchmarkSample(
            id="s3",
            question="Q3",
            reference="Ref3",
            setting="noisy_context",
            ground_truth_label="Entailment",
        ),
    ]

    preds = [
        ResponseEvaluationResult(
            sample_id="s1",
            setting="accurate_context",
            verifier="nli_deberta",
            granularity="triplet",
            unit_results=[],
            response_label="Entailment",
            is_hallucinated=False,
            contradiction_count=0,
            neutral_count=0,
            entailment_count=1,
            score=1.0,
        ),
        ResponseEvaluationResult(
            sample_id="s2",
            setting="accurate_context",
            verifier="nli_deberta",
            granularity="triplet",
            unit_results=[],
            response_label="Contradiction",
            is_hallucinated=True,
            contradiction_count=1,
            neutral_count=0,
            entailment_count=0,
            score=0.0,
        ),
        ResponseEvaluationResult(
            sample_id="s3",
            setting="noisy_context",
            verifier="nli_deberta",
            granularity="triplet",
            unit_results=[],
            response_label="Contradiction",  # False positive
            is_hallucinated=True,
            contradiction_count=1,
            neutral_count=0,
            entailment_count=0,
            score=0.0,
        ),
    ]

    metrics = Evaluator.evaluate_predictions(preds, gt_samples)

    assert metrics["num_samples"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3, 0.01)
    assert "hallucination_detection" in metrics
    assert "by_setting" in metrics
    assert "accurate_context" in metrics["by_setting"]
    assert metrics["by_setting"]["accurate_context"]["accuracy"] == 1.0


def test_evaluator_save_metrics():
    """Verify exporting metrics to JSON and CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "metrics.json"
        csv_file = Path(tmpdir) / "metrics.csv"

        sample_metrics = {
            "num_samples": 2,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "hallucination_detection": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "by_setting": {"accurate_context": {"count": 2, "accuracy": 1.0, "macro_f1": 1.0}},
        }

        Evaluator.save_metrics(sample_metrics, str(json_file), str(csv_file))

        assert json_file.exists()
        assert csv_file.exists()
        assert json_file.stat().st_size > 0
        assert csv_file.stat().st_size > 0
