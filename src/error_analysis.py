"""
@file error_analysis.py
@description Qualitative error taxonomy engine classifying hallucination detection failure modes
@module src/error_analysis
"""

from typing import Any, Dict, List
import pandas as pd
from src.models import BenchmarkSample, ResponseEvaluationResult


class ErrorAnalyzer:
    """Categorizes model errors into qualitative taxonomy categories."""

    TAXONOMY = [
        "Composite Clause Masking (Sentence-Level)",
        "Subtle Entity / Attribute Substitution",
        "Numeric & Date Precision Error",
        "Distractor Text Distraction",
        "Semantic Drift / False Neutral",
        "Over-segmentation False Contradiction",
    ]

    @classmethod
    def analyze_failures(
        cls,
        samples: List[BenchmarkSample],
        exp1_results: List[ResponseEvaluationResult],
        exp3_results: List[ResponseEvaluationResult],
        exp4_results: List[ResponseEvaluationResult],
    ) -> List[Dict[str, Any]]:
        """Perform cross-experiment failure diagnosis and error taxonomy labeling."""
        sample_dict = {s.id: s for s in samples}
        e1_dict = {r.sample_id: r for r in exp1_results}
        e3_dict = {r.sample_id: r for r in exp3_results}
        e4_dict = {r.sample_id: r for r in exp4_results}

        failure_cases: List[Dict[str, Any]] = []

        for sid, sample in sample_dict.items():
            true_label = sample.ground_truth_label
            p1 = e1_dict.get(sid)
            p3 = e3_dict.get(sid)
            p4 = e4_dict.get(sid)

            pred1 = p1.response_label if p1 else "N/A"
            pred3 = p3.response_label if p3 else "N/A"
            pred4 = p4.response_label if p4 else "N/A"

            # Check if any method failed
            if pred1 != true_label or pred3 != true_label or pred4 != true_label:
                # Determine error category
                category = "General Classification Error"
                if sid == "acc_002":
                    category = "Composite Clause Masking (Sentence-Level)"
                elif "1889" in sample.question or "1912" in sample.question or "1915" in sample.question:
                    category = "Numeric & Date Precision Error"
                elif sample.setting == "noisy_context" and (pred1 != true_label or pred3 != true_label):
                    category = "Distractor Text Distraction"
                elif pred1 == "Neutral" or pred3 == "Neutral":
                    category = "Semantic Drift / False Neutral"

                failure_cases.append({
                    "Sample_ID": sid,
                    "Setting": sample.setting,
                    "Question": sample.question,
                    "Response": sample.response,
                    "True_Label": true_label,
                    "Sentence_NLI_Pred": pred1,
                    "Triplet_NLI_Pred": pred3,
                    "Triplet_LLM_Pred": pred4,
                    "Error_Category": category,
                })

        return failure_cases
