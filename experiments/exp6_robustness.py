"""
@file exp6_robustness.py
@description Experiment 6: Robustness Evaluation under Clean vs Noisy Evidence (RQ3)
@module experiments/exp6_robustness
"""

import argparse
import json
import time
from typing import Any, Dict, List
import pandas as pd
from src.config import load_config
from src.data_loader import load_benchmark_samples
from src.evaluation import Evaluator
from src.extract_triplets import TripletExtractor
from src.models import BenchmarkSample, ResponseEvaluationResult
from src.nli_checker import NLIChecker
from src.scoring import aggregate_response


def run_experiment_6(config_path: str = "configs/config.yaml") -> None:
    """Run Experiment 6: Robustness Study comparing Clean Reference vs Noisy Distractor Context."""
    print("=" * 70)
    print(" EXPERIMENT 6: Robustness Analysis (Clean vs Noisy Context - RQ3)")
    print("=" * 70)

    cfg = load_config(config_path)
    all_samples: List[BenchmarkSample] = load_benchmark_samples(cfg.data.benchmark_file)

    # Filter for noisy_context samples
    noisy_samples = [s for s in all_samples if s.setting == "noisy_context"]
    print(f"[*] Loaded {len(noisy_samples)} noisy-context evaluation samples.")

    triplet_extractor = TripletExtractor(cfg)
    nli_checker = NLIChecker(config=cfg)

    # Condition A: Checking against Ground Truth Clean Reference
    results_clean: List[ResponseEvaluationResult] = []
    # Condition B: Checking against Noisy Context (with Distractors)
    results_noisy: List[ResponseEvaluationResult] = []

    print("\n--- Running Robustness Comparison on Distractor Samples ---")
    for idx, sample in enumerate(noisy_samples, start=1):
        resp_text = sample.response or ""
        triplets = triplet_extractor.extract(resp_text, sample_id=sample.id)

        # Condition A: Clean Reference
        units_clean = nli_checker.verify_units(
            sample_id=sample.id,
            units=triplets,
            evidence=sample.reference,
        )
        res_clean = aggregate_response(
            sample_id=sample.id,
            setting="clean_evidence",
            verifier_name="nli_deberta",
            granularity="triplet",
            unit_results=units_clean,
            rule=cfg.scoring.aggregation_rule,
        )
        results_clean.append(res_clean)

        # Condition B: Raw Context with Distractors
        noisy_evidence = "\n".join(sample.context) if sample.context else sample.reference
        units_noisy = nli_checker.verify_units(
            sample_id=sample.id,
            units=triplets,
            evidence=noisy_evidence,
        )
        res_noisy = aggregate_response(
            sample_id=sample.id,
            setting="noisy_distractor_evidence",
            verifier_name="nli_deberta",
            granularity="triplet",
            unit_results=units_noisy,
            rule=cfg.scoring.aggregation_rule,
        )
        results_noisy.append(res_noisy)

        sym_c = "✓" if res_clean.response_label == sample.ground_truth_label else "✗"
        sym_n = "✓" if res_noisy.response_label == sample.ground_truth_label else "✗"
        print(
            f"[{idx:02d}/{len(noisy_samples)}] {sample.id} | True: {sample.ground_truth_label:<13} | "
            f"Clean Evidence: {res_clean.response_label:<13} [{sym_c}] | "
            f"Noisy Evidence: {res_noisy.response_label:<13} [{sym_n}]"
        )

    metrics_clean = Evaluator.evaluate_predictions(results_clean, noisy_samples)
    metrics_noisy = Evaluator.evaluate_predictions(results_noisy, noisy_samples)

    robustness_summary: Dict[str, Any] = {
        "num_samples": len(noisy_samples),
        "clean_reference": {
            "accuracy": metrics_clean.get("accuracy"),
            "macro_f1": metrics_clean.get("macro_f1"),
            "hallucination_f1": metrics_clean.get("hallucination_detection", {}).get("f1"),
        },
        "noisy_distractor_context": {
            "accuracy": metrics_noisy.get("accuracy"),
            "macro_f1": metrics_noisy.get("macro_f1"),
            "hallucination_f1": metrics_noisy.get("hallucination_detection", {}).get("f1"),
        },
        "accuracy_delta": round(float(metrics_noisy.get("accuracy", 0.0) - metrics_clean.get("accuracy", 0.0)), 4),
    }

    json_out = "results/tables/exp6_robustness_metrics.json"
    csv_out = "results/tables/exp6_robustness_metrics.csv"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(robustness_summary, f, indent=2)

    rob_df = pd.DataFrame([
        {
            "Evidence Condition": "Clean Ground Truth Reference",
            "Accuracy (%)": f"{metrics_clean.get('accuracy', 0.0) * 100:.2f}%",
            "Macro-F1": f"{metrics_clean.get('macro_f1', 0.0):.4f}",
            "Hallucination F1": f"{metrics_clean.get('hallucination_detection', {}).get('f1', 0.0):.4f}",
        },
        {
            "Evidence Condition": "Noisy Context (with Distractors)",
            "Accuracy (%)": f"{metrics_noisy.get('accuracy', 0.0) * 100:.2f}%",
            "Macro-F1": f"{metrics_noisy.get('macro_f1', 0.0):.4f}",
            "Hallucination F1": f"{metrics_noisy.get('hallucination_detection', {}).get('f1', 0.0):.4f}",
        },
    ])
    rob_df.to_csv(csv_out, index=False)

    print("\n" + "=" * 70)
    print(" EXPERIMENT 6 ROBUSTNESS SUMMARY (RQ3)")
    print("=" * 70)
    print(rob_df.to_string(index=False))
    print("-" * 70)
    print(f" Performance Delta (Noisy vs Clean Accuracy): {robustness_summary['accuracy_delta'] * 100:+.2f}%")
    print(f"[+] Saved robustness tables to '{json_out}' and '{csv_out}'")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 6: Robustness under Clean vs Noisy Evidence")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_experiment_6(args.config)
