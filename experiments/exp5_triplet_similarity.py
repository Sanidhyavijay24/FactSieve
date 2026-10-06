"""
@file exp5_triplet_similarity.py
@description Experiment 5: Semantic Similarity Baseline (Triplets -> Semantic Similarity)
@module experiments/exp5_triplet_similarity
"""

import argparse
import time
from typing import List
import pandas as pd
from src.config import load_config
from src.data_loader import load_benchmark_samples
from src.evaluation import Evaluator
from src.extract_triplets import TripletExtractor
from src.models import BenchmarkSample, ResponseEvaluationResult
from src.scoring import aggregate_response
from src.semantic_checker import SemanticSimilarityChecker


def run_experiment_5(config_path: str = "configs/config.yaml") -> None:
    """Run Experiment 5: Triplets -> Semantic Similarity Verification."""
    print("=" * 70)
    print(" EXPERIMENT 5: Triplets -> Semantic Similarity Baseline")
    print("=" * 70)

    cfg = load_config(config_path)
    samples: List[BenchmarkSample] = load_benchmark_samples(cfg.data.benchmark_file)
    print(f"[*] Loaded {len(samples)} benchmark samples from '{cfg.data.benchmark_file}'")

    triplet_extractor = TripletExtractor(cfg)
    print(f"[*] TripletExtractor ready.")

    sim_checker = SemanticSimilarityChecker(config=cfg)
    print(f"[*] Loaded SentenceTransformer '{cfg.verifiers.semantic_similarity.model_name}' on device '{sim_checker.device}'")
    print(f"[*] Similarity Threshold: {sim_checker.threshold}")

    start_time = time.time()
    evaluation_results: List[ResponseEvaluationResult] = []

    print("\n--- Running Evaluation Pipeline ---")
    for idx, sample in enumerate(samples, start=1):
        resp_text = sample.response or ""
        triplets = triplet_extractor.extract(resp_text, sample_id=sample.id)

        evidence = sample.reference

        # Verify triplets using Semantic Embedding Similarity
        unit_results = sim_checker.verify_units(
            sample_id=sample.id,
            units=triplets,
            evidence=evidence,
        )

        # Aggregate response decision
        eval_result = aggregate_response(
            sample_id=sample.id,
            setting=sample.setting,
            verifier_name=f"semantic_{cfg.verifiers.semantic_similarity.model_name.split('/')[-1]}",
            granularity="triplet",
            unit_results=unit_results,
            rule=cfg.scoring.aggregation_rule,
        )
        evaluation_results.append(eval_result)

        pred_symbol = "✓" if eval_result.response_label == sample.ground_truth_label else "✗"
        print(
            f"[{idx:02d}/{len(samples)}] Sample {sample.id} ({sample.setting:<16}) | "
            f"Triplets: {len(triplets):02d} | True: {sample.ground_truth_label:<13} | "
            f"Pred: {eval_result.response_label:<13} [{pred_symbol}]"
        )

    elapsed = time.time() - start_time
    print(f"\n[*] Completed evaluation in {elapsed:.2f}s")

    # Compute metrics
    metrics = Evaluator.evaluate_predictions(evaluation_results, samples)

    # Save results
    json_out = "results/tables/exp5_triplet_similarity_metrics.json"
    csv_out = "results/tables/exp5_triplet_similarity_metrics.csv"
    Evaluator.save_metrics(metrics, json_out, csv_out)
    Evaluator.append_markdown_log("Experiment 5: Triplets -> Semantic Similarity Baseline", metrics)

    print("\n" + "=" * 70)
    print(" EXPERIMENT 5 SUMMARY RESULTS (Triplets -> Semantic Similarity)")
    print("=" * 70)
    print(f" Total Samples Evaluated : {metrics.get('num_samples')}")
    print(f" 3-Way Accuracy          : {metrics.get('accuracy') * 100:.2f}%")
    print(f" Macro-F1                : {metrics.get('macro_f1'):.4f}")
    print(f" Hallucination Precision : {metrics.get('hallucination_detection', {}).get('precision'):.4f}")
    print(f" Hallucination Recall    : {metrics.get('hallucination_detection', {}).get('recall'):.4f}")
    print(f" Hallucination F1        : {metrics.get('hallucination_detection', {}).get('f1'):.4f}")
    print("-" * 70)
    print(" Performance by Setting:")
    for st, st_data in metrics.get("by_setting", {}).items():
        print(f"   • {st:<18} : Acc = {st_data.get('accuracy') * 100:.1f}% | Macro-F1 = {st_data.get('macro_f1'):.4f}")
    print("-" * 70)
    print(" Confusion Matrix (Labels: Entailment, Neutral, Contradiction):")
    cm_df = pd.DataFrame(
        metrics.get("confusion_matrix", {}).get("matrix"),
        index=[f"True_{l}" for l in Evaluator.LABELS],
        columns=[f"Pred_{l}" for l in Evaluator.LABELS],
    )
    print(cm_df.to_string())
    print(f"\n[+] Saved metrics to '{json_out}' and '{csv_out}'")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 5: Triplets -> Semantic Similarity")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_experiment_5(args.config)
