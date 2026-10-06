"""
@file run_rq1_comparison.py
@description Comparative analysis table generator for RQ1 (Sentence vs Claims vs Triplets)
@module experiments/run_rq1_comparison
"""

import json
from pathlib import Path
import pandas as pd


def generate_rq1_comparison_table() -> None:
    """Read experiment 1, 2, and 3 metric results and print consolidated comparison."""
    files = {
        "Sentence -> NLI (Baseline)": "results/tables/exp1_sentence_nli_metrics.json",
        "Claims -> NLI (Atomic)": "results/tables/exp2_claim_nli_metrics.json",
        "Triplets -> NLI (RefChecker)": "results/tables/exp3_triplet_nli_metrics.json",
    }

    rows = []
    for exp_name, filepath in files.items():
        p = Path(filepath)
        if not p.exists():
            print(f"[-] Warning: '{filepath}' not found. Please run the corresponding experiment script first.")
            continue

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows.append({
            "Representation": exp_name,
            "Accuracy (%)": f"{data.get('accuracy', 0.0) * 100:.2f}%",
            "Macro-F1": f"{data.get('macro_f1', 0.0):.4f}",
            "Hallucination Precision": f"{data.get('hallucination_detection', {}).get('precision', 0.0):.4f}",
            "Hallucination Recall": f"{data.get('hallucination_detection', {}).get('recall', 0.0):.4f}",
            "Hallucination F1": f"{data.get('hallucination_detection', {}).get('f1', 0.0):.4f}",
            "Accurate Context Acc": f"{data.get('by_setting', {}).get('accurate_context', {}).get('accuracy', 0.0) * 100:.1f}%",
            "Noisy Context Acc": f"{data.get('by_setting', {}).get('noisy_context', {}).get('accuracy', 0.0) * 100:.1f}%",
            "Zero Context Acc": f"{data.get('by_setting', {}).get('zero_context', {}).get('accuracy', 0.0) * 100:.1f}%",
        })

    if not rows:
        print("No completed experiment records found yet.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(" RESEARCH QUESTION 1 (RQ1) GRANULARITY COMPARISON TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    out_csv = "results/tables/rq1_granularity_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"[+] Exported comparative summary to '{out_csv}'\n")


if __name__ == "__main__":
    generate_rq1_comparison_table()
