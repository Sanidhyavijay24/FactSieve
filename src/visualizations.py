"""
@file visualizations.py
@description Generates publication-quality charts, confusion matrix heatmaps, and comparison plots
@module src/visualizations
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 14


def plot_rq1_granularity_comparison(
    csv_path: str = "results/tables/rq1_granularity_comparison.csv",
    output_path: str = "results/plots/rq1_granularity_comparison.png",
) -> None:
    """Plot bar chart comparing Sentence vs Claims vs Triplets across accuracy, Macro-F1, and Hallucination F1."""
    p = Path(csv_path)
    if not p.exists():
        return

    df = pd.read_csv(p)
    labels = ["Sentence -> NLI\n(Baseline)", "Claims -> NLI\n(Atomic)", "Triplets -> NLI\n(RefChecker)"]
    
    # Parse percentages and floats
    acc = [float(x.replace("%", "")) for x in df["Accuracy (%)"]]
    macro_f1 = [float(x) * 100 for x in df["Macro-F1"]]
    h_f1 = [float(x) * 100 for x in df["Hallucination F1"]]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - width, acc, width, label="3-Way Accuracy (%)", color="#3b82f6")
    rects2 = ax.bar(x, macro_f1, width, label="Macro-F1 (x100)", color="#10b981")
    rects3 = ax.bar(x + width, h_f1, width, label="Hallucination F1 (x100)", color="#8b5cf6")

    ax.set_ylabel("Score (%)")
    ax.set_title("RQ1: Representation Granularity Comparison", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="semibold")
    ax.set_ylim(40, 105)
    ax.legend(loc="upper left")

    # Add data labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved RQ1 chart to '{output_path}'")


def plot_rq2_checker_comparison(
    csv_path: str = "results/tables/rq2_checker_comparison.csv",
    output_path: str = "results/plots/rq2_checker_comparison.png",
) -> None:
    """Plot bar chart comparing NLI vs LLM Judge vs Semantic Similarity."""
    p = Path(csv_path)
    if not p.exists():
        return

    df = pd.read_csv(p)
    labels = ["DeBERTa-v3 NLI\n(Triplets)", "LLM Judge\n(Gemini)", "Semantic Similarity\n(MiniLM)"]
    
    acc = [float(x.replace("%", "")) for x in df["Accuracy (%)"]]
    prec = [float(x) * 100 for x in df["Hallucination Precision"]]
    rec = [float(x) * 100 for x in df["Hallucination Recall"]]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    rects1 = ax.bar(x - width, acc, width, label="3-Way Accuracy (%)", color="#2563eb")
    rects2 = ax.bar(x, prec, width, label="Hallucination Precision (%)", color="#059669")
    rects3 = ax.bar(x + width, rec, width, label="Hallucination Recall (%)", color="#d97706")

    ax.set_ylabel("Score (%)")
    ax.set_title("RQ2: Fact-Checking Strategy Ablation (Fixed Triplets)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="semibold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved RQ2 chart to '{output_path}'")


def plot_all_confusion_matrices(
    output_path: str = "results/plots/confusion_matrices_grid.png",
) -> None:
    """Plot a 2x2 grid of confusion matrices for all 4 primary experiments."""
    files = {
        "Exp 1: Sentence -> NLI": "results/tables/exp1_sentence_nli_metrics.json",
        "Exp 2: Claims -> NLI": "results/tables/exp2_claim_nli_metrics.json",
        "Exp 3: Triplets -> NLI": "results/tables/exp3_triplet_nli_metrics.json",
        "Exp 4: Triplets -> LLM Judge": "results/tables/exp4_triplet_llm_metrics.json",
    }

    labels = ["Entailment", "Neutral", "Contradiction"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()

    for idx, (title, fpath) in enumerate(files.items()):
        p = Path(fpath)
        if not p.exists():
            continue

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        cm = data.get("confusion_matrix", {}).get("matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Entail", "Neutral", "Contra"],
            yticklabels=["Entail", "Neutral", "Contra"],
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(f"{title}\nAcc: {data.get('accuracy', 0)*100:.1f}% | F1: {data.get('macro_f1', 0):.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("Confusion Matrix Grid across Granularities and Checkers", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrices grid to '{output_path}'")


def generate_all_plots() -> None:
    """Generate all publication-ready visual artifacts."""
    plot_rq1_granularity_comparison()
    plot_rq2_checker_comparison()
    plot_all_confusion_matrices()
