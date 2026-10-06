"""
@file evaluation.py
@description Comprehensive evaluation harness calculating Accuracy, Macro-F1, Precision/Recall, and Confusion Matrices
@module src/evaluation
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from src.models import BenchmarkSample, ResponseEvaluationResult


class Evaluator:
    """Computes rigorous classification metrics and generates evaluation summaries."""

    LABELS = ["Entailment", "Neutral", "Contradiction"]

    @classmethod
    def evaluate_predictions(
        cls,
        eval_results: List[ResponseEvaluationResult],
        ground_truth_samples: List[BenchmarkSample],
    ) -> Dict[str, Any]:
        """Compute full evaluation metrics comparing predicted responses against gold labels.

        Args:
            eval_results (List[ResponseEvaluationResult]): Predicted evaluation outputs.
            ground_truth_samples (List[BenchmarkSample]): Reference ground truth benchmark instances.

        Returns:
            Dict[str, Any]: Structured dictionary of computed metrics.
        """
        sample_map = {s.id: s for s in ground_truth_samples}

        y_true: List[str] = []
        y_pred: List[str] = []
        settings: List[str] = []

        for res in eval_results:
            sample = sample_map.get(res.sample_id)
            if sample and sample.ground_truth_label:
                # Map gold labels if needed (e.g. Factual -> Entailment, Hallucinated -> Contradiction)
                gold = sample.ground_truth_label
                if gold == "Factual":
                    gold = "Entailment"
                elif gold == "Hallucinated":
                    gold = "Contradiction"

                y_true.append(gold)
                y_pred.append(res.response_label)
                settings.append(res.setting)

        if not y_true:
            return {"error": "No matching ground truth labels available for evaluation."}

        # Overall 3-way metrics
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        # Binary Hallucination / Contradiction Detection Metrics
        # True positive = Contradiction (Hallucination detected)
        y_true_binary = [1 if y == "Contradiction" else 0 for y in y_true]
        y_pred_binary = [1 if y == "Contradiction" else 0 for y in y_pred]
        p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average="binary", zero_division=0
        )

        # Confusion matrix for 3-way classification
        cm = confusion_matrix(y_true, y_pred, labels=cls.LABELS)

        metrics = {
            "num_samples": len(y_true),
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "hallucination_detection": {
                "precision": round(float(p_bin), 4),
                "recall": round(float(r_bin), 4),
                "f1": round(float(f1_bin), 4),
            },
            "confusion_matrix": {
                "labels": cls.LABELS,
                "matrix": cm.tolist(),
            },
            "by_setting": {},
        }

        # Per-setting breakdown (Zero, Noisy, Accurate)
        unique_settings = sorted(list(set(settings)))
        for st in unique_settings:
            sub_true = [yt for yt, s in zip(y_true, settings) if s == st]
            sub_pred = [yp for yp, s in zip(y_pred, settings) if s == st]
            st_acc = accuracy_score(sub_true, sub_pred)
            st_f1 = f1_score(sub_true, sub_pred, average="macro", zero_division=0)
            metrics["by_setting"][st] = {
                "count": len(sub_true),
                "accuracy": round(float(st_acc), 4),
                "macro_f1": round(float(st_f1), 4),
            }

        return metrics

    @classmethod
    def save_metrics(
        cls,
        metrics: Dict[str, Any],
        output_json_path: str,
        output_csv_path: Optional[str] = None,
    ) -> None:
        """Save evaluation metrics to disk in JSON and summary CSV formats.

        Args:
            metrics (Dict[str, Any]): Metrics dictionary.
            output_json_path (str): Filepath for JSON export.
            output_csv_path (Optional[str]): Filepath for CSV summary table.
        """
        json_path = Path(output_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if output_csv_path:
            csv_path = Path(output_csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create tabular summary row
            row = {
                "Accuracy": metrics.get("accuracy"),
                "Macro_F1": metrics.get("macro_f1"),
                "Hallucination_Precision": metrics.get("hallucination_detection", {}).get("precision"),
                "Hallucination_Recall": metrics.get("hallucination_detection", {}).get("recall"),
                "Hallucination_F1": metrics.get("hallucination_detection", {}).get("f1"),
            }
            for st, st_data in metrics.get("by_setting", {}).items():
                row[f"{st}_Acc"] = st_data.get("accuracy")
                row[f"{st}_F1"] = st_data.get("macro_f1")

            df = pd.DataFrame([row])
            df.to_csv(csv_path, index=False)

    @classmethod
    def append_markdown_log(
        cls,
        exp_name: str,
        metrics: Dict[str, Any],
        log_md_path: str = "results/experiment_log.md",
        notes: str = "",
    ) -> None:
        """Append formatted markdown summary to experiment log file."""
        log_path = Path(log_md_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"\n## {exp_name}",
            f"* **Total Samples:** {metrics.get('num_samples')}",
            f"* **3-Way Accuracy:** **{metrics.get('accuracy', 0.0) * 100:.2f}%**",
            f"* **Macro-F1:** **{metrics.get('macro_f1', 0.0):.4f}**",
            f"* **Hallucination Precision:** {metrics.get('hallucination_detection', {}).get('precision', 0.0):.4f}",
            f"* **Hallucination Recall:** {metrics.get('hallucination_detection', {}).get('recall', 0.0):.4f}",
            f"* **Hallucination F1:** {metrics.get('hallucination_detection', {}).get('f1', 0.0):.4f}",
            "",
            "### Setting Breakdown",
            "| Setting | Samples | Accuracy | Macro-F1 |",
            "| :--- | :---: | :---: | :---: |",
        ]

        for st, st_data in metrics.get("by_setting", {}).items():
            lines.append(
                f"| `{st}` | {st_data.get('count', '-')} | {st_data.get('accuracy', 0.0) * 100:.1f}% | {st_data.get('macro_f1', 0.0):.4f} |"
            )

        cm = metrics.get("confusion_matrix", {}).get("matrix", [])
        if cm and len(cm) == 3:
            lines.extend([
                "",
                "### Confusion Matrix",
                "| True \\ Pred | Pred Entailment | Pred Neutral | Pred Contradiction |",
                "| :--- | :---: | :---: | :---: |",
                f"| **True Entailment** | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} |",
                f"| **True Neutral** | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} |",
                f"| **True Contradiction** | {cm[2][0]} | {cm[2][1]} | {cm[2][2]} |",
            ])

        if notes:
            lines.extend(["", f"### Notes", notes])

        lines.extend(["", "---"])

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
