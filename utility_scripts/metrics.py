# utils/metrics.py — Evaluation metrics including EER

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)
from typing import Dict, Tuple
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Main Metric for Spoof detection
def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.
    EER is where FAR (False Acceptance Rate) == FRR (False Rejection Rate).

    Args:
        labels : ground truth binary labels (0=bonafide, 1=spoof)
        scores : predicted probability of being spoof (class 1)
    Returns:
        eer       : Equal Error Rate (0–1)
        threshold : decision threshold at EER
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find threshold where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer      = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = thresholds[eer_idx]
    return float(eer), float(threshold)


def compute_all_metrics(labels: np.ndarray,
                        preds: np.ndarray,
                        scores: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        labels : ground truth binary labels
        preds  : predicted class labels (0 or 1)
        scores : predicted probability for class 1 (spoof score)
    Returns:
        dict of metric_name -> value
    """
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    eer, _ = compute_eer(labels, scores)
    cm   = confusion_matrix(labels, preds)

    # TN, FP, FN, TP
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn + 1e-8)  # False Acceptance Rate
        frr = fn / (fn + tp + 1e-8)  # False Rejection Rate
    else:
        far = frr = 0.0

    return {
        "accuracy":  round(acc  * 100, 2),
        "precision": round(prec * 100, 2),
        "recall":    round(rec  * 100, 2),
        "f1_score":  round(f1   * 100, 2),
        "eer":       round(eer  * 100, 2),
        "far":       round(far  * 100, 2),
        "frr":       round(frr  * 100, 2),
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "", feature: str = ""):
    """Pretty-print metrics table."""
    header = f"{'─'*50}"
    title  = f"  Model: {model_name}  |  Feature: {feature}"
    print(f"\n{header}")
    print(title)
    print(header)
    for k, v in metrics.items():
        bar_len = int(v / 2) if v <= 100 else 50
        bar = "█" * bar_len
        print(f"  {k:<12}: {v:>7.2f}%  {bar}")
    print(header)


def format_metrics_row(model_name: str, feature: str, metrics: Dict[str, float]) -> dict:
    """Format metrics as a flat dict for DataFrame creation."""
    row = {"model": model_name, "feature": feature}
    row.update(metrics)
    return row