"""Per-label and aggregate multi-label metrics (Precision, Recall, F1)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.config import LABELS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return per-label P/R/F1 and macro/micro F1.

    Inputs are (N, 6) binary numpy arrays. zero_division=0 by sklearn default
    set explicitly to avoid warnings on rare labels with empty predicted positive sets.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape, f"shape mismatch: {y_true.shape} vs {y_pred.shape}"

    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), average=None, zero_division=0
    )
    per_label = {}
    for i, lbl in enumerate(LABELS):
        per_label[lbl] = {
            "precision": float(p[i]),
            "recall":    float(r[i]),
            "f1":        float(f[i]),
            "support":   int(support[i]),
        }

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    return {"per_label": per_label, "macro_f1": macro_f1, "micro_f1": micro_f1}


def aggregate_folds(fold_metrics: list[dict]) -> dict:
    """Mean and std across folds for every metric."""
    keys_per_label = ["precision", "recall", "f1", "support"]
    agg_per_label = {}
    for lbl in LABELS:
        agg_per_label[lbl] = {}
        for k in keys_per_label:
            vals = [fm["per_label"][lbl][k] for fm in fold_metrics]
            agg_per_label[lbl][f"{k}_mean"] = float(np.mean(vals))
            agg_per_label[lbl][f"{k}_std"] = float(np.std(vals, ddof=0))

    macro_vals = [fm["macro_f1"] for fm in fold_metrics]
    micro_vals = [fm["micro_f1"] for fm in fold_metrics]
    return {
        "per_label": agg_per_label,
        "macro_f1_mean": float(np.mean(macro_vals)),
        "macro_f1_std":  float(np.std(macro_vals, ddof=0)),
        "micro_f1_mean": float(np.mean(micro_vals)),
        "micro_f1_std":  float(np.std(micro_vals, ddof=0)),
        "per_fold_macro_f1": macro_vals,
        "per_fold_micro_f1": micro_vals,
    }
