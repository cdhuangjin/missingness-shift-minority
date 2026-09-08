"""Primary and secondary classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true,
    y_score,
    y_pred,
    minority_label: int = 1,
    majority_label: int = 0,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred)

    eps = 1e-12
    rec_min = float(recall_score(y_true, y_pred, pos_label=minority_label))
    rec_maj = float(recall_score(y_true, y_pred, pos_label=majority_label))
    gmean = float(np.sqrt(max(rec_min * rec_maj, 0.0)))

    return {
        "AUPRC": float(average_precision_score(y_true, y_score)),
        "AUROC": float(roc_auc_score(y_true, y_score)),
        "minority_recall": rec_min,
        "minority_f1": float(f1_score(y_true, y_pred, pos_label=minority_label)),
        "minority_precision": float(
            precision_score(y_true, y_pred, pos_label=minority_label, zero_division=0)
        ),
        "majority_recall": rec_maj,
        "majority_f1": float(f1_score(y_true, y_pred, pos_label=majority_label)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Gmean": gmean,
    }


def threshold_05(probabilities: np.ndarray) -> np.ndarray:
    return (np.asarray(probabilities) >= 0.5).astype(int)
