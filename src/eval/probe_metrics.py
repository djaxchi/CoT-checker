"""Binary step-classification metrics for evaluating a dense probe on PRM800K.

Label convention (matches the loaders): y = 1 is INCORRECT, y = 0 is correct,
and the probe score is P(incorrect) = sigmoid(h.w + b), so a higher score means
"more likely an error" and `roc_auc_score(y, score)` is oriented correctly.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)


def classification_metrics(score: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Threshold-free AUC plus point metrics at a given decision threshold."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = (score >= threshold).astype(int)
    two = len(np.unique(y)) > 1
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y, score)) if two else float("nan"),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "pred_pos_rate": float(pred.mean()),
        "mean_score_incorrect": float(score[y == 1].mean()) if (y == 1).any() else float("nan"),
        "mean_score_correct": float(score[y == 0].mean()) if (y == 0).any() else float("nan"),
    }


def oracle_threshold(score: np.ndarray, y: np.ndarray,
                     grid: np.ndarray | None = None,
                     metric: str = "balanced_accuracy") -> tuple[float, float]:
    """Best threshold on THIS set (a ceiling, not deployable). Returns (thr, value)."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    scorer = balanced_accuracy_score if metric == "balanced_accuracy" else \
        (lambda yt, yp: f1_score(yt, yp, pos_label=1, zero_division=0))
    best_v, best_t = -1.0, 0.5
    for t in grid:
        v = scorer(y, (score >= t).astype(int))
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return best_t, best_v
