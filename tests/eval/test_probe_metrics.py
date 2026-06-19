"""Unit tests for probe classification metrics (synthetic, no model)."""

from __future__ import annotations

import numpy as np

from src.eval.probe_metrics import classification_metrics, oracle_threshold


def test_perfect_separation():
    # correct (y=0) low scores, incorrect (y=1) high scores
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1])
    m = classification_metrics(score, y, threshold=0.5)
    assert m["auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["n"] == 6
    assert m["mean_score_incorrect"] > m["mean_score_correct"]


def test_threshold_changes_predictions():
    score = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    lo = classification_metrics(score, y, threshold=0.3)  # flags 0.4 as positive
    assert lo["pred_pos_rate"] == 0.75
    hi = classification_metrics(score, y, threshold=0.5)
    assert hi["pred_pos_rate"] == 0.5
    assert hi["accuracy"] == 1.0


def test_oracle_threshold_recovers_separator():
    score = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    thr, val = oracle_threshold(score, y)
    assert 0.2 < thr < 0.8
    assert val == 1.0
