"""Tests for the in-domain step metrics added to train_easy_probe_method.

auroc_numpy is the sklearn-free AUROC used in the slurm env (no sklearn there);
it must match sklearn on ordinary inputs, handle ties, and return NaN for a
single-class target. step_binary_metrics must be a correct macro-F1 readout.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_easy_probe_method import auroc_numpy, step_binary_metrics  # noqa: E402


def test_auroc_matches_sklearn_random():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    scores = rng.random(500).astype(np.float32)
    assert auroc_numpy(y, scores) == abs(roc_auc_score(y, scores)) or math.isclose(
        auroc_numpy(y, scores), roc_auc_score(y, scores), abs_tol=1e-9
    )


def test_auroc_known_value():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert math.isclose(auroc_numpy(y, scores), 0.75, abs_tol=1e-12)


def test_auroc_perfect_and_reversed():
    y = np.array([0, 0, 1, 1])
    assert auroc_numpy(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc_numpy(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auroc_all_ties_is_half():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert math.isclose(auroc_numpy(y, scores), 0.5, abs_tol=1e-12)


def test_auroc_single_class_is_nan():
    assert math.isnan(auroc_numpy(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9])))


def test_step_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = step_binary_metrics(y, scores, 0.5)
    assert math.isclose(m["macro_f1"], 1.0, abs_tol=1e-9)
    assert math.isclose(m["accuracy"], 1.0, abs_tol=1e-9)
    assert math.isclose(m["pos_pred_rate"], 0.5, abs_tol=1e-9)


def test_step_metrics_all_one_class_prediction():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.9, 0.9, 0.9])  # everything predicted incorrect
    m = step_binary_metrics(y, scores, 0.5)
    assert m["pos_pred_rate"] == 1.0
    assert math.isclose(m["f1_correct"], 0.0, abs_tol=1e-12)
