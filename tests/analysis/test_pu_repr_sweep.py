"""Unit tests for progress_usefulness_v0 P2 sweep pure core (numpy only)."""

import numpy as np

from scripts.analysis.pu_repr_sweep import (
    auc_from_scores,
    fit_direction,
    pair_accuracy,
    permutation_accuracy,
    random_direction_accuracy,
)


def test_fit_direction_is_unit_mean():
    dh = np.array([[2.0, 0.0], [4.0, 0.0]])  # all along +x
    d = fit_direction(dh)
    assert np.allclose(d, [1.0, 0.0])


def test_pair_accuracy_aligned_and_anti():
    dh = np.array([[1.0, 0.0], [2.0, 0.0]])
    assert pair_accuracy(dh, np.array([1.0, 0.0])) == 1.0
    assert pair_accuracy(dh, np.array([-1.0, 0.0])) == 0.0


def test_auc_perfect_and_reversed():
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert auc_from_scores(scores, labels) == 1.0
    assert auc_from_scores(-scores, labels) == 0.0


def test_auc_degenerate_labels_is_nan():
    assert np.isnan(auc_from_scores(np.array([1.0, 2.0]), np.array([1, 1])))


def test_random_and_permutation_nulls_near_half():
    rng = np.random.default_rng(0)
    # symmetric deltas: half +x, half -x -> any fixed direction ~0.5
    dh = np.vstack([np.tile([1.0, 0.0], (50, 1)), np.tile([-1.0, 0.0], (50, 1))])
    d = np.array([1.0, 0.0])
    assert abs(random_direction_accuracy(dh, 200, rng) - 0.5) < 0.1
    assert abs(permutation_accuracy(dh, d, 200, rng) - 0.5) < 0.1


def test_pair_accuracy_empty_is_nan():
    assert np.isnan(pair_accuracy(np.empty((0, 3)), np.ones(3)))
