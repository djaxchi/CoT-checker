"""Step and ProcessBench metrics for bidirectional_token_probe_v1."""

import numpy as np
import pytest

from src.eval.contextual_probe_metrics import (
    auroc, best_f1_threshold, best_pb_threshold, pb_trace_metrics, pb_trivial, step_metrics,
)


def reference_pb(rows, thr):
    """Copy of scripts/evaluate_processbench_from_scores.py's loop (strict >)."""
    n_error = n_correct = ae = ac = 0
    for scores, label in rows:
        pred = next((i for i, s in enumerate(scores) if s > thr), -1)
        if label == -1:
            n_correct += 1; ac += pred == -1
        else:
            n_error += 1; ae += pred == label
    acc_e = ae / max(n_error, 1); acc_c = ac / max(n_correct, 1)
    return acc_e, acc_c, (2 * acc_e * acc_c / (acc_e + acc_c) if acc_e + acc_c else 0.0)


def test_pb_matches_reference_including_error_free_and_post_error():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(300):
        T = int(rng.integers(1, 10))
        lab = int(rng.integers(-1, T))
        rows.append((list(rng.random(T)), lab))
    thr = 0.73  # never equal to a sampled score
    m = pb_trace_metrics([r[0] for r in rows], [r[1] for r in rows], thr)
    ae, ac, f1 = reference_pb(rows, thr)
    assert m["Acc_error"] == pytest.approx(ae) and m["Acc_correct"] == pytest.approx(ac)
    assert m["F1_PB"] == pytest.approx(f1)


def test_pb_hand_case_post_error_scores_do_not_matter_before_crossing():
    seqs = [[0.1, 0.9, 0.95], [0.1, 0.2, 0.3], [0.8, 0.1], [0.1, 0.2, 0.99]]
    labels = [1, -1, 1, 2]
    m = pb_trace_metrics(seqs, labels, 0.5)
    assert m["Acc_error"] == pytest.approx(2 / 3)  # trace 0 and 3 hit, trace 2 premature
    assert m["Acc_correct"] == 1.0
    assert m["premature_alarm_rate_err"] == pytest.approx(1 / 3)
    triv = pb_trivial(labels, [3, 3, 2, 3])
    assert triv["always_no_error"]["F1_PB"] == 0.0
    assert triv["always_no_error"]["Acc_correct"] == 1.0


def test_best_f1_threshold_ties_prefer_higher():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([1, 0, 1, 0])
    # thr 0.9 -> F1 = 2/3 ; thr 0.7 -> P=2/3,R=1 -> F1=0.8 ; thr 0.6 -> F1=2/3
    t, f = best_f1_threshold(s, y)
    assert t == 0.7 and f == pytest.approx(0.8)
    s2 = np.array([0.9, 0.5, 0.4, 0.1])
    y2 = np.array([1, 0, 0, 1])
    # thr 0.9 -> F1 2/3 ; thr 0.1 -> P=.5,R=1 -> 2/3 : tie, higher wins
    t2, f2 = best_f1_threshold(s2, y2)
    assert t2 == 0.9 and f2 == pytest.approx(2 / 3)


def test_best_f1_threshold_matches_bruteforce():
    rng = np.random.default_rng(1)
    s = np.round(rng.random(500), 2)
    y = (rng.random(500) < 0.2).astype(int)
    t, f = best_f1_threshold(s, y)
    best = max((step_metrics(s, y, c)["f1"], c) for c in np.unique(s))
    assert f == pytest.approx(best[0])
    ties = [c for c in np.unique(s) if step_metrics(s, y, c)["f1"] == pytest.approx(best[0])]
    assert t == max(ties)


def test_auroc_and_baseline():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(2)
    s = np.round(rng.random(300), 1)
    y = (rng.random(300) < 0.3).astype(int)
    assert auroc(s, y) == pytest.approx(roc_auc_score(y, s))
    m = step_metrics(s, y, 0.5)
    p = y.mean()
    assert m["always_positive_f1"] == pytest.approx(2 * p / (1 + p))


def test_oracle_pb_threshold_is_at_least_any_fixed():
    rng = np.random.default_rng(3)
    seqs = [list(rng.random(int(rng.integers(1, 8)))) for _ in range(100)]
    labels = [int(rng.integers(-1, len(x))) for x in seqs]
    t, f = best_pb_threshold(seqs, labels)
    for thr in (0.3, 0.5, 0.7, 0.9):
        assert f >= pb_trace_metrics(seqs, labels, thr)["F1_PB"] - 1e-12
    assert pb_trace_metrics(seqs, labels, t)["F1_PB"] == pytest.approx(f)
