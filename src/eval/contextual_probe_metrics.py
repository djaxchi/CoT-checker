"""Metrics for bidirectional_token_probe_v1 (plan sections 9-10).

Conventions: y = 1 is an INCORRECT step, scores are P(incorrect), a step is
flagged when score >= threshold. Only label-mask==True steps enter step metrics.

Threshold search is explicit and deterministic: candidates are the distinct
observed scores plus +inf (flag nothing); F1 is evaluated at each and the
HIGHER threshold wins ties.

ProcessBench trace prediction is the first step whose score crosses the
threshold (-1 if none), derived from the saved complete score sequence. Under
the >= rule this matches scripts/evaluate_processbench_from_scores.py (strict >)
whenever no score equals the threshold exactly.
"""

from __future__ import annotations

import numpy as np


def auroc(score, y) -> float:
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    # average ranks over ties
    _, first, counts = np.unique(ss, return_index=True, return_counts=True)
    r = first + (counts - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(r, counts)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def prf_at(score, y, thr: float) -> dict:
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    p = s >= thr
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(thr), "f1": f1, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "n": int(len(y)), "pred_pos_rate": float(p.mean()) if len(y) else 0.0}


def best_f1_threshold(score, y) -> tuple[float, float]:
    """(threshold, F1) maximizing F1 on this set; higher threshold wins ties."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    if len(s) == 0 or not y.any():
        return float("inf"), 0.0
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    tp = np.cumsum(yy); fp = np.cumsum(~yy)
    # last index of each distinct score in descending order = predict-all >= that score
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    tp, fp, thr = tp[last], fp[last], ss[last]
    P = int(y.sum())
    f1 = 2 * tp / (tp + fp + P)
    # descending thresholds: argmax returns the first max -> the highest threshold
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def step_metrics(score, y, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    m = prf_at(score, y, thr)
    m["auroc"] = auroc(score, y)
    m["prevalence"] = float(y.mean()) if len(y) else float("nan")
    m["always_positive_f1"] = (2 * m["prevalence"] / (1 + m["prevalence"])) if len(y) and y.any() else 0.0
    return m


def first_crossing(scores, thr: float) -> int:
    for i, s in enumerate(scores):
        if s >= thr:
            return i
    return -1


def pb_trace_metrics(score_seqs: list, labels: list[int], thr: float) -> dict:
    """ProcessBench first-error metrics from complete per-step score sequences.

    labels: annotated first-error index, -1 for an error-free trace."""
    preds = [first_crossing(s, thr) for s in score_seqs]
    return pb_from_preds(preds, labels) | {"threshold": float(thr)}


def pb_from_preds(preds: list[int], labels: list[int]) -> dict:
    preds = np.asarray(preds); labels = np.asarray(labels)
    err = labels >= 0
    acc_e = float((preds[err] == labels[err]).mean()) if err.any() else 0.0
    acc_c = float((preds[~err] == -1).mean()) if (~err).any() else 0.0
    f1 = 2 * acc_e * acc_c / (acc_e + acc_c) if acc_e + acc_c else 0.0
    premature = err & (preds >= 0) & (preds < labels)
    late = err & ((preds > labels) | (preds == -1))
    return {"F1_PB": f1, "Acc_error": acc_e, "Acc_correct": acc_c,
            "exact_localization": float((preds == labels).mean()) if len(labels) else 0.0,
            "n_traces": int(len(labels)), "n_error": int(err.sum()), "n_correct": int((~err).sum()),
            "premature_alarm_rate_err": float(premature[err].mean()) if err.any() else 0.0,
            "late_or_missed_rate_err": float(late[err].mean()) if err.any() else 0.0,
            "false_alarm_rate_correct": float((preds[~err] >= 0).mean()) if (~err).any() else 0.0}


def pb_trivial(labels: list[int], n_steps: list[int]) -> dict:
    """Always-no-error and always-flag-step-0 trace baselines."""
    return {"always_no_error": pb_from_preds([-1] * len(labels), labels),
            "always_step0": pb_from_preds([0] * len(labels), labels)}


def best_pb_threshold(score_seqs: list, labels: list[int]) -> tuple[float, float]:
    """Oracle PB threshold (ceiling only): candidates are the distinct prefix maxima;
    higher threshold wins ties."""
    cands = sorted({float(x) for s in score_seqs for x in s} | {float("inf")}, reverse=True)
    best_t, best_f = float("inf"), -1.0
    # first crossing depends only on running maxima
    T = max((len(x) for x in score_seqs), default=1)
    R = np.full((len(score_seqs), max(T, 1)), -np.inf)
    for i, x in enumerate(score_seqs):
        if len(x):
            R[i, :len(x)] = np.maximum.accumulate(np.asarray(x, dtype=np.float64))
    labels_a = np.asarray(labels)
    for t in cands:
        c = R >= t
        preds = np.where(c.any(1), c.argmax(1), -1)
        f = pb_from_preds(preds, labels_a)["F1_PB"]
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f
