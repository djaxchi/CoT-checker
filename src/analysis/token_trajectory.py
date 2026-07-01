"""Pure, GPU-free core for the per-token incorrectness-trajectory study.

The mechanistic 6k study (REPORT.md §15) read the L28 dense probe only at the LAST
token of each step. This module supports the finer question: *within a step that is
labelled incorrect, does one particular token make the probe fire, and does that token
coincide with a dip in the model's certainty?* All functions here are numpy-only so they
are unit-testable without a model; the GPU extractor
(scripts/analysis/s3_token_incorrectness_extract.py) computes the same quantities on the
device and this module is the reference semantics.

Conventions (matched to encode_prm800k_multitoken_multilayer.tokenize_span):
  - A step's tokens are ``full_ids[first_idx : last_idx + 1]`` (length T).
  - The logits that *predict* step token at position t are ``logits[t - 1]``; so the T
    predictive rows for the step span are ``logits[first_idx - 1 : last_idx]``.
  - Probe score = ``h @ w + b`` (higher = more incorrect), applied per token.
"""

from __future__ import annotations

import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def per_token_certainty(pred_logits: np.ndarray, target_ids: np.ndarray) -> dict:
    """Per-token certainty of the model over one step's realized tokens.

    pred_logits : (T, V) logits that predict each of the T step tokens.
    target_ids  : (T,) the realized step token ids.

    Returns a dict of (T,) float arrays (nll/entropy in nats):
      nll         -log p(realized token)         higher = more surprised
      entropy     predictive entropy             higher = less certain
      logit_gap   top1 - top2 logit              higher = more certain
      p_top1      max softmax prob               higher = more certain
      p_realized  prob of the realized token     higher = more certain
    """
    pred_logits = np.asarray(pred_logits, dtype=np.float64)
    target_ids = np.asarray(target_ids).astype(np.int64).reshape(-1)
    T = target_ids.shape[0]
    if T == 0 or pred_logits.ndim != 2 or pred_logits.shape[0] != T:
        raise ValueError(
            f"pred_logits {pred_logits.shape} must be (T, V) with T={T} > 0")
    logp = log_softmax(pred_logits)
    p = np.exp(logp)
    rows = np.arange(T)
    nll = -logp[rows, target_ids]
    entropy = -(p * logp).sum(axis=-1)
    p_top1 = p.max(axis=-1)
    p_realized = p[rows, target_ids]
    part = np.partition(pred_logits, -2, axis=-1)
    logit_gap = part[:, -1] - part[:, -2]
    return {
        "nll": nll, "entropy": entropy, "logit_gap": logit_gap,
        "p_top1": p_top1, "p_realized": p_realized,
    }


def probe_scores(H: np.ndarray, w: np.ndarray, b: float = 0.0) -> np.ndarray:
    """Per-token probe score ``H @ w + b`` (higher = incorrect).

    H : (T, hidden) per-token hidden states.  w : (hidden,).  b : scalar bias.
    """
    H = np.asarray(H, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if H.ndim != 2 or H.shape[1] != w.shape[0]:
        raise ValueError(f"H {H.shape} incompatible with w {w.shape}")
    return H @ w + float(b)


def spike_stats(scores: np.ndarray) -> dict:
    """Localization of the incorrectness signal across a step's T token scores.

    Answers "is there one token that fires?": ``peakiness`` is how many standard
    deviations the top token sits above the step mean (high => a localized spike, low
    => a plateau); ``argmax_frac`` is where that peak sits within the step (0 = first
    token, 1 = last), so a value near 1 means the signal is a step-end effect.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    T = s.shape[0]
    if T == 0:
        raise ValueError("empty score vector")
    amax = int(s.argmax())
    mean = float(s.mean())
    std = float(s.std())
    return {
        "n": T,
        "peak": float(s.max()),
        "min": float(s.min()),
        "mean": mean,
        "median": float(np.median(s)),
        "std": std,
        "argmax_idx": amax,
        "argmax_frac": float(amax / (T - 1)) if T > 1 else 0.0,
        "peakiness": float((s.max() - mean) / (std + 1e-9)),
        "prominence": float(s.max() - np.median(s)),
    }


def coincidence(scores: np.ndarray, uncertainty: np.ndarray) -> dict:
    """Does the token that fires the probe coincide with the least-certain token?

    scores      : (T,) per-token probe score (higher = incorrect).
    uncertainty : (T,) per-token uncertainty aligned to the same tokens (e.g. nll or
                  entropy; higher = less certain).

    Returns argmax indices, their normalized distance in [0, 1] (0 = same token), and
    the within-step Pearson correlation (nan for constant vectors).
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    u = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if s.shape != u.shape or s.shape[0] == 0:
        raise ValueError(f"scores {s.shape} and uncertainty {u.shape} must match, non-empty")
    T = s.shape[0]
    i, j = int(s.argmax()), int(u.argmax())
    if s.std() < 1e-12 or u.std() < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(s, u)[0, 1])
    return {
        "argmax_score": i,
        "argmax_uncertainty": j,
        "argmax_distance_frac": float(abs(i - j) / (T - 1)) if T > 1 else 0.0,
        "within_step_corr": corr,
    }
