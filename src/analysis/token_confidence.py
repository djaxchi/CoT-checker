"""Token-confidence statistics, implemented to DeepConf's equations.

REPORT.md §20.11 and §20.12 leave the on-policy tie-break claim with one
competitor untested: token confidence. DeepConf (arXiv:2508.15260) is the
method a reviewer names first, it needs no training, and §20.12's "beats free"
currently means only "beats free *length* rules". This module is that
competitor, written from the paper's equations rather than reinvented, so the
comparison in docs/onpolicy_tiebreak_v2_plan.md Phase 1 is against the
published rule and not a strawman.

The equations, verbatim from the paper:

    token confidence      C_i = -(1/k) sum_{j=1..k} log P_i(j)          (Eq 2)
    average trace conf    C_avg = (1/N) sum_i C_i                       (Eq 3)
    group confidence      C_Gi = (1/|G_i|) sum_{t in G_i} C_t           (Eq 4)
    bottom-10% group      C_bottom-10 = mean of the lowest 10% of C_G   (Eq 5)
    lowest group          C_least = min_Gj C_Gj                         (Eq 6)
    tail confidence       C_tail = mean of C_t over the final tokens    (Eq 7)

**On orientation.** The paper says higher C means more confident and that it
retains "the top eta% highest-confidence traces". Read literally, Eq 2 does not
have that property: a flat distribution over a 151K vocabulary gives every
top-k token log P = -11.93 and so C = 11.93, while a peaked distribution whose
top token carries 0.99 gives a smaller C. Rather than guess which way the paper
means it, every rule here returns the raw statistic and the orientation is a
separate, recorded decision fixed on the exploratory half (see `ORIENTATION`
below). Giving the competitor whichever sign serves it better is the
conservative choice when the competitor is the thing our claim has to beat.

**On window sizes.** The paper's groups are 1024 or 2048 tokens and its tail is
2048. Our traces average 470 generation tokens over 9.3 steps, so those windows
span the whole trace and collapse Eq 4 to Eq 3. Windows are therefore a
parameter, run at scaled sizes, and one additional rule aggregates by *step*
under the same minimum-over-steps rule the verifier uses, so that the verifier
and its competitor differ in signal and not in aggregation.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# Which direction of each raw statistic is treated as "better candidate". Fixed
# on the exploratory half and recorded in the run manifest; never re-chosen
# after the confirmatory pool is touched.
ORIENTATION_HIGHER_IS_BETTER = "higher"
ORIENTATION_LOWER_IS_BETTER = "lower"


def token_confidence(topk_logprobs: np.ndarray) -> np.ndarray:
    """Eq 2, per position. `topk_logprobs` is (T, k) natural-log probabilities.

    Returns (T,). No clipping: a -inf logprob means the model assigned a top-k
    slot zero probability under the dtype, which is information, and turning it
    into a finite number here would hide it. Callers that cannot tolerate inf
    should say so at the aggregation step.
    """
    arr = np.asarray(topk_logprobs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (T, k) logprobs, got shape {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return -arr.mean(axis=1)


def group_confidence(conf: Sequence[float], window: int) -> np.ndarray:
    """Eq 4: means over overlapping windows of `window` tokens, stride 1.

    A trace shorter than one window yields a single group covering it, which is
    what makes Eq 5 and Eq 6 well defined on short traces rather than empty.
    """
    c = np.asarray(conf, dtype=np.float64)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if c.size == 0:
        return np.zeros(0, dtype=np.float64)
    if c.size <= window:
        return np.array([c.mean()], dtype=np.float64)
    cumsum = np.concatenate([[0.0], np.cumsum(c)])
    return (cumsum[window:] - cumsum[:-window]) / window


def bottom_percent_group(groups: Sequence[float], pct: float = 10.0) -> float:
    """Eq 5: mean of the lowest `pct`% of group confidences.

    At least one group is always taken, so this degrades to Eq 6 rather than to
    a nan when a trace has few groups.
    """
    g = np.asarray(groups, dtype=np.float64)
    if g.size == 0:
        return float("nan")
    n = max(1, int(np.floor(g.size * pct / 100.0)))
    return float(np.sort(g)[:n].mean())


def lowest_group(groups: Sequence[float]) -> float:
    """Eq 6."""
    g = np.asarray(groups, dtype=np.float64)
    return float(g.min()) if g.size else float("nan")


def tail_confidence(conf: Sequence[float], n_tail: int) -> float:
    """Eq 7: mean over the final `n_tail` positions (all of them if shorter)."""
    c = np.asarray(conf, dtype=np.float64)
    if c.size == 0:
        return float("nan")
    return float(c[-min(n_tail, c.size):].mean())


def mean_confidence(conf: Sequence[float]) -> float:
    """Eq 3."""
    c = np.asarray(conf, dtype=np.float64)
    return float(c.mean()) if c.size else float("nan")


def step_min_confidence(conf: Sequence[float], step_spans: Sequence[tuple[int, int]]) -> float:
    """Minimum over per-step mean confidence.

    Not a DeepConf rule. This is the aggregation the verifier uses (ReProbe's
    Q_offline, the minimum over steps, §20.6), applied to the confidence signal
    so the head-to-head isolates the signal rather than the aggregation. Spans
    are half-open [start, end) into the generated-token sequence.
    """
    c = np.asarray(conf, dtype=np.float64)
    per_step = [c[a:b].mean() for a, b in step_spans if b > a and a < c.size]
    return float(min(per_step)) if per_step else float("nan")


def answer_token_margin(topk_logprobs: np.ndarray, answer_span: tuple[int, int]) -> float:
    """Mean top1-minus-top2 logprob gap over the answer tokens, in nats.

    The rule "When Self-Consistency Backfires" (arXiv:2608.11403) reports as
    weak: on GPQA-Diamond, samples contradicting the plurality still emitted
    their answer at a median margin of 20.52 nats with 75.7% above 10 nats. It
    is included because that paper's finding is a prediction this study can
    check, not because it is expected to win.
    """
    arr = np.asarray(topk_logprobs, dtype=np.float64)
    a, b = answer_span
    a, b = max(0, a), min(arr.shape[0], b)
    if b <= a or arr.shape[1] < 2:
        return float("nan")
    return float((arr[a:b, 0] - arr[a:b, 1]).mean())


def trace_rules(windows: Sequence[int] = (32, 64),
                tails: Sequence[int] = (64, 128)) -> dict[str, Callable]:
    """The Tier-1 rule family, keyed by the name used in every results file.

    Each value maps (token_confidence array, step_spans, topk_logprobs) to one
    scalar per trace. Window and tail sizes are scaled to our ~470-token traces;
    the paper's 1024/2048 are included so the degenerate case is visible rather
    than silently omitted.
    """
    rules: dict[str, Callable] = {
        "mean_token_conf": lambda c, s, lp: mean_confidence(c),
    }
    for w in list(windows) + [1024, 2048]:
        rules[f"bottom10_group_w{w}"] = (
            lambda c, s, lp, w=w: bottom_percent_group(group_confidence(c, w)))
        rules[f"lowest_group_w{w}"] = (
            lambda c, s, lp, w=w: lowest_group(group_confidence(c, w)))
    for t in list(tails) + [2048]:
        rules[f"tail_conf_n{t}"] = lambda c, s, lp, t=t: tail_confidence(c, t)
    rules["step_min_conf"] = lambda c, s, lp: step_min_confidence(c, s)
    return rules
