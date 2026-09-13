"""Who decides when the vote is tied, and does the verifier decide better?

The on-policy arm's one surviving positive result was a gain "inside the majority
bloc". The 2026-09-10 audit showed the bloc is not one answer: it is every
solution tied at the top answer frequency, so on 66 of 300 problems it spans
several distinct answers, and every correctness change the verifier produced came
from breaking one of those ties (REPORT.md §20.2). That leaves a real question,
just a narrower one than was claimed: on a tied vote, does a hidden-state score
pick the right answer more often than something free?

This module is the arithmetic for that comparison. A selector sees the tied
candidates and returns the one it wants; `random` is scored as its expectation
over the bloc rather than by drawing, so it carries no sampling noise of its own.
Everything is a pure function of the saved rows so the whole comparison runs on
CPU from what is already on disk.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np

# A candidate is a dict carrying at least:
#   answer   normalized final answer, or None if it did not parse
#   correct  bool, the recorded final-answer outcome
#   scores   per-step suspicion, higher meaning more likely faulty
#   n_chars  length of the solution text
#   n_steps  number of steps
#   index    position in the sampling order


def answer_groups(rows: Sequence[dict]) -> dict[str, list[int]]:
    """Group candidate indices by normalized answer, dropping unparsed ones."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row["answer"] is not None:
            groups[row["answer"]].append(i)
    return dict(groups)


def top_bloc(groups: dict[str, list[int]]) -> tuple[list[int], list[str]]:
    """Every candidate tied at the top answer frequency, and which answers those are.

    This reproduces `onpolicy_agreement_redundancy.py`'s bloc, including the part
    the original reading missed: with several answers at the top count, the bloc
    spans more than one answer.
    """
    if not groups:
        return [], []
    top = max(len(v) for v in groups.values())
    answers = sorted(k for k, v in groups.items() if len(v) == top)
    return sorted(i for k in answers for i in groups[k]), answers


# ---- selectors ------------------------------------------------------------
#
# Each returns the index it picks out of `bloc`. Ties inside a selector's own key
# are broken by sampling order, so no selector gets a free coin flip the others
# do not.

def _argmin(rows: Sequence[dict], bloc: Sequence[int],
            key: Callable[[dict], float]) -> int:
    return min(bloc, key=lambda i: (key(rows[i]), rows[i]["index"]))


SELECTORS: dict[str, Callable[[Sequence[dict], Sequence[int]], int]] = {
    # the verifier, aggregated the three ways the arm reports
    "verifier_worst_step": lambda r, b: _argmin(r, b, lambda s: max(s["scores"])),
    "verifier_mean_step": lambda r, b: _argmin(r, b, lambda s: float(np.mean(s["scores"]))),
    "verifier_last_step": lambda r, b: _argmin(r, b, lambda s: s["scores"][-1]),
    # free alternatives that need no forward pass at all
    "shortest": lambda r, b: _argmin(r, b, lambda s: s["n_chars"]),
    "longest": lambda r, b: _argmin(r, b, lambda s: -s["n_chars"]),
    "fewest_steps": lambda r, b: _argmin(r, b, lambda s: s["n_steps"]),
    "most_steps": lambda r, b: _argmin(r, b, lambda s: -s["n_steps"]),
    "first_sampled": lambda r, b: _argmin(r, b, lambda s: 0.0),
}

RANDOM = "random"


def selector_accuracy(rows: Sequence[dict], bloc: Sequence[int], rule: str) -> float:
    """Outcome of applying one rule to one problem's bloc.

    `random` is the mean over the bloc, which is exactly the expected accuracy of
    picking uniformly, without the variance of actually drawing.
    """
    if not bloc:
        return 0.0
    if rule == RANDOM:
        return float(np.mean([bool(rows[i]["correct"]) for i in bloc]))
    return float(bool(rows[SELECTORS[rule](rows, bloc)]["correct"]))


def problem_record(rows: Sequence[dict], rules: Iterable[str]) -> dict:
    """Decompose one problem into the decision the vote leaves open, and who wins it.

    `tied` is the only case where a selector can change the final answer: with a
    unique plurality every bloc member carries the same answer, so the choice is
    between derivations of one answer and the recorded outcome label cannot move.
    """
    groups = answer_groups(rows)
    bloc, answers = top_bloc(groups)
    tied = len(answers) > 1
    labels_by_answer = {a: sorted({bool(rows[i]["correct"]) for i in ids})
                        for a, ids in groups.items()}
    return {
        "n_candidates": len(rows),
        "n_answer_groups": len(groups),
        "n_top_answers": len(answers),
        "tied": tied,
        "bloc_size": len(bloc),
        # An answer group holding both labels would mean the grader or the
        # normalizer disagrees with itself; A0 found none, and this keeps it checked.
        "mixed_label_answer_groups": [a for a, v in labels_by_answer.items() if len(v) > 1],
        "oracle": float(any(bool(r["correct"]) for r in rows)),
        "accuracy": {rule: selector_accuracy(rows, bloc, rule)
                     for rule in [*rules, RANDOM]},
    }


# ---- inference ------------------------------------------------------------

def cluster_bootstrap(paired: np.ndarray, clusters: Sequence[str],
                      draws: int = 10000, seed: int = 910) -> dict:
    """Paired mean difference, resampling whole question texts.

    The 300 evaluation ids are 284 distinct questions, so resampling ids treats a
    repeated question as two independent observations and reports an interval that
    is too narrow. The unit here is the question.
    """
    paired = np.asarray(paired, dtype=float)
    order = defaultdict(list)
    for i, c in enumerate(clusters):
        order[c].append(i)
    blocks = [np.array(v) for v in order.values()]
    # A resampled mean is (sum of the drawn blocks) / (size of the drawn blocks),
    # so the draws only need each block's sum and size, not its members.
    sums = np.array([paired[b].sum() for b in blocks])
    sizes = np.array([b.size for b in blocks], dtype=float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(blocks), size=(draws, len(blocks)))
    boot = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"delta": float(paired.mean()), "ci95": [float(lo), float(hi)],
            "n_problems": int(paired.size), "n_questions": len(blocks),
            "crosses_zero": bool(lo <= 0.0 <= hi)}
