"""Did the model box its answer at all, and is that the tie-break signal?

Phase 1 of docs/onpolicy_tiebreak_v2_plan.md raced the verifier against the
token-confidence family and turned up a result the plan did not anticipate: the
strongest competitor was `answer_token_margin`, and its strength did not come
from the margin. It came from which candidates had a \\boxed{} answer at all.

On the on-policy pool the presence of a boxed answer is asymmetric by outcome,
1.8% of correct trajectories lack one against 31.1% of incorrect ones, because
a trajectory that runs out of its 768-token budget never reaches the box. The
grader still parses an answer for those via its fallbacks, so they enter the
vote and can enter a tied bloc, where any rule that happens to deprioritise
them picks up a large and entirely free correctness signal.

That makes `has_boxed` a competitor in its own right, and a much harsher one
than the length rules of §20.12: it costs no logprobs, no forward pass and no
model at all. §20.12 concluded that "no cheap ordering reproduces" the
verifier's tie-break gain; it tested orderings of the saved text by length and
step count, and this is the ordering it did not test.

Two readings, and the study should carry both rather than choose. As a
deployable signal it is real and free. As a scientific finding it is partly an
artifact of the generation budget, and would shrink under a larger one, which
is a prediction the four-set core can check.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def has_boxed(margin: float) -> bool:
    """Whether a trajectory reached a \\boxed{} answer.

    Derived from the answer-margin statistic rather than re-parsed: the margin
    is nan exactly when `answer_char_span` found no closed box, so this is the
    same predicate the encoder already applied, not a second one that could
    disagree with it.
    """
    return bool(np.isfinite(margin))


def select_has_boxed(rows: Sequence[dict], bloc: Sequence[int]) -> int:
    """Prefer any candidate with a boxed answer; sampling order otherwise.

    Deliberately the crudest possible form of the rule. Anything richer would
    be reading the margin, which is the thing this is meant to be separated
    from.
    """
    return min(bloc, key=lambda i: (0 if rows[i]["boxed"] else 1, rows[i]["index"]))


def select_boxed_then(rows: Sequence[dict], bloc: Sequence[int], key: str) -> int:
    """Boxing first, then a secondary score. The composition, not either half."""
    return min(bloc, key=lambda i: (0 if rows[i]["boxed"] else 1,
                                    rows[i][key], rows[i]["index"]))


def bloc_purity(rows: Sequence[dict], bloc: Sequence[int]) -> str:
    """Whether a rule reading `boxed` can act on this bloc at all.

    `all_boxed` blocs are the only place a margin or a verifier score is being
    asked to do work that the formatting indicator cannot already do, so they
    are the subset any claim about the *signal* has to survive on.
    """
    n = sum(1 for i in bloc if rows[i]["boxed"])
    if n == len(bloc):
        return "all_boxed"
    return "none_boxed" if n == 0 else "mixed"


def within_bloc_z(values: Sequence[float]) -> np.ndarray:
    """Z-score within one bloc, with a missing value pushed well below the rest.

    Two tie-break signals on different scales cannot be summed as they are: the
    verifier is a probability and the answer margin is nats. Standardising
    *within the bloc* puts them on a common footing and makes the combination a
    statement about relative ranking rather than about units.

    A nan becomes -3.0 rather than 0.0. Zero would be the bloc mean, which would
    make a trajectory whose statistic failed to compute an average candidate
    instead of a disqualified one.
    """
    v = np.asarray(values, dtype=float)
    fin = np.isfinite(v)
    out = np.full(v.size, -3.0)
    if fin.sum() < 2 or v[fin].std() == 0:
        out[fin] = 0.0
        return out
    out[fin] = (v[fin] - v[fin].mean()) / v[fin].std()
    return out


def select_combined(rows: Sequence[dict], bloc: Sequence[int],
                    w_verifier: float, w_margin: float) -> int:
    """Pick by a weighted sum of the two standardised signals.

    The verifier's score is suspicion, so it enters negated: higher is worse.
    """
    zv = within_bloc_z([-rows[i]["w"] for i in bloc])
    zm = within_bloc_z([rows[i]["margin"] for i in bloc])
    score = w_verifier * zv + w_margin * zm
    return bloc[int(np.argmax(score))]
