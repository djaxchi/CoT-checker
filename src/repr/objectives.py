"""Fork-based preference objectives for representation shaping.

Label convention in this codebase (kept consistent everywhere):
  rating +1  -> label 0  -> "correct" / preferred / viable continuation
  rating -1  -> label 1  -> "error"   / rejected  / non-viable continuation

A *fork* is a single reasoning prefix (same problem, solution, step index) with
two or more rated candidate next steps. Forks give us, for free, a contrast that
is matched on everything except the continuation itself:

    anchor   = the reasoning prefix
    positive = a preferred (rating +1) continuation of that prefix
    negative = a rejected  (rating -1) continuation of that prefix

Two objective families consume this structure:

* Ranking (RankNet / hinge): a scalar head s(.) must order the two siblings,
  s(positive) > s(negative). Requires no anchor. Gradients flow through the
  encoder latents into the representation.

* Triplet: in latent space, pull the anchor toward the positive and push it
  away from the negative. Requires the anchor (prefix) latent.

Both heads/criteria are auxiliary and discarded after training.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ranking loss (operates on scalar scores of preferred vs rejected siblings)
# ---------------------------------------------------------------------------

def ranking_loss(
    score_pos: torch.Tensor,
    score_neg: torch.Tensor,
    kind: str = "logistic",
    margin: float = 1.0,
) -> torch.Tensor:
    """Pairwise ranking loss enforcing score_pos > score_neg.

    Args:
        score_pos: scalar scores for preferred continuations, shape (n,).
        score_neg: scalar scores for rejected continuations, shape (n,).
        kind: "logistic" for RankNet (-log sigmoid(d)) or "margin" for a
            hinge relu(margin - d), where d = score_pos - score_neg.
        margin: hinge margin (ignored for the logistic variant).

    Returns:
        Scalar mean loss.
    """
    if score_pos.shape != score_neg.shape:
        raise ValueError(
            f"score_pos {tuple(score_pos.shape)} != score_neg {tuple(score_neg.shape)}"
        )
    diff = score_pos - score_neg
    if kind == "logistic":
        # -log sigmoid(diff) == softplus(-diff); minimized as diff -> +inf.
        return F.softplus(-diff).mean()
    if kind == "margin":
        return F.relu(margin - diff).mean()
    raise ValueError(f"Unknown ranking kind {kind!r}; expected 'logistic' or 'margin'.")


# ---------------------------------------------------------------------------
# Triplet loss (operates on latent vectors)
# ---------------------------------------------------------------------------

def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    metric: str = "l2",
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet margin loss: pull anchor->positive, push anchor->negative.

    Args:
        anchor: latent vectors of the reasoning prefix, shape (n, d).
        positive: latents of preferred continuations, shape (n, d).
        negative: latents of rejected continuations, shape (n, d).
        metric: "l2" for squared Euclidean distance, or "cosine" for
            (1 - cosine_similarity) as the distance.
        margin: required separation d(a,n) - d(a,p) >= margin.

    Returns:
        Scalar mean loss.
    """
    if not (anchor.shape == positive.shape == negative.shape):
        raise ValueError(
            "anchor/positive/negative must share shape, got "
            f"{tuple(anchor.shape)}, {tuple(positive.shape)}, {tuple(negative.shape)}"
        )

    if metric == "l2":
        d_pos = (anchor - positive).pow(2).sum(dim=-1)
        d_neg = (anchor - negative).pow(2).sum(dim=-1)
    elif metric == "cosine":
        d_pos = 1.0 - F.cosine_similarity(anchor, positive, dim=-1)
        d_neg = 1.0 - F.cosine_similarity(anchor, negative, dim=-1)
    else:
        raise ValueError(f"Unknown metric {metric!r}; expected 'l2' or 'cosine'.")

    return F.relu(d_pos - d_neg + margin).mean()


# ---------------------------------------------------------------------------
# Pair enumeration from a fork's siblings
# ---------------------------------------------------------------------------

def enumerate_fork_pairs(
    positives: Sequence[Any],
    negatives: Sequence[Any],
    mode: str = "one",
    rng: random.Random | None = None,
) -> list[tuple[Any, Any]]:
    """Build (positive, negative) pairs from one fork's siblings.

    Args:
        positives: preferred-continuation items (opaque; ids, dicts, ...).
        negatives: rejected-continuation items.
        mode: "one" samples a single positive and single negative (balanced,
            one pair per fork) or "all" takes the full Cartesian product
            (every positive against every negative).
        rng: random source for "one" (required for reproducibility there).

    Returns:
        List of (positive_item, negative_item) tuples. Empty if the fork is
        degenerate (missing a positive or a negative).
    """
    if not positives or not negatives:
        return []
    if mode == "one":
        if rng is None:
            raise ValueError("mode='one' requires an explicit rng for reproducibility.")
        return [(rng.choice(list(positives)), rng.choice(list(negatives)))]
    if mode == "all":
        return [(p, n) for p in positives for n in negatives]
    raise ValueError(f"Unknown mode {mode!r}; expected 'one' or 'all'.")
