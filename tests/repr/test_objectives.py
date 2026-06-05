"""Unit tests for Sprint 2 representation-shaping objectives.

All CPU-only, no model downloads.
"""

import random

import pytest
import torch

from src.repr.objectives import enumerate_fork_pairs, ranking_loss, triplet_loss

# ---------------------------------------------------------------------------
# ranking_loss
# ---------------------------------------------------------------------------

def test_ranking_logistic_lower_when_correctly_ordered():
    pos = torch.tensor([2.0, 3.0])
    neg = torch.tensor([-2.0, -3.0])
    well = ranking_loss(pos, neg, kind="logistic")
    badly = ranking_loss(neg, pos, kind="logistic")
    assert well < badly
    assert well.item() >= 0.0


def test_ranking_margin_zero_when_separated_beyond_margin():
    pos = torch.tensor([5.0])
    neg = torch.tensor([1.0])
    loss = ranking_loss(pos, neg, kind="margin", margin=1.0)
    assert loss.item() == pytest.approx(0.0)


def test_ranking_margin_active_within_margin():
    pos = torch.tensor([1.2])
    neg = torch.tensor([1.0])
    loss = ranking_loss(pos, neg, kind="margin", margin=1.0)
    assert loss.item() == pytest.approx(0.8, abs=1e-6)


def test_ranking_gradients_push_pos_up_neg_down():
    pos = torch.tensor([0.0], requires_grad=True)
    neg = torch.tensor([0.0], requires_grad=True)
    ranking_loss(pos, neg, kind="logistic").backward()
    # Lowering loss should increase pos and decrease neg.
    assert pos.grad.item() < 0
    assert neg.grad.item() > 0


def test_ranking_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ranking_loss(torch.zeros(3), torch.zeros(2))


def test_ranking_unknown_kind_raises():
    with pytest.raises(ValueError):
        ranking_loss(torch.zeros(2), torch.zeros(2), kind="bogus")


# ---------------------------------------------------------------------------
# triplet_loss
# ---------------------------------------------------------------------------

def test_triplet_l2_zero_when_well_separated():
    anchor = torch.tensor([[0.0, 0.0]])
    positive = torch.tensor([[0.1, 0.0]])      # close to anchor
    negative = torch.tensor([[10.0, 0.0]])     # far from anchor
    loss = triplet_loss(anchor, positive, negative, metric="l2", margin=1.0)
    assert loss.item() == pytest.approx(0.0)


def test_triplet_l2_active_when_negative_too_close():
    anchor = torch.tensor([[0.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[1.0, 0.0]])      # same distance as positive
    loss = triplet_loss(anchor, positive, negative, metric="l2", margin=1.0)
    assert loss.item() == pytest.approx(1.0, abs=1e-6)  # d_pos - d_neg + margin = margin


def test_triplet_cosine_runs_and_is_nonnegative():
    anchor = torch.randn(4, 8)
    positive = torch.randn(4, 8)
    negative = torch.randn(4, 8)
    loss = triplet_loss(anchor, positive, negative, metric="cosine", margin=0.5)
    assert loss.item() >= 0.0


def test_triplet_gradient_flows_to_anchor():
    anchor = torch.zeros(1, 2, requires_grad=True)
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[1.5, 0.0]])  # farther, but margin keeps loss active
    triplet_loss(anchor, positive, negative, metric="l2", margin=5.0).backward()
    assert anchor.grad is not None
    assert anchor.grad.abs().sum().item() > 0


def test_triplet_shape_mismatch_raises():
    with pytest.raises(ValueError):
        triplet_loss(torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# enumerate_fork_pairs
# ---------------------------------------------------------------------------

def test_pairs_all_is_cartesian_product():
    pairs = enumerate_fork_pairs(["p1", "p2"], ["n1", "n2", "n3"], mode="all")
    assert len(pairs) == 6
    assert ("p1", "n3") in pairs
    assert ("p2", "n1") in pairs


def test_pairs_one_returns_single_pair():
    rng = random.Random(0)
    pairs = enumerate_fork_pairs(["p1", "p2"], ["n1", "n2"], mode="one", rng=rng)
    assert len(pairs) == 1
    p, n = pairs[0]
    assert p in {"p1", "p2"} and n in {"n1", "n2"}


def test_pairs_one_is_deterministic_under_seed():
    a = enumerate_fork_pairs(["p1", "p2"], ["n1", "n2"], mode="one", rng=random.Random(7))
    b = enumerate_fork_pairs(["p1", "p2"], ["n1", "n2"], mode="one", rng=random.Random(7))
    assert a == b


def test_pairs_degenerate_fork_returns_empty():
    assert enumerate_fork_pairs([], ["n1"], mode="all") == []
    assert enumerate_fork_pairs(["p1"], [], mode="all") == []


def test_pairs_one_requires_rng():
    with pytest.raises(ValueError):
        enumerate_fork_pairs(["p1"], ["n1"], mode="one")


def test_pairs_unknown_mode_raises():
    with pytest.raises(ValueError):
        enumerate_fork_pairs(["p1"], ["n1"], mode="bogus")
