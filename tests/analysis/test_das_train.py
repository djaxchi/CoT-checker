"""Unit tests for das_train DAS subspace core (pure torch, no model)."""

from __future__ import annotations

import torch

from src.analysis.das_train import (
    SubspaceU,
    dist_match_loss,
    interchange_states,
    margin_ce_loss,
    margin_match_loss,
    smooth_margin,
    subspace_overlap,
)


def test_u_columns_orthonormal():
    Q = SubspaceU(16, 4, seed=1)()
    gram = Q.transpose(0, 1) @ Q
    assert torch.allclose(gram, torch.eye(4), atol=1e-5)


def test_interchange_full_rank_recovers_donor():
    torch.manual_seed(0)
    base = torch.randn(3, 8)
    donor = torch.randn(3, 8)
    Q, _ = torch.linalg.qr(torch.randn(8, 8))   # full-rank orthonormal (k=d)
    out = interchange_states(base, donor, Q)
    assert torch.allclose(out, donor, atol=1e-4)   # UU^T = I


def test_interchange_identity_when_donor_equals_base():
    base = torch.randn(4, 6)
    Q = SubspaceU(6, 2, seed=3)()
    out = interchange_states(base, base, Q)
    assert torch.allclose(out, base, atol=1e-6)     # zero diff -> no change


def test_interchange_projects_into_subspace():
    torch.manual_seed(0)
    base = torch.zeros(1, 10)
    donor = torch.randn(1, 10)
    Q = SubspaceU(10, 3, seed=2)()
    out = interchange_states(base, donor, Q)          # = proj of donor onto span(Q)
    # the moved part must lie in span(Q): projecting again is idempotent
    reproj = (out @ Q) @ Q.transpose(0, 1)
    assert torch.allclose(reproj, out, atol=1e-5)


def test_grad_flows_to_u():
    u = SubspaceU(12, 4, seed=0)
    base = torch.randn(5, 12)
    donor = torch.randn(5, 12)
    states = interchange_states(base, donor, u())
    states.pow(2).sum().backward()
    assert u.weight.grad is not None
    assert u.weight.grad.abs().sum() > 0


def test_margin_ce_lower_when_gold_higher():
    high_gold = torch.tensor([2.0, -1.0, -1.0])
    low_gold = torch.tensor([-1.0, 2.0, 0.0])
    assert margin_ce_loss(high_gold, 0) < margin_ce_loss(low_gold, 0)


def test_dist_match_loss_minimal_at_target():
    target = torch.tensor([1.5, -0.5, 0.2, -1.0])
    # loss is minimised when the patched logprobs match the donor distribution
    at_target = dist_match_loss(target.clone(), target)
    far = dist_match_loss(torch.tensor([-2.0, 2.0, 0.0, 0.0]), target)
    assert at_target < far
    # matching target equals the donor entropy (CE at the minimum)
    q = torch.softmax(target, 0)
    assert torch.allclose(at_target, -(q * torch.log(q)).sum(), atol=1e-5)


def test_dist_match_loss_grad_flows():
    u = SubspaceU(10, 3, seed=0)
    base, donor = torch.randn(4, 10), torch.randn(4, 10)
    target = torch.tensor([0.5, -0.5, 1.0])
    # a toy readout: candidate logprobs = projections of the mean patched state
    states = interchange_states(base, donor, u())
    cand = states.mean(0)[:3]
    dist_match_loss(cand, target).backward()
    assert u.weight.grad is not None and u.weight.grad.abs().sum() > 0


def test_smooth_margin_matches_gap():
    import math
    m = smooth_margin(torch.tensor([2.0, 0.0, 0.0]))
    assert abs(float(m) - (2.0 - math.log(2.0))) < 1e-5
    # a bigger gold-distractor gap gives a bigger margin
    assert smooth_margin(torch.tensor([3.0, 0.0, 0.0])) > m


def test_margin_match_loss_bounded_and_targets_donor():
    donor = torch.tensor([1.0, -1.0, -1.0])          # moderate donor margin
    at_target = margin_match_loss(donor.clone(), donor)
    # overshooting the donor margin costs MORE than matching it (bounded objective)
    overshoot = margin_match_loss(torch.tensor([9.0, -1.0, -1.0]), donor)
    assert at_target < 1e-4 < overshoot


def test_margin_match_loss_grad_flows():
    u = SubspaceU(10, 3, seed=0)
    base, donor = torch.randn(4, 10), torch.randn(4, 10)
    target = torch.tensor([0.7, -0.3, -0.4])
    cand = interchange_states(base, donor, u()).mean(0)[:3]
    margin_match_loss(cand, target).backward()
    assert u.weight.grad is not None and u.weight.grad.abs().sum() > 0


def test_subspace_overlap_bounds():
    torch.manual_seed(0)
    full = torch.linalg.qr(torch.randn(20, 20))[0]
    Q, Qc = full[:, :8], full[:, 8:16]        # disjoint cols of ONE basis -> orthogonal
    assert abs(subspace_overlap(Q, Q) - 1.0) < 1e-5           # identical span
    assert subspace_overlap(Q, Qc) < 1e-4                     # truly orthogonal
