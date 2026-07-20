"""Unit tests for das_train DAS subspace core (pure torch, no model)."""

from __future__ import annotations

import torch

from src.analysis.das_train import (
    SubspaceU,
    interchange_states,
    margin_ce_loss,
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


def test_subspace_overlap_bounds():
    torch.manual_seed(0)
    full = torch.linalg.qr(torch.randn(20, 20))[0]
    Q, Qc = full[:, :8], full[:, 8:16]        # disjoint cols of ONE basis -> orthogonal
    assert abs(subspace_overlap(Q, Q) - 1.0) < 1e-5           # identical span
    assert subspace_overlap(Q, Qc) < 1e-4                     # truly orthogonal
