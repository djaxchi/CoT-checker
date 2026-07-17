"""Unit tests for das_branch_subspace_v0 core (pure torch, no model download)."""

from __future__ import annotations

import torch

from src.analysis.das_branch import (
    fork_branch_prompts,
    make_boundary_patch_hook,
)


class _Block(torch.nn.Module):
    """Toy decoder block returning a tuple (hidden_states, cache) like Qwen2."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = torch.nn.Identity()

    def forward(self, hs):
        return (self.lin(hs), "cache")


def test_hook_patches_boundary_during_prefill():
    d = 4
    block = _Block(d)
    donor = torch.arange(d, dtype=torch.float32) + 100.0
    h = block.register_forward_hook(make_boundary_patch_hook(2, donor))
    hs = torch.zeros(1, 5, d)  # prefill: seq covers boundary_pos=2
    out = block(hs)[0]
    h.remove()
    assert torch.allclose(out[0, 2], donor)          # boundary replaced
    assert torch.allclose(out[0, 0], torch.zeros(d))  # others untouched
    assert torch.allclose(out[0, 3], torch.zeros(d))


def test_hook_is_noop_during_decode():
    d = 4
    block = _Block(d)
    donor = torch.ones(d) * 7.0
    h = block.register_forward_hook(make_boundary_patch_hook(2, donor))
    hs = torch.full((1, 1, d), 3.0)  # single new token: seq len 1 <= boundary_pos
    out = block(hs)[0]
    h.remove()
    assert torch.allclose(out, torch.full((1, 1, d), 3.0))  # unchanged


def test_hook_identity_when_donor_is_original():
    d = 6
    block = _Block(d)
    hs = torch.randn(2, 5, d)
    original_boundary = hs[:, 3, :].clone()
    # donor equals the current boundary state -> output must be unchanged
    h = block.register_forward_hook(
        make_boundary_patch_hook(3, original_boundary[0]))
    out = block(hs)[0]
    h.remove()
    # row 0 boundary equals donor by construction; check the whole tensor for row 0
    assert torch.allclose(out[0], hs[0])


def test_hook_non_tuple_output():
    d = 3
    donor = torch.ones(d) * 5.0
    layer = torch.nn.Identity()
    h = layer.register_forward_hook(make_boundary_patch_hook(1, donor))
    hs = torch.zeros(1, 4, d)
    out = layer(hs)
    h.remove()
    assert not isinstance(out, tuple)
    assert torch.allclose(out[0, 1], donor)


def test_fork_branch_prompts_format():
    tr = {
        "question": "Q?",
        "steps": ["s0", "s1", "s2"],
        "fork_t": 1,
        "wrong_step": "bad1",
    }
    p = fork_branch_prompts(tr)
    assert p["correct"] == "Q?\ns0\ns1\n"
    assert p["wrong"] == "Q?\ns0\nbad1\n"
    assert p["correct"].endswith("\n") and p["wrong"].endswith("\n")
