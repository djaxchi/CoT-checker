"""Unit tests for das_span whole-step-span interchange core (pure torch)."""

from __future__ import annotations

import pytest
import torch

from src.analysis.das_span import (
    aligned_positions,
    fork_span_ids,
    make_span_patch_hook,
)


class _FakeTok:
    """Whitespace tokenizer: each word -> a stable positive id."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1000 + (hash(w) % 500) for w in text.split()]}


def test_span_hook_patches_window_in_prefill():
    d = 4
    layer = torch.nn.Identity()
    states = torch.stack([torch.full((d,), 10.0), torch.full((d,), 20.0)])  # (2,d)
    h = layer.register_forward_hook(make_span_patch_hook(1, 3, states))
    hs = torch.zeros(1, 5, d)
    out = layer(hs)
    h.remove()
    assert torch.allclose(out[0, 1], states[0])
    assert torch.allclose(out[0, 2], states[1])
    assert torch.allclose(out[0, 0], torch.zeros(d))   # before window
    assert torch.allclose(out[0, 3], torch.zeros(d))   # after window


def test_span_hook_noop_during_decode():
    d = 3
    layer = torch.nn.Identity()
    states = torch.ones(2, d)
    h = layer.register_forward_hook(make_span_patch_hook(1, 3, states))
    hs = torch.full((1, 1, d), 4.0)  # single new token, seq len 1 < hi
    out = layer(hs)
    h.remove()
    assert torch.allclose(out, torch.full((1, 1, d), 4.0))


def test_span_hook_tuple_output():
    d = 2

    class Blk(torch.nn.Module):
        def forward(self, hs):
            return (hs, "cache")

    blk = Blk()
    states = torch.ones(1, d) * 9
    h = blk.register_forward_hook(make_span_patch_hook(0, 1, states))
    out = blk(torch.zeros(1, 3, d))
    h.remove()
    assert isinstance(out, tuple)
    assert torch.allclose(out[0][0, 0], states[0])


def test_aligned_positions_equal():
    assert aligned_positions((5, 10), (5, 10), "equal") == (5, 10, 5, 10)
    with pytest.raises(ValueError):
        aligned_positions((5, 10), (5, 9), "equal")


def test_aligned_positions_lastk():
    # wrong span len 8, correct span len 5, k=8 -> min = 5
    assert aligned_positions((10, 18), (3, 8), "lastk", k=8) == (13, 18, 3, 8)
    # k smaller than both
    assert aligned_positions((10, 18), (3, 11), "lastk", k=3) == (15, 18, 8, 11)


def test_fork_span_ids_equal_positions_when_same_length():
    tok = _FakeTok()
    tr = {"question": "a b", "steps": ["c d", "e f g"], "fork_t": 1,
          "wrong_step": "x y z"}  # golden step 'e f g' (3) vs wrong 'x y z' (3)
    fc, lo_c, hi_c = fork_span_ids(tok, tr, "correct")
    fw, lo_w, hi_w = fork_span_ids(tok, tr, "wrong")
    assert (lo_c, hi_c) == (lo_w, hi_w)        # identical absolute span
    assert hi_c - lo_c == 3                     # three step tokens
    assert fc[:lo_c] == fw[:lo_w]               # shared prefix ids
