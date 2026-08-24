"""The learner axis: every learner must honour one contract so the grid is fair.

A cell of the grid is (representation, learner). For the comparison to mean
anything, the learner must be the only thing that changes between two cells on
the same representation, which requires a single forward contract, an honest
parameter count for the capacity axis, and padding that provably does not leak
into the answer for the sequence learners.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.harness.learners import (  # noqa: E402
    build_learner, is_sequence, param_count, parse_spec,
)


def test_spec_parsing():
    assert parse_spec("linear") == ("linear", {})
    assert parse_spec("mlp:h1024") == ("mlp", {"h": 1024})
    assert parse_spec("mlp:h512x2") == ("mlp", {"h": 512, "depth": 2})
    head, o = parse_spec("transformer:d256,l2,f1024,h4")
    assert head == "transformer"
    assert (o["d"], o["l"], o["f"], o["h"]) == (256, 2, 1024, 4)


def test_is_sequence_splits_the_grid():
    assert not is_sequence("linear")
    assert not is_sequence("mlp:h1024")
    assert is_sequence("attn_query")
    assert is_sequence("transformer:d128,l1")
    with pytest.raises(ValueError):
        is_sequence("lightgbm")


def test_linear_param_count_is_d_plus_bias():
    assert param_count(build_learner("linear", 3584)) == 3584 + 1


def test_mlp_depth_changes_capacity_not_contract():
    d = 16
    one = build_learner("mlp:h32", d)
    two = build_learner("mlp:h32x2", d)
    assert param_count(two) > param_count(one)
    x = torch.randn(4, d)
    assert one(x, None).shape == (4,) == two(x, None).shape


@pytest.mark.parametrize("spec", ["attn_query", "transformer:d16,l1,f32,h2"])
def test_sequence_learners_ignore_padding(spec):
    """Appending pad rows with mask 0 must not move the logit.

    Without this, a cell's score would depend on how long its neighbours in the
    batch happened to be, which would make the sequence rows incomparable to
    each other, let alone to the vector rows.
    """
    torch.manual_seed(0)
    d, T = 8, 5
    model = build_learner(spec, d, t_max=32).eval()
    x = torch.randn(2, T, d)
    mask = torch.ones(2, T)
    mask[1, 3:] = 0.0            # second item is genuinely 3 tokens long
    with torch.no_grad():
        base = model(x, mask)
        x_noise = x.clone()
        x_noise[1, 3:] = torch.randn(T - 3, d) * 100  # garbage in the pad slots
        noised = model(x_noise, mask)
    torch.testing.assert_close(base, noised, rtol=1e-4, atol=1e-4)


def test_transformer_refuses_a_sequence_past_t_max():
    model = build_learner("transformer:d16,l1,f32,h2", 8, t_max=4)
    with pytest.raises(ValueError, match="t_max"):
        model(torch.randn(1, 6, 8), torch.ones(1, 6))


def test_unknown_learner_is_rejected():
    with pytest.raises(ValueError, match="unknown learner"):
        build_learner("randomforest", 8)
