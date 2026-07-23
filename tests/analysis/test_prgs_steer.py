"""Unit tests for parametric_retrieval_steer_v1: detector, fact->neuron
selection, and the persistent ClampNeuron hook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.parametric_retrieval.prga_common import ClampNeuron
from scripts.parametric_retrieval.prgs_steer import (
    build_fact_neuron,
    mentions,
    normalize,
)


def test_normalize_strips_punct_and_case():
    assert normalize("You Rock, My World!") == "you rock my world"


def test_mentions_substring_and_min_len():
    assert mentions('the answer is "You Rock My World" indeed',
                    "You Rock My World")
    assert mentions("about DEKAPENTASYLLABOS verse", "dekapentasyllabos")
    assert not mentions("nothing relevant here", "You Rock My World")
    assert not mentions("a b c", "of")          # <3 chars entity skipped


def test_build_fact_neuron_rank_weighted():
    # fact f: two pairs; neuron 5 is rank-0 in both -> should win over 9
    df = pd.DataFrame([
        {"task": "top_neurons", "fact_id": "f", "neuron_ids": [5, 9, 1]},
        {"task": "top_neurons", "fact_id": "f", "neuron_ids": [5, 2, 9]},
        {"task": "top_neurons", "fact_id": "g", "neuron_ids": [7, 7, 7]},
        {"task": "coord", "fact_id": "f", "neuron_ids": None},
    ])
    fn = build_fact_neuron(df)
    assert fn["f"] == 5
    assert fn["g"] == 7


# --- ClampNeuron hook -------------------------------------------------------

class _DownProj(nn.Module):
    """Stand-in for mlp.down_proj: records the (possibly clamped) input g."""
    def __init__(self, inter, hidden):
        super().__init__()
        self.lin = nn.Linear(inter, hidden, bias=False)
        self.seen = None

    def forward(self, g):
        self.seen = g.clone()
        return self.lin(g)


class _Mlp(nn.Module):
    def __init__(self, inter, hidden):
        super().__init__()
        self.down_proj = _DownProj(inter, hidden)


class _Layer(nn.Module):
    def __init__(self, inter, hidden):
        super().__init__()
        self.mlp = _Mlp(inter, hidden)


class _Inner(nn.Module):
    def __init__(self, n, inter, hidden):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(inter, hidden) for _ in range(n)])


class _Model(nn.Module):
    def __init__(self, n=2, inter=6, hidden=4):
        super().__init__()
        self.model = _Inner(n, inter, hidden)


def test_clamp_sets_selected_neurons_all_positions():
    m = _Model(inter=6, hidden=4)
    clamp = ClampNeuron(m, 1)
    g = torch.zeros(2, 3, 6)               # (batch, seq, inter)
    with clamp:
        clamp.set([2, 4], [9.0, -5.0])
        m.model.layers[1].mlp.down_proj(g)
    seen = m.model.layers[1].mlp.down_proj.seen
    assert torch.allclose(seen[:, :, 2], torch.full((2, 3), 9.0))   # all tokens
    assert torch.allclose(seen[:, :, 4], torch.full((2, 3), -5.0))
    assert torch.allclose(seen[:, :, 0], torch.zeros(2, 3))         # untouched


def test_clamp_empty_is_noop_after_exit():
    m = _Model(inter=5, hidden=3)
    clamp = ClampNeuron(m, 0)
    with clamp:
        clamp.set([], [])
        m.model.layers[0].mlp.down_proj(torch.ones(1, 2, 5))
    assert torch.allclose(m.model.layers[0].mlp.down_proj.seen,
                          torch.ones(1, 2, 5))
    # after exit the hook is removed: a fresh forward is untouched
    m.model.layers[0].mlp.down_proj(torch.full((1, 2, 5), 3.0))
    assert torch.allclose(m.model.layers[0].mlp.down_proj.seen,
                          torch.full((1, 2, 5), 3.0))


def test_specificity_matrix_diag_offdiag():
    # sanity for the diag/offdiag reduction used in analyze
    M = np.array([[0.8, 0.1], [0.05, 0.6]])
    diag = np.mean(np.diag(M))
    off = (M.sum() - np.trace(M)) / (M.size - len(M))
    assert abs(diag - 0.7) < 1e-9
    assert abs(off - 0.075) < 1e-9
