"""Unit tests for parametric_retrieval_minimal_v1: masked/subspace edit
construction, neuron ranking, the additive ResidualEdit mode, and NeuronStore."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from scripts.parametric_retrieval.prga_common import NeuronStore, ResidualEdit
from scripts.parametric_retrieval.prgm_minimal import (
    build_subspace_basis,
    rank_indices,
    randk_mask_vec,
    subspace_vec,
    topk_mask_vec,
)


def test_topk_mask_keeps_largest_only():
    d = np.array([0.1, -3.0, 0.2, 2.0, -0.05], dtype=np.float32)
    v = topk_mask_vec(d, 2)
    assert v[1] == -3.0 and v[3] == 2.0            # two largest |.|
    assert v[0] == 0 and v[2] == 0 and v[4] == 0
    assert np.count_nonzero(v) == 2


def test_topk_mask_full_returns_delta():
    d = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    assert np.allclose(topk_mask_vec(d, 3), d)
    assert np.allclose(topk_mask_vec(d, 99), d)


def test_randk_mask_size_and_values():
    rng = np.random.default_rng(0)
    d = np.arange(1, 11, dtype=np.float32)
    v = randk_mask_vec(d, 4, rng)
    assert np.count_nonzero(v) == 4
    nz = v != 0
    assert np.allclose(v[nz], d[nz])               # kept coords keep their value


def test_subspace_projection_reconstructs_in_span():
    rng = np.random.default_rng(1)
    d = 6
    basis = np.linalg.qr(rng.standard_normal((d, d)))[0].T   # orthonormal rows
    delta = rng.standard_normal(d).astype(np.float64)
    # full-rank projection == identity
    assert np.allclose(subspace_vec(delta, basis, d), delta, atol=1e-10)
    # rank-1 projection lies along basis[0]
    v1 = subspace_vec(delta, basis, 1)
    assert np.allclose(v1, (basis[0] @ delta) * basis[0])


def test_build_subspace_basis_orthonormal_rows():
    rng = np.random.default_rng(2)
    deltas = rng.standard_normal((50, 8))
    U = build_subspace_basis(deltas, r=4)
    assert U.shape == (4, 8)
    assert np.allclose(U @ U.T, np.eye(4), atol=1e-8)   # orthonormal


def test_rank_indices_descending_topk():
    s = np.array([0.1, 5.0, 2.0, 5.0, -1.0])
    idx = rank_indices(s, 2)
    assert set(idx.tolist()) == {1, 3}                  # the two 5.0s
    assert s[idx[0]] >= s[idx[1]]


def test_residual_edit_add_mode_prefill_only():
    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

            class _Cfg:
                hidden_size = 4
            self.config = _Cfg()

    m = _M()
    e = ResidualEdit(m, 1, "add")                       # hooks layers[0]
    v = torch.full((4,), 2.0)
    hs = torch.ones(1, 3, 4)
    with e:
        e.set([2], [v], 0.5)
        out = m.model.layers[0](hs)
    assert torch.allclose(out[0, 2], torch.full((4,), 2.0))   # 1 + 0.5*2
    assert torch.allclose(out[0, 0], torch.ones(4))           # untouched


def test_residual_edit_add_noop_during_decode():
    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

            class _Cfg:
                hidden_size = 3
            self.config = _Cfg()

    m = _M()
    e = ResidualEdit(m, 1, "add")
    with e:
        e.set([0], [torch.full((3,), 9.0)], 1.0)
        out = m.model.layers[0](torch.ones(1, 1, 3))    # seq_len 1
    assert torch.allclose(out, torch.ones(1, 1, 3))      # unchanged


def test_neuron_reconstruction_matches_full_mlp_delta():
    # swapping ALL neurons reconstructs the full MLP-output delta:
    # W_down @ (g_donor - g_recip)  ==  mlp_out_donor - mlp_out_recip
    rng = np.random.default_rng(3)
    hidden, inter = 5, 12
    Wd = torch.tensor(rng.standard_normal((hidden, inter)), dtype=torch.float32)
    gd = torch.tensor(rng.standard_normal(inter), dtype=torch.float32)
    gr = torch.tensor(rng.standard_normal(inter), dtype=torch.float32)
    recon = Wd @ (gd - gr)
    assert torch.allclose(recon, (Wd @ gd) - (Wd @ gr), atol=1e-5)


def test_neuron_store_roundtrip(tmp_path):
    from safetensors.numpy import save_file

    ns = tmp_path / "neuron_states_v1"
    ns.mkdir()
    pd.DataFrame({"instance_id": ["a", "b"], "fact_id": ["f", "g"],
                  "direction": ["direct", "direct"],
                  "is_correct": [True, False]}).to_parquet(
        ns / "neuron_meta.parquet", index=False)
    g = np.arange(8, dtype=np.float16).reshape(2, 4)
    save_file({"h": g}, str(ns / "g_L27.safetensors"))
    store = NeuronStore(tmp_path)
    assert np.allclose(store.vec("b", 27), g[1])
    assert store.vec("missing", 27) is None
