"""Sparse-input sequence learners must compute what the dense ones would.

A padded (B, T, 65536) batch of SAE codes is 4.3 GB against 47 KB of real data,
so these learners read the sparse form directly. That is a performance change and
must not alter the function: these tests build the same batch both ways and
assert the logits match.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.harness.learners import (  # noqa: E402
    AttnQueryPool, SparseAttnQueryPool, SparseTransformerPool, TransformerPool,
    build_sparse_learner,
)
from src.harness.sparse_vec import SparseTokenSplit, write_token_csr  # noqa: E402

D = 24


def _steps(tmp_path, n=9, seed=0, t_max=512):
    """Random per-token codes, plus the dense (n, T, D) tensor they represent."""
    rng = np.random.default_rng(seed)
    steps, labels, dense_rows = [], [], []
    for k in range(n):
        n_tok = int(rng.integers(1, 6))
        toks, rows = [], []
        for _ in range(n_tok):
            nnz = int(rng.integers(1, 5))
            idx = np.sort(rng.choice(D, nnz, replace=False))
            val = rng.uniform(0.2, 2.0, nnz).astype(np.float16)
            toks.append((idx, val))
            r = np.zeros(D, np.float32); r[idx] = val.astype(np.float32)
            rows.append(r)
        steps.append(toks); labels.append(k % 2); dense_rows.append(np.stack(rows))
    write_token_csr(tmp_path / "t.npz", steps, np.array(labels), D)
    return SparseTokenSplit(tmp_path / "t.npz", t_max=t_max), dense_rows


def _densify(dense_rows, idx):
    T = max(dense_rows[i].shape[0] for i in idx)
    x = np.zeros((len(idx), T, D), np.float32)
    m = np.zeros((len(idx), T), np.float32)
    for j, i in enumerate(idx):
        r = dense_rows[i]
        x[j, :r.shape[0]] = r
        m[j, :r.shape[0]] = 1.0
    return torch.from_numpy(x), torch.from_numpy(m)


def test_sparse_attn_query_matches_the_dense_one(tmp_path):
    sp, rows = _steps(tmp_path, seed=1)
    idx = np.array([0, 3, 7, 5])
    batch, mask, y = sp.collate(idx)
    x_dense, m_dense = _densify(rows, idx)
    torch.testing.assert_close(mask, m_dense)

    torch.manual_seed(0); dense = AttnQueryPool(D).eval()
    torch.manual_seed(0); sparse = SparseAttnQueryPool(D).eval()
    with torch.no_grad():
        torch.testing.assert_close(sparse(batch), dense(x_dense, m_dense),
                                   rtol=1e-4, atol=1e-5)


def test_sparse_transformer_matches_the_dense_one(tmp_path):
    sp, rows = _steps(tmp_path, seed=2)
    idx = np.array([1, 2, 8])
    batch, mask, _ = sp.collate(idx)
    x_dense, m_dense = _densify(rows, idx)

    torch.manual_seed(0); sparse = SparseTransformerPool(D, d_model=16, nhead=2,
                                                         nlayers=1, ff=32, t_max=32).eval()
    torch.manual_seed(0); dense = TransformerPool(D, d_model=16, nhead=2,
                                                  nlayers=1, ff=32, t_max=32).eval()
    # the dense model's Linear projection is the sparse model's embedding-bag weight
    with torch.no_grad():
        dense.proj.weight.copy_(sparse.proj_w.T)
        dense.proj.bias.copy_(sparse.proj_b)
        dense.pos.copy_(sparse.pos)
        dense.enc.load_state_dict(sparse.enc.state_dict())
        dense.head.load_state_dict(sparse.head.state_dict())
        torch.testing.assert_close(sparse(batch), dense(x_dense, m_dense),
                                   rtol=1e-4, atol=1e-5)


def test_truncation_keeps_the_last_tokens(tmp_path):
    """Same rule as the dense span loader: the end of a step is what is kept."""
    sp, rows = _steps(tmp_path, n=6, seed=4, t_max=2)
    batch, mask, _ = sp.collate(np.arange(6))
    assert batch.T <= 2
    for j in range(6):
        assert int(mask[j].sum()) == min(rows[j].shape[0], 2)


def test_a_single_token_step_is_handled(tmp_path):
    steps = [[(np.array([1, 4]), np.array([1.0, 2.0], np.float16))]]
    write_token_csr(tmp_path / "one.npz", steps, np.array([1]), D)
    sp = SparseTokenSplit(tmp_path / "one.npz")
    batch, mask, y = sp.collate(np.array([0]))
    assert batch.B == 1 and batch.T == 1 and mask.sum() == 1
    out = SparseAttnQueryPool(D).eval()(batch)
    assert out.shape == (1,) and torch.isfinite(out).all()


def test_build_sparse_learner_rejects_a_vector_learner():
    with pytest.raises(ValueError, match="not a sequence learner"):
        build_sparse_learner("linear", D)
