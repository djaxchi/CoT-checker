"""Sparse SAE codes must densify to exactly the vector the learner would have got.

A pooled Qwen-Scope code is 65,536 wide with a few hundred non-zeros, so it is
stored CSR and scattered into a dense batch at collate time. That is a storage
change (67 GB -> ~1.5 GB for the train split), so it must not change a single
number the model sees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.harness.sparse_vec import SparseVecSplit, write_csr  # noqa: E402

D = 64


def _dense_and_sparse(tmp_path, n=25, seed=0):
    """Build a known dense matrix and its CSR form."""
    rng = np.random.default_rng(seed)
    dense = np.zeros((n, D), dtype=np.float32)
    rows, labels = [], []
    for k in range(n):
        nnz = int(rng.integers(1, 8))
        idx = np.sort(rng.choice(D, nnz, replace=False))
        val = rng.uniform(0.1, 2.0, nnz).astype(np.float16)
        dense[k, idx] = val.astype(np.float32)
        rows.append((idx, val))
        labels.append(k % 2)
    stats = write_csr(tmp_path / "r.npz", rows, np.array(labels), D)
    return dense, np.array(labels, dtype=np.float32), stats


def test_collate_reconstructs_the_dense_vector_exactly(tmp_path):
    dense, labels, _ = _dense_and_sparse(tmp_path)
    sp = SparseVecSplit(tmp_path / "r.npz")
    idx = np.array([7, 0, 24, 13])
    x, mask, y = sp.collate(idx)
    assert mask is None                       # vector-learner contract
    np.testing.assert_allclose(x.numpy(), dense[idx], rtol=0, atol=0)
    np.testing.assert_allclose(y.numpy(), labels[idx])


def test_every_row_round_trips(tmp_path):
    dense, _, _ = _dense_and_sparse(tmp_path, n=40, seed=3)
    sp = SparseVecSplit(tmp_path / "r.npz")
    x, _, _ = sp.collate(np.arange(len(sp)))
    np.testing.assert_array_equal(x.numpy(), dense)


def test_an_all_zero_row_survives(tmp_path):
    """A step whose pooled code is empty must still produce a zero vector, not
    shift every later row's offsets."""
    rows = [(np.array([1, 5]), np.array([1.0, 2.0], np.float16)),
            (np.zeros(0, np.int64), np.zeros(0, np.float16)),
            (np.array([3]), np.array([4.0], np.float16))]
    write_csr(tmp_path / "z.npz", rows, np.array([0, 1, 0]), D)
    sp = SparseVecSplit(tmp_path / "z.npz")
    x, _, _ = sp.collate(np.arange(3))
    assert x[1].abs().sum() == 0
    assert x[0, 1] == 1.0 and x[0, 5] == 2.0 and x[2, 3] == 4.0


def test_reported_stats_describe_the_store(tmp_path):
    _, _, stats = _dense_and_sparse(tmp_path, n=30, seed=5)
    sp = SparseVecSplit(tmp_path / "r.npz")
    assert stats["items"] == len(sp) == 30
    assert abs(stats["mean_nnz"] - sp.mean_nnz) < 1e-9
    assert 0 < stats["density"] < 1


def test_sparse_storage_is_much_smaller_than_dense(tmp_path):
    _, _, stats = _dense_and_sparse(tmp_path, n=200, seed=7)
    dense_bytes = 200 * D * 2                 # float16
    assert stats["bytes"] < dense_bytes       # the whole reason for CSR
