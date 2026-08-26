"""The fast loader must produce exactly what the slow one produced.

Grid B was I/O bound, so this replaces per-item Python reads with one vectorized
gather and adds length bucketing. Both are performance changes that must not
change a single number the model sees, which is what these tests pin against the
original `collate_seq` implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_rep_learner_cell import build_handles, collate_seq  # noqa: E402

from src.harness.spanloader import (  # noqa: E402
    SpanLoader, batch_gather_indices, length_bucketed_batches, span_bounds,
)
from src.repstore import TOKEN_SEQ, RepSpec, write_split  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402

D = 6


def _store(tmp_path, n=40, seed=0):
    """One shard of span items: [boundary] ++ [step tokens], lengths 1..30."""
    rng = np.random.default_rng(seed)
    items, metas, labels = [], [], []
    for k in range(n):
        span = int(rng.integers(1, 30))
        items.append(rng.normal(size=(span + 1, D)).astype(np.float32))
        metas.append({"uid": f"u{k}", "n_tokens": span + 1, "step_start_idx": 1,
                      "pre_step_boundary_idx": 0, "global_index": k,
                      "label": k % 2, "step_idx": 1})
        labels.append(k % 2)
    write_split(tmp_path / "split" / "shard_00", items, labels, metas,
                RepSpec(name="step_spans", kind=TOKEN_SEQ, dim=D, layer=-1,
                        backbone="t", readout="step_span_with_boundary"))
    handles, _ = build_handles(ShardedRepSplit(tmp_path / "split"))
    return handles


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("t_max", [512, 8])
def test_collate_matches_the_original_loader(tmp_path, preload, t_max):
    handles = _store(tmp_path)
    dev = torch.device("cpu")
    loader = SpanLoader(handles, t_max=t_max, device=dev, preload=preload)
    idx = np.array([3, 17, 0, 39, 12], dtype=np.int64)

    x_ref, m_ref, y_ref = collate_seq(handles, idx.tolist(), t_max, dev)
    x, m, y = loader.collate(idx)

    torch.testing.assert_close(x, x_ref)
    torch.testing.assert_close(m, m_ref)
    torch.testing.assert_close(y, y_ref)


def test_truncation_keeps_the_end_of_the_step(tmp_path):
    handles = _store(tmp_path)
    starts, lengths = span_bounds(handles, t_max=4)
    assert lengths.max() <= 4
    rs, li, step_start, n_tokens, _ = handles[0]
    end = int(rs.offsets[li + 1])
    assert int(starts[0]) + int(lengths[0]) == end     # the span's last row is kept


def test_gather_indices_cover_every_real_token_once():
    lengths = np.array([3, 1, 4], dtype=np.int64)
    starts = np.array([10, 100, 50], dtype=np.int64)
    src, dest = batch_gather_indices(starts, lengths, T=4)
    assert list(src) == [10, 11, 12, 100, 50, 51, 52, 53]
    assert list(dest) == [0, 1, 2, 4, 8, 9, 10, 11]     # row-major (B=3, T=4)
    assert len(set(dest)) == len(dest)


def test_bucketing_is_a_permutation_of_the_epoch():
    lengths = np.random.default_rng(0).integers(1, 200, size=257)
    rng = np.random.default_rng(1)
    batches = length_bucketed_batches(lengths, batch_size=16, rng=rng)
    seen = np.concatenate(batches)
    assert sorted(seen.tolist()) == list(range(257))     # every index exactly once


def test_bucketing_cuts_padding_waste(tmp_path):
    """The point of bucketing: batches stop being padded to an outlier."""
    handles = _store(tmp_path, n=512, seed=3)
    loader = SpanLoader(handles, t_max=512, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    plain = loader.padding_waste(loader.batches(32, rng, bucketed=False))
    bucketed = loader.padding_waste(loader.batches(32, rng, bucketed=True))
    assert bucketed < plain
    assert bucketed < 0.25


def test_preload_detaches_from_the_store(tmp_path):
    """After preloading, an epoch must touch no disk; corrupting the mmap-backed
    store afterwards must not change a batch."""
    handles = _store(tmp_path)
    dev = torch.device("cpu")
    loader = SpanLoader(handles, t_max=512, device=dev, preload=True)
    idx = np.arange(5)
    before, _, _ = loader.collate(idx)

    h_path = tmp_path / "split" / "shard_00" / "h.npy"
    arr = np.load(h_path)
    np.save(h_path, np.zeros_like(arr))
    handles2, _ = build_handles(ShardedRepSplit(tmp_path / "split"))
    loader.handles = handles2

    after, _, _ = loader.collate(idx)
    torch.testing.assert_close(before, after)


def test_eval_batches_preserve_order(tmp_path):
    handles = _store(tmp_path)
    loader = SpanLoader(handles, t_max=512, device=torch.device("cpu"))
    assert np.array_equal(np.concatenate(loader.eval_batches(7)),
                          np.arange(len(handles)))


def test_preload_bytes_is_reported_before_allocating(tmp_path):
    handles = _store(tmp_path)
    loader = SpanLoader(handles, t_max=512, device=torch.device("cpu"))
    assert loader.preload_bytes() == int(loader.lengths.sum()) * D * 2
