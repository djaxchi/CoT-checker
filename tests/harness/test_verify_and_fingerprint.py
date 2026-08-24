"""The equivalence check must actually fail when the stores differ.

A verification that only ever prints IDENTICAL is worth nothing as a gate on
deleting a 984G store, so these tests corrupt a compact store in each of the ways
it could realistically go wrong and assert the checker catches it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_step_span_store import compact_shard  # noqa: E402
from verify_step_span_store import compare, compare_rows  # noqa: E402

from src.repstore import TOKEN_SEQ, RepSpec, split_fingerprint, write_split  # noqa: E402

D = 3


def _build(tmp_path: Path):
    master, compact = tmp_path / "master", tmp_path / "compact"
    rng = np.random.default_rng(0)
    items, metas, labels = [], [], []
    for k in range(6):
        n, start = 12, 5
        items.append(rng.normal(size=(n, D)).astype(np.float32))
        metas.append({"uid": f"u{k}", "n_tokens": n, "step_start_idx": start,
                      "pre_step_boundary_idx": start - 1, "global_index": k,
                      "label": k % 2, "step_idx": 1})
        labels.append(k % 2)
    write_split(master / "probe_train_full" / "shard_00", items, labels, metas,
                RepSpec(name="tokens_last_layer", kind=TOKEN_SEQ, dim=D, layer=-1,
                        backbone="t", readout="token_all_last_layer"))
    compact_shard(master / "probe_train_full" / "shard_00",
                  compact / "probe_train_full" / "shard_00", "step_spans")
    return master, compact


def test_clean_compaction_passes_both_modes(tmp_path):
    master, compact = _build(tmp_path)
    assert compare_rows(master, compact, "probe_train_full", 0)
    assert compare(master, compact, "probe_train_full", "multistat")


def test_a_single_flipped_row_is_caught(tmp_path):
    master, compact = _build(tmp_path)
    h_path = compact / "probe_train_full" / "shard_00" / "h.npy"
    h = np.load(h_path)
    h[3, 0] = np.float16(h[3, 0] + np.float16(1.0))
    np.save(h_path, h)
    assert not compare_rows(master, compact, "probe_train_full", 0)


def test_a_truncated_item_is_caught(tmp_path):
    master, compact = _build(tmp_path)
    shard = compact / "probe_train_full" / "shard_00"
    lengths = np.load(shard / "lengths.npy")
    lengths[0] -= 1
    np.save(shard / "lengths.npy", lengths)
    h = np.load(shard / "h.npy")
    np.save(shard / "h.npy", np.delete(h, 0, axis=0))
    assert not compare_rows(master, compact, "probe_train_full", 0)


def test_a_reordered_item_is_caught(tmp_path):
    """Items must line up by identity, not only by count."""
    master, compact = _build(tmp_path)
    meta_path = compact / "probe_train_full" / "shard_00" / "meta.jsonl"
    rows = meta_path.read_text().splitlines()
    rows[0], rows[1] = rows[1], rows[0]
    meta_path.write_text("\n".join(rows) + "\n")
    assert not compare_rows(master, compact, "probe_train_full", 0)


def test_fingerprint_is_stable_and_sensitive(tmp_path):
    master, compact = _build(tmp_path)
    split = compact / "probe_train_full"
    before = split_fingerprint(split)
    assert before == split_fingerprint(split)          # deterministic
    assert len(before) == 32

    y_path = split / "shard_00" / "y.npy"
    y = np.load(y_path)
    y[0] = 1 - y[0]
    np.save(y_path, y)
    assert split_fingerprint(split) != before          # a flipped label changes it


def test_fingerprint_separates_the_two_stores(tmp_path):
    master, compact = _build(tmp_path)
    assert split_fingerprint(master / "probe_train_full") != \
           split_fingerprint(compact / "probe_train_full")


def test_full_mode_hashes_every_byte(tmp_path):
    """The sampled fingerprint may miss a single row between sample points; the
    full mode is the fallback for when that matters."""
    master, compact = _build(tmp_path)
    split = compact / "probe_train_full"
    before = split_fingerprint(split, full=True)
    h_path = split / "shard_00" / "h.npy"
    h = np.load(h_path)
    h[2, 1] = np.float16(h[2, 1] + np.float16(0.5))
    np.save(h_path, h)
    assert split_fingerprint(split, full=True) != before
