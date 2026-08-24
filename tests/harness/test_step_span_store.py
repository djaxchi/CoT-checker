"""Compacting the token store must not change a single derived vector.

The compact store drops the question and prior-step token states and keeps only
the pre-step boundary row followed by the step's own tokens. Since no
representation reads the dropped rows, every readout derived from the compact
store must equal the one derived from the master store, bit for bit. These tests
pin that equivalence, because it is the entire justification for the 6.7x saving.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_step_span_store import compact_meta, compact_shard, span_bounds  # noqa: E402
from derive_delta_from_token_store import derive_split  # noqa: E402

from src.repstore import TOKEN_SEQ, RepSpec, write_split  # noqa: E402
from src.repstore.store import RepSplit  # noqa: E402

D = 3


def _item(n_tokens: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_tokens, D)).astype(np.float32)


def _write_master(root: Path) -> None:
    """Two shards, items whose step span sits well inside a long prefix."""
    specs = [  # (n_tokens, step_start_idx, global_index, label)
        [(10, 7, 0, 0), (12, 3, 2, 1)],
        [(9, 8, 1, 1)],
    ]
    for si, shard in enumerate(specs):
        items, metas, labels = [], [], []
        for n_tokens, start, gi, label in shard:
            items.append(_item(n_tokens, seed=gi + 1))
            metas.append({
                "uid": f"u{gi}", "n_tokens": n_tokens, "step_start_idx": start,
                "pre_step_boundary_idx": start - 1, "global_index": gi,
                "label": label, "step_idx": 1,
            })
            labels.append(label)
        write_split(
            root / f"shard_{si:02d}", items, labels, metas,
            RepSpec(name="tokens_last_layer", kind=TOKEN_SEQ, dim=D, layer=-1,
                    backbone="Qwen2.5-7B", readout="token_all_last_layer"),
        )


def _compact(master: Path, out: Path) -> None:
    for sd in sorted(master.glob("shard_*")):
        compact_shard(sd, out / sd.name, "step_spans")


@pytest.mark.parametrize(
    "readout", ["last", "mean", "max", "delta", "multistat", "boundary_stats"])
def test_every_readout_is_identical_after_compaction(tmp_path, readout):
    master, compact = tmp_path / "master", tmp_path / "compact"
    _write_master(master)
    _compact(master, compact)

    v_master, y_master, meta_master = derive_split(master, readout, sort=True)
    v_compact, y_compact, meta_compact = derive_split(compact, readout, sort=True)

    np.testing.assert_array_equal(v_master, v_compact)
    np.testing.assert_array_equal(y_master, y_compact)
    assert [m["global_index"] for m in meta_master] == \
           [m["global_index"] for m in meta_compact]


def test_compaction_keeps_only_boundary_plus_span(tmp_path):
    master, compact = tmp_path / "master", tmp_path / "compact"
    _write_master(master)
    _compact(master, compact)

    for sd in sorted(master.glob("shard_*")):
        src = RepSplit(sd)
        dst = RepSplit(compact / sd.name)
        for k, m in enumerate(src.meta()):
            kept = int(m["n_tokens"]) - int(m["pre_step_boundary_idx"])
            assert int(dst.lengths[k]) == kept
            np.testing.assert_array_equal(dst.item(k), src.item(k)[-kept:])


def test_compacted_meta_reindexes_and_records_the_original(tmp_path):
    row = {"uid": "u0", "n_tokens": 10, "step_start_idx": 7,
           "pre_step_boundary_idx": 6, "global_index": 0}
    out = compact_meta(row, new_len=4)
    assert (out["pre_step_boundary_idx"], out["step_start_idx"], out["n_tokens"]) == (0, 1, 4)
    assert (out["orig_pre_step_boundary_idx"], out["orig_step_start_idx"],
            out["orig_n_tokens"]) == (6, 7, 10)
    # The compacted row is itself a fixed point: compacting twice is a no-op on
    # the offsets, so a store can be rebuilt from a compact store safely.
    assert span_bounds(out, out["n_tokens"]) == (0, 4)


def test_rejects_a_meta_row_whose_offsets_disagree(tmp_path):
    bad = {"uid": "u", "n_tokens": 10, "step_start_idx": 7, "pre_step_boundary_idx": 3}
    with pytest.raises(ValueError, match="step_start_idx"):
        span_bounds(bad, 10)


def test_rejects_a_boundary_outside_the_item(tmp_path):
    bad = {"uid": "u", "n_tokens": 4, "step_start_idx": 5, "pre_step_boundary_idx": 4}
    with pytest.raises(ValueError, match="outside"):
        span_bounds(bad, 4)
