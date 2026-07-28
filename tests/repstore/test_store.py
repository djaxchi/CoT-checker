"""Tests for the uniform representation store."""

from __future__ import annotations

import numpy as np
import pytest

from src.repstore import (
    STEP_SEQ,
    TOKEN_SEQ,
    VECTOR,
    RepSpec,
    RepSplit,
    ShardedRepSplit,
    write_split,
    write_vector_split,
)


def _meta(n):
    return [{"uid": f"u{i}", "trace_id": f"t{i//2}", "step_idx": i % 2} for i in range(n)]


def test_ragged_roundtrip(tmp_path):
    d = 4
    items = [
        np.arange(1 * d, dtype=np.float32).reshape(1, d),
        np.arange(3 * d, dtype=np.float32).reshape(3, d),
        np.arange(2 * d, dtype=np.float32).reshape(2, d),
    ]
    y = [1, 0, 1]
    spec = RepSpec(name="step_tokens", kind=STEP_SEQ, dim=d, layer=-1,
                   backbone="test", readout="step_tokens")
    write_split(tmp_path, items, y, _meta(3), spec)

    rs = RepSplit(tmp_path)
    assert len(rs) == 3
    assert not rs.is_vector
    assert list(rs.lengths) == [1, 3, 2]
    np.testing.assert_array_equal(rs.item(1), items[1])
    np.testing.assert_array_equal(rs.item(2), items[2])
    assert list(rs.y) == [1, 0, 1]
    assert rs.spec.name == "step_tokens"


def test_vector_reductions(tmp_path):
    d = 3
    items = [np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32)]  # one item, 2 rows
    spec = RepSpec(name="s", kind=STEP_SEQ, dim=d, layer=-1, backbone="t", readout="r")
    write_split(tmp_path, items, [1], _meta(1), spec)
    rs = RepSplit(tmp_path)
    np.testing.assert_allclose(rs.vectors("mean")[0], [2, 3, 4])
    np.testing.assert_allclose(rs.vectors("max")[0], [3, 4, 5])
    np.testing.assert_allclose(rs.vectors("last")[0], [3, 4, 5])
    np.testing.assert_allclose(rs.vectors("first")[0], [1, 2, 3])


def test_vector_fastpath_matches_and_is_vector(tmp_path):
    d = 5
    h = np.random.default_rng(0).standard_normal((10, d)).astype(np.float32)
    spec = RepSpec(name="dense_last", kind=VECTOR, dim=d, layer=-1,
                   backbone="Qwen2.5-7B", readout="last")
    write_vector_split(tmp_path, h, list(range(10)), _meta(10), spec)
    rs = RepSplit(tmp_path)
    assert rs.is_vector
    # vectors(reduce) is identity for a vector rep (up to float16 storage)
    np.testing.assert_allclose(rs.vectors("mean"), h.astype(np.float16), rtol=0, atol=1e-2)
    np.testing.assert_allclose(rs.vectors("mean"), rs.vectors("last"))


def test_kind_vector_rejects_sequences(tmp_path):
    d = 2
    items = [np.zeros((2, d), dtype=np.float32)]
    spec = RepSpec(name="bad", kind=VECTOR, dim=d, layer=-1, backbone="t", readout="r")
    with pytest.raises(ValueError):
        write_split(tmp_path, items, [0], _meta(1), spec)


def test_dim_mismatch_rejected(tmp_path):
    spec = RepSpec(name="s", kind=STEP_SEQ, dim=4, layer=-1, backbone="t", readout="r")
    with pytest.raises(ValueError):
        write_split(tmp_path, [np.zeros((1, 3), dtype=np.float32)], [0], _meta(1), spec)


def test_bad_kind_rejected():
    with pytest.raises(ValueError):
        RepSpec(name="s", kind="nope", dim=4, layer=-1, backbone="t", readout="r")


def test_sharded_view_reconstructs_global_order(tmp_path):
    d = 3
    spec = RepSpec(name="tok", kind=TOKEN_SEQ, dim=d, layer=-1, backbone="t", readout="r")
    # 4 items with global_index 0..3, striped across 2 shards by parity.
    items = {
        0: np.full((1, d), 0.0, np.float32),
        1: np.full((2, d), 1.0, np.float32),
        2: np.full((1, d), 2.0, np.float32),
        3: np.full((3, d), 3.0, np.float32),
    }
    for shard, gidxs in {0: [0, 2], 1: [1, 3]}.items():
        sd = tmp_path / f"shard_{shard:02d}"
        write_split(
            sd, [items[g] for g in gidxs], [g % 2 for g in gidxs],
            [{"global_index": g} for g in gidxs], spec,
        )
    view = ShardedRepSplit(tmp_path)
    assert len(view) == 4
    for g in range(4):
        np.testing.assert_array_equal(view.item(g), items[g])
    np.testing.assert_array_equal(view.vectors("mean")[3], [3, 3, 3])
    assert list(view.y) == [0, 1, 0, 1]
