"""Test delta derivation from a token store: delta = last_row - pre_boundary_row."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from derive_delta_from_token_store import derive_delta_split, derive_split  # noqa: E402

from src.repstore import TOKEN_SEQ, RepSpec, write_split


def _write_item(sd, items, metas, d=2):
    spec = RepSpec(name="tok", kind=TOKEN_SEQ, dim=d, layer=-1, backbone="t", readout="r")
    write_split(sd, items, [0] * len(items), metas, spec)


def test_delta_values_and_global_order(tmp_path):
    d = 2
    # global 0: last[5,5]-pre[1,1]=[4,4] ; global 2: last[9,9]-pre[2,2]=[7,7]
    _write_item(
        tmp_path / "shard_00",
        [np.array([[1, 1], [5, 5]], np.float32), np.array([[0, 0], [2, 2], [9, 9]], np.float32)],
        [{"pre_step_boundary_idx": 0, "global_index": 0},
         {"pre_step_boundary_idx": 1, "global_index": 2}],
    )
    # global 1: last[8,8]-pre[3,3]=[5,5]
    _write_item(
        tmp_path / "shard_01",
        [np.array([[3, 3], [8, 8]], np.float32)],
        [{"pre_step_boundary_idx": 0, "global_index": 1}],
    )
    delta, y, meta = derive_delta_split(tmp_path)
    assert delta.shape == (3, d)
    np.testing.assert_allclose(delta[0], [4, 4])
    np.testing.assert_allclose(delta[1], [5, 5])   # global order 0,1,2
    np.testing.assert_allclose(delta[2], [7, 7])
    assert [m["global_index"] for m in meta] == [0, 1, 2]


def test_last_readout_reproduces_last_row(tmp_path):
    # global 0 last=[5,5]; global 1 last=[8,8]; global 2 last=[9,9]
    _write_item(
        tmp_path / "shard_00",
        [np.array([[1, 1], [5, 5]], np.float32), np.array([[0, 0], [2, 2], [9, 9]], np.float32)],
        [{"pre_step_boundary_idx": 0, "global_index": 0},
         {"pre_step_boundary_idx": 1, "global_index": 2}],
    )
    _write_item(
        tmp_path / "shard_01",
        [np.array([[3, 3], [8, 8]], np.float32)],
        [{"pre_step_boundary_idx": 0, "global_index": 1}],
    )
    v, _, _ = derive_split(tmp_path, "last")
    np.testing.assert_allclose(v, [[5, 5], [8, 8], [9, 9]])


def test_mean_max_over_step_span(tmp_path):
    # item rows [[2,2],[4,4],[10,10]], pre=0 -> step span = rows[1:] = [[4,4],[10,10]]
    _write_item(
        tmp_path / "shard_00",
        [np.array([[2, 2], [4, 4], [10, 10]], np.float32)],
        [{"pre_step_boundary_idx": 0, "global_index": 0}],
    )
    np.testing.assert_allclose(derive_split(tmp_path, "mean")[0][0], [7, 7])
    np.testing.assert_allclose(derive_split(tmp_path, "max")[0][0], [10, 10])
    np.testing.assert_allclose(derive_split(tmp_path, "last")[0][0], [10, 10])


def test_multistat_concat(tmp_path):
    # span = rows[1:] = [[4,4],[10,10]] -> mean[7,7] max[10,10] min[4,4] std[3,3] last[10,10]
    _write_item(
        tmp_path / "shard_00",
        [np.array([[2, 2], [4, 4], [10, 10]], np.float32)],
        [{"pre_step_boundary_idx": 0, "global_index": 0}],
    )
    v = derive_split(tmp_path, "multistat")[0][0]
    assert v.shape == (10,)  # 5 stats x d(=2)
    np.testing.assert_allclose(v, [7, 7, 10, 10, 4, 4, 3, 3, 10, 10])


def test_boundary_stats_prepends_boundary_and_matches_multistat(tmp_path):
    """boundary_stats = concat[pre-step boundary row, multistat(step span)].

    Item global 0: rows [1,1] (boundary), [5,5] (the only step token), so the
    5-stat pool of a one-token span is that token repeated and the prepended
    boundary is [1,1]. Item global 2: boundary [2,2], span rows [9,9] only.
    """
    d = 2
    _write_item(
        tmp_path / "shard_00",
        [np.array([[1, 1], [5, 5]], np.float32), np.array([[0, 0], [2, 2], [9, 9]], np.float32)],
        [{"pre_step_boundary_idx": 0, "global_index": 0},
         {"pre_step_boundary_idx": 1, "global_index": 2}],
    )
    _write_item(
        tmp_path / "shard_01",
        [np.array([[3, 3], [8, 8]], np.float32)],
        [{"pre_step_boundary_idx": 0, "global_index": 1}],
    )
    v, _, _ = derive_split(tmp_path, "boundary_stats")
    ms, _, _ = derive_split(tmp_path, "multistat")
    assert v.shape == (3, 6 * d)
    assert ms.shape == (3, 5 * d)
    # the trailing 5*d block is exactly multistat
    np.testing.assert_allclose(v[:, d:], ms)
    # the leading d block is the pre-step boundary row, in global order
    np.testing.assert_allclose(v[:, :d], [[1, 1], [3, 3], [2, 2]])


def test_boundary_stats_is_more_expressive_than_delta(tmp_path):
    """A linear map over boundary_stats can reproduce delta (-boundary + last),
    which is what makes concat strictly more expressive than the forced
    subtraction. Guards the claim the leaderboard row is testing."""
    d = 2
    _write_item(
        tmp_path / "shard_00",
        [np.array([[1, 1], [5, 5]], np.float32), np.array([[0, 0], [2, 2], [9, 9]], np.float32)],
        [{"pre_step_boundary_idx": 0, "global_index": 0},
         {"pre_step_boundary_idx": 1, "global_index": 2}],
    )
    _write_item(
        tmp_path / "shard_01",
        [np.array([[3, 3], [8, 8]], np.float32)],
        [{"pre_step_boundary_idx": 0, "global_index": 1}],
    )
    v, _, _ = derive_split(tmp_path, "boundary_stats")
    delta, _, _ = derive_delta_split(tmp_path)
    # blocks are [boundary, mean, max, min, std, last]; last block minus first
    reconstructed = v[:, 5 * d:].astype(np.float32) - v[:, :d].astype(np.float32)
    np.testing.assert_allclose(reconstructed, delta.astype(np.float32))
