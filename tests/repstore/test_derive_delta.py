"""Test delta derivation from a token store: delta = last_row - pre_boundary_row."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from derive_delta_from_token_store import derive_delta_split  # noqa: E402

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
