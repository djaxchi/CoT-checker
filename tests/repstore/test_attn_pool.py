"""Plumbing test for the attention-pool probe over a token store."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from train_attn_pool_probe import AttnPoolProbe, build_handles, collate, score_split  # noqa: E402

from src.repstore import TOKEN_SEQ, RepSpec, ShardedRepSplit, write_split


def test_attn_pool_forward_and_score(tmp_path):
    d = 4
    spec = RepSpec(name="tok", kind=TOKEN_SEQ, dim=d, layer=-1, backbone="t", readout="r")
    items = [
        np.arange(3 * d, dtype=np.float32).reshape(3, d),
        np.arange(2 * d, dtype=np.float32).reshape(2, d),
    ]
    metas = [
        {"step_start_idx": 1, "pre_step_boundary_idx": 0, "n_tokens": 3, "global_index": 0},
        {"step_start_idx": 1, "pre_step_boundary_idx": 0, "n_tokens": 2, "global_index": 1},
    ]
    write_split(tmp_path / "shard_00", items, [1, 0], metas, spec)

    view = ShardedRepSplit(tmp_path)
    handles, meta = build_handles(view)
    assert len(handles) == 2
    # step_start passed through
    assert handles[0][2] == 1 and handles[0][3] == 3

    model = AttnPoolProbe(d)
    xb, mb, yb = collate(handles, [0, 1], t_max=8, device=torch.device("cpu"))
    assert xb.shape[0] == 2 and xb.shape[2] == d
    out = model(xb, mb)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()

    scores = score_split(model, handles, t_max=8, batch_size=2, device=torch.device("cpu"))
    assert scores.shape == (2,)
    assert ((scores >= 0) & (scores <= 1)).all()
