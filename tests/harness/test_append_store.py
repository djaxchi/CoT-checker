"""--append_store: one vector per item appended to its step span (read_token_v1 Stage F-C)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_rep_learner_cell import appended  # noqa: E402
from src.repstore import VECTOR, RepSpec, ShardedRepSplit  # noqa: E402
from src.repstore.store import write_vector_split  # noqa: E402
from tests.harness.test_rep_learner_cell import D, _run, store  # noqa: E402,F401


def _mirror(src: Path, dst: Path, reverse: bool = False) -> None:
    """A vector store with src's items, each row = its label sign; optionally reversed."""
    for split in (p for p in src.iterdir() if p.is_dir()):
        view = ShardedRepSplit(split)
        meta = view.meta()
        y = view.y
        if reverse:
            meta = [dict(m, global_index=i) for i, m in enumerate(meta[::-1])]
            y = y[::-1]
        h = np.repeat((np.asarray(y, np.float32) * 4 - 2)[:, None], D, 1)
        write_vector_split(dst / split.name / "shard_00", h, y, meta,
                           RepSpec(name="rt_append", kind=VECTOR, dim=D, layer=35,
                                   backbone="test", readout="test"))


def test_appended_row_follows_each_items_last_token():
    x = torch.zeros(2, 3, 2)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    rows = np.array([[0, 0], [7, 7], [9, 9]], np.float16)
    f = appended(lambda idx: (x.clone(), mask.clone(), torch.zeros(2)), rows)
    xb, mb, _ = f(np.array([1, 2]))
    assert xb.shape == (2, 4, 2)
    assert torch.equal(mb, torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]))
    assert torch.equal(xb[0, 2], torch.tensor([7.0, 7.0])) and torch.equal(xb[1, 3], torch.tensor([9.0, 9.0]))


def test_cell_with_append_runs_and_records_it(store, tmp_path):
    prm, pb = store
    _mirror(prm, tmp_path / "aprm")
    _mirror(pb, tmp_path / "apb")
    out = tmp_path / "cell"
    r = _run(prm, pb, out, "step_tokens", "attn_query",
             ["--append_store", str(tmp_path / "aprm"), "--append_pb_store", str(tmp_path / "apb")])
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((out / "results.json").read_text())
    assert res["protocol"]["append_store"] == str(tmp_path / "aprm")
    assert set(res["append_inputs"]) >= {"prm/probe_train_full", "pb/gsm8k"}
    assert res["in_domain"]["auroc"] > 0.9


def test_append_store_in_another_order_is_refused(store, tmp_path):
    prm, pb = store
    _mirror(prm, tmp_path / "aprm", reverse=True)
    _mirror(pb, tmp_path / "apb")
    r = _run(prm, pb, tmp_path / "cell", "step_tokens", "attn_query",
             ["--append_store", str(tmp_path / "aprm"), "--append_pb_store", str(tmp_path / "apb")])
    assert r.returncode != 0 and "same order" in r.stderr


def test_append_store_needs_a_sequence_learner(store, tmp_path):
    prm, pb = store
    r = _run(prm, pb, tmp_path / "cell", "last_token", "linear",
             ["--append_store", str(tmp_path), "--append_pb_store", str(tmp_path)])
    assert r.returncode != 0 and "--append_store needs" in r.stderr
