import json

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts import read_token_store as rts  # noqa: E402
from scripts import read_token_traj as rtt  # noqa: E402
from scripts.read_token_probe import ReadToken, read_all_states, reduce_states, tokenize  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402
from tests.test_read_token_probe import tiny_model  # noqa: E402
from tests.test_read_token_store import D, CharTok, ref_store  # noqa: E402

# u0: no previous step; u1: two; u2: one, and u2 itself is u1's first previous step
ROWS = [("p0", "", "s0"), ("p1", "first\n\nsecond", "s1"), ("p1", "", "first")]


def test_prior_nodes_are_oldest_first_each_under_its_own_prefix():
    it = {"problem": "p", "prefix": "a\n\nb\n\nc", "prior_steps": ["a", "b", "c"], "step": "d"}
    assert [(n["prefix"], n["step"]) for n in rtt.prior_nodes(it)] == [
        ("", "a"), ("a", "b"), ("a\n\nb", "c")]


def test_traj_rows_are_current_then_previous_steps_most_recent_first(tmp_path, monkeypatch):
    src = tmp_path / "s.jsonl"
    with src.open("w") as f:
        for k, (p, pre, step) in enumerate(ROWS):
            f.write(json.dumps({"uid": f"u{k}", "problem": p, "prefix": pre,
                                "candidate_step": step, "label": k % 2}) + "\n")
    order = [2, 0, 1]
    ref = tmp_path / "ref" / "val_5k"
    ref_store(ref, 3, order)
    model = tiny_model()
    rt = ReadToken(2, D, torch.randn(2, D) * 0.02, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, 0.3)
    ck = tmp_path / "rt.pt"
    torch.save({"soft": rt.soft.detach(), "head": rt.head.state_dict(), "m": 2, "read": "last",
                "head_kind": "linear", "deep": rt.deep.detach()}, ck)
    monkeypatch.setattr(rts, "load_model", lambda *_a: (model, CharTok()))
    monkeypatch.setattr(rtt, "load_model", lambda *_a: (model, CharTok()))
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--ref", str(ref), "--source",
                                     f"prm800k:{src}", "--out_root", str(out), "--blocks", "0,1",
                                     "--append_block", "1", "--chunk", "2"])
    rts.main()
    layers = out / "rt_layers" / "val_5k"
    common = ["--layers", str(layers), "--source", f"prm800k:{src}", "--out", str(out),
              "--chunk", "1"]
    for i in (0, 1):
        monkeypatch.setattr("sys.argv", ["x", "nodes", "--ckpt", str(ck), "--shards", f"{i}/2"] + common)
        rtt.main()
    # u1's second previous step is the only one encoded: its first is item u2
    assert len(list((out / "rt_traj_nodes" / "val_5k").glob("part_*.npy"))) == 1
    monkeypatch.setattr("sys.argv", ["x", "assemble"] + common)
    rtt.main()

    traj = ShardedRepSplit(out / "rt_traj" / "val_5k")
    assert [m["uid"] for m in traj.meta()] == [f"u{k}" for k in order]
    assert list(traj.y) == [k % 2 for k in order]

    def read(p, pre, step):
        ids = tokenize(CharTok(), {"problem": p, "prefix": pre, "step": step})
        with torch.no_grad():
            return reduce_states(rt, read_all_states(model, rt, [ids], 0, torch.device("cpu"),
                                                     grad=False))[0].numpy()

    want = {"u0": [read(*ROWS[0])], "u1": [read(*ROWS[1]), read("p1", "first", "second"),
                                           read("p1", "", "first")],
            "u2": [read(*ROWS[2])]}
    for g, m in enumerate(traj.meta()):
        item = traj.item(g)
        rows = [want[m["uid"]][0]] + want[m["uid"]]
        assert item.shape == (len(rows), D) and m["n_prev_steps"] == len(rows) - 2
        assert m["step_start_idx"] == 1 and m["n_tokens"] == len(rows)
        for r, w in zip(item, rows):
            assert np.allclose(r, w, atol=2e-2)
