import json
import sys

import numpy as np


def _write(states, name, uids, y, d=8, rng=None):
    rng = rng or np.random.default_rng(0)
    x = rng.normal(size=(len(uids), 1, d)).astype(np.float16)
    x[:, 0, 0] += 2.0 * np.asarray(y)          # a learnable signal
    np.savez(states / f"{name}.shard0.npz", y=np.asarray(y, np.int8), kept=np.ones(len(uids), bool),
             passive_final=x)
    (states / f"{name}.shard0.meta.jsonl").write_text("".join(json.dumps({"uid": u}) + "\n" for u in uids))


def test_step_prm_trains_each_seed_on_its_draw(tmp_path, monkeypatch):
    import scripts.instep_v1.step_prm as sp
    st = tmp_path / "st"
    st.mkdir()
    rng = np.random.default_rng(1)
    uids = [f"u{i}" for i in range(300)]
    y = (rng.random(300) < 0.3).astype(int)
    _write(st, "union_p0", uids[:150], y[:150], rng=rng)
    _write(st, "union_p1", uids[150:], y[150:], rng=rng)
    _write(st, "val5k", [f"v{i}" for i in range(80)], (rng.random(80) < 0.3).astype(int), rng=rng)
    for s in ("gsm8k", "math", "olympiadbench", "omnimath"):
        _write(st, f"pb_{s}", [f"{s}-{i}::0" for i in range(20)], (rng.random(20) < 0.3).astype(int), rng=rng)
    dr = tmp_path / "dr"
    dr.mkdir()
    for s in (0, 1):
        (dr / f"draw_seed{s}.uids.json").write_text(json.dumps(list(rng.choice(uids, 200, replace=False))))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--train_states", str(st), "--train_names", "union_p0,union_p1",
                                      "--draws", str(dr), "--val_states", str(st), "--val_name", "val5k",
                                      "--pb_states", str(st), "--out", str(out), "--seeds", "0,1", "--epochs", "3"])
    sp.main()
    js = json.loads((out / "summary.json").read_text())
    assert set(js["per_seed"]) == {"0", "1"} and all(v["n_train"] == 200 for v in js["per_seed"].values())
    rows = [json.loads(l) for l in open(out / "seed1/pb_math.jsonl")]
    assert len(rows) == 20 and all(0 <= r["p"] <= 1 for r in rows)
