import json
import sys

import numpy as np
import pytest

from scripts.instep_v1 import passive_fit as pf

D = 16


def write_set(states, name, uids, y, xs, kept=None):
    """xs: {key: (n, D)}; npz arrays stored (n, 1, D) fp16 as read_token_inline.py encode does."""
    kept = np.ones(len(y), bool) if kept is None else kept
    np.savez(states / f"{name}.shard0.npz", layers=np.array([-1]), y=np.array(y, np.int8), kept=kept,
             head_p_correct=np.full(len(y), 0.5, np.float32),
             **{k: v[:, None].astype(np.float16) for k, v in xs.items()})
    with open(states / f"{name}.shard0.meta.jsonl", "w") as f:
        for u, l in zip(uids, y):
            f.write(json.dumps({"uid": u, "label": int(l)}) + "\n")


def make(n, rng, pos=0.1):
    y = (rng.random(n) < pos).astype(int)
    sig = rng.normal(size=(n, D))
    sig[:, 0] += 2.5 * y  # block7 and read_block7 carry the signal
    weak = rng.normal(size=(n, D))
    weak[:, 1] += 0.5 * y
    return y, {"passive_block3": rng.normal(size=(n, D)), "passive_block7": sig, "passive_final": weak,
               "read_block3": rng.normal(size=(n, D)), "read_block7": sig + 0.1 * rng.normal(size=(n, D))}


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(0)
    states, data = tmp_path / "states", tmp_path / "data"
    for d in (states, data / "rp", data / "rt"):
        d.mkdir(parents=True)
    # train40k: uids and labels as in the source items file
    y, xs = make(2000, rng)
    tu = [f"tr{i}::0" for i in range(2000)]
    write_set(states, "train40k", tu, y, xs)
    with open(data / "rt" / "ds_train.items.jsonl", "w") as f:
        for u, l in zip(tu, y):
            f.write(json.dumps({"uid": u, "label": int(l)}) + "\n")
    # val: 300 traces of 2 steps; the npz label is the first set's, the traces carry ds (and -1s)
    yv, xv = make(600, rng)
    vu = [f"rp{i // 2:05d}::{i % 2}" for i in range(600)]
    write_set(states, "val", vu, np.zeros(600, int), xv)
    with open(data / "rt" / "all_val.items.jsonl", "w") as f:
        for u in vu:
            f.write(json.dumps({"uid": u, "label": 0}) + "\n")
    with open(data / "rp" / "val.traces.jsonl", "w") as f:
        for t in range(300):
            lab = [int(yv[2 * t]), int(yv[2 * t + 1])]
            if t == 0:
                lab[1] = -1
            f.write(json.dumps({"trace_id": f"rp{t:05d}", "labels": {"ds": lab, "self": [0, 0]}}) + "\n")
    for name in ("math", "pb_gsm8k"):
        ym, xm = make(400, rng)
        kept = np.ones(400, bool)
        kept[5] = False
        write_set(states, name, [f"{name}{i}::{i % 3}" for i in range(400)], ym, xm, kept)
    return tmp_path, states, data


def run(monkeypatch, tmp_path, states, data, *extra):
    argv = ["passive_fit.py", "--states", str(states), "--data", str(data), "--val_label", "ds",
            "--train_items", str(data / "rt" / "ds_train.items.jsonl"), "--out", str(tmp_path / "scores"),
            "--grid", "0.01,1", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    pf.main()


def test_picks_signal_layer_and_writes_rp_scores(setup, monkeypatch):
    tmp_path, states, data = setup
    run(monkeypatch, tmp_path, states, data, "--per_layer_out", str(tmp_path / "pl.json"))
    pl = json.loads((tmp_path / "pl.json").read_text())
    assert pl["picked"]["key"] == "passive_block7"
    by = {r["key"]: r for r in pl["layers"]}
    assert set(by) == {"passive_block3", "passive_block7", "passive_final", "read_block3", "read_block7"}
    assert by["passive_block7"]["val_auroc"] > 0.9 > 0.7 > by["passive_block3"]["val_auroc"]
    assert by["read_block7"]["val_auroc"] > 0.9 and by["read_block7"]["kind"] == "read"
    assert pl["n_val"] == 599  # one -1 step dropped
    assert pl["test"]["math"]["auroc"] > 0.9
    # every set but train40k scored, one line per kept step, uids in meta order
    assert sorted(p.name for p in (tmp_path / "scores").iterdir()) == ["math.jsonl", "pb_gsm8k.jsonl", "val.jsonl"]
    rows = [json.loads(l) for l in open(tmp_path / "scores" / "math.jsonl")]
    meta = [json.loads(l)["uid"] for l in open(states / "math.shard0.meta.jsonl")]
    assert [r["uid"] for r in rows] == meta[:5] + meta[6:]
    assert all(set(r) == {"uid", "p"} and 0 <= r["p"] <= 1 for r in rows)
    # p is P(incorrect): higher on label-1 steps
    ym = {json.loads(l)["uid"]: json.loads(l)["label"] for l in open(states / "math.shard0.meta.jsonl")}
    p1 = np.mean([r["p"] for r in rows if ym[r["uid"]] == 1])
    p0 = np.mean([r["p"] for r in rows if ym[r["uid"]] == 0])
    assert p1 > 3 * p0


def test_without_per_layer_fits_passive_only(setup, monkeypatch):
    tmp_path, states, data = setup
    run(monkeypatch, tmp_path, states, data)
    assert (tmp_path / "scores" / "val.jsonl").exists()
    assert len(open(tmp_path / "scores" / "val.jsonl").readlines()) == 600


def test_rejects_val_uid_mismatch(setup, monkeypatch):
    tmp_path, states, data = setup
    rows = [json.loads(l) for l in open(data / "rt" / "all_val.items.jsonl")]
    rows[3]["uid"] = "other::0"
    with open(data / "rt" / "all_val.items.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with pytest.raises(SystemExit, match="val meta uids"):
        run(monkeypatch, tmp_path, states, data)


def test_rejects_train_label_mismatch(setup, monkeypatch):
    tmp_path, states, data = setup
    rows = [json.loads(l) for l in open(data / "rt" / "ds_train.items.jsonl")]
    rows[0]["label"] = 1 - rows[0]["label"]
    with open(data / "rt" / "ds_train.items.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with pytest.raises(SystemExit, match="labels differ"):
        run(monkeypatch, tmp_path, states, data)


def test_fit_path_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    import torch
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 8))
    y = (X[:, 0] + 0.5 * rng.normal(size=500) > 1).astype(float)
    (C, w, b), = pf.fit_path(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), (0.05,))
    sk = LogisticRegression(C=0.05, max_iter=5000, tol=1e-10).fit(X, y)
    np.testing.assert_allclose(w.numpy(), sk.coef_[0], atol=2e-3)
    np.testing.assert_allclose(b.numpy(), sk.intercept_, atol=2e-3)
