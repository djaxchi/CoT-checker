"""step_probe: the STEP baseline (2-layer MLP on the last-layer step-end state, weighted BCE, early stop)."""

import json
import sys

import numpy as np
import pytest
import torch

from scripts.instep_v1 import step_probe as SP

D = 8


def write_set(states, name, uids, y, x, kept=None):
    kept = np.ones(len(y), bool) if kept is None else kept
    np.savez(states / f"{name}.shard0.npz", passive_final=x.astype(np.float32), y=y, kept=kept)
    with open(states / f"{name}.shard0.meta.jsonl", "w") as f:
        for u in uids:
            f.write(json.dumps({"uid": u}) + "\n")


def make(n, rng, pos=0.1):
    y = (rng.random(n) < pos).astype(int)
    x = rng.normal(size=(n, D))
    x[:, 0] += 3.0 * y
    x[:, 1] = np.abs(x[:, 1]) * (1 - 2 * y)  # a non-linear (sign) cue the MLP can use too
    return y, x


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(0)
    states, train, data = tmp_path / "states", tmp_path / "train", tmp_path / "data"
    for d in (states, train, data / "rp", data / "rt"):
        d.mkdir(parents=True)
    for p in ("train_p0", "train_p1"):
        y, x = make(1500, rng)
        write_set(train, p, [f"{p}_{i}::0" for i in range(1500)], y, x)
    yv, xv = make(400, rng)
    vu = [f"rp{i // 2:05d}::{i % 2}" for i in range(400)]
    write_set(states, "val", vu, np.zeros(400, int), xv)
    with open(data / "rt" / "all_val.items.jsonl", "w") as f:
        for u in vu:
            f.write(json.dumps({"uid": u, "label": 0}) + "\n")
    with open(data / "rp" / "val.traces.jsonl", "w") as f:
        for t in range(200):
            f.write(json.dumps({"trace_id": f"rp{t:05d}", "labels": {"ds": [int(yv[2 * t]), int(yv[2 * t + 1])]}}) + "\n")
    ym, xm = make(300, rng)
    kept = np.ones(300, bool)
    kept[3] = False
    write_set(states, "math", [f"m{i}::0" for i in range(300)], ym, xm, kept)
    write_set(states, "train40k", [f"x{i}::0" for i in range(10)], np.zeros(10, int), rng.normal(size=(10, D)))
    return tmp_path, states, train, data


def run(monkeypatch, tmp_path, states, train, data, *extra):
    argv = ["step_probe.py", "--states", str(states), "--train_states", str(train),
            "--train_names", "train_p0,train_p1", "--data", str(data), "--val_label", "ds",
            "--out", str(tmp_path / "out"), "--seeds", "0,1", "--epochs", "6", "--patience", "2", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    SP.main()
    return json.loads((tmp_path / "out" / "summary.json").read_text())


def test_mlp_shape_and_step_hyperparameters():
    m = SP.StepScorer(4096)
    assert [tuple(p.shape) for p in m.parameters()] == [(512, 4096), (512,), (1, 512), (1,)]
    assert SP.n_params(4096) == 4096 * 512 + 512 + 512 + 1
    assert SP.DEFAULTS == {"hidden": 512, "lr": 1e-4, "wd": 1e-5, "batch": 128, "epochs": 20, "patience": 5}


def test_pos_weight_is_negatives_over_positives():
    y = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert float(SP.pos_weight(y)) == 3.0


def test_trains_per_seed_scores_every_non_train_set(setup, monkeypatch):
    tmp_path, states, train, data = setup
    s = run(monkeypatch, tmp_path, states, train, data)
    assert s["n_train"] == 3000 and s["n_val"] == 400 and s["seeds"] == [0, 1]
    for seed in ("0", "1"):
        r = s["per_seed"][seed]
        assert 1 <= r["best_epoch"] <= 6 and r["test"]["math"]["auroc"] > 0.9
        names = sorted(p.name for p in (tmp_path / "out" / f"seed{seed}").iterdir())
        assert names == ["math.jsonl", "val.jsonl"]  # train40k and the training parts are not scored
        rows = [json.loads(l) for l in open(tmp_path / "out" / f"seed{seed}" / "math.jsonl")]
        assert len(rows) == 299 and all(0 <= r["p"] <= 1 for r in rows)
    # the reported number per set is the mean of the per-seed APs
    ap = [s["per_seed"][k]["test"]["math"]["ap"] for k in ("0", "1")]
    assert np.isclose(s["test_seed_mean"]["math"], np.mean(ap))
