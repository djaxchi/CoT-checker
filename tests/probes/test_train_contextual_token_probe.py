"""Trainer: paired init, resume equivalence, all-step predictions (synthetic store)."""

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
D = 12


def make_store(root: Path, n_train=24, n_dev=8, seed=0):
    rng = np.random.default_rng(seed)
    sd = root / "shard_00"
    sd.mkdir(parents=True)
    hs, lengths, metas, labs = [], [], [], []
    for i in range(n_train + n_dev):
        T = int(rng.integers(2, 5))
        pos, ss, se = 3, [], []
        for j in range(T):
            if j:
                pos += 1
            ss.append(pos); pos += int(rng.integers(1, 4)); se.append(pos)
        H = rng.standard_normal((pos, D)).astype(np.float16)
        y = [int(rng.random() < 0.3) for _ in range(T)]
        mask = [True] * (T - 1) + [False]  # last step unknown
        y[-1] = -1
        hs.append(H); lengths.append(pos)
        metas.append({"trace_id": f"t{i:03d}", "split": "train" if i < n_train else "dev",
                      "n_tokens": pos, "n_steps": T, "step_starts": ss, "step_ends": se})
        labs.append({"trace_id": f"t{i:03d}", "y": y, "label_mask": mask})
    np.save(sd / "h.npy", np.concatenate(hs))
    np.save(sd / "lengths.npy", np.array(lengths, dtype=np.int32))
    np.save(sd / "y.npy", np.zeros(len(lengths), dtype=np.int8))
    (sd / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas))
    (sd / "step_labels.jsonl").write_text("".join(json.dumps(m) + "\n" for m in labs))
    (sd / "DONE").write_text("x")


def run(store, out, *extra):
    cmd = [sys.executable, str(ROOT / "scripts/train_contextual_token_probe.py"), "--store", str(store),
           "--out_dir", str(out), "--arms", "causal", "future1", "--seed", "3", "--lr", "1e-3",
           "--wd", "0.01", "--eff_batch", "8", "--d_model", "16", "--ff", "32", "--heads", "2",
           "--score_splits", "dev", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_resume_matches_straight_run_and_predictions_cover_all_steps(tmp_path):
    store = tmp_path / "store"
    make_store(store)
    a = run(store, tmp_path / "straight", "--max_epochs", "2", "--patience", "9")
    assert a.returncode == 0, a.stderr[-2000:]
    # interrupted after the first effective batch, then resumed twice
    b = run(store, tmp_path / "resumed", "--max_epochs", "2", "--patience", "9", "--max_hours", "1e-9")
    assert b.returncode == 3, b.stderr[-2000:]
    c = run(store, tmp_path / "resumed", "--max_epochs", "1", "--patience", "9")
    assert c.returncode == 0, c.stderr[-2000:]
    for f in (tmp_path / "resumed").glob("*/done.json"):
        f.unlink()  # allow the longer run to rescore
    d = run(store, tmp_path / "resumed", "--max_epochs", "2", "--patience", "9")
    assert d.returncode == 0, d.stderr[-2000:]
    for arm in ("causal", "future1"):
        s1 = torch.load(tmp_path / "straight" / arm / "last.pt", weights_only=False)
        s2 = torch.load(tmp_path / "resumed" / arm / "last.pt", weights_only=False)
        assert s1["state"]["epoch"] == s2["state"]["epoch"] == 2
        for k in s1["model"]:
            assert torch.equal(s1["model"][k], s2["model"][k]), (arm, k)
        assert [h["dev_f1"] for h in s1["state"]["history"]] == [h["dev_f1"] for h in s2["state"]["history"]]
    # paired init across arms
    d1 = json.loads((tmp_path / "straight/causal/done.json").read_text())
    d2 = json.loads((tmp_path / "straight/future1/done.json").read_text())
    assert d1["init_hash"] == d2["init_hash"]
    # every dev step is scored, including unknown-label steps
    rows = [json.loads(l) for l in gzip.open(tmp_path / "straight/future1/predictions.jsonl.gz", "rt")]
    meta = [json.loads(l) for l in open(store / "shard_00/meta.jsonl")]
    n_dev_steps = sum(m["n_steps"] for m in meta if m["split"] == "dev")
    assert len(rows) == d2["n_prediction_rows"] == n_dev_steps
    assert any(not r["label_known"] for r in rows)
    assert all(r["visible_through_step"] == min(r["step"] + 1, r["n_steps"] - 1) for r in rows)


def test_label_override_replaces_only_listed_traces(tmp_path):
    import sys as _s
    _s.path.insert(0, str(ROOT / "scripts"))
    from train_contextual_token_probe import TrajStore
    store = tmp_path / "store"
    make_store(store)
    base = TrajStore(store)
    ov = tmp_path / "ov"
    ov.mkdir()
    tid = "t000"
    T = len(base.items[tid]["y"])
    (ov / "train.jsonl").write_text(json.dumps({"trace_id": tid, "y": [1] * T, "label_mask": [True] * T}) + "\n")
    s2 = TrajStore(store, label_override=ov)
    assert s2.n_overridden == 1
    assert s2.items[tid]["y"] == [1] * T and s2.items[tid]["label_mask"] == [True] * T
    assert s2.items["t001"]["y"] == base.items["t001"]["y"]
