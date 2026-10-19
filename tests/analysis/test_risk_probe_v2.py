"""risk_probe_v2: soft targets, splits, label jsonl, the d512-protocol trainer and the
ProcessBench combination metrics (docs/risk_probe_v2_plan.md)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.analysis.risk_probe_v2 import (calib20_weighted, is_val, parse_header,  # noqa: E402
                                        soft_harmful)


# ----------------------------------------------------------------- targets, splits

def test_soft_harmful_limits_and_monte_carlo():
    assert soft_harmful(5, 16, 5, 16) == pytest.approx(0.5, abs=1e-6)
    assert soft_harmful(0, 16, 16, 16) > 0.999
    assert soft_harmful(16, 16, 0, 16) < 0.001
    rng = np.random.default_rng(0)
    a = rng.beta(1 + 9, 1 + 7, 400_000); b = rng.beta(1 + 5, 1 + 11, 400_000)
    assert soft_harmful(5, 16, 9, 16) == pytest.approx(float((a > b).mean()), abs=0.005)
    # vectorized
    q = soft_harmful(np.array([5, 0]), np.array([16, 16]), np.array([5, 16]), np.array([16, 16]))
    assert q.shape == (2,) and q[0] == pytest.approx(0.5, abs=1e-6)


def test_is_val_deterministic_and_about_ten_percent():
    keys = [f"pk_{i}" for i in range(5000)]
    v = [is_val(k) for k in keys]
    assert v == [is_val(k) for k in keys]
    assert 0.08 < np.mean(v) < 0.12


def test_parse_header():
    assert parse_header("Problem:\nWhat is 1+1?\n\nSolution:\n") == "What is 1+1?"
    with pytest.raises(ValueError):
        parse_header("Question: x")


# ----------------------------------------------------------------- label jsonl

def _views(tmp_path):
    v = tmp_path / "views"; (v / "inputs").mkdir(parents=True)
    hdr = "Problem:\nFind x.\n\nSolution:\n"
    inputs = [{"trace_id": "rp_a#0", "header": hdr, "steps": ["s0"]},
              {"trace_id": "rp_a#2", "header": hdr, "steps": ["s0", "s1", "s2"]},
              {"trace_id": "rp_other#0", "header": hdr, "steps": ["t0"]}]       # a validity-only view
    (v / "inputs" / "xf3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inputs))
    lab = [{"view_id": "rp_a#0", "trace_id": "rp_a", "step": 0, "claim_idx": 0, "problem_key": "pk_a", "fold": 3},
           {"view_id": "rp_a#2", "trace_id": "rp_a", "step": 1, "claim_idx": 2, "problem_key": "pk_a", "fold": 3}]
    (v / "lab_views.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lab))
    sv = [{"trace_id": "rp_a", "step": 0, "view_id": "rp_a#0", "err_before": 0.25, "err_after": 0.25, "k_before": 16,
           "k_after": 16, "risk": 0.0, "se": 0.15, "harmful": False, "y_ds": 0, "mask_ds": True, "phase": "pre_error"},
          {"trace_id": "rp_a", "step": 1, "view_id": "rp_a#2", "err_before": 0.25, "err_after": 1.0, "k_before": 16,
           "k_after": 16, "risk": 0.75, "se": 0.11, "harmful": True, "y_ds": 1, "mask_ds": True, "phase": "first_error"}]
    (tmp_path / "sv.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sv))
    return v


def test_build_label_jsonl(tmp_path):
    v = _views(tmp_path)
    out = tmp_path / "label.jsonl"
    subprocess.run([sys.executable, str(ROOT / "scripts/risk_probe_v2/build_label_jsonl.py"), "--views", str(v),
                    "--step_values", str(tmp_path / "sv.jsonl"), "--out", str(out)], check=True, cwd=ROOT)
    rows = [json.loads(l) for l in open(out)]
    assert [r["uid"] for r in rows] == ["rp_a#0", "rp_a#2"]
    a, b = rows
    assert a["problem"] == "Find x." and a["prefix"] == "" and a["candidate_step"] == "s0"
    assert b["prefix"] == "s0\n\ns1" and b["candidate_step"] == "s2" and b["label"] == 1 and b["step_idx"] == 1
    assert b["problem_id"] == "pk_a" and b["fold"] == 3 and b["y_ds"] == 1
    assert b["soft"] > 0.99 and a["soft"] == pytest.approx(0.5, abs=1e-6)
    audit = json.loads((tmp_path / "label.audit.json").read_text())
    assert audit["n_rows"] == 2 and audit["n_harmful"] == 1


def test_build_label_jsonl_fails_on_missing_view(tmp_path):
    v = _views(tmp_path)
    (v / "lab_views.jsonl").write_text((v / "lab_views.jsonl").read_text()
                                       + json.dumps({"view_id": "rp_zz#0", "trace_id": "rp_zz", "step": 0,
                                                     "problem_key": "pk_z", "fold": 1}) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/risk_probe_v2/build_label_jsonl.py"), "--views", str(v),
                        "--step_values", str(tmp_path / "sv.jsonl"), "--out", str(tmp_path / "x.jsonl")],
                       cwd=ROOT, capture_output=True)
    assert r.returncode != 0


# ----------------------------------------------------------------- trainer

D = 8


def _write_span_split(path, items, metas, labels):
    from src.repstore import TOKEN_SEQ, RepSpec, write_split
    write_split(path / "shard_00", items, labels, metas,
                RepSpec(name="step_spans", kind=TOKEN_SEQ, dim=D, layer=35, backbone="t",
                        readout="step_span_with_boundary"))


def _synthetic(tmp_path, n_traces=160, seed=0):
    """Label store + label jsonl (feature 0 of the step tokens carries the harmful
    signal) and a ProcessBench store with two subsets."""
    from src.analysis.risk_probe_v1 import fold_of
    rng = np.random.default_rng(seed)
    items, metas, labels, rows = [], [], [], []
    gi = 0
    for t in range(n_traces):
        pk = f"pk_{t // 2}"
        for s in range(3):
            h = int(rng.random() < 0.2)
            span = int(rng.integers(2, 9))
            x = rng.normal(size=(span + 1, D)).astype(np.float32)
            x[1:, 0] += 2.0 * h
            uid = f"rp_{t}#{s}"
            items.append(x); labels.append(h)
            metas.append({"uid": uid, "problem_id": pk, "step_idx": s, "label": h, "n_tokens": span + 1,
                          "step_start_idx": 1, "pre_step_boundary_idx": 0, "global_index": gi})
            rows.append({"uid": uid, "problem_id": pk, "solution_id": f"rp_{t}", "step_idx": s, "label": h,
                         "soft": 0.9 if h else 0.2, "fold": fold_of(pk), "y_ds": h, "mask_ds": True,
                         "risk": 0.3 * h, "phase": "first_error" if h else "pre_error"})
            gi += 1
    store = tmp_path / "store"
    _write_span_split(store / "label", items, metas, labels)
    (tmp_path / "label.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pb = tmp_path / "pb"
    for sub in ("gsm8k", "math"):
        items, metas, labels = [], [], []
        gi = 0
        for k in range(30):
            fe = int(rng.integers(0, 3)) if k % 2 else -1
            for s in range(3):
                x = rng.normal(size=(4, D)).astype(np.float32)
                x[1:, 0] += 2.0 * (s == fe)
                items.append(x); labels.append(int(s == fe))
                metas.append({"id": f"{sub}-{k}", "step_idx": s, "label": fe, "n_steps": 3, "n_tokens": 4,
                              "step_start_idx": 1, "pre_step_boundary_idx": 0, "global_index": gi})
                gi += 1
        _write_span_split(pb / sub, items, metas, labels)
    return store, pb


def _train(tmp_path, store, pb, fold, target="soft", extra=()):
    out = tmp_path / f"cell_{target}_{fold}"
    cmd = [sys.executable, str(ROOT / "scripts/risk_probe_v2/train_risk_cell.py"), "--store", str(store),
           "--labels", str(tmp_path / "label.jsonl"), "--target", target, "--fold", str(fold),
           "--pb_store", str(pb), "--pb_subsets", "gsm8k", "math", "--out_dir", str(out),
           "--learner", "transformer:d16,l1,f32,h2", "--epochs", "4", "--batch_size", "32",
           "--lr_grid", "1e-3", "--wd_grid", "0.0", *extra]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    return out


def test_train_risk_cell_fold_and_full(tmp_path):
    store, pb = _synthetic(tmp_path)
    labels = {json.loads(l)["uid"]: json.loads(l) for l in open(tmp_path / "label.jsonl")}
    out0 = _train(tmp_path, store, pb, 0)
    res = json.loads((out0 / "results.json").read_text())
    oof = [json.loads(l) for l in open(out0 / "oof.jsonl")]
    want = {u for u, r in labels.items() if r["fold"] == 0}
    assert {r["uid"] for r in oof} == want                     # exactly the held-out fold
    assert not (set(res["split_problems"]["train"]) & set(res["split_problems"]["test"]))
    assert not (set(res["split_problems"]["val"]) & set(res["split_problems"]["test"]))
    y = np.array([labels[r["uid"]]["label"] for r in oof]); p = np.array([r["score"] for r in oof])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p) > 0.8                            # the planted signal is learned
    assert res["hp"]["selected"] == {"lr": 1e-3, "weight_decay": 0.0}
    outf = _train(tmp_path, store, pb, "full", extra=("--hp_from", str(out0 / "results.json"), "--seed", "43"))
    resf = json.loads((outf / "results.json").read_text())
    assert resf["hp"]["reused_from"] and resf["seed"] == 43
    rows = [json.loads(l) for l in open(outf / "pb_step_scores_math.jsonl")]
    assert len(rows) == 30 and all(len(r["scores"]) == 3 for r in rows)   # leaderboard per-trace format
    assert not (outf / "oof.jsonl").exists()


def test_train_risk_cell_hard_target_uses_binary_labels(tmp_path):
    store, pb = _synthetic(tmp_path, n_traces=80)
    out = _train(tmp_path, store, pb, 1, target="harmful")
    res = json.loads((out / "results.json").read_text())
    assert res["target"] == "harmful" and res["n_train"] > 0


# ----------------------------------------------------------------- PB combination metric

def test_calib20_weighted_reduces_to_calib20_at_zero_weight():
    from merge_rep_grid_leaderboard import calib20_subset
    rng = np.random.default_rng(1)
    tv, tr = [], []
    for k in range(120):
        n = int(rng.integers(2, 8)); fe = int(rng.integers(0, n)) if k % 3 else -1
        v = rng.normal(size=n) + 2.0 * (np.arange(n) == fe)
        tv.append((fe, v.tolist())); tr.append((fe, rng.normal(size=n).tolist()))
    assert calib20_weighted(tv, tr, [0.0]) == pytest.approx(calib20_subset(tv), abs=1e-12)


def test_calib20_weighted_uses_an_informative_second_score():
    rng = np.random.default_rng(2)
    tv, tr = [], []
    for k in range(150):
        n = int(rng.integers(2, 8)); fe = int(rng.integers(0, n)) if k % 2 else -1
        hit = (np.arange(n) == fe).astype(float)
        tv.append((fe, (rng.normal(size=n) + 1.0 * hit).tolist()))
        tr.append((fe, (rng.normal(size=n) + 1.0 * hit).tolist()))
    only_v = calib20_weighted(tv, tr, [0.0])
    both = calib20_weighted(tv, tr, [0.0, 0.5, 1.0, 2.0])
    assert both > only_v + 0.03


def test_score_cells_reproduces_a_cells_saved_pb_scores(tmp_path):
    store, pb = _synthetic(tmp_path)
    cell = _train(tmp_path, store, pb, "full")
    res = json.loads((cell / "results.json").read_text())
    # leaderboard cells record dim and protocol; mimic that layout
    res["dim"] = D
    (cell / "results.json").write_text(json.dumps(res))
    for sub in ("gsm8k", "math"):                       # the leaderboard saves probabilities
        rows = [json.loads(l) for l in open(cell / f"pb_step_scores_{sub}.jsonl")]
        for r in rows:
            r["scores"] = [float(1 / (1 + np.exp(-v))) for v in r["scores"]]
        (cell / f"pb_step_scores_{sub}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "scored"
    subprocess.run([sys.executable, str(ROOT / "scripts/risk_probe_v2/score_cells.py"), "--cells", f"v42={cell}",
                    "--label_store", str(store), "--pb_store", str(pb), "--pb_subsets", "gsm8k", "math",
                    "--check_saved", "--out_dir", str(out)], check=True, cwd=ROOT, capture_output=True)
    gate = json.loads((out / "v42__gate.json").read_text())
    assert gate["pass"] and gate["n"] == 2 * 30 * 3
    assert len((out / "v42__label.jsonl").read_text().splitlines()) == 160 * 3


def test_analyze_end_to_end(tmp_path):
    """Folds and full fits on the synthetic set, synthetic validity scores, analyze.py."""
    store, pb = _synthetic(tmp_path, n_traces=120)
    labels = [json.loads(l) for l in open(tmp_path / "label.jsonl")]
    fits = tmp_path / "fits"
    for t in ("soft", "harmful"):
        for k in range(5):
            o = _train(tmp_path, store, pb, k, target=t)
            o.rename(fits / f"f{k}_{t}") if fits.exists() else (fits.mkdir(), o.rename(fits / f"f{k}_{t}"))
        for s in (42, 43):
            o = _train(tmp_path, store, pb, "full", target=t, extra=("--seed", str(s)))
            o.rename(fits / f"full_{t}_s{s}")
    val = tmp_path / "validity"; val.mkdir()
    rng = np.random.default_rng(4)
    for s in (42, 43):
        (val / f"v{s}__label.jsonl").write_text("".join(
            json.dumps({"uid": r["uid"], "logit": float(rng.normal() + 2 * r["y_ds"])}) + "\n" for r in labels))
        for sub in ("gsm8k", "math"):
            rows = [json.loads(l) for l in open(fits / f"full_soft_s{s}" / f"pb_step_scores_{sub}.jsonl")]
            with open(val / f"v{s}__pb_{sub}.jsonl", "w") as f:
                for r in rows:
                    for k in range(r["n_steps"]):
                        f.write(json.dumps({"id": r["id"], "step_idx": k,
                                            "logit": float(rng.normal() + 1.5 * (k == r["label"]))}) + "\n")
        (val / f"v{s}__gate.json").write_text(json.dumps({"pass": True}))
    out = tmp_path / "analysis"
    subprocess.run([sys.executable, str(ROOT / "scripts/risk_probe_v2/analyze.py"), "--labels", str(tmp_path / "label.jsonl"),
                    "--fits", str(fits), "--validity", str(val), "--pb_subsets", "gsm8k", "math", "--seeds", "42", "43",
                    "--n_boot", "8", "--workers", "2", "--out_dir", str(out)], check=True, cwd=ROOT)
    res = json.loads((out / "summary.json").read_text())
    assert res["gates"]["oof_complete_soft"] and res["gates"]["oof_complete_harmful"]
    assert res["e2"]["soft"]["all"]["auroc"] > 0.8
    pbm = res["pb_seed_mean"]
    assert pbm["risk_soft"]["auroc_known"] > 0.8 and pbm["validity"]["auroc_known"] > 0.6
    # the risk score carries the planted first-error signal, so adding it helps here
    assert res["delta_f1_pb_calib20"]["soft"]["delta"] > 0
    assert set(pbm) >= {"validity", "risk_soft", "risk_harmful", "combo_soft", "combo_harmful", "combo_w_soft"}
