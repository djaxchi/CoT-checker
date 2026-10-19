"""risk_probe_v1: folds, view rows, solvers against sklearn, null permutation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis.risk_probe_v1 import (RidgeSolver, fold_of, group_permutation, logistic_fit, residualize,
                                        step_rows)


def test_fold_of_is_stable_and_spread():
    assert fold_of("pk_abc") == fold_of("pk_abc")
    f = [fold_of(f"pk_{i}") for i in range(2000)]
    assert set(f) == {0, 1, 2, 3, 4} and min(np.bincount(f)) > 300
    assert [fold_of(f"pk_{i}", salt="inner:", n=3) for i in range(5)] != [fold_of(f"pk_{i}", n=3) for i in range(5)]


def test_step_rows_before_after_mean():
    H = np.arange(20, dtype=np.float32).reshape(10, 2)     # rows 0..9
    b, a, m = step_rows(H, 4, 7)                            # step rows 4, 5, 6
    assert b.tolist() == H[3].tolist() and a.tolist() == H[6].tolist()
    assert m.tolist() == H[4:7].mean(0).tolist()
    with pytest.raises(ValueError):
        step_rows(H, 0, 3)


def test_step_rows_match_compact_layout():
    """The stored row ahead of the step is the separator token before it (build_context_views)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from build_context_views import compact
    # forward ids: header 0..4, step0 5..7, sep 8, step1 9..11, sep 12, step2 13..15
    ss, se = [5, 9, 13], [8, 12, 16]
    keep, n, st_s, st_e = compact(ss, se)
    fwd = np.arange(16)
    stored = np.concatenate([fwd[a:b] for a, b in keep])
    assert len(stored) == n
    assert stored[st_s[0] - 1] == 12 and stored[st_e[0] - 1] == 15     # separator, last step token
    assert stored[st_s[0]:st_e[0]].tolist() == [13, 14, 15]
    keep, n, st_s, st_e = compact([5], [8])                               # step 0: last header token
    stored = np.concatenate([fwd[a:b] for a, b in keep])
    assert stored[st_s[0] - 1] == 4


def data(n=400, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, d) + rng.normal(size=d)
    w = rng.normal(size=d)
    return X, w, rng


def test_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    X, w, rng = data()
    y = X @ w + rng.normal(size=len(X))
    rs = RidgeSolver(torch.tensor(X))
    Zs = StandardScaler().fit_transform(X)
    for alpha in (1.0, 100.0):
        ref = Ridge(alpha=alpha).fit(Zs, y)
        m = rs.fit(y, alpha)
        assert np.allclose(m.w.numpy(), ref.coef_, atol=1e-8)
        assert np.allclose(m.predict(X), ref.predict(Zs), atol=1e-8)


def test_logistic_counts_match_sklearn_expanded_rows():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, w, rng = data(300, 8, 1)
    p = 1 / (1 + np.exp(-(X - X.mean(0)) @ w / 4))
    n = np.full(len(X), 16.0)
    k = rng.binomial(16, p).astype(float)
    m = logistic_fit(torch.tensor(X), k, n, C=0.5)
    Zs = StandardScaler().fit_transform(X)
    Xe = np.vstack([Zs, Zs]); ye = np.r_[np.ones(len(X)), np.zeros(len(X))]; sw = np.r_[k, n - k]
    ref = LogisticRegression(C=0.5, max_iter=10000, tol=1e-10).fit(Xe, ye, sample_weight=sw)
    assert np.allclose(m.w.numpy(), ref.coef_[0], atol=1e-4)
    assert np.allclose(m.predict(X), ref.predict_proba(Zs)[:, 1], atol=1e-5)


def test_group_permutation_stays_in_groups():
    rng = np.random.default_rng(0)
    keys = [(i % 3, i % 2) for i in range(60)]
    p = group_permutation(keys, rng)
    assert sorted(p.tolist()) == list(range(60))
    assert all(keys[i] == keys[j] for i, j in enumerate(p))
    assert (p != np.arange(60)).any()


def test_residualize_removes_covariates():
    rng = np.random.default_rng(0)
    c = rng.normal(size=200)
    r = residualize(3 * c + 1 + rng.normal(size=200) * 0.1, [c])
    assert abs(np.corrcoef(r, c)[0, 1]) < 1e-8 and abs(r.mean()) < 1e-8


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_build_views_copies_builds_and_checks_text():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "risk_probe_v1"))
    from build_context_views import Tok
    from build_views import build
    tok = CharTokenizer()
    tk = Tok(tok)
    steps = ["a = 1", "b = 2", "c = 3"]
    lines = [f"- Step {j + 1}: {s}\n" for j, s in enumerate(steps)]
    reply = [ord(ch) for ch in "".join(lines)]
    spans, lo = [], 0
    for l in lines:
        spans.append([lo, lo + len(l)]); lo += len(l)
    trace = {"trace_id": "rp_1", "problem_key": "pk_1", "step_claim_idx": [0, 1, 2], "step_spans": spans,
             "reply_ids": reply, "y_ds": [0, 1, 1], "mask_ds": [True, True, False]}
    manifest = {"rp_1": {"trace_id": "rp_1", "problem": "P?", "steps": steps}}
    v0_meta = {"trace_id": "rp_1#0", "split": "v2_train_xfit", "problem_key": "pk_1", "y": [0], "label_mask": [True]}
    other = {"trace_id": "rp_9#0", "split": "v2_train_xfit", "problem_key": "pk_9", "y": [1], "label_mask": [True]}
    v0 = {"v2_train_xfit": {"meta": [v0_meta, other], "inputs": [{"trace_id": "rp_1#0"}, {"trace_id": "rp_9#0"}],
                            "enc": [{"trace_id": "rp_1#0", "n_forward": 5}, {"trace_id": "rp_9#0", "n_forward": 5}]}}
    for sp in ("v2_dev_xfit", "fsl_forks", "gate_rp_test"):
        v0[sp] = {"meta": [{"trace_id": f"{sp}#0", "problem_key": f"pk_{sp}"}], "inputs": [{"trace_id": f"{sp}#0"}],
                  "enc": [{"trace_id": f"{sp}#0", "n_forward": 3}]}
    out_in, out_meta, out_enc, lab, audit = build(v0, [trace], manifest, tk, tok.decode, {"pk_fsl_forks"})
    assert audit["counts"] == {"label_steps": 3, "copied_from_v0": 1, "built": 2}
    assert [r["view_id"] for r in lab] == ["rp_1#0", "rp_1#1", "rp_1#2"] and [r["source"] for r in lab] == ["v0", "built", "built"]
    f = lab[0]["fold"]
    metas = {m["trace_id"]: m for m in out_meta[f"xf{f}"]}
    assert metas["rp_1#2"]["y"] == [1] and metas["rp_1#2"]["label_mask"] == [False]
    enc = {e["trace_id"]: e for e in out_enc}
    # the built view of step 2 is header + steps 0..2; its stored step is the last step's text
    ids = [ord(ch) for ch in "Problem:\nP?\n\nSolution:\n" + "\n\n".join(steps)]
    e = enc["rp_1#2"]
    assert e["n_forward"] == len(ids) and e["fwd_step_starts"][-1] == len(ids) - len(steps[2])
    assert sum(b - a for a, b in e["keep"]) == e["n_tokens"]
    assert {e["split"] for e in out_enc} >= {"v2_dev_xfit", "fsl_forks", "gate_rp_test"}
    # a text mismatch drops the step
    manifest["rp_1"]["steps"] = ["a = 1", "b = 22", "c = 3"]
    _, _, _, lab2, audit2 = build(v0, [trace], manifest, tk, tok.decode, set())
    assert audit2["counts"]["drop_text_mismatch"] == 1 and len(lab2) == 2


def test_dump_rows_from_store(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sd = tmp_path / "store" / "shard_000"
    sd.mkdir(parents=True)
    # two views: lab "rp_1#1" (6 rows, step rows 3..5) and fork "rp_f#2" (4 rows, step rows 1..3)
    H = np.arange(10 * 4, dtype=np.float16).reshape(10, 4)
    np.save(sd / "h.npy", H)
    np.save(sd / "lengths.npy", np.array([6, 4]))
    metas = [{"trace_id": "rp_1#1", "split": "xf2", "step_starts": [3], "step_ends": [6]},
             {"trace_id": "rp_f#2", "split": "fsl_forks", "step_starts": [1], "step_ends": [4]}]
    (sd / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas))
    (sd / "step_labels.jsonl").write_text("".join(json.dumps({"trace_id": m["trace_id"], "y": [0], "label_mask": [True]}) + "\n" for m in metas))
    (sd / "DONE").write_text("")
    lab = tmp_path / "lab_views.jsonl"
    lab.write_text(json.dumps({"view_id": "rp_1#1"}) + "\n")
    out = tmp_path / "rows.npz"
    subprocess.run([sys.executable, str(root / "scripts/risk_probe_v1/dump_rows.py"), "--store", str(tmp_path / "store"),
                    "--lab_views", str(lab), "--out", str(out)], check=True, cwd=root)
    z = np.load(out)
    assert z["ids"].tolist() == ["rp_1#1", "rp_f#2"] and z["split"].tolist() == ["xf2", "fsl_forks"]
    assert z["before"][0].tolist() == H[2].tolist() and z["after"][0].tolist() == H[5].tolist()
    assert z["after"][1].tolist() == H[6 + 3].tolist() and z["before"][1].tolist() == H[6].tolist()
    assert np.allclose(z["mean"][0], H[3:6].astype(np.float32).mean(0))


def test_dump_rows_other_splits_without_label_views(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sd = tmp_path / "store" / "shard_000"
    sd.mkdir(parents=True)
    H = np.arange(9 * 4, dtype=np.float16).reshape(9, 4)
    np.save(sd / "h.npy", H)
    np.save(sd / "lengths.npy", np.array([5, 4]))
    metas = [{"trace_id": "pb_math_a#0", "split": "pb_math", "step_starts": [2], "step_ends": [5]},
             {"trace_id": "rp_f#2", "split": "fsl_forks", "step_starts": [1], "step_ends": [4]}]
    (sd / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas))
    (sd / "step_labels.jsonl").write_text("".join(json.dumps({"trace_id": m["trace_id"], "y": [0], "label_mask": [True]}) + "\n" for m in metas))
    (sd / "DONE").write_text("")
    out = tmp_path / "rows.npz"
    subprocess.run([sys.executable, str(root / "scripts/risk_probe_v1/dump_rows.py"), "--store", str(tmp_path / "store"),
                    "--splits", "pb_math", "--out", str(out)], check=True, cwd=root)
    z = np.load(out)
    assert z["ids"].tolist() == ["pb_math_a#0"] and z["split"].tolist() == ["pb_math"]
    assert z["before"][0].tolist() == H[1].tolist() and z["after"][0].tolist() == H[4].tolist()


def synth_label_set(n_traces=120, d=6, seed=0):
    """Traces whose cut error rate is a logistic function of a 1-d feature of the row."""
    import math
    rng = np.random.default_rng(seed)
    traces, step_values, lab, B, A, M, ids = {}, [], [], [], [], [], []
    w = rng.normal(size=d)
    for t in range(n_traces):
        n = 3
        tid, pk = f"rp_{t}", f"pk_{t // 2}"
        H = rng.normal(size=(n + 1, d))                     # one row per cut
        p = 1 / (1 + np.exp(-(H @ w)))
        wrong = rng.binomial(16, p)
        traces[tid] = {"trace_id": tid, "n_steps": n, "reply_ids": list(range(40)), "cut_offsets": [0, 10, 20, 30]}
        for s in range(n):
            vid = f"{tid}#{s}"
            eb, ea = wrong[s] / 16, wrong[s + 1] / 16
            se = math.sqrt(eb * (1 - eb) / 16 + ea * (1 - ea) / 16)
            step_values.append({"trace_id": tid, "step": s, "risk": ea - eb, "se": se, "harmful": bool(se > 0 and ea - eb > 2 * se),
                                "err_before": eb, "err_after": ea, "k_before": 16, "k_after": 16, "relpos": s / 2,
                                "y_ds": int(ea > eb), "mask_ds": True, "y_self": 0})
            from src.analysis.risk_probe_v1 import fold_of
            lab.append({"view_id": vid, "trace_id": tid, "step": s, "problem_key": pk, "fold": fold_of(pk)})
            ids.append(vid); B.append(H[s]); A.append(H[s + 1]); M.append((H[s] + H[s + 1]) / 2)
    for f in range(20):                                         # forks
        H = rng.normal(size=(2, d))
        ids.append(f"fork_{f}"); B.append(H[0]); A.append(H[1]); M.append(H.mean(0))
    rows = {"ids": np.array(ids), "split": np.array(["xf0"] * (len(ids) - 20) + ["fsl_forks"] * 20),
            "before": np.array(B, np.float32), "after": np.array(A, np.float32), "mean": np.array(M, np.float32)}
    return rows, lab, step_values, traces, w


def test_fit_readouts_end_to_end_and_null():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "risk_probe_v1"))
    from fit_readouts import ARMS, Fitter, assemble, permuted_targets, run
    rows, lab, sv, traces, w = synth_label_set()
    D = assemble(rows, lab, sv, traces)
    assert len(D["step"]) == 360 and D["before"].shape == (360, 6)
    assert np.allclose(D["log_n_cont"][:3], np.log1p([30, 20, 10]))
    fk = rows["split"] == "fsl_forks"
    forks = {k: rows[k][fk] for k in ("before", "after", "mean")}
    e2, e1, hp = run(D, forks, Fitter("cpu"))
    assert e2.shape == (360, len(ARMS)) and e1.shape == (20, len(ARMS)) and not np.isnan(e2).any()
    j = ARMS.index
    # the value probe recovers the cut error rate, so its difference tracks risk out of fold
    assert np.corrcoef(e2[:, j("B_after")], D["err_after"])[0, 1] > 0.8
    assert np.corrcoef(e2[:, j("B")], D["risk"])[0, 1] > 0.5
    assert np.allclose(e2[:, j("B")], e2[:, j("B_after")] - e2[:, j("B_before")])
    assert set(hp) == {"fold0", "fold1", "fold2", "fold3", "fold4", "full"}
    Dp = permuted_targets(D, np.random.default_rng(0))
    assert sorted(Dp["risk"].tolist()) == sorted(D["risk"].tolist())
    for f in range(5):                                          # labels stay in their fold
        m = D["fold"] == f
        assert sorted(Dp["risk"][m].tolist()) == sorted(D["risk"][m].tolist())
    n2, n1, _ = run(Dp, forks, Fitter("cpu"), hp)
    assert np.corrcoef(n2[:, j("B")], D["risk"])[0, 1] < 0.3


def test_analyze_end_to_end(tmp_path):
    """fit_readouts (real + null) and analyze.py on a synthetic label set, forks and run dir."""
    import gzip
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    rows, lab, sv, traces, w = synth_label_set(n_traces=150)
    rng = np.random.default_rng(3)
    fk = [str(v) for v in rows["ids"] if str(v).startswith("fork_")]
    # label set files
    for r in sv:
        r["problem_key"] = f"pk_{int(r['trace_id'].split('_')[1]) // 2}"
        r["phase"] = "first_error" if r["step"] == 1 else "pre_error"
    ts = {"traces": [{**t, "step_claim_idx": [0, 1, 2], "finished": True} for t in traces.values()]}
    (tmp_path / "ts.json").write_text(json.dumps(ts))
    (tmp_path / "sv.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sv))
    views = tmp_path / "views"; (views / "meta").mkdir(parents=True)
    (views / "lab_views.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lab))
    (views / "meta" / "fsl_forks.jsonl").write_text("".join(
        json.dumps({"trace_id": v, "fork_id": f"F{v}", "problem_key": f"fpk_{v}"}) + "\n" for v in fk))
    (views / "build_audit.json").write_text(json.dumps({"counts": {"label_steps": len(lab)}, "aligned_share": 1.0}))
    # run dir: rows, cross-fitted validity, v0 rescoring
    run = tmp_path / "run"; (run / "features").mkdir(parents=True); (run / "existing_scores").mkdir()
    np.savez(run / "features" / "rows_L35.npz", **{k: (v.astype(np.float16) if v.dtype.kind == "f" else v)
                                                    for k, v in rows.items()})
    v0 = tmp_path / "v0"; (v0 / "features").mkdir(parents=True)
    fi = [i for i, v in enumerate(rows["ids"]) if str(v).startswith("fork_")]
    np.savez(v0 / "features" / "fsl_forks_L35.npz", ids=np.array(fk), last=rows["after"][fi], mean=rows["mean"][fi])
    for k in range(5):
        d = run / "fits" / f"xfit5_f{k}" / "local"; d.mkdir(parents=True)
        (d.parent / "config.json").write_text(json.dumps({"train_split": ",".join(f"xf{j}" for j in range(5) if j != k),
                                                          "score_splits": [f"xf{k}", "fsl_forks"]}))
        with gzip.open(d / "predictions.jsonl.gz", "wt") as f:
            for r in lab:
                if r["fold"] == k:
                    f.write(json.dumps({"split": f"xf{k}", "trace_id": r["view_id"], "step": 0,
                                        "logit": float(rng.normal())}) + "\n")
    for s in (42, 43, 44):
        sc = [{"split": "fsl_forks", "trace_id": v, "step": 0, "logit": float(rng.normal())} for v in fk]
        (run / "existing_scores" / f"xfit_v2ds_s{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sc))
        d = v0 / "fits" / f"xfit_v2ds_s{s}" / "local"; d.mkdir(parents=True)
        with gzip.open(d / "predictions.jsonl.gz", "wt") as f:
            f.write("".join(json.dumps(r) + "\n" for r in sc))
    fac = [{"fork_id": f"F{v}", "incorrect": int(rng.random() < 0.5), "risk_diff": float(rng.normal() * 0.3),
            "risk_se": 0.1, "err_a": float(rng.random()), "log_n_cont": 4.0, "relpos": float(rng.random())} for v in fk]
    (tmp_path / "fac.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fac))
    fit = tmp_path / "fit"
    py = [sys.executable]
    common = ["--rows", str(run / "features" / "rows_L35.npz"), "--lab_views", str(views / "lab_views.jsonl"),
              "--step_values", str(tmp_path / "sv.jsonl"), "--trace_set", str(tmp_path / "ts.json"),
              "--out_dir", str(fit), "--device", "cpu"]
    subprocess.run(py + [str(root / "scripts/risk_probe_v1/fit_readouts.py")] + common, check=True, cwd=root)
    subprocess.run(py + [str(root / "scripts/risk_probe_v1/fit_readouts.py")] + common +
                   ["--stage", "null", "--perm_start", "0", "--perm_end", "3"], check=True, cwd=root)
    out = tmp_path / "out"
    subprocess.run(py + [str(root / "scripts/risk_probe_v1/analyze.py"), "--fit_dir", str(fit), "--run_dir", str(run),
                         "--v0_run", str(v0), "--views", str(views), "--step_values", str(tmp_path / "sv.jsonl"),
                         "--trace_set", str(tmp_path / "ts.json"), "--fork_factors", str(tmp_path / "fac.jsonl"),
                         "--out_dir", str(out)], check=True, cwd=root)
    S = json.loads((out / "summary.json").read_text())
    assert S["gates"]["reproduction_scores"]["pass"] and S["gates"]["reproduction_features"]["pass"]
    assert S["gates"]["leakage"]["pass"] and S["n_null_perms"] == 3
    assert S["q2_e2"]["all"]["B"]["spearman_risk"] > 0.3          # the synthetic value probe works out of fold
    assert "verdict" in S["reading"] and (out / "summary.md").exists()


def _score_pb():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "scripts" / "risk_probe_v1"))
    import score_pb
    return score_pb


def _pb_meta(sp, tid, ys):
    """Views of one PB trace; ys per step: 0 correct, 1 incorrect, None unknown."""
    return [{"trace_id": f"{tid}#{i}", "split": sp, "orig_trace_id": tid, "orig_step": i, "orig_n_steps": len(ys),
             "y": [0 if y is None else y], "label_mask": [y is not None]} for i, y in enumerate(ys)]


def test_pb_traces_first_error_and_order():
    sp = _score_pb()
    meta = _pb_meta("pb_math", "t1", [0, 0, 1, None]) + _pb_meta("pb_math", "t2", [0, 0])
    tr = sp.pb_traces(list(reversed(meta)))["pb_math"]
    assert tr["t1"]["label"] == 2 and tr["t1"]["views"] == ["t1#0", "t1#1", "t1#2", "t1#3"]
    assert tr["t1"]["y"] == [0, 0, 1, -1] and tr["t2"]["label"] == -1
    with pytest.raises(SystemExit):
        sp.pb_traces(_pb_meta("pb_math", "t3", [0, None]))


def test_pb_metrics_perfect_and_inverted_scores():
    sp = _score_pb()
    meta = []
    for k in range(30):                       # 30 error traces (error at step 1), 30 correct traces
        meta += _pb_meta("pb_math", f"e{k}", [0, 1, None]) + _pb_meta("pb_math", f"c{k}", [0, 0, 0])
    tr = sp.pb_traces(meta)
    good = {m["trace_id"]: float(m["y"][0] == 1) + (0.5 if not m["label_mask"][0] else 0) for m in meta}
    r = sp.pb_metrics(good, tr)["pb_math"]
    assert r["auroc_known"] == 1.0 and r["f1_pb_oracle"] == 1.0 and r["f1_pb_calib20"] == 1.0
    bad = {k: -v for k, v in good.items()}
    assert sp.pb_metrics(bad, tr)["pb_math"]["auroc_known"] == 0.0


def test_score_pb_end_to_end(tmp_path):
    """fit_readouts real on a synthetic label set, then score_pb.py on synthetic PB views."""
    import gzip
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    rows, lab, sv, traces, w = synth_label_set(n_traces=150)
    for r in sv:
        r["view_id"] = f"{r['trace_id']}#{r['step']}"
    (tmp_path / "ts.json").write_text(json.dumps({"traces": list(traces.values())}))
    (tmp_path / "sv.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sv))
    (tmp_path / "lab.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lab))
    np.savez(tmp_path / "rows.npz", **rows)
    ro = tmp_path / "readouts"
    subprocess.run([sys.executable, str(root / "scripts/risk_probe_v1/fit_readouts.py"), "--rows", str(tmp_path / "rows.npz"),
                    "--lab_views", str(tmp_path / "lab.jsonl"), "--step_values", str(tmp_path / "sv.jsonl"),
                    "--trace_set", str(tmp_path / "ts.json"), "--out_dir", str(ro), "--device", "cpu"],
                   check=True, cwd=root, capture_output=True)
    rng = np.random.default_rng(5)
    (tmp_path / "e2.jsonl").write_text("".join(json.dumps({"view_id": r["view_id"], "xfit5": float(rng.normal() + 2 * r["y_ds"])}) + "\n" for r in sv))
    # PB: per subset 25 error traces (first error at step 1 of 3) and 25 correct traces
    meta_dir = tmp_path / "pb_meta"; meta_dir.mkdir()
    pb_ids, B, A, M = [], [], [], []
    scores = tmp_path / "scores"; scores.mkdir()
    saved = tmp_path / "saved"
    logit = {}
    for sp in ("pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath"):
        meta = []
        for k in range(25):
            meta += _pb_meta(sp, f"{sp}_e{k}", [0, 1, None]) + _pb_meta(sp, f"{sp}_c{k}", [0, 0, 0])
        (meta_dir / f"{sp}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
        for m in meta:
            H = rng.normal(size=(2, 6))
            pb_ids.append(m["trace_id"]); B.append(H[0]); A.append(H[1]); M.append(H.mean(0))
            logit[m["trace_id"]] = float(rng.normal() + 3 * m["y"][0])
    np.savez(tmp_path / "pb_rows.npz", ids=np.array(pb_ids), split=np.array(["pb"] * len(pb_ids)),
             before=np.array(B, np.float16), after=np.array(A, np.float16), mean=np.array(M, np.float16))
    for name in [f"xfit5_f{k}" for k in range(5)] + [f"full_v2ds_s{s}" for s in (42, 43, 44)]:
        (scores / f"{name}.jsonl").write_text("".join(json.dumps({"trace_id": v, "step": 0, "logit": logit[v]}) + "\n" for v in pb_ids))
    for s in (42, 43, 44):
        d = saved / f"full_v2ds_s{s}" / "local"; d.mkdir(parents=True)
        with gzip.open(d / "predictions.jsonl.gz", "wt") as f:
            f.write("".join(json.dumps({"trace_id": v, "logit": logit[v]}) + "\n" for v in pb_ids))
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(root / "scripts/risk_probe_v1/score_pb.py"), "--label_rows", str(tmp_path / "rows.npz"),
                    "--lab_views", str(tmp_path / "lab.jsonl"), "--step_values", str(tmp_path / "sv.jsonl"),
                    "--trace_set", str(tmp_path / "ts.json"), "--readouts", str(ro), "--e2_scores", str(tmp_path / "e2.jsonl"),
                    "--pb_rows", str(tmp_path / "pb_rows.npz"), "--pb_meta", str(meta_dir), "--scores_dir", str(scores),
                    "--saved_v2ds", str(saved), "--out_dir", str(out), "--device", "cpu"], check=True, cwd=root)
    res = json.loads((out / "pb_metrics.json").read_text())
    assert res["gates"]["refit_reproduces_e1"]["pass"] and res["gates"]["v2ds_rescoring"]["pass"]
    assert res["metrics"]["validity xfit5"]["mean"]["auroc_known"] > 0.9          # the planted validity signal
    assert 0.3 < res["metrics"]["risk A_last"]["mean"]["auroc_known"] < 0.7       # random PB rows
    assert len((out / "pb_scores.jsonl").read_text().splitlines()) == len(pb_ids)


def test_build_pb_manifest_keeps_only_processbench(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    src = tmp_path / "full"
    for d in ("inputs", "meta"):
        (src / d).mkdir(parents=True)
    rows = []
    for sp, n, shard in (("pb_gsm8k", 2, 0), ("pb_math", 1, 3), ("pb_olympiadbench", 1, 3), ("pb_omnimath", 1, 5), ("v1_train", 3, 1)):
        for i in range(n):
            rows.append({"trace_id": f"{sp}_{i}#0", "split": sp, "shard": shard})
        for d in ("inputs", "meta"):
            (src / d / f"{sp}.jsonl").write_text("".join(json.dumps({"trace_id": f"{sp}_{i}#0"}) + "\n" for i in range(n)))
    (src / "encode_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "pb"
    cmd = [sys.executable, str(root / "scripts/risk_probe_v1/build_pb_manifest.py"), "--src", str(src), "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=root, capture_output=True)
    audit = json.loads((out / "build_audit.json").read_text())
    assert audit["n_views"] == 5 and audit["shards"] == [0, 3, 5]
    assert not (out / "inputs" / "v1_train.jsonl").exists()
    assert all(json.loads(l)["split"].startswith("pb_") for l in open(out / "encode_manifest.jsonl"))
    assert subprocess.run(cmd, cwd=root, capture_output=True).returncode != 0      # never overwrites
