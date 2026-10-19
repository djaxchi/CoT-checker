#!/usr/bin/env python3
"""risk_probe_v1 readouts on ProcessBench (post hoc, not part of the plan).

Fits every readout arm once on all label steps at the real run's `full`
hyperparameters (hparams.json), which is the E1 fit, and applies it to the L35
rows of ProcessBench's `full` step views (dump_rows.py --splits pb_*). Validity
scores for the same views come from score_with_checkpoints.py: the five xfit5
fold models (mean logit) and §21.18's full_v2ds fits (mean logit, reference).

Scores compared, all oriented so that higher means "flag this step":
  * each risk readout alone (A_last, A_mean, A_last_h, A_mean_h, B);
  * each validity probe alone;
  * validity + readout: a logistic regression of DeepSeek invalidity on
    [xfit5, readout], fit on the label set's out-of-fold scores (preds_e2.npz and
    step_scores.jsonl). No ProcessBench label is used to fit it.

Metrics per subset: AUROC over steps with a known label (steps up to and
including the first error), F1_PB at calib-20 (scripts/merge_rep_grid_leaderboard
.calib20_subset) and the oracle F1_PB; then the mean over the four subsets.

Gates: the refit reproduces the saved E1 fork predictions, and the rescored
full_v2ds logits match context_ablation_v3's saved ProcessBench predictions
(median |dP| <= 0.002, max <= 0.02).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "risk_probe_v1"))
from fit_readouts import ARMS, Fitter, assemble, jl  # noqa: E402
from scripts.merge_rep_grid_leaderboard import calib20_subset  # noqa: E402
from src.eval.contextual_probe_metrics import auroc, best_pb_threshold  # noqa: E402

PB = ("pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath")
READOUTS = ("A_last", "A_mean", "A_last_h", "A_mean_h", "B")


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


def pb_traces(meta: list[dict]) -> dict:
    """Per subset, per original trace: ordered view ids, step labels and the
    first-error label (-1 when every step is known and correct)."""
    by = defaultdict(dict)
    for m in meta:
        by[m["split"]].setdefault(m["orig_trace_id"], []).append(m)
    out = {}
    for sp, trs in by.items():
        out[sp] = {}
        for t, ms in trs.items():
            ms = sorted(ms, key=lambda m: m["orig_step"])
            if [m["orig_step"] for m in ms] != list(range(ms[0]["orig_n_steps"])):
                raise SystemExit(f"[pb] {t}: steps {[m['orig_step'] for m in ms]} of {ms[0]['orig_n_steps']}")
            y = [int(m["y"][0]) if m["label_mask"][0] else -1 for m in ms]
            fe = next((i for i, v in enumerate(y) if v == 1), -1)
            if fe == -1 and min(y) < 0:
                raise SystemExit(f"[pb] {t}: unknown labels without an error")
            out[sp][t] = {"views": [m["trace_id"] for m in ms], "y": y, "label": fe}
    return out


def pb_metrics(score: dict, traces: dict) -> dict:
    """score: view id -> value. Per-subset and mean metrics."""
    res = {}
    for sp, trs in traces.items():
        seqs = [(t["label"], [score[v] for v in t["views"]]) for t in trs.values()]
        s, y = [], []
        for t in trs.values():
            for v, yy in zip(t["views"], t["y"]):
                if yy >= 0:
                    s.append(score[v]); y.append(yy)
        _, oracle = best_pb_threshold([x for _, x in seqs], [lab for lab, _ in seqs])
        res[sp] = {"auroc_known": float(auroc(np.array(s), np.array(y))), "f1_pb_calib20": calib20_subset(seqs),
                   "f1_pb_oracle": float(oracle), "n_traces": len(seqs), "n_error": int(sum(lab >= 0 for lab, _ in seqs))}
    res["mean"] = {k: float(np.mean([res[sp][k] for sp in traces])) for k in ("auroc_known", "f1_pb_calib20", "f1_pb_oracle")}
    return res


def load_logits(path: Path) -> dict:
    return {r["trace_id"]: float(r["logit"]) for r in jl(path)}


def combiner(x_val: np.ndarray, x_read: np.ndarray, y: np.ndarray):
    """Logistic regression of invalidity on standardized [validity, readout]."""
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([x_val, x_read])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    m = LogisticRegression(C=1.0, max_iter=1000).fit((X - mu) / sd, y)
    return lambda a, b: m.decision_function((np.column_stack([a, b]) - mu) / sd), m.coef_[0].tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label_rows", type=Path, required=True, help="risk_probe_v1 rows_L35.npz")
    ap.add_argument("--lab_views", type=Path, required=True)
    ap.add_argument("--step_values", type=Path, required=True)
    ap.add_argument("--trace_set", type=Path, required=True)
    ap.add_argument("--readouts", type=Path, required=True, help="dir with hparams.json, preds_e1.npz, preds_e2.npz")
    ap.add_argument("--e2_scores", type=Path, required=True, help="analysis step_scores.jsonl (xfit5 per label view)")
    ap.add_argument("--pb_rows", type=Path, required=True)
    ap.add_argument("--pb_meta", type=Path, required=True, help="context_ablation_v3 views/full/meta")
    ap.add_argument("--scores_dir", type=Path, required=True, help="score_with_checkpoints.py outputs")
    ap.add_argument("--saved_v2ds", type=Path, default=None, help="context_ablation_v3 prod/fits")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    gates = {}

    # ---- readouts: the E1 fit on all label steps
    z = np.load(a.label_rows)
    rows = {k: z[k] for k in ("ids", "split", "before", "after", "mean")}
    traces = {t["trace_id"]: t for t in json.loads(a.trace_set.read_text())["traces"]}
    D = assemble(rows, jl(a.lab_views), jl(a.step_values), traces)
    hp = json.loads((a.readouts / "hparams.json").read_text())["hparams"]["full"]
    fitter = Fitter(device)
    models, _ = fitter.fit_all(D, np.arange(len(D["step"])), "full", hp)
    fk = np.where(rows["split"] == "fsl_forks")[0]
    pf = Fitter.predict(models, {k: rows[k][fk] for k in ("before", "after", "mean")})
    e1 = np.load(a.readouts / "preds_e1.npz")
    assert e1["view_id"].tolist() == rows["ids"][fk].tolist()
    dmax = {arm: float(np.max(np.abs(pf[arm] - e1["pred"][:, j]))) for j, arm in enumerate(e1["arms"].tolist())}
    gates["refit_reproduces_e1"] = {"max_abs_diff": dmax, "pass": bool(max(dmax.values()) < 1e-3)}
    zp = np.load(a.pb_rows)
    pb_ids = zp["ids"].tolist()
    pp = Fitter.predict(models, {k: zp[k] for k in ("before", "after", "mean")})

    # ---- validity scores on ProcessBench
    val = {}
    for name, files in {"xfit5": [f"xfit5_f{k}" for k in range(5)],
                        "v2ds_21_18": [f"full_v2ds_s{s}" for s in (42, 43, 44)]}.items():
        L = [load_logits(a.scores_dir / f"{f}.jsonl") for f in files]
        val[name] = {v: float(np.mean([l[v] for l in L])) for v in pb_ids}
    if a.saved_v2ds:
        dp = []
        for s in (42, 43, 44):
            new = load_logits(a.scores_dir / f"full_v2ds_s{s}.jsonl")
            with gzip.open(a.saved_v2ds / f"full_v2ds_s{s}" / "local" / "predictions.jsonl.gz", "rt") as f:
                for l in f:
                    r = json.loads(l)
                    if r["trace_id"] in new:
                        dp.append(abs(sigmoid(r["logit"]) - sigmoid(new[r["trace_id"]])))
        gates["v2ds_rescoring"] = {"n": len(dp), "median_abs_dP": float(np.median(dp)), "max_abs_dP": float(np.max(dp)),
                                   "pass": bool(len(dp) == 3 * len(pb_ids) and np.median(dp) <= 0.002 and np.max(dp) <= 0.02)}

    # ---- validity + readout, fit on the label set's out-of-fold scores
    e2 = np.load(a.readouts / "preds_e2.npz")
    e2_arms = e2["arms"].tolist()
    xf = {r["view_id"]: r["xfit5"] for r in jl(a.e2_scores)}
    sv = {r["view_id"]: r for r in jl(a.step_values) if "view_id" in r}
    lab = [i for i, v in enumerate(e2["view_id"].tolist()) if v in sv and sv[v]["mask_ds"] and v in xf]
    y_ds = np.array([sv[e2["view_id"][i]]["y_ds"] for i in lab])
    x_val = np.array([xf[e2["view_id"][i]] for i in lab])
    pb_val = np.array([val["xfit5"][v] for v in pb_ids])
    scores = {f"validity {k}": v for k, v in val.items()}
    coefs = {}
    for arm in READOUTS:
        scores[f"risk {arm}"] = dict(zip(pb_ids, pp[arm].tolist()))
        f, coefs[arm] = combiner(x_val, e2["pred"][lab, e2_arms.index(arm)], y_ds)
        scores[f"xfit5 + {arm}"] = dict(zip(pb_ids, f(pb_val, pp[arm]).tolist()))

    # ---- metrics
    tr = pb_traces([m for sp in PB for m in jl(a.pb_meta / f"{sp}.jsonl")])
    missing = [v for t in tr.values() for x in t.values() for v in x["views"] if v not in scores["risk A_last"]]
    if missing:
        sys.exit(f"[pb] {len(missing)} PB views without rows, e.g. {missing[:3]}")
    res = {name: pb_metrics(s, tr) for name, s in scores.items()}
    out = {"gates": gates, "combiner_coefs_std": coefs, "metrics": res, "hparams_full": hp, "n_pb_views": len(pb_ids)}
    (a.out_dir / "pb_metrics.json").write_text(json.dumps(out, indent=1))
    with open(a.out_dir / "pb_scores.jsonl", "w") as f:
        for v in pb_ids:
            f.write(json.dumps({"view_id": v, **{k: s[v] for k, s in scores.items()}}) + "\n")
    lines = ["| score | mean AUROC (known steps) | mean F1_PB calib-20 | mean F1_PB oracle | "
             + " | ".join(f"{sp[3:]} calib-20" for sp in PB) + " |", "|" + "---|" * (4 + len(PB))]
    for name, r in res.items():
        lines.append(f"| {name} | {r['mean']['auroc_known']:.3f} | {r['mean']['f1_pb_calib20']:.3f} | "
                     f"{r['mean']['f1_pb_oracle']:.3f} | " + " | ".join(f"{r[sp]['f1_pb_calib20']:.3f}" for sp in PB) + " |")
    (a.out_dir / "pb_metrics.md").write_text("\n".join(lines) + "\n\ngates: " + json.dumps(gates) + "\n")
    print("\n".join(lines)); print(json.dumps(gates))


if __name__ == "__main__":
    main()
