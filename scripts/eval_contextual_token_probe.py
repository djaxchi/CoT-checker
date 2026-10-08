#!/usr/bin/env python3
"""Metrics and paired differences for bidirectional_token_probe_v1 (plan 9-12).

Reads every final fit's predictions (fits/<name>/<arm>/predictions.jsonl.gz),
freezes each run's threshold on labeled PRM calibration steps (best F1, higher
threshold on ties), applies it unchanged to PRM test and the four ProcessBench
subsets, and reports oracle-threshold results only as marked ceilings.

Paired differences (full - causal, future1 - causal, causal - local) use the same
examples per seed, with 10,000 bootstrap resamples clustered by problem
(multinomial problem weights shared by both arms and all seeds), recomputing
metrics at the fixed calibrated thresholds.

Outputs (out_dir): metrics.json, leaderboard.csv, paired_differences.json.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.contextual_probe_metrics import (  # noqa: E402
    auroc, best_f1_threshold, best_pb_threshold, pb_trace_metrics, pb_trivial, step_metrics,
)

PB = ("pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath")
CONTRASTS = (("full", "causal"), ("future1", "causal"), ("causal", "local"), ("full", "future1"))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def load_meta(manifest: Path) -> dict:
    meta = {}
    for p in (manifest / "meta").glob("*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            meta[r["trace_id"]] = r
    return meta


def load_preds(path: Path) -> dict[str, dict]:
    """split -> {trace_id: np.array of P(incorrect) per step}"""
    out: dict[str, dict] = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            d = out[r["split"]].setdefault(r["trace_id"], [None] * r["n_steps"])
            d[r["step"]] = float(sigmoid(r["logit"]))
    return {sp: {t: np.asarray(v) for t, v in d.items()} for sp, d in out.items()}


def flat(preds: dict, meta: dict, ids: list[str], extra=None, slice_: str | None = None):
    """Labeled steps: scores, y, problem index, slices. slice_ (v2): 'pre' = up to
    and including the first labeled error, 'post' = labeled steps after it,
    'finished' / 'unfinished' = by trace flag, 'future' = targets with a later step."""
    s, y, prob, rating, has_future = [], [], [], [], []
    for t in ids:
        m = meta[t]
        T = len(m["y"])
        fe = next((k for k in range(T) if m["label_mask"][k] and m["y"][k] == 1), None)
        if slice_ == "finished" and not m.get("finished", True):
            continue
        if slice_ == "unfinished" and m.get("finished", True):
            continue
        for k in range(T):
            if not m["label_mask"][k]:
                continue
            if slice_ == "pre" and fe is not None and k > fe:
                continue
            if slice_ == "post" and (fe is None or k <= fe):
                continue
            if slice_ == "future" and k >= T - 1:
                continue
            s.append(preds[t][k]); y.append(m["y"][k]); prob.append(m["problem_key"])
            r = (m.get("ratings") or [None] * T)[k]
            rating.append(-9 if r is None else r)
            has_future.append(k < T - 1)
    return (np.asarray(s), np.asarray(y, dtype=int), np.asarray(prob), np.asarray(rating),
            np.asarray(has_future, dtype=bool))


# v2 step-level sets: name -> (split, slice)
V2_STEP_SETS = {"rp_test": ("test", None), "rp_test_post": ("test", "post"), "rp_test_pre": ("test", "pre"),
                "rp_test_finished": ("test", "finished"), "rp_test_unfinished": ("test", "unfinished"),
                "rp_test_future": ("test", "future"), "prm_human_test": ("prm_human_test", None)}
STEP_SETS: dict = {}


def run_metrics(preds: dict, meta: dict) -> dict:
    out = {}
    cs, cy, *_ = flat(preds["calib"], meta, sorted(preds["calib"]))
    thr, cf1 = best_f1_threshold(cs, cy)
    out["calib"] = {"threshold": thr, "f1": cf1, "n": int(len(cy))}
    ds, dy, *_ = flat(preds["dev"], meta, sorted(preds["dev"]))
    out["dev"] = step_metrics(ds, dy, thr) | {"oracle_f1": best_f1_threshold(ds, dy)[1]}
    ts, ty, _, tr, tf = flat(preds["test"], meta, sorted(preds["test"]))
    m = step_metrics(ts, ty, thr)
    m["oracle_threshold"], m["oracle_f1"] = best_f1_threshold(ts, ty)
    pos, r0, r1 = ty == 1, tr == 0, tr == 1
    m["rating0"] = {"n": int(r0.sum()), "flag_rate": float((ts[r0] >= thr).mean()),
                    "auroc_incorrect_vs_rating0": auroc(np.r_[ts[pos], ts[r0]], np.r_[np.ones(pos.sum()), np.zeros(r0.sum())])}
    m["rating1"] = {"n": int(r1.sum()), "flag_rate": float((ts[r1] >= thr).mean()),
                    "auroc_incorrect_vs_rating1": auroc(np.r_[ts[pos], ts[r1]], np.r_[np.ones(pos.sum()), np.zeros(r1.sum())])}
    m["with_future_slice"] = step_metrics(ts[tf], ty[tf], thr)
    out["prm_test"] = m
    for name, (sp, sl) in STEP_SETS.items():
        if sp not in preds:
            continue
        ss, yy, *_ = flat(preds[sp], meta, sorted(preds[sp]), slice_=sl)
        mm = step_metrics(ss, yy, thr)
        mm["oracle_threshold"], mm["oracle_f1"] = best_f1_threshold(ss, yy)
        out[name] = mm
    pbf = []
    for sp in PB:
        ids = sorted(preds[sp])
        s, y, *_ = flat(preds[sp], meta, ids)
        st = step_metrics(s, y, thr)
        st["oracle_threshold"], st["oracle_f1"] = best_f1_threshold(s, y)
        seqs = [preds[sp][t] for t in ids]
        labels = [int(meta[t]["pb_label"]) for t in ids]
        tm = pb_trace_metrics(seqs, labels, thr)
        ot, of = best_pb_threshold(seqs, labels)
        out[sp] = {"step_known_labels": st, "trace": tm,
                   "trace_oracle": {"threshold": ot, "F1_PB": of},
                   "trivial": pb_trivial(labels, [len(x) for x in seqs]),
                   "coverage": {"traces": len(ids), "steps": int(sum(len(x) for x in seqs)),
                                "known_label_steps": int(len(y))}}
        pbf.append(tm["F1_PB"])
    out["pb_mean_F1_PB"] = float(np.mean(pbf))
    return out


# ------------------------------------------------------------------ bootstrap

def problem_weights(problems: list[str], n_boot: int, seed: int) -> tuple[dict, np.ndarray]:
    uniq = sorted(set(problems))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n_boot).astype(np.float32)
    return {p: i for i, p in enumerate(uniq)}, W


def boot_f1(s, y, pidx, thr, W):
    """Bootstrap F1 at a fixed threshold: per-problem tp/fp/fn sums."""
    P = W.shape[1]
    pred = s >= thr
    tp = np.bincount(pidx, (pred & (y == 1)).astype(float), P)
    fp = np.bincount(pidx, (pred & (y == 0)).astype(float), P)
    fn = np.bincount(pidx, (~pred & (y == 1)).astype(float), P)
    TP, FP, FN = W @ tp, W @ fp, W @ fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(2 * TP + FP + FN > 0, 2 * TP / (2 * TP + FP + FN), 0.0)


def boot_auc(s, y, pidx, W, chunk=250):
    """Weighted AUROC per resample (sample weight = its problem's multiplicity)."""
    order = np.argsort(s, kind="mergesort")
    s, y, pidx = s[order], y[order], pidx[order]
    _, grp = np.unique(s, return_inverse=True)
    out = []
    pos = y == 1
    for i in range(0, W.shape[0], chunk):
        w = W[i:i + chunk][:, pidx].astype(np.float64)  # (B, n)
        wn = np.where(pos, 0.0, w)
        wp = np.where(pos, w, 0.0)
        G = grp.max() + 1
        # per tie-group sums
        gn = np.zeros((w.shape[0], G)); gp = np.zeros((w.shape[0], G))
        np.add.at(gn.T, grp, wn.T); np.add.at(gp.T, grp, wp.T)
        below = np.cumsum(gn, axis=1) - gn  # negatives strictly below each group
        num = (gp * (below + 0.5 * gn)).sum(1)
        den = gp.sum(1) * gn.sum(1)
        out.append(num / den)
    return np.concatenate(out)


def boot_pb(seqs, labels, pidx, thr, W):
    from src.eval.contextual_probe_metrics import first_crossing
    P = W.shape[1]
    preds = np.array([first_crossing(x, thr) for x in seqs])
    labels = np.asarray(labels)
    err = labels >= 0
    he = np.bincount(pidx, (err & (preds == labels)).astype(float), P)
    ne = np.bincount(pidx, err.astype(float), P)
    hc = np.bincount(pidx, (~err & (preds == -1)).astype(float), P)
    nc = np.bincount(pidx, (~err).astype(float), P)
    with np.errstate(invalid="ignore", divide="ignore"):
        ae = (W @ he) / (W @ ne)
        ac = (W @ hc) / (W @ nc)
        return np.where(ae + ac > 0, 2 * ae * ac / (ae + ac), 0.0)


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def paired(runs: dict, meta: dict, n_boot: int, seed: int) -> dict:
    """runs: {(arm, seed): (preds, metrics)}"""
    seeds = sorted({s for _, s in runs})
    arms = sorted({a for a, _ in runs})
    out = {}
    datasets = ["prm_test"] + list(STEP_SETS) + list(PB)
    for ds in datasets:
        sp, sl = ("test", None) if ds == "prm_test" else STEP_SETS.get(ds, (ds, None))
        if sp not in next(iter(runs.values()))[0]:
            continue
        any_run = next(iter(runs.values()))[0]
        ids = sorted(any_run[sp])
        problems = [meta[t]["problem_key"] for t in ids]
        pmap, W = problem_weights(problems, n_boot, seed)
        trace_p = np.array([pmap[p] for p in problems])
        per = {}
        for (arm, sd), (preds, met) in runs.items():
            thr = met["calib"]["threshold"]
            s, y, prob, *_ = flat(preds[sp], meta, ids, slice_=sl)
            pidx = np.array([pmap[p] for p in prob])
            r = {"f1": boot_f1(s, y, pidx, thr, W), "auroc": boot_auc(s, y, pidx, W),
                 "f1_point": step_metrics(s, y, thr)["f1"], "auroc_point": auroc(s, y)}
            if ds in PB:
                seqs = [preds[sp][t] for t in ids]
                labels = [int(meta[t]["pb_label"]) for t in ids]
                r["F1_PB"] = boot_pb(seqs, labels, trace_p, thr, W)
                r["F1_PB_point"] = pb_trace_metrics(seqs, labels, thr)["F1_PB"]
            per[(arm, sd)] = r
        res = {}
        for a, b in CONTRASTS:
            if a not in arms or b not in arms:
                continue
            res[f"{a}-{b}"] = {}
            for metric in (["f1", "auroc"] + (["F1_PB"] if ds in PB else [])):
                by_seed = {sd: per[(a, sd)][f"{metric}_point"] - per[(b, sd)][f"{metric}_point"] for sd in seeds}
                boots = np.stack([per[(a, sd)][metric] - per[(b, sd)][metric] for sd in seeds])
                res[f"{a}-{b}"][metric] = {
                    "per_seed": {str(k): v for k, v in by_seed.items()},
                    "per_seed_ci95": {str(sd): ci(boots[i]) for i, sd in enumerate(seeds)},
                    "mean": float(np.mean(list(by_seed.values()))),
                    "seed_sd": float(np.std(list(by_seed.values()), ddof=1)) if len(seeds) > 1 else 0.0,
                    "seed_min": float(min(by_seed.values())), "seed_max": float(max(by_seed.values())),
                    "mean_ci95": ci(boots.mean(0))}
        out[ds] = res
        out[ds + "_per_run_boot"] = {f"{a}/{s}": {"f1_ci95": ci(v["f1"]), "auroc_ci95": ci(v["auroc"])}
                                     for (a, s), v in per.items()}
        if ds in PB:
            out.setdefault("_pb_boot", {})[ds] = {k: v["F1_PB"] for k, v in per.items()}
    # mean F1_PB over the four subsets (independent problem resampling per subset)
    pbb = out.pop("_pb_boot", {})
    if len(pbb) == 4:
        res = {}
        for a, b in CONTRASTS:
            if a not in arms or b not in arms:
                continue
            per_seed = {}
            boots = []
            for sd in seeds:
                da = np.mean([pbb[d][(a, sd)] for d in PB], 0)
                db = np.mean([pbb[d][(b, sd)] for d in PB], 0)
                boots.append(da - db)
                per_seed[str(sd)] = float(np.mean([runs[(a, sd)][1][d]["trace"]["F1_PB"] for d in PB])
                                          - np.mean([runs[(b, sd)][1][d]["trace"]["F1_PB"] for d in PB]))
            boots = np.stack(boots)
            res[f"{a}-{b}"] = {"per_seed": per_seed, "mean": float(np.mean(list(per_seed.values()))),
                               "seed_sd": float(np.std(list(per_seed.values()), ddof=1)) if len(seeds) > 1 else 0.0,
                               "per_seed_ci95": {sd: ci(boots[i]) for i, sd in enumerate(per_seed)},
                               "mean_ci95": ci(boots.mean(0))}
        out["pb_mean_F1_PB"] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits_root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--fit_prefix", default="final_s")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--v2", action="store_true", help="add the v2 ReProbe step sets and slices")
    ap.add_argument("--step_set", action="append", default=[],
                    help="extra step set NAME=SPLIT[:SLICE] (slice: pre|post|finished|unfinished|future)")
    ap.add_argument("--contrasts", default=None, help="comma list of A-B arm contrasts")
    a = ap.parse_args()
    if a.v2:
        STEP_SETS.update(V2_STEP_SETS)
    for spec in a.step_set:
        name, rest = spec.split("=", 1)
        sp, _, sl = rest.partition(":")
        STEP_SETS[name] = (sp, sl or None)
    if a.contrasts:
        global CONTRASTS
        CONTRASTS = tuple(tuple(c.split("-", 1)) for c in a.contrasts.split(","))
    meta = load_meta(a.manifest)
    runs = {}
    rows = []
    for fd in sorted(a.fits_root.glob(f"{a.fit_prefix}*")):
        for arm_dir in sorted(p for p in fd.iterdir() if p.is_dir()):
            done = arm_dir / "done.json"
            if not done.exists():
                raise SystemExit(f"[FATAL] incomplete run {arm_dir}")
            d = json.loads(done.read_text())
            preds = load_preds(arm_dir / "predictions.jsonl.gz")
            met = run_metrics(preds, meta)
            met["run"] = d
            runs[(d["arm"], d["seed"])] = (preds, met)
            row = {"arm": d["arm"], "seed": d["seed"], "best_epoch": d["best_epoch"],
                   "dev_f1": d["dev_f1"], "calib_thr": met["calib"]["threshold"],
                   "prm_test_f1": met["prm_test"]["f1"], "prm_test_oracle_f1": met["prm_test"]["oracle_f1"],
                   "prm_test_auroc": met["prm_test"]["auroc"],
                   "prm_test_always_pos_f1": met["prm_test"]["always_positive_f1"],
                   "pb_mean_F1_PB": met["pb_mean_F1_PB"]}
            for name in STEP_SETS:
                if name in met:
                    row[f"{name}_f1"] = met[name]["f1"]; row[f"{name}_auroc"] = met[name]["auroc"]
            for sp in PB:
                row[f"{sp}_F1_PB"] = met[sp]["trace"]["F1_PB"]
                row[f"{sp}_F1_PB_oracle"] = met[sp]["trace_oracle"]["F1_PB"]
                row[f"{sp}_step_f1"] = met[sp]["step_known_labels"]["f1"]
                row[f"{sp}_step_auroc"] = met[sp]["step_known_labels"]["auroc"]
            rows.append(row)
            print(f"[run] {d['arm']} s{d['seed']} prm F1={row['prm_test_f1']:.4f} AUC={row['prm_test_auroc']:.4f} "
                  f"PB F1={row['pb_mean_F1_PB']:.4f}", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "metrics.json").write_text(json.dumps(
        {f"{k[0]}/{k[1]}": v[1] for k, v in sorted(runs.items())}, indent=2))
    with open(a.out_dir / "leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r["arm"], r["seed"])))
    pdiff = paired(runs, meta, a.n_boot, a.seed)
    (a.out_dir / "paired_differences.json").write_text(json.dumps(pdiff, indent=2))
    print(json.dumps({k: pdiff[k] for k in ("prm_test", "pb_mean_F1_PB") if k in pdiff}, indent=1)[:4000])


if __name__ == "__main__":
    main()
