#!/usr/bin/env python3
"""ProcessBench numbers for stored per-trace step scores: the paper's first-error AUROC / PR-AUC,
F1_PB at calib-20, at the oracle threshold and at a fixed P = 0.5, and paired bootstraps.

Arms, one directory per seed:
  --arm name=dir[,dir...]            dirs holding pb_step_scores_<subset>.jsonl (analyze.write_pb's
                                     format, the files paper_v1.py scores calib-20 from)
  --raw name=kind:dir[,dir...]       raw score dirs (analyze.seed_scores kinds: rp = {"uid", "p"}
                                     jsonl, rt = encode npz); converted with analyze.write_pb into
                                     <out dir>/pb_scores/<name>/seed<i> first (needs --data)
Per subset and seed:
  first-error  AUROC and PR-AUC over each trace's steps up to its first error, that step the
               only positive (pb_metrics.py, the paper's ProcessBench table)
  calib-20     merge_rep_grid_leaderboard.calib20_subset (the paper's thresholded number)
  oracle       the best F1_PB over the uniform grid 0.01..0.99 plus 999 quantiles of the subset's
               scores (the same candidates for every arm)
  p05          F1_PB at P(incorrect) > 0.5, a PRM's deployed rule
Seed means, then the 4-subset mean. --pair A-B: paired trace bootstrap (--n_boot resamples of the
traces inside each subset, the same for both arms) of A minus B on first-error AUROC and PR-AUC.

    python scripts/instep_v1/pb_oracle.py --out $OUT/final/pb_oracle.json --data $V1/data \\
        --arm instep_ds=$OUT/final/pb_scores/qwen3_8b/instep_ds/seed0,... \\
        --raw step=rp:$OUT/scores/qwen3_8b/step/seed0,... --pair instep_ds-step
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.pb_threshold_calibration import load_traces  # noqa: E402
from scripts.instep_v1.pb_metrics import fast_ap, fast_auroc  # noqa: E402
from scripts.merge_rep_grid_leaderboard import (CALIB_GRID, PB_SUBSETS, calib20_subset,  # noqa: E402
                                                f1_pb_from_preds, pred_matrix)

POINT = ("auroc", "ap", "oracle", "p05")


def oracle_subset(traces: list) -> tuple[float, float]:
    """(best F1_PB, its threshold) over the uniform grid and the score quantiles."""
    s = np.concatenate([np.asarray(sc, dtype=np.float64) for _, sc in traces])
    grid = np.unique(np.concatenate([CALIB_GRID, np.quantile(s, np.linspace(0.0005, 0.9995, 999))]))
    preds, labels = pred_matrix(traces, grid)
    f1 = f1_pb_from_preds(preds, labels)
    i = int(np.argmax(f1))
    return float(f1[i]), float(grid[i])


def p05_subset(traces: list) -> float:
    preds, labels = pred_matrix(traces, np.array([0.5]))
    return float(f1_pb_from_preds(preds, labels)[0])


def first_error(traces: list) -> tuple[float, float]:
    """(AUROC, PR-AUC) of the first-error step against the steps before it and correct traces."""
    y, s = [], []
    for lab, sc in traces:
        last = len(sc) - 1 if lab < 0 else min(lab, len(sc) - 1)
        y += [1 if k == lab else 0 for k in range(last + 1)]
        s += sc[:last + 1]
    y, s = np.array(y), np.array(s, dtype=float)
    return fast_auroc(y, s), fast_ap(y, s)


def point(traces: list, metrics=POINT) -> dict:
    out = dict(zip(("auroc", "ap"), first_error(traces)))
    if "oracle" in metrics:
        out["oracle"] = oracle_subset(traces)[0]
    if "p05" in metrics:
        out["p05"] = p05_subset(traces)
    return {m: out[m] for m in metrics}


def mean_point(seeds: list[dict[str, list]], metrics=POINT) -> dict:
    """4-subset mean of the seed-mean point metrics; seeds: [{subset: traces}]."""
    per = [{s: point(sd[s], metrics) for s in PB_SUBSETS} for sd in seeds]
    return {m: float(np.mean([np.mean([p[s][m] for p in per]) for s in PB_SUBSETS])) for m in metrics}


def bootstrap_pair(A: list[dict], B: list[dict], n_boot: int, seed: int = 0, metrics=("auroc", "ap")) -> dict:
    """A minus B, traces resampled within each subset (same indices for both arms and every seed)."""
    rng = np.random.default_rng(seed)
    n = {s: len(A[0][s]) for s in PB_SUBSETS}
    for X in (A, B):
        for sd in X:
            if {s: len(sd[s]) for s in PB_SUBSETS} != n:
                raise SystemExit("arms differ in their ProcessBench traces")
    d = []
    for _ in range(n_boot):
        idx = {s: rng.integers(0, n[s], n[s]) for s in PB_SUBSETS}
        res = [mean_point([{s: [sd[s][i] for i in idx[s]] for s in PB_SUBSETS} for sd in X], metrics) for X in (A, B)]
        d.append([res[0][m] - res[1][m] for m in metrics])
    d = np.array(d)
    full = [mean_point(A, metrics), mean_point(B, metrics)]
    return {m: {"delta": full[0][m] - full[1][m], "ci95": [float(np.percentile(d[:, j], 2.5)),
                                                         float(np.percentile(d[:, j], 97.5))]}
            for j, m in enumerate(metrics)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", nargs="*", default=[], help="name=dir[,dir...] of pb_step_scores files")
    p.add_argument("--raw", nargs="*", default=[], help="name=kind:dir[,dir...] of raw score dirs")
    p.add_argument("--data", type=Path, help="ReProbe data root (data/pb/<subset>.traces.jsonl), for --raw")
    p.add_argument("--pair", nargs="*", default=[], help="A-B")
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dirs = {}
    for spec in a.arm:
        name, ds = spec.split("=", 1)
        dirs[name] = ds.split(",")
    for spec in a.raw:
        from scripts.reprobe_cmp.analyze import write_pb
        name, rest = spec.split("=", 1)
        kind, ds = rest.split(":", 1)
        dirs[name] = []
        for i, d in enumerate(ds.split(",")):
            o = a.out.parent / "pb_scores" / name / f"seed{i}"
            if not write_pb(kind, d, a.data, o):
                raise SystemExit(f"{name}: {d} has no ProcessBench scores")
            dirs[name].append(str(o))
    traces = {k: [{s: load_traces(Path(d) / f"pb_step_scores_{s}.jsonl") for s in PB_SUBSETS} for d in v]
              for k, v in dirs.items()}
    res = {"arms": {}, "pairs": []}
    for name, seeds in traces.items():
        r = mean_point(seeds)
        r["calib20"] = float(np.mean([np.mean([calib20_subset(sd[s]) for sd in seeds]) for s in PB_SUBSETS]))
        r["seeds"] = len(seeds)
        res["arms"][name] = r
        print(f"{name:12s} seeds {r['seeds']} | first-error AUROC {r['auroc']:.3f} PR-AUC {r['ap']:.3f} | "
              f"F1_PB calib-20 {r['calib20']:.3f} oracle {r['oracle']:.3f} P>0.5 {r['p05']:.3f}", flush=True)
    for pr in a.pair:
        A, B = pr.split("-", 1)
        r = bootstrap_pair(traces[A], traces[B], a.n_boot)
        res["pairs"].append({"a": A, "b": B, **r})
        print(f"{A} - {B}: " + ", ".join(f"{m} {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                                         for m, v in r.items()), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
