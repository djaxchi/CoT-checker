#!/usr/bin/env python3
"""Reviewer Q1: ProcessBench MATH split by whether PRM800K's training split holds the problem.

--seen: JSON {subset: {"n", "seen": [trace ids]}}, problems matched after normalize_problem against
prm800k_probe_train_full.jsonl (884 of 1,000 MATH solutions, none elsewhere). For every arm (seed
dirs of pb_step_scores_<subset>.jsonl, as scripts/instep_v1/pb_oracle.py reads them) and every part
(MATH seen, MATH unseen, MATH all): the paper's first-error AUROC and PR-AUC and calib-20 F1_PB
(merge_rep_grid_leaderboard.calib20_subset, 20 calibration solutions drawn inside the part), per
seed then the seed mean. --pair A-B: paired bootstrap over the part's traces (the same resamples
for both arms and every seed) of A minus B on the three metrics.

    python scripts/instep_v1/pb_overlap.py --seen pb_seen_prm800k.json --out item1_pb_overlap.json \\
        --arm instep_prm=<dir0>,<dir1>,<dir2> prm800k=<dir> ... --pair instep_prm-prm800k --n_boot 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.instep_v1.pb_oracle import first_error  # noqa: E402
from scripts.merge_rep_grid_leaderboard import calib20_subset  # noqa: E402

PARTS = ("math_seen", "math_unseen", "math_all")


def load(path: Path) -> list[tuple[str, int, list[float]]]:
    return [(r["id"], int(r["label"]), [float(x) for x in r["scores"]])
            for r in map(json.loads, open(path)) if r]


def parts_of(rows: list, seen: set) -> dict:
    tr = {"math_seen": [], "math_unseen": [], "math_all": []}
    for i, lab, sc in rows:
        tr["math_seen" if i in seen else "math_unseen"].append((lab, sc))
        tr["math_all"].append((lab, sc))
    return tr


def point(traces: list) -> dict:
    au, ap = first_error(traces)
    return {"auroc": au, "ap": ap, "calib20": calib20_subset(traces)}


def mean_seeds(seeds: list[list]) -> dict:
    per = [point(t) for t in seeds]
    return {m: float(np.mean([p[m] for p in per])) for m in ("auroc", "ap", "calib20")} | {
        "seeds": len(seeds), "per_seed": per}


def boot(A: list[list], B: list[list], n_boot: int, seed: int = 0, calib: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    n = len(A[0])
    ms = ("auroc", "ap", "calib20") if calib else ("auroc", "ap")
    d = {m: [] for m in ms}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        res = []
        for X in (A, B):
            vals = {m: [] for m in ms}
            for t in X:
                tt = [t[i] for i in idx]
                au, ap = first_error(tt)
                vals["auroc"].append(au)
                vals["ap"].append(ap)
                if calib:
                    vals["calib20"].append(calib20_subset(tt))
            res.append({m: np.mean(v) for m, v in vals.items()})
        for m in ms:
            d[m].append(res[0][m] - res[1][m])
    return {m: {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for m, v in d.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seen", type=Path, required=True)
    p.add_argument("--arm", nargs="+", required=True, help="name=dir[,dir...]")
    p.add_argument("--pair", nargs="*", default=[], help="A-B")
    p.add_argument("--pair_parts", default="math_unseen,math_seen")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    seen_js = json.loads(a.seen.read_text())
    seen = set(seen_js["math"]["seen"])
    arms = {}
    for spec in a.arm:
        name, ds = spec.split("=", 1)
        arms[name] = [parts_of(load(Path(d) / "pb_step_scores_math.jsonl"), seen) for d in ds.split(",")]
    ref = arms[next(iter(arms))][0]
    res = {"seen_source": "PRM800K training split (prm800k_probe_train_full.jsonl), problems after normalize_problem",
           "parts": {}, "pairs": []}
    for pn in PARTS:
        tr = ref[pn]
        res["parts"][pn] = {"solutions": len(tr), "erroneous": sum(l != -1 for l, _ in tr),
                            "first_error_steps_kept": int(sum((len(s) if l < 0 else min(l, len(s) - 1) + 1)
                                                              for l, s in tr)),
                            "arms": {}}
        for name, seeds in arms.items():
            if [len(s[pn]) for s in seeds] != [len(tr)] * len(seeds):
                raise SystemExit(f"{name}: traces differ from the first arm's on {pn}")
            r = mean_seeds([s[pn] for s in seeds])
            res["parts"][pn]["arms"][name] = r
            print(f"{pn:12s} {name:14s} n={len(tr)} AUROC {r['auroc']:.3f} PR-AUC {r['ap']:.3f} "
                  f"calib-20 {r['calib20']:.3f}", flush=True)
    for pr in a.pair:
        A, B = pr.split("-", 1)
        for pn in a.pair_parts.split(","):
            r = boot([s[pn] for s in arms[A]], [s[pn] for s in arms[B]], a.n_boot)
            for m in r:
                r[m]["delta"] = res["parts"][pn]["arms"][A][m] - res["parts"][pn]["arms"][B][m]
            res["pairs"].append({"a": A, "b": B, "part": pn, "n_boot": a.n_boot, **r})
            print(f"{A} - {B} on {pn}: " + ", ".join(f"{m} {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                                                   for m, v in r.items()), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
