#!/usr/bin/env python3
"""Reviewer W9: the step split behind the paper's first-error claim on ReProbe's eight test sets
(Qwen3-8B). Every labelled step of a trace falls in one of two parts, by the trace's first error
under the test sets' only labels (DeepSeek-R1, ReProbe's released StepFactCheck labels):

  upto   steps up to and including the first incorrect step (every step of a trace with none)
  after  steps after the first incorrect step (correct or incorrect)

Per set and part: steps, incorrect steps, and each arm's PR-AUC of "incorrect" and AUROC
(per seed, then the seed mean, as scripts/instep_v1/paper_v1.py); "overall" is the mean over
the eight sets. --pair A-B: paired trace bootstrap of A minus B on each part's overall PR-AUC and
AUROC, traces resampled within each set (paper_v1.make_test_boot: generator seed 0, the same
resamples for every arm and part).

  export   on the cluster: every arm's per-seed test scores ({uid: p}, analyze.seed_scores) and
           the test labels into --export (small JSON files)
  analyze  from --export: the table, written to --out (JSON) with the pairs

    python scripts/instep_v1/first_error_split.py analyze --export <dir> --out first_error_split.json \\
        --pair instep_ds-prm7b instep_ds-prm800k --n_boot 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reprobe_cmp.analyze import TEST, fast_ap  # noqa: E402
from scripts.instep_v1.pb_metrics import fast_auroc  # noqa: E402

PARTS = ("upto", "after")


def split_set(traces: list[dict]) -> dict:
    """uids, y, trace index and part of every labelled step (label >= 0)."""
    uids, y, tr, part = [], [], [], []
    for ti, t in enumerate(traces):
        lab = t["labels"]
        fe = next((k for k, v in enumerate(lab) if v == 1), None)
        for k, v in enumerate(lab):
            if v < 0:
                continue
            uids.append(f"{t['trace_id']}::{k}")
            y.append(int(v))
            tr.append(ti)
            part.append(0 if fe is None or k <= fe else 1)
    return {"uids": uids, "y": np.array(y), "trace": np.array(tr), "part": np.array(part)}


def metrics(y: np.ndarray, arrs: list[np.ndarray]) -> dict:
    if y.sum() == 0 or y.sum() == len(y):
        return {"ap": None, "auroc": None}
    return {"ap": float(np.mean([fast_ap(y, x) for x in arrs])),
            "auroc": float(np.mean([fast_auroc(y, x) for x in arrs]))}


def trace_groups(tr: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(tr, kind="mergesort")
    _, start = np.unique(tr[order], return_index=True)
    end = np.r_[start[1:], len(tr)]
    return [order[a:b] for a, b in zip(start, end)]


def make_boot(tests: dict, n_boot: int, seed: int = 0) -> dict:
    """paper_v1.make_test_boot: per set, n_boot resamples of the traces (step indices)."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in TEST:
        g = trace_groups(tests[s]["trace"])
        out[s] = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).astype(np.int64)
                  for _ in range(n_boot)]
    return out


def arm_arrays(arm: dict, tests: dict) -> dict:
    """{set: [per-seed score array over the set's labelled steps]}."""
    out = {}
    for s in TEST:
        arrs = []
        for sd in arm["seeds"]:
            sc = sd.get(s) or {}
            miss = [u for u in tests[s]["uids"] if u not in sc]
            if miss:
                raise SystemExit(f"{s}: {len(miss)} labelled steps without a score (e.g. {miss[0]})")
            arrs.append(np.array([sc[u] for u in tests[s]["uids"]], dtype=np.float64))
        out[s] = arrs
    return out


def table(tests: dict, arms: dict) -> dict:
    res = {"sets": {}, "overall": {}}
    for s in TEST:
        y, part = tests[s]["y"], tests[s]["part"]
        row = {"traces": int(len(np.unique(tests[s]["trace"]))),
               "traces_with_error": int(len(np.unique(tests[s]["trace"][y == 1])))}
        for pi, pn in enumerate(PARTS):
            m = part == pi
            row[pn] = {"steps": int(m.sum()), "incorrect": int(y[m].sum()),
                       "arms": {a: metrics(y[m], [x[m] for x in arrs[s]]) for a, arrs in arms.items()}}
        row["all"] = {"steps": int(len(y)), "incorrect": int(y.sum()),
                      "arms": {a: metrics(y, arrs[s]) for a, arrs in arms.items()}}
        inc = row["upto"]["incorrect"] + row["after"]["incorrect"]
        row["after_share_of_incorrect"] = row["after"]["incorrect"] / inc if inc else None
        res["sets"][s] = row
    for pn in PARTS + ("all",):
        o = {"steps": sum(res["sets"][s][pn]["steps"] for s in TEST),
             "incorrect": sum(res["sets"][s][pn]["incorrect"] for s in TEST), "arms": {}}
        for a in arms:
            v = [res["sets"][s][pn]["arms"][a] for s in TEST]
            o["arms"][a] = {m: (float(np.mean([x[m] for x in v])) if all(x[m] is not None for x in v) else None)
                            for m in ("ap", "auroc")}
        res["overall"][pn] = o
    sh = [res["sets"][s]["after_share_of_incorrect"] for s in TEST]
    res["overall"]["after_share_of_incorrect"] = {"min": min(sh), "max": max(sh),
                                                  "pooled": res["overall"]["after"]["incorrect"]
                                                  / res["overall"]["all"]["incorrect"]}
    return res


def boot_pair(tests: dict, A: dict, B: dict, boot: dict) -> dict:
    out = {}
    for pi, pn in enumerate(PARTS + ("all",)):
        d = {"ap": [], "auroc": []}
        n = len(next(iter(boot.values())))
        for b in range(n):
            va = {"ap": [], "auroc": []}
            for s in TEST:
                ii = boot[s][b]
                if pn != "all":
                    ii = ii[tests[s]["part"][ii] == pi]
                y = tests[s]["y"][ii]
                ma = metrics(y, [x[ii] for x in A[s]])
                mb = metrics(y, [x[ii] for x in B[s]])
                for m in ("ap", "auroc"):
                    va[m].append(None if ma[m] is None else ma[m] - mb[m])
            for m in ("ap", "auroc"):
                if all(v is not None for v in va[m]):
                    d[m].append(float(np.mean(va[m])))
        out[pn] = {m: {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "n_boot": len(v)}
                   for m, v in d.items() if v}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["export", "analyze"])
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--data", type=Path, help="export: <R>/data")
    p.add_argument("--arm", nargs="*", default=[], help="export: name=kind:dir[,dir...]")
    p.add_argument("--arms", default=None, help="analyze: comma list (default: every exported arm)")
    p.add_argument("--pair", nargs="*", default=[])
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.phase == "export":
        from scripts.reprobe_cmp.analyze import jl, seed_scores
        a.export.mkdir(parents=True, exist_ok=True)
        for spec in a.arm:
            name, rest = spec.split("=", 1)
            kind, dirs = rest.split(":", 1)
            dirs = dirs.split(",")
            out = {"kind": kind, "dirs": dirs,
                   "seeds": [{s: seed_scores(kind, d, s, a.data) for s in TEST} for d in dirs]}
            (a.export / f"test_{name}.json").write_text(json.dumps(out))
        lab = {s: [{"trace_id": t["trace_id"], "labels": t["labels"]} for t in jl(a.data / "test" / f"{s}.traces.jsonl")]
               for s in TEST}
        (a.export / "test_labels.json").write_text(json.dumps(lab))
        return
    lab = json.loads((a.export / "test_labels.json").read_text())
    tests = {s: split_set(lab[s]) for s in TEST}
    names = a.arms.split(",") if a.arms else sorted(f.stem[5:] for f in a.export.glob("test_*.json")
                                                    if f.stem != "test_labels")
    arms = {n: arm_arrays(json.loads((a.export / f"test_{n}.json").read_text()), tests) for n in names}
    res = table(tests, arms)
    res["label_source"] = "DeepSeek-R1 (the test sets' released labels; the first error is the first step labelled 1)"
    res["n_seeds"] = {n: len(arms[n]["math"]) for n in arms}
    res["pairs"] = []
    if a.pair:
        boot = make_boot(tests, a.n_boot)
        for pr in a.pair:
            A, B = pr.split("-", 1)
            r = boot_pair(tests, arms[A], arms[B], boot)
            for pn in r:
                for m in r[pn]:
                    pa, pb = res["overall"][pn]["arms"][A][m], res["overall"][pn]["arms"][B][m]
                    r[pn][m]["delta"] = pa - pb
            res["pairs"].append({"a": A, "b": B, **r})
            print(f"{A} - {B}: " + "; ".join(f"{pn} PR-AUC {r[pn]['ap']['delta']:+.3f} [{r[pn]['ap']['ci95'][0]:+.3f}, "
                                            f"{r[pn]['ap']['ci95'][1]:+.3f}]" for pn in r), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for pn in PARTS + ("all",):
        o = res["overall"][pn]
        print(f"[{pn}] steps {o['steps']} incorrect {o['incorrect']}: "
              + ", ".join(f"{n} {v['ap']:.3f}/{v['auroc']:.3f}" for n, v in o["arms"].items()), flush=True)


if __name__ == "__main__":
    main()
