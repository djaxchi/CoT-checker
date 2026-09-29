#!/usr/bin/env python3
"""Instruct downstream TTS: accuracy table and ProcessBench-to-selection rank transfer.

Reads the paired rule contrasts (`tts_rule_contrast.py`, baseline majority) for
one pool, the chosen checkpoint per leaderboard cell (`best_seed.json`, which
carries that seed's calib-20 F1_PB) and the seed-averaged leaderboard
(`base_vs_instruct_core.json`). Reports, per scorer, accuracy and lift over
majority for tie-break, best-of-N (rerank) and weighted vote, and the Spearman
and Kendall correlations across the 19 cells between calib-20 F1_PB and the
selection lift. The frozen primary endpoint is MATH-500 best-of-4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def kendall_b(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a); c = d = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if x == 0 and y == 0:
                continue
            if x == 0:
                ta += 1
            elif y == 0:
                tb += 1
            elif x == y:
                c += 1
            else:
                d += 1
    den = np.sqrt((c + d + ta) * (c + d + tb))
    return float((c - d) / den) if den else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contrasts", type=Path, nargs="+", required=True)
    p.add_argument("--best_seed", type=Path, required=True)
    p.add_argument("--leaderboard", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    rows = [r for f in a.contrasts for r in json.loads(f.read_text())]
    acc, lift, ci, base = {}, {}, {}, {}
    for r in rows:
        k = (r["dataset"], r["n"], r["rule"])
        acc[k] = r["rule_acc"]; lift[k] = -r["delta"]
        ci[k] = (-r["ci95"][1], -r["ci95"][0])
        base[(r["dataset"], r["n"])] = r["baseline_acc"]

    best = json.loads(a.best_seed.read_text())
    lb = {(c["rep"], c["learner"]): c for c in json.loads(a.leaderboard.read_text())["cells"]}
    cells = []
    for b in best:
        tag = f"{b['rep']}__{b['learner'].replace(':', '_').replace(',', '_')}__seed{b['seed']}"
        cells.append({"rep": b["rep"], "learner": b["learner"], "seed": b["seed"],
                      "scorer": f"probe::{tag}__gen::worst",
                      "calib20_seed": b["calib20"],
                      "calib20_mean": lb[(b["rep"], b["learner"])]["instruct"]["calib20"]})

    def L(ds, n, rule, sc):
        return lift.get((ds, n, f"{rule}::{sc}"))

    endpoints = [("math500", 4, "rerank"), ("math500", 2, "tiebreak"), ("math500", 4, "tiebreak"),
                 ("math500", 10, "wvote"), ("gsm8k", 4, "rerank"), ("gsm8k", 2, "tiebreak")]
    corr = {}
    for ds, n, rule in endpoints:
        y = [L(ds, n, rule, c["scorer"]) for c in cells]
        if any(v is None for v in y):
            continue
        for key in ("calib20_seed", "calib20_mean"):
            x = [c[key] for c in cells]
            corr[f"{ds}_N{n}_{rule}__{key}"] = {"spearman": spearman(x, y), "kendall_b": kendall_b(x, y)}

    ref = ["probe::prm_qwen25_math_7b::worst", "conf::bottom10_group_w32", "conf::mean_token_conf"]
    lines = ["| scorer | calib20 | " + " | ".join(
        f"{ds} N={n} {rule}" for ds, n, rule in endpoints) + " |", "|" + "---|" * (2 + len(endpoints))]
    order = sorted(cells, key=lambda c: -(L("math500", 4, "rerank", c["scorer"]) or -9))
    for c in order:
        lines.append(f"| {c['rep']} x {c['learner']} (s{c['seed']}) | {c['calib20_seed']:.4f} | " +
                     " | ".join(f"{100*L(ds,n,r,c['scorer']):+.2f}" for ds, n, r in endpoints) + " |")
    for sc in ref:
        vals = [L(ds, n, r, sc) for ds, n, r in endpoints]
        lines.append(f"| {sc.split('::')[1]} | - | " +
                     " | ".join("-" if v is None else f"{100*v:+.2f}" for v in vals) + " |")
    lines.append("| majority accuracy | - | " + " | ".join(
        f"{100*base[(ds, n)]:.2f}" for ds, n, _ in endpoints) + " |")
    lines += ["", "Rank transfer across the 19 cells (calib-20 F1_PB vs lift):"]
    lines += [f"- {k}: Spearman {v['spearman']:+.3f}, Kendall tau-b {v['kendall_b']:+.3f}"
              for k, v in corr.items()]
    md = "\n".join(lines)
    print(md)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".md").write_text(md + "\n")
    a.out.write_text(json.dumps({"cells": cells, "correlations": corr,
                                 "lift": {"|".join(map(str, k)): v for k, v in lift.items()},
                                 "ci95": {"|".join(map(str, k)): v for k, v in ci.items()},
                                 "accuracy": {"|".join(map(str, k)): v for k, v in acc.items()},
                                 "majority": {f"{k[0]}|{k[1]}": v for k, v in base.items()}},
                                indent=1))


if __name__ == "__main__":
    main()
