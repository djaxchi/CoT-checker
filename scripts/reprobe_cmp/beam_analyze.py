#!/usr/bin/env python3
"""Beam-search accuracy per arm (reprobe_comparison_v2, B2), from beam_search.py's output.

Only questions every arm has finished enter a set's numbers, so arms are compared on the
same questions; the mean over sets weights sets equally. --pairs A-B: paired bootstrap
over questions (every set resampled for the mean). Also reports the mean step draws per
question (the search's sampling cost) and the share of answers that finished.

    python scripts/reprobe_cmp/beam_analyze.py --beam $R/beam --out $R/beam/analysis \\
        --pairs rt_self-rp_self rt_self-random rp_self-random
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beam", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", nargs="*", default=[])
    p.add_argument("--n_boot", type=int, default=2000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = defaultdict(lambda: defaultdict(dict))  # set -> arm -> qi -> row
    for f in sorted(a.beam.glob("*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            rows[r["set"]][r["arm"]][r["qi"]] = r
    sets = sorted(rows)
    arms = sorted({arm for s in sets for arm in rows[s]})
    acc, table = {}, defaultdict(dict)
    for s in sets:
        common = sorted(set.intersection(*(set(rows[s][arm]) for arm in arms)))
        for arm in arms:
            v = np.array([float(rows[s][arm][q]["correct"]) for q in common])
            acc[(s, arm)] = v
            table[arm][s] = {"acc": 100 * float(v.mean()) if len(v) else float("nan"), "n": len(common),
                             "draws": float(np.mean([rows[s][arm][q]["step_draws"] for q in common])),
                             "finished": float(np.mean([rows[s][arm][q]["finished"] for q in common]))}
    for arm in arms:
        table[arm]["mean"] = float(np.mean([table[arm][s]["acc"] for s in sets]))
    rng = np.random.default_rng(0)
    pairs = []
    for pair in a.pairs:
        A, B = next((x, z) for x in arms for z in arms if f"{x}-{z}" == pair)
        res, dm = {"pair": pair}, np.zeros(a.n_boot)
        for s in sets:
            d = acc[(s, A)] - acc[(s, B)]
            bs = 100 * d[rng.integers(0, len(d), (a.n_boot, len(d)))].mean(1)
            dm += bs / len(sets)
            res[s] = {"delta": 100 * float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        res["mean"] = {"delta": float(np.mean([res[s]["delta"] for s in sets])),
                       "ci95": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))]}
        pairs.append(res)
    (a.out / "summary.json").write_text(json.dumps({"table": table, "pairs": pairs}, indent=1))
    lines = ["| arm | " + " | ".join(f"{s} (n={table[arms[0]][s]['n']})" for s in sets) + " | mean | draws/question |",
             "|---" * (len(sets) + 3) + "|"]
    for arm in arms:
        lines.append(f"| {arm} | " + " | ".join(f"{table[arm][s]['acc']:.1f}" for s in sets)
                     + f" | {table[arm]['mean']:.1f} | "
                     + " / ".join(f"{table[arm][s]['draws']:.0f}" for s in sets) + " |")
    if pairs:
        lines += ["", "| pair | " + " | ".join(sets) + " | mean |", "|---" * (len(sets) + 2) + "|"]
        for r in pairs:
            lines.append(f"| {r['pair']} | " + " | ".join(
                f"{r[s]['delta']:+.1f} [{r[s]['ci95'][0]:+.1f}, {r[s]['ci95'][1]:+.1f}]" for s in sets + ["mean"]) + " |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
