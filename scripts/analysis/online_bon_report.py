#!/usr/bin/env python3
"""Did letting the checker choose each step produce better reasoning?

Two questions, and conflating them is the easy mistake:

  1. does branching help?      plain  -> random   (search alone, no checker)
  2. does the CHECKER help?    random -> guided   (the same search, checker picks)

Only the second is evidence about hidden-state verification. Reporting
guided-against-plain would credit the checker with whatever the wider search
bought, which for N=5 at temperature 1.5 is not small.

Both are paired per problem, so McNemar applies: the arms ran on the same
problems with the same seeds, and only the problems where two arms disagree
carry information.

COST. Accuracy alone flatters branching, because guided and random sample ~N
times the tokens that plain does. So accuracy per thousand generation tokens is
reported alongside, and the discarded branches are already in that denominator.
The comparison a practitioner actually faces is not "guided or plain" but "given
this many tokens, spend them on branching with a checker, or on more independent
samples and a majority vote", which is the matched-budget line.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ("plain", "random", "guided")


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict:
    shared = set(a) & set(b)
    a_only = sum(1 for k in shared if a[k] and not b[k])
    b_only = sum(1 for k in shared if b[k] and not a[k])
    n = a_only + b_only
    if n == 0:
        return {"n_discordant": 0, "p": float("nan"), "a_wins": 0, "b_wins": 0}
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(max(a_only, b_only), n + 1)) / 2 ** n)
    return {"n_discordant": n, "a_wins": a_only, "b_wins": b_only, "p": float(p)}


def paired_bootstrap(a: dict[str, bool], b: dict[str, bool],
                     n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    keys = sorted(set(a) & set(b))
    rng = np.random.default_rng(seed)
    d = np.array([float(b[k]) - float(a[k]) for k in keys])
    out = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", nargs="+", required=True, type=Path)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    by_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in a.rollouts:
        for line in open(f):
            r = json.loads(line)
            by_arm[r["arm"]][r["problem_id"]] = r
    arms = [x for x in ARMS if x in by_arm]
    if not arms:
        raise SystemExit("no rollouts found")

    common = set.intersection(*(set(by_arm[x]) for x in arms))
    print(f"{len(common)} problems complete in all of {arms}\n")

    report = {"n_problems": len(common), "arms": {}}
    print(f"{'arm':8s} {'accuracy':>9s} {'steps':>7s} {'gen tokens':>11s} "
          f"{'acc / 1k tok':>13s} {'checker calls':>14s}")
    for arm in arms:
        rows = [by_arm[arm][k] for k in sorted(common)]
        acc = float(np.mean([r["correct"] for r in rows]))
        tok = float(np.mean([r.get("gen_tokens", 0) for r in rows]))
        steps = float(np.mean([r["n_steps"] for r in rows]))
        calls = float(np.mean([r.get("scored_candidates", 0) for r in rows]))
        eff = 1000.0 * acc / tok if tok else float("nan")
        report["arms"][arm] = {"accuracy": acc, "mean_gen_tokens": tok,
                               "mean_steps": steps, "mean_scored_candidates": calls,
                               "accuracy_per_1k_tokens": eff}
        print(f"{arm:8s} {acc:9.3f} {steps:7.1f} {tok:11.0f} {eff:13.4f} {calls:14.1f}")

    print()
    contrasts = [("plain", "random", "does branching alone help?"),
                 ("random", "guided", "does the CHECKER help? (the real test)"),
                 ("plain", "guided", "combined, not attributable to the checker")]
    report["contrasts"] = []
    for lo, hi, what in contrasts:
        if lo not in by_arm or hi not in by_arm:
            continue
        A = {k: bool(by_arm[lo][k]["correct"]) for k in common}
        B = {k: bool(by_arm[hi][k]["correct"]) for k in common}
        mc = mcnemar(A, B)
        d, l, h = paired_bootstrap(A, B, a.n_boot)
        report["contrasts"].append({"from": lo, "to": hi, "question": what,
                                    "delta": d, "lo95": l, "hi95": h, "mcnemar": mc})
        print(f"{lo:6s} -> {hi:6s}  {d:+.3f} [{l:+.3f}, {h:+.3f}]  "
              f"McNemar p={mc['p']:.4g} ({hi} wins {mc['b_wins']}, {lo} wins {mc['a_wins']})"
              f"   {what}")

    if "guided" in by_arm and "plain" in by_arm:
        g = report["arms"]["guided"]
        pl = report["arms"]["plain"]
        ratio = g["mean_gen_tokens"] / pl["mean_gen_tokens"] if pl["mean_gen_tokens"] else float("nan")
        print(f"\nguided spends {ratio:.1f}x the generation tokens of plain. "
              f"At that budget the honest comparison is against self-consistency "
              f"over ~{ratio:.0f} independent samples, not against one sample.")
        report["guided_token_multiple_of_plain"] = ratio

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
        print(f"\n[online-report] wrote {a.out}")


if __name__ == "__main__":
    main()
