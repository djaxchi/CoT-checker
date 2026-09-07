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
    p.add_argument("--outcomes", type=Path, default=None,
                   help="stage-1 outcomes jsonl. Baselines are recomputed on "
                        "EXACTLY the problems the rollouts cover; the headline "
                        "0.560 self-consistency is over 300 problems and four of "
                        "them have no source record, so quoting it against a "
                        "296-problem run would be comparing different sets.")
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

    if a.outcomes:
        pool = defaultdict(list)
        for line in open(a.outcomes):
            r = json.loads(line)
            if r["problem_id"] in common:
                pool[r["problem_id"]].append(r)
        pass1, sc, oracle = [], [], []
        for pid, sols in pool.items():
            ok = [bool(s["correct"]) for s in sols]
            pass1.append(float(np.mean(ok)))
            oracle.append(float(any(ok)))
            counts = defaultdict(int)
            for s in sols:
                counts[str(s.get("pred"))] += 1
            best = max(counts.values())
            tied = [p_ for p_, c in counts.items() if c == best]
            hits = [float(np.mean([s["correct"] for s in sols
                                   if str(s.get("pred")) == t])) for t in tied]
            sc.append(float(np.mean(hits)))
        report["baselines_on_covered_problems"] = {
            "n_problems": len(pool),
            "pass@1": float(np.mean(pass1)),
            "self_consistency": float(np.mean(sc)),
            "oracle": float(np.mean(oracle)),
        }
        b = report["baselines_on_covered_problems"]
        print(f"baselines on these same {b['n_problems']} problems: "
              f"pass@1 {b['pass@1']:.3f}  self-consistency {b['self_consistency']:.3f}  "
              f"oracle {b['oracle']:.3f}\n")
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

    # Length control. The arms do not produce equally long solutions, and on this
    # pool length alone gives trajectory AUROC 0.561 while picking the shortest
    # solution scores 0.350. So if guided writes shorter solutions than random,
    # part of any gain could be length rather than verification. This asks
    # directly: on the problems where guided wins and random loses, is guided
    # also the shorter one more often than chance?
    if "guided" in by_arm and "random" in by_arm:
        shorter_when_winning, n_win = 0, 0
        shorter_when_losing, n_lose = 0, 0
        dsteps = []
        for k in sorted(common):
            g, r = by_arm["guided"][k], by_arm["random"][k]
            dsteps.append(g["n_steps"] - r["n_steps"])
            if g["correct"] and not r["correct"]:
                n_win += 1
                shorter_when_winning += int(g["n_steps"] < r["n_steps"])
            elif r["correct"] and not g["correct"]:
                n_lose += 1
                shorter_when_losing += int(g["n_steps"] < r["n_steps"])
        rate_w = shorter_when_winning / n_win if n_win else float("nan")
        rate_l = shorter_when_losing / n_lose if n_lose else float("nan")
        report["length_control"] = {
            "mean_step_delta_guided_minus_random": float(np.mean(dsteps)),
            "guided_shorter_when_it_wins": rate_w, "n_wins": n_win,
            "guided_shorter_when_it_loses": rate_l, "n_losses": n_lose,
        }
        print(f"\nlength control: guided runs {np.mean(dsteps):+.2f} steps vs random on average.")
        print(f"  when guided wins ({n_win} problems) it is the shorter solution {rate_w:.2f} of the time")
        print(f"  when guided loses ({n_lose} problems) it is the shorter solution {rate_l:.2f} of the time")
        print("  a large gap between those two rates means the gain tracks length, not verification")

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
