#!/usr/bin/env python3
"""Price a verifier onto the token axis and compare it to more sampling.

The frontier's token axis counts generated tokens only, which makes every
verifier look free. Here each verifier is charged for the candidates it
actually reads: all N for rerank and the weighted vote, only the tied bloc for
the tie-break (the frontier's `scored_mean`). A read costs one prefill of the
verifier's input, converted to generation-token equivalents by the ratio of
parameter counts, since a forward pass costs about 2 x params FLOPs per token
whether it is a prefill or a decode step. FLOPs are the only fair common unit
here; prefill is far faster than decoding in wall-clock, so this charge is an
upper bound on the verifier's real cost.

The comparison is then: at the same total compute, is the verifier rule more
accurate than majority voting with more samples? Majority accuracy at an
arbitrary budget is linearly interpolated over its N = 1..10 curve; budgets
beyond N = 10 are reported as out of range rather than extrapolated.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GEN_PARAMS = 8.2e9  # Qwen3-8B-Base


def majority_at(curve: list[tuple[float, float]], budget: float) -> float | None:
    xs, ys = zip(*sorted(curve))
    if budget > xs[-1] or budget < xs[0]:
        return None
    return float(np.interp(budget, xs, ys))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frontier", type=Path, required=True)
    p.add_argument("--input_tokens", type=Path, required=True,
                   help="traj_uid -> verifier input tokens (JSON)")
    p.add_argument("--scorer", action="append", required=True,
                   help="name=RULE_SUFFIX:PARAMS, e.g. "
                        "prm=probe::prm_qwen25_math_7b::worst:7.6e9")
    p.add_argument("--ns", type=int, nargs="+", default=[2, 4, 10])
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    toks = json.loads(a.input_tokens.read_text())
    per_ds: dict[str, list[int]] = defaultdict(list)
    for uid, n in toks.items():
        per_ds["math500" if "math500" in uid else "gsm8k"].append(n)
    mean_in = {ds: float(np.mean(v)) for ds, v in per_ds.items()}

    curves = [c for c in json.loads(a.frontier.read_text())["curves"] if c["split"] == "all"]
    idx = {(c["dataset"], c["n"], c["rule"]): c for c in curves}
    rows = []
    for ds in sorted({c["dataset"] for c in curves}):
        maj = [(c["tokens_mean"], c["accuracy"]) for c in curves
               if c["dataset"] == ds and c["rule"] == "majority"]
        for spec in a.scorer:
            name, rest = spec.split("=", 1)
            suffix, params = rest.rsplit(":", 1)
            ratio = float(params) / GEN_PARAMS
            for rule in ("tiebreak", "wvote", "rerank"):
                for n in a.ns:
                    c = idx.get((ds, n, f"{rule}::{suffix}"))
                    if c is None:
                        continue
                    reads = c["scored_mean"] if rule == "tiebreak" else n
                    cost = c["tokens_mean"] + reads * mean_in[ds] * ratio
                    m = majority_at(maj, cost)
                    rows.append({"dataset": ds, "scorer": name, "rule": rule, "n": n,
                                 "accuracy": c["accuracy"], "gen_tokens": c["tokens_mean"],
                                 "verifier_reads": reads, "total_cost": cost,
                                 "majority_at_cost": m,
                                 "delta_vs_majority": None if m is None else c["accuracy"] - m})
                    ms = "out of range" if m is None else f"{m:.3f} ({c['accuracy'] - m:+.3f})"
                    print(f"{ds:8s} {name:6s} {rule:8s} N={n:<3d} acc {c['accuracy']:.3f} "
                          f"cost {cost:7.0f} (gen {c['tokens_mean']:.0f}, reads {reads:.2f}) "
                          f"majority at cost {ms}")
    if a.out:
        a.out.write_text(json.dumps({"mean_input_tokens": mean_in, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
