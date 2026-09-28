#!/usr/bin/env python3
"""A 7B PRM against the generation-state probe at matched compute.

The question: at the compute a PRM rule spends, what does spending the same
compute on more samples plus our probe buy? The probe reads the hidden states
the sampler already computed, so its cost is the head alone; the PRM re-reads
every trace it scores with a 7B forward pass.

Cost per problem is counted two ways, both in units of one generated token of
the policy:

  flops   2 x params per token for every model. A PRM read of T tokens costs
          T x PRM_PARAMS / GEN_PARAMS; the probe head costs its parameter share
          of every generated token.
  gpu     measured seconds. Decode seconds per generated token, PRM seconds
          per input token and head seconds per generated token are passed in
          from the jobs' own logs, then divided by the decode rate.

For each PRM rule and budget N the PRM's cost C is computed, and the probe's
tie-break accuracy is read off its own N = 1..10 cost curve at C by linear
interpolation. A cost past the probe's N = 10 point is reported as out of
range rather than extrapolated. PRM input tokens are the trace's generated
tokens plus the problem, the latter estimated at 3.8 characters per token
because the PRM's tokenised lengths were not saved.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def per_ds(spec: str) -> dict[str, float]:
    """'gsm8k=2.75e-3,math500=3.26e-3' -> {'gsm8k': 0.00275, 'math500': 0.00326}."""
    return {k: float(v) for k, v in (x.split("=") for x in spec.split(","))}


def interp_at(curve: list[tuple[float, float]], x: float) -> float | None:
    xs, ys = zip(*sorted(curve))
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frontier", type=Path, required=True)
    p.add_argument("--run_root", type=Path, required=True,
                   help="pool directory, for the PRM input lengths")
    p.add_argument("--probe", required=True, help="scorer suffix, e.g. probe::<cell>__gen::worst")
    p.add_argument("--prm", required=True, help="scorer suffix, e.g. probe::prm_qwen25_math_7b::worst")
    p.add_argument("--gen_params", type=float, default=8.19e9)
    p.add_argument("--prm_params", type=float, default=7.6e9)
    p.add_argument("--head_params", type=float, default=8.665e6)
    p.add_argument("--s_decode_tok", type=per_ds, required=True,
                   help="measured decode seconds per generated token (one GPU)")
    p.add_argument("--s_prm_tok", type=per_ds, required=True,
                   help="measured PRM seconds per input token (one GPU)")
    p.add_argument("--s_head_tok", type=per_ds, required=True,
                   help="measured probe-head seconds per generated token (one GPU)")
    p.add_argument("--ns", type=int, nargs="+", default=[2, 4, 6, 10])
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    in_toks: dict[str, list[float]] = defaultdict(list)
    for f in sorted(a.run_root.glob("tts_*.shard*_trajectories.jsonl")):
        ds = f.name.split(".")[0].replace("tts_", "")
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                in_toks[ds].append(r.get("n_gen_tokens", 0) + len(r["problem"]) / 3.8)
    mean_in = {ds: float(np.mean(v)) for ds, v in in_toks.items()}

    curves = [c for c in json.loads(a.frontier.read_text())["curves"] if c["split"] == "all"]
    idx = {(c["dataset"], c["n"], c["rule"]): c for c in curves}
    flops = {"prm": a.prm_params / a.gen_params, "head": a.head_params / a.gen_params}

    rows = []
    for ds in sorted({c["dataset"] for c in curves}):
        ratio = {"flops": flops,
                 "gpu": {"prm": a.s_prm_tok[ds] / a.s_decode_tok[ds],
                         "head": a.s_head_tok[ds] / a.s_decode_tok[ds]}}
        for model, r in ratio.items():
            probe_curve, maj_curve = [], []
            for n in range(1, 11):
                c = idx.get((ds, n, f"tiebreak::{a.probe}"))
                m = idx.get((ds, n, "majority"))
                if c:
                    probe_curve.append((c["tokens_mean"] * (1 + r["head"]), c["accuracy"]))
                if m:
                    maj_curve.append((m["tokens_mean"], m["accuracy"]))
            for rule in ("tiebreak", "wvote", "rerank"):
                for n in a.ns:
                    c = idx.get((ds, n, f"{rule}::{a.prm}"))
                    if c is None:
                        continue
                    reads = c["scored_mean"] if rule == "tiebreak" else n
                    cost = c["tokens_mean"] + reads * mean_in[ds] * r["prm"]
                    pa = interp_at(probe_curve, cost)
                    ma = interp_at(maj_curve, cost)
                    rows.append({"dataset": ds, "cost_model": model, "prm_rule": rule, "n": n,
                                 "prm_acc": c["accuracy"], "cost": cost,
                                 "prm_reads": reads,
                                 "probe_tiebreak_at_cost": pa, "majority_at_cost": ma})
                    f = lambda v: "out of range" if v is None else f"{v:.4f}"
                    print(f"{ds:8s} {model:5s} PRM {rule:8s} N={n:<2d} acc {c['accuracy']:.4f} "
                          f"cost {cost:7.0f} | probe at cost {f(pa)} | majority at cost {f(ma)}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"mean_prm_input_tokens": mean_in, "flops_ratios": flops,
                                     "args": {k: str(v) for k, v in vars(a).items()},
                                     "rows": rows}, indent=1))
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
