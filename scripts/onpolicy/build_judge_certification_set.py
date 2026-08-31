#!/usr/bin/env python3
"""Sample the human-labelled traces the judge is certified on.

ProcessBench, all four subsets, at their natural error prevalence. Prevalence is
kept rather than balanced because F1_PB is built from Acc_error and Acc_correct
separately, and reading a judge's number against the benchmark everyone else
reports means sampling the benchmark, not a reshaped version of it.

Traces are shuffled per subset under a fixed seed and the counts are recorded, so
the certification set is reproducible and the same one every judge is scored on.
Judges are only comparable if they answered the same questions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, write_jsonl  # noqa: E402
from scripts.encode_processbench_token_store import load_traces  # noqa: E402


def sample_subset(traces: list[dict], n: int, seed: int) -> list[dict]:
    import random
    rng = random.Random(seed)
    idx = list(range(len(traces)))
    rng.shuffle(idx)
    return [traces[i] for i in idx[:n]] if n > 0 else [traces[i] for i in idx]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pb_dir", required=True, type=Path)
    p.add_argument("--subsets", nargs="+",
                   default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--n_per_subset", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=40,
                   help="Drop traces too long to fit a judge prompt; counted.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    out, tally = [], Counter()
    for sub in args.subsets:
        f = args.pb_dir / f"processbench_{sub}.jsonl"
        if not f.exists():
            print(f"[cert] missing {f}, skipping", flush=True)
            continue
        traces = load_traces(f)
        keep = [t for t in traces if len(t["steps"]) <= args.max_steps]
        tally[f"{sub}_dropped_long"] = len(traces) - len(keep)
        picked = sample_subset(keep, args.n_per_subset, args.seed)
        for t in picked:
            t["pb_subset"] = sub
            out.append(t)
        tally[f"{sub}_n"] = len(picked)
        tally[f"{sub}_error"] = sum(1 for t in picked if int(t["label"]) != -1)
    write_jsonl(args.out, out)
    manifest = {"counts": dict(sorted(tally.items())), "n": len(out),
                "n_error": sum(1 for t in out if int(t["label"]) != -1),
                "seed": args.seed, "n_per_subset": args.n_per_subset,
                "max_steps": args.max_steps, "source": str(args.pb_dir),
                "code_commit": git_commit()}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    for k, v in manifest["counts"].items():
        print(f"[cert] {k:28s} {v}")
    print(f"[cert] {len(out)} traces, {manifest['n_error']} with an error -> {args.out}")


if __name__ == "__main__":
    main()
