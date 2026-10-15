"""Problem-text split for step_bert_probe_v0 (S1).

Groups on-policy traces by canonical problem text (NFKC, lowercase, whitespace
collapsed), drops every problem whose text also appears in ProcessBench, and
assigns problems 70/15/15 to train/val/test, stratified by whether any trace of
the problem has an incorrect step. Written once; every later stage reads it.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np


def canon(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def make_split(trace_problem: dict[str, str], trace_has_error: dict[str, bool],
               pb_problems: set[str], seed: int = 1729,
               fracs: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> dict:
    by_prob: dict[str, list[str]] = defaultdict(list)
    for t, p in trace_problem.items():
        if t in trace_has_error:
            by_prob[canon(p)].append(t)
    pb = {canon(p) for p in pb_problems}
    excluded = sorted(p for p in by_prob if p in pb)

    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for stratum in (True, False):
        probs = sorted(p for p in by_prob if p not in pb
                       and any(trace_has_error[t] for t in by_prob[p]) == stratum)
        probs = [probs[i] for i in rng.permutation(len(probs))]
        n_tr = int(round(fracs[0] * len(probs)))
        n_va = int(round(fracs[1] * len(probs)))
        for name, chunk in (("train", probs[:n_tr]), ("val", probs[n_tr:n_tr + n_va]),
                            ("test", probs[n_tr + n_va:])):
            for p in chunk:
                out[name].extend(by_prob[p])
    for k in out:
        out[k] = sorted(out[k])
    out["audit"] = {
        "seed": seed, "fracs": list(fracs),
        "problems": len(by_prob), "pb_overlap_problems": len(excluded),
        "pb_overlap_traces": sum(len(by_prob[p]) for p in excluded),
        "traces": {k: len(out[k]) for k in ("train", "val", "test")},
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pooled", type=Path, required=True)
    p.add_argument("--trace_problems", type=Path, required=True)
    p.add_argument("--pb_problems", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()

    z = np.load(args.pooled)
    has_err: dict[str, bool] = defaultdict(bool)
    for t, y in zip(z["trace_id"].tolist(), z["y"].tolist()):
        has_err[t] |= bool(y)
    tp = {r["trace_id"]: r["problem"] for r in map(json.loads, args.trace_problems.open())}
    missing = sorted(set(has_err) - set(tp))
    if missing:
        raise SystemExit(f"{len(missing)} pooled traces have no problem text, e.g. {missing[:3]}")
    pb = {r["problem"] for r in map(json.loads, args.pb_problems.open())}
    split = make_split(tp, dict(has_err), pb, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split))
    print(json.dumps(split["audit"], indent=2))


if __name__ == "__main__":
    main()
