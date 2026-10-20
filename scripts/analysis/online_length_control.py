#!/usr/bin/env python3
"""online_length_control_v0 readings (docs/online_length_control_v0_plan.md).

For the probe arm (`reject`) and the length arm (`reject_length`): the share of
swaps (a step whose first draft was condemned and a later draft kept) where the
PRM, scoring every draft without steering, prefers the kept draft; and accuracy
against the blind arm, paired by problem. Usage: online_length_control.py RUNS_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_v4_control import boot, extra_draws, load  # noqa: E402

PRM = "prm_qwen25_math_7b"


def swaps(R: dict) -> list[tuple[float, int, str]]:
    out = []
    for pid, r in R.items():
        for texts, scores, k in zip(r["draft_texts"], r["draft_scores"], r["kept_attempt"]):
            if k > 0 and scores[0] and scores[k]:
                out.append((scores[0][PRM] - scores[k][PRM], len(texts[0]) - len(texts[k]), pid))
    return out


def clustered(sw, rng, n=2000):
    d = np.array([s[0] for s in sw])
    pids = sorted({s[2] for s in sw})
    idx = {p: [] for p in pids}
    for j, s in enumerate(sw):
        idx[s[2]].append(j)
    bs = [(d[[j for p in rng.choice(pids, len(pids)) for j in idx[p]]] > 0).mean() for _ in range(n)]
    return [float((d > 0).mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), len(sw), len(pids)]


def main() -> None:
    root = sys.argv[1]
    A = {k: load(root, k) for k in ("plain", "reject_blind", "reject", "reject_length")}
    ids = sorted(set.intersection(*(set(v) for v in A.values())))
    rng = np.random.default_rng(0)
    out = {"n": len(ids), "arms": {}, "prm_prefers_kept": {}, "diffs": {}}
    for k, R in A.items():
        Ri = {i: R[i] for i in ids}
        out["arms"][k] = {"acc": float(np.mean([r["correct"] for r in Ri.values()])),
                          "tokens": float(np.mean([r.get("gen_tokens") or 0 for r in Ri.values()])),
                          "steps": float(np.mean([r["n_steps"] for r in Ri.values()])),
                          "extra_draws_per_step": float(np.mean([extra_draws(r) if "attempts_per_step" in r else 0.0 for r in Ri.values()]))}
        if k in ("reject", "reject_length"):
            sw = swaps(Ri)
            ld = np.array([s[1] for s in sw])
            out["prm_prefers_kept"][k] = {
                "all": clustered(sw, rng),
                "kept_longer_by_15": clustered([s for s, l in zip(sw, ld) if l < -15], rng),
                "within_15": clustered([s for s, l in zip(sw, ld) if abs(l) < 15], rng),
                "kept_shorter_by_15": clustered([s for s, l in zip(sw, ld) if l > 15], rng)}
    acc = {k: np.array([float(A[k][i]["correct"]) for i in ids]) for k in A}
    for x, y in (("reject", "reject_blind"), ("reject_length", "reject_blind"), ("reject", "reject_length")):
        d = acc[x] - acc[y]
        out["diffs"][f"{x} - {y}"] = {"mean_ci": boot(d), "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
