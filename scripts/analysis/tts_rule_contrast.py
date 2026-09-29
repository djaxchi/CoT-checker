#!/usr/bin/env python3
"""Paired cluster bootstrap of the gap between two selection rules.

`tts_build_frontier.py` reports each rule's own interval, and two overlapping
intervals do not settle whether one rule beats the other: the rules are applied
to the *same* draws on the *same* problems, so the comparison that matters is
the paired one. This recomputes the per-problem outcomes for a short list of
rules and bootstraps the difference directly, resampling question text for the
reason REPORT.md §20.12 records.

It exists because sprint 8's question is a ranking question. "Our tie-break
leads the field's weighted vote by 0.004" is a claim about a difference, and a
difference needs its own interval.
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

from scripts.analysis.tts_build_frontier import (AGGS, CONF_WITH_REWARD,  # noqa: E402
                                                 load_many, read_jsonl)
from src.analysis.tts_frontier import quality_from, simulate_problem  # noqa: E402
from src.eval.math_grade import answer_key  # noqa: E402


def paired_bootstrap(a: np.ndarray, b: np.ndarray, clusters: list[str],
                     draws: int = 4000, seed: int = 915) -> tuple:
    """Interval on mean(a) - mean(b), resampling whole questions."""
    diff = a - b
    by = defaultdict(list)
    for i, c in enumerate(clusters):
        by[c].append(i)
    blocks = [np.array(v) for v in by.values()]
    sums = np.array([diff[x].sum() for x in blocks])
    sizes = np.array([x.size for x in blocks], dtype=float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(blocks), size=(draws, len(blocks)))
    boot = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=Path, required=True)
    p.add_argument("--stems", nargs="+", default=["tts_gsm8k", "tts_math500"])
    p.add_argument("--scorers", nargs="+", required=True,
                   help="scorer ids as the frontier names them")
    p.add_argument("--baseline", required=True, help="rule every other is measured against")
    p.add_argument("--ns", type=int, nargs="+", default=[2, 4, 10])
    p.add_argument("--n_orders", type=int, default=32)
    p.add_argument("--betas", type=float, nargs="*", default=[1.0])
    p.add_argument("--etas", type=float, nargs="*", default=[0.5])
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    results = []
    for stem in a.stems:
        tr = load_many(sorted(a.run_root.glob(f"{stem}.shard*_trajectories.jsonl")))
        conf = {r["traj_uid"]: r["rules"]
                for r in load_many(sorted(a.run_root.glob(f"{stem}.shard*_conf.jsonl")))}
        cells: dict[str, dict[str, list[float]]] = defaultdict(dict)
        for f in sorted((a.run_root / "scores").glob(f"{stem}__*.shard*.jsonl")):
            cell = f.name.split("__", 1)[1].rsplit(".shard", 1)[0]
            for r in read_jsonl(f):
                cells[cell][r["traj_uid"]] = r["scores"]

        by_problem: dict[str, list[dict]] = defaultdict(list)
        for r in tr:
            if r.get("gradeable"):
                r["_conf"] = conf.get(r["traj_uid"], {})
                r["_cells"] = {c: v.get(r["traj_uid"]) for c, v in cells.items()}
                by_problem[r["fork_id"]].append(r)

        per_problem: dict[str, list[dict]] = {}
        qhash: dict[str, str] = {}
        rng = np.random.default_rng(915)
        for pid, cands in by_problem.items():
            cands.sort(key=lambda r: r["traj_uid"])
            answers = [(r["answer_key"] if "answer_key" in r else answer_key(r.get("pred"))) for r in cands]
            correct = [bool(r["correct"]) for r in cands]
            tokens = [int(r.get("n_gen_tokens", 0)) for r in cands]
            qualities, rewards = {}, {}
            for sc in a.scorers:
                kind, rest = sc.split("::", 1)
                if kind == "probe":
                    cell, agg = rest.rsplit("::", 1)
                    vals = [AGGS[agg](r["_cells"][cell]) if r["_cells"].get(cell)
                            else float("nan") for r in cands]
                    qualities[sc] = quality_from(vals, False)
                    rewards[sc] = 1.0 - np.asarray(vals, dtype=float)
                else:
                    vals = [r["_conf"].get(rest, float("nan")) for r in cands]
                    qualities[sc] = quality_from(vals, True)
                    if rest in CONF_WITH_REWARD:
                        rewards[sc] = np.asarray(vals, dtype=float)
            per_problem[pid] = simulate_problem(
                answers, correct, tokens, qualities, a.ns, a.n_orders, rng,
                rewards=rewards, betas=a.betas, etas=a.etas)
            qhash[pid] = cands[0].get("question_hash", pid)

        rules = sorted({k for rs in per_problem.values() for r in rs for k in r
                        if k not in ("order", "n", "tokens", "tied", "n_scored_lazy")})
        for n in a.ns:
            pids = [p for p in per_problem if any(r["n"] == n for r in per_problem[p])]
            clusters = [qhash[p] for p in pids]
            col = {}
            for rule in rules:
                col[rule] = np.array([
                    float(np.mean([r[rule] for r in per_problem[p] if r["n"] == n]))
                    for p in pids])
            base = col[a.baseline]
            for rule in rules:
                if rule == a.baseline:
                    continue
                m, lo, hi = paired_bootstrap(base, col[rule], clusters)
                results.append({"dataset": stem.replace("tts_", ""), "n": n,
                                "baseline": a.baseline, "rule": rule,
                                "baseline_acc": float(base.mean()),
                                "rule_acc": float(col[rule].mean()),
                                "delta": m, "ci95": [lo, hi],
                                "n_problems": len(pids)})
                star = "" if lo <= 0 <= hi else "  *"
                print(f"{stem.replace('tts_',''):8s} N={n:<3d} {a.baseline} - {rule}: "
                      f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}", flush=True)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=1))
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
