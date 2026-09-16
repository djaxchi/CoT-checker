#!/usr/bin/env python3
"""Join traces, logprobs and verifier scores into the explorer's input.

Three jobs write three kinds of atom and none of them aggregates:

  tts_*.shard*_trajectories.jsonl   one row per sampled solution
  tts_*.shard*_conf.jsonl           token-confidence statistics per solution
  scores/*__<cell>.shard*.jsonl     the verifier's per-step scores per solution

This joins them by trajectory id, replays each problem's candidates in random
draw orders, and applies every selection rule at every budget. The output is a
tidy set of curves the HTML explorer reads: pick a metric, pick one
hyperparameter for the x-axis, fix the rest.

Aggregation happens here and only here. The atoms stay on disk untouched, so a
rule invented next week can be added by rerunning this rather than by
regenerating anything on a GPU.

The unit of analysis is the problem after averaging over draw orders. Doing it
in that order matters: averaging orders first keeps order-to-order variance out
of the interval, which would otherwise be mistaken for problem-to-problem
variance and make every curve look tighter than it is. Intervals resample
question text, not problem id, for the reason §20.12 records.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.tts_frontier import quality_from, simulate_problem  # noqa: E402
from src.eval.math_grade import normalize_answer  # noqa: E402

# Aggregations of a per-step score into one number per solution. `worst` is the
# primary: it is ReProbe's Q_offline and what §20.6 found ranks identically.
AGGS = {"worst": max, "mean": lambda s: float(np.mean(s)), "last": lambda s: s[-1]}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_many(paths: list[Path]) -> list[dict]:
    return [r for p in paths for r in read_jsonl(p)]


def bootstrap_mean(values: np.ndarray, clusters: list[str], draws: int = 2000,
                   seed: int = 915) -> tuple[float, float, float]:
    """Cluster bootstrap of a mean, resampling whole question texts."""
    by = defaultdict(list)
    for i, c in enumerate(clusters):
        by[c].append(i)
    blocks = [np.array(v) for v in by.values()]
    sums = np.array([values[b].sum() for b in blocks])
    sizes = np.array([b.size for b in blocks], dtype=float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(blocks), size=(draws, len(blocks)))
    boot = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", type=Path, required=True)
    p.add_argument("--stems", nargs="+", default=["tts_gsm8k", "tts_math500"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    p.add_argument("--n_orders", type=int, default=64)
    p.add_argument("--conf_rules", nargs="+",
                   default=["bottom10_group_w32", "mean_token_conf",
                            "mean_sampled_logprob", "answer_token_margin"])
    p.add_argument("--seed", type=int, default=915)
    a = p.parse_args()

    rows_by_stem: dict[str, list[dict]] = {}
    for stem in a.stems:
        tr = load_many(sorted(a.run_root.glob(f"{stem}.shard*_trajectories.jsonl")))
        conf = {r["traj_uid"]: r["rules"]
                for r in load_many(sorted(a.run_root.glob(f"{stem}.shard*_conf.jsonl")))}
        cells: dict[str, dict[str, list[float]]] = defaultdict(dict)
        for f in sorted((a.run_root / "scores").glob(f"{stem}__*.shard*.jsonl")):
            cell = f.name.split("__", 1)[1].rsplit(".shard", 1)[0]
            for r in read_jsonl(f):
                cells[cell][r["traj_uid"]] = r["scores"]
        print(f"[load] {stem}: {len(tr)} traces, {len(conf)} with confidence, "
              f"{ {c: len(v) for c, v in cells.items()} } scored", flush=True)
        for r in tr:
            r["_conf"] = conf.get(r["traj_uid"], {})
            r["_cells"] = {c: v.get(r["traj_uid"]) for c, v in cells.items()}
        rows_by_stem[stem] = tr

    curves: list[dict] = []
    for stem, tr in rows_by_stem.items():
        by_problem: dict[str, list[dict]] = defaultdict(list)
        for r in tr:
            if r.get("gradeable"):
                by_problem[r["fork_id"]].append(r)

        # Every scorer, named as it will appear in the explorer's dropdown.
        scorer_names: list[str] = []
        for cell in sorted({c for r in tr for c in r["_cells"]}):
            scorer_names += [f"probe::{cell}::{agg}" for agg in AGGS]
        scorer_names += [f"conf::{c}" for c in a.conf_rules]

        per_problem: dict[str, list[dict]] = defaultdict(list)
        meta: dict[str, dict] = {}
        rng = np.random.default_rng(a.seed)
        for pid, cands in by_problem.items():
            cands.sort(key=lambda r: r["traj_uid"])
            answers = [normalize_answer(r.get("pred")) for r in cands]
            correct = [bool(r["correct"]) for r in cands]
            tokens = [int(r.get("n_gen_tokens", 0)) for r in cands]
            qualities: dict[str, np.ndarray] = {}
            for cell in sorted({c for r in cands for c in r["_cells"]}):
                for agg, fn in AGGS.items():
                    vals = [fn(r["_cells"][cell]) if r["_cells"].get(cell)
                            else float("nan") for r in cands]
                    # Probe scores are suspicion: lower is a better candidate.
                    qualities[f"probe::{cell}::{agg}"] = quality_from(vals, False)
            for c in a.conf_rules:
                vals = [r["_conf"].get(c, float("nan")) for r in cands]
                # Orientation for the confidence family is fixed on the
                # exploratory half and recorded; higher is treated as better
                # here, which src/analysis/token_confidence.py explains is not
                # derivable from DeepConf's Eq 2 and must be measured.
                qualities[f"conf::{c}"] = quality_from(vals, True)
            rows = simulate_problem(answers, correct, tokens, qualities,
                                    a.ns, a.n_orders, rng)
            for r in rows:
                per_problem[pid].append(r)
            meta[pid] = {"question_hash": cands[0].get("question_hash", pid),
                         "split": cands[0].get("split"),
                         "level": cands[0].get("level")}

        rules = ["majority", "oracle", "pass1"] + \
                [f"tiebreak::{s}" for s in scorer_names] + \
                [f"rerank::{s}" for s in scorer_names]
        for split in (None, "expl", "conf"):
            pids = [p for p in per_problem
                    if split is None or meta[p]["split"] == split]
            if not pids:
                continue
            clusters = [meta[p]["question_hash"] for p in pids]
            for n in a.ns:
                for rule in rules:
                    vals, toks, scored = [], [], []
                    for pid in pids:
                        rs = [r for r in per_problem[pid] if r["n"] == n]
                        if not rs:
                            continue
                        vals.append(float(np.mean([r[rule] for r in rs])))
                        toks.append(float(np.mean([r["tokens"] for r in rs])))
                        scored.append(float(np.mean([r["n_scored_lazy"] for r in rs])))
                    if not vals:
                        continue
                    m, lo, hi = bootstrap_mean(np.array(vals), clusters)
                    curves.append({
                        "dataset": stem.replace("tts_", ""),
                        "split": split or "all", "n": n, "rule": rule,
                        "accuracy": m, "ci95": [lo, hi],
                        "tokens_mean": float(np.mean(toks)),
                        "scored_mean": float(np.mean(scored)),
                        "n_problems": len(vals),
                        "n_questions": len(set(clusters))})
        print(f"[curves] {stem}: {len(curves)} rows so far", flush=True)

    out = {"created_at": datetime.now(timezone.utc).isoformat(),
           "ns": a.ns, "n_orders": a.n_orders, "curves": curves}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out))
    print(f"[out] {a.out}  ({len(curves)} curve rows)")


if __name__ == "__main__":
    main()
