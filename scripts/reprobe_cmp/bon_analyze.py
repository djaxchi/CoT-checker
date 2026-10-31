#!/usr/bin/env python3
"""Best-of-N accuracy on ReProbe's released pools (reprobe_comparison_v2, B1).

Joins the step scores of every arm (written by slurm/reprobe_cmp_tamia.sh STAGE=bon) with
our grading (bon_d0.py graded.jsonl). A candidate's score is its highest step P(incorrect),
the paper's minimum-step aggregation; best-of-N keeps the candidate with the lowest score
(first on ties). Arms:

  NAME=rp:DIR[,DIR]   ReProbe heads, DIR/bon_<set>.jsonl {uid, p}, one DIR per seed
  NAME=rt:DIR[,DIR]   read tokens, DIR/bon_<set>.shard0.{npz,meta.jsonl}
  NAME=stored:COLUMN  a trace score stored in the pool (lower = better)

Per set: accuracy (mean over seeds), with pass@1, pass@N and majority vote; the mean over
sets; --pairs A-B paired bootstrap over questions (within each set; for the mean every set
is resampled), on per-question accuracy averaged over seeds. Writes summary.{json,md}.

    python scripts/reprobe_cmp/bon_analyze.py --bon $R/bon --out $R/bon/analysis \\
        --arm RT-self=rt:$U/rt/rt_self_seed0,... RP-rel-self=rp:$U/released/scores/rel_self \\
              PRM7B=stored:Qwen/Qwen2.5-Math-PRM-7B --pairs RT-self-RP-rel-self
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reprobe_cmp.analyze import jl, parse_arm, seed_scores  # noqa: E402


def trace_scores(sc: dict) -> dict:
    """{trace_id: max step P(incorrect)} from {uid: p}."""
    out = defaultdict(lambda: -np.inf)
    for uid, p in sc.items():
        t = uid.split("::")[0]
        out[t] = max(out[t], p)
    return dict(out)


def pick(qi: np.ndarray, score: np.ndarray) -> dict:
    """{question: row index of its lowest-scoring candidate (first on ties)}."""
    best = {}
    for i in np.lexsort((np.arange(len(qi)), score)):
        best.setdefault(int(qi[i]), int(i))
    return best


def within_auc(qi: np.ndarray, y: np.ndarray, score: np.ndarray) -> float:
    """Mean over questions with both outcomes of P(a correct candidate scores below an
    incorrect one), ties half: how well the arm ranks a question's own candidates."""
    out = []
    for q in np.unique(qi):
        m = qi == q
        c, w = score[m][y[m] == 1], score[m][y[m] == 0]
        if len(c) and len(w):
            out.append(((c[:, None] < w[None]).mean() + 0.5 * (c[:, None] == w[None]).mean()))
    return float(np.mean(out)) if out else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bon", type=Path, required=True, help="dir with <set>.graded.jsonl and inputs/<set>.meta.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sets", default="math,gsm8k,strategyqa,scienceqa,calendar")
    p.add_argument("--arm", nargs="+", required=True)
    p.add_argument("--pairs", nargs="*", default=[])
    p.add_argument("--n_boot", type=int, default=2000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sets = a.sets.split(",")
    arms = {}
    for spec in a.arm:
        name, kind, dirs = parse_arm(spec)
        arms[name] = (kind, dirs)

    table, perq = defaultdict(dict), defaultdict(dict)  # perq[arm][set] = per-question accuracy
    for s in sets:
        ok = {(g["qi"], g["k"]): (g["ok"], g["key"]) for g in jl(a.bon / f"{s}.graded.jsonl")}
        meta = jl(a.bon / "inputs" / f"{s}.meta.jsonl")
        tid = [m["trace_id"] for m in meta]
        qi = np.array([m["qi"] for m in meta])
        y = np.array([ok[(m["qi"], m["k"])][0] for m in meta], dtype=float)
        keys = [ok[(m["qi"], m["k"])][1] for m in meta]
        keys = [k if isinstance(k, str) else None for k in keys]  # no answer: None (NaN in older files)
        qs = sorted(set(qi.tolist()))
        by_q = defaultdict(list)
        for i, q in enumerate(qi):
            by_q[int(q)].append(i)
        maj = []
        for q in qs:
            c = Counter(keys[i] for i in by_q[q] if keys[i] is not None)
            maj.append(y[next(i for i in by_q[q] if keys[i] == c.most_common(1)[0][0])] if c else 0.0)
        base = {"pass@1": np.array([y[by_q[q]].mean() for q in qs]),
                "pass@N": np.array([y[by_q[q]].max() for q in qs]), "majority": np.array(maj)}
        for k, v in base.items():
            table[k][s] = 100 * float(v.mean())
            perq[k][s] = v
        for name, (kind, dirs) in arms.items():
            accs, wauc = [], []
            for d in dirs:
                if kind == "stored":
                    score = np.array([m["stored"][d] for m in meta])
                else:
                    sc = seed_scores(kind, d, f"bon_{s}", a.bon)
                    if sc is None:
                        raise SystemExit(f"{name}: no scores for bon_{s} in {d}")
                    ts = trace_scores(sc)
                    score = np.array([ts.get(t, np.inf) for t in tid])
                best = pick(qi, score)
                accs.append(np.array([y[best[q]] for q in qs]))
                wauc.append(within_auc(qi, y, score))
            v = np.mean(accs, 0)
            table[name][s] = 100 * float(v.mean())
            table[name][f"{s}_within_auc"] = float(np.mean(wauc))
            table[name][f"{s}_seeds"] = [100 * float(x.mean()) for x in accs]
            perq[name][s] = v
    for name in table:
        table[name]["mean"] = float(np.mean([table[name][s] for s in sets]))
        if f"{sets[0]}_within_auc" in table[name]:
            table[name]["within_auc_mean"] = float(np.mean([table[name][f"{s}_within_auc"] for s in sets]))

    rng = np.random.default_rng(0)
    boots = []
    for pair in a.pairs:
        A, B = next((x, z) for x in perq for z in perq if f"{x}-{z}" == pair)
        res = {"pair": pair}
        dm = np.zeros(a.n_boot)
        for s in sets:
            d = perq[A][s] - perq[B][s]
            idx = rng.integers(0, len(d), (a.n_boot, len(d)))
            bs = 100 * d[idx].mean(1)
            dm += bs / len(sets)
            res[s] = {"delta": 100 * float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        res["mean"] = {"delta": float(np.mean([res[s]["delta"] for s in sets])),
                       "ci95": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
                       "frac_le0": float((dm <= 0).mean())}
        boots.append(res)

    (a.out / "summary.json").write_text(json.dumps({"table": table, "pairs": boots}, indent=1))
    lines = ["| arm | " + " | ".join(sets) + " | mean |", "|---" * (len(sets) + 2) + "|"]
    for name, row in table.items():
        lines.append(f"| {name} | " + " | ".join(f"{row[s]:.1f}" for s in sets) + f" | {row['mean']:.1f} |")
    lines += ["", "within-question AUROC (a correct candidate scores below an incorrect one)", "",
              "| arm | " + " | ".join(sets) + " | mean |", "|---" * (len(sets) + 2) + "|"]
    for name, row in table.items():
        if "within_auc_mean" in row:
            lines.append(f"| {name} | " + " | ".join(f"{row[f'{s}_within_auc']:.3f}" for s in sets)
                         + f" | {row['within_auc_mean']:.3f} |")
    if boots:
        lines += ["", "| pair | " + " | ".join(sets) + " | mean |", "|---" * (len(sets) + 2) + "|"]
        for r in boots:
            lines.append(f"| {r['pair']} | " + " | ".join(
                f"{r[s]['delta']:+.1f} [{r[s]['ci95'][0]:+.1f}, {r[s]['ci95'][1]:+.1f}]" for s in sets + ["mean"]) + " |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
