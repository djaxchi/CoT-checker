#!/usr/bin/env python3
"""Does the off-policy leaderboard predict on-policy usefulness? Rank and size.

Two questions that are routinely conflated, kept apart here.

**Rank transfer.** Does a benchmark put the verifiers in the order their
on-policy behaviour puts them in? Spearman and Kendall, with the interval that
matters: resampling PROBLEMS, not steps and not cells. Every downstream number
is an average over problems, so a problem-level bootstrap is the one that
propagates the uncertainty the measurement actually has.

**Effect-size transfer.** A benchmark can preserve order perfectly while
exaggerating how much the ordering is worth. For the contrasts the leaderboard's
argument rests on, the same gap is reported three ways: on the benchmark, on the
frozen on-policy downstream metric, and (when it exists) on the on-policy-trained
one. Order transferring and magnitude transferring are separate findings and are
never merged into one sentence.

Two units are reported separately and labelled, because a correlation across
representation x learner cells cannot isolate representation quality when the
learner changes along with it:

  cell    all 19 verifier cells, representation and learner both varying
  family  one row per representation, learner held at linear

Both benchmarks are compared on the same footing, since "ProcessBench predicts
this better than PRM800K does" is only meaningful if both were asked.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.analysis.onpolicy_downstream import (  # noqa: E402
    aggregate, auroc, best_of_n_hits, cell_solutions, load_outcomes,
    within_problem_auroc,
)
from scripts.analysis.onpolicy_rank_transfer import kendall, rankdata, spearman  # noqa: E402

# The comparisons the off-policy leaderboard's argument rests on, as
# (name, worse cell, better cell). Learner held fixed inside each pair so the
# contrast is about the representation.
CONTRASTS = [
    ("last_token -> step_mean (same dim, same params)",
     ("last_token", "linear"), ("step_mean", "linear")),
    ("step_delta -> last_token",
     ("step_delta", "linear"), ("last_token", "linear")),
    ("step_mean -> step_stats (5x wider input)",
     ("step_mean", "linear"), ("step_stats", "linear")),
    ("fixed pooling -> learned pooling",
     ("step_mean", "linear"), ("step_tokens", "attn_query")),
]


def spearman_p(rho: float, n: int) -> float:
    """Two-sided p for a rank correlation, via the t approximation.

    Honest about itself: at n=19 this is an approximation and the bootstrap
    interval is the number to read. It is reported because a reviewer will ask.
    """
    if n < 4 or not np.isfinite(rho) or abs(rho) >= 1:
        return float("nan")
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    # survival function of |t| with n-2 df, via the incomplete beta
    df = n - 2
    x = df / (df + t * t)
    return float(_betainc(df / 2, 0.5, x))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta, continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def load_cells(grid_root: Path, scores_name: str, outcomes: dict) -> dict:
    """(rep, learner) -> per-seed dicts of every metric this report needs."""
    per: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in sorted(grid_root.iterdir()):
        rj, sj = d / "results.json", d / f"pb_step_scores_{scores_name}.jsonl"
        if not (rj.exists() and sj.exists()):
            continue
        res = json.loads(rj.read_text())
        if res["rep"].startswith("sae"):
            continue
        groups = cell_solutions(sj, outcomes)
        if not groups:
            continue
        hits = best_of_n_hits(groups, "worst_step")
        ys, ss = [], []
        for sols in groups.values():
            for s in sols:
                ys.extend([0 if s["correct"] else 1] * len(s["scores"]))
                ss.extend(s["scores"])
        per[(res["rep"], res["learner"])].append({
            "prm800k_auroc": float(res["in_domain"]["auroc"]),
            "onpolicy_step_auroc": auroc(np.array(ys), np.array(ss)),
            "onpolicy_within_auroc": within_problem_auroc(groups, "worst_step"),
            "bon_hits": hits,
            "bon": float(np.mean(list(hits.values()))),
        })
    return dict(per)


def mean_metric(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows]))


def bootstrap_rank(x: list[float], cells: list[tuple[str, str]],
                   per: dict, problems: list[str], key: str, b: int, seed: int) -> dict:
    """Spearman CI from resampling PROBLEMS.

    Every downstream metric is a mean over problems, so this is the resampling
    that reflects how the numbers were produced. Resampling cells would answer a
    different question (how much the correlation depends on the grid's
    composition) and resampling steps would understate it.
    """
    rng = np.random.default_rng(seed)
    hits = {c: {p: np.mean([r["bon_hits"].get(p, np.nan) for r in per[c]])
                for p in problems} for c in cells}
    vals = []
    for _ in range(b):
        draw = rng.choice(len(problems), len(problems), replace=True)
        picked = [problems[i] for i in draw]
        y = [float(np.nanmean([hits[c][p] for p in picked])) for c in cells]
        if len(set(np.round(y, 12))) < 3:
            continue
        vals.append(spearman(x, y))
    v = np.array([z for z in vals if np.isfinite(z)])
    if v.size == 0:
        return {"ci95": [float("nan")] * 2, "n_resamples": 0}
    return {"ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
            "sd": float(v.std()), "n_resamples": int(v.size)}


def report_unit(name: str, cells: list[tuple[str, str]], per: dict,
                problems: list[str], b: int, seed: int) -> dict:
    x_prm = [mean_metric(per[c], "prm800k_auroc") for c in cells]
    x_pb = [PB.get(f"{c[0]} x {c[1]}", float("nan")) for c in cells]
    ys = {k: [mean_metric(per[c], k) for c in cells]
          for k in ("onpolicy_step_auroc", "onpolicy_within_auroc", "bon")}
    n = len(cells)
    out = {"unit": name, "n_cells": n, "correlations": {}}
    print(f"\n=== rank transfer, unit = {name} (n={n}) ===")
    print(f"{'benchmark':<14}{'against':<26}{'Spearman':>10}{'Kendall':>9}{'p':>8}"
          f"{'  problem-bootstrap 95% CI'}")
    for bname, x in (("PRM800K", x_prm), ("ProcessBench", x_pb)):
        if not np.isfinite(x).all():
            print(f"{bname:<14}{'(missing scores)':<26}")
            continue
        for yk, y in ys.items():
            rho, tau = spearman(x, y), kendall(x, y)
            ci = bootstrap_rank(x, cells, per, problems, yk, b, seed) \
                if yk == "bon" else {"ci95": [float("nan")] * 2}
            cis = (f"  [{ci['ci95'][0]:+.3f}, {ci['ci95'][1]:+.3f}]"
                   if np.isfinite(ci["ci95"][0]) else "")
            print(f"{bname:<14}{yk:<26}{rho:>+10.3f}{tau:>+9.3f}"
                  f"{spearman_p(rho, n):>8.3f}{cis}")
            out["correlations"][f"{bname}|{yk}"] = {
                "spearman": rho, "kendall": tau, "p": spearman_p(rho, n), **ci}
    rho = spearman(ys["onpolicy_step_auroc"], ys["bon"])
    out["correlations"]["onpolicy_step|bon"] = {"spearman": rho,
                                                "kendall": kendall(ys["onpolicy_step_auroc"], ys["bon"])}
    print(f"{'on-policy':<14}{'step AUROC vs best-of-N':<26}{rho:>+10.3f}"
          f"{kendall(ys['onpolicy_step_auroc'], ys['bon']):>+9.3f}")
    return out


def report_effect_sizes(per: dict) -> list[dict]:
    print("\n=== effect-size transfer ===")
    print(f"{'contrast':<44}{'d PRM800K':>11}{'d PB':>8}{'d step':>9}{'d BoN':>8}")
    rows = []
    for name, w, b_ in CONTRASTS:
        if w not in per or b_ not in per:
            continue
        d_prm = mean_metric(per[b_], "prm800k_auroc") - mean_metric(per[w], "prm800k_auroc")
        kw, kb = f"{w[0]} x {w[1]}", f"{b_[0]} x {b_[1]}"
        d_pb = PB.get(kb, float("nan")) - PB.get(kw, float("nan"))
        d_step = mean_metric(per[b_], "onpolicy_step_auroc") - mean_metric(per[w], "onpolicy_step_auroc")
        d_bon = mean_metric(per[b_], "bon") - mean_metric(per[w], "bon")
        rows.append({"contrast": name, "d_prm800k": d_prm, "d_processbench": d_pb,
                     "d_onpolicy_step": d_step, "d_best_of_n": d_bon})
        print(f"{name:<44}{d_prm:>+11.3f}{d_pb:>+8.3f}{d_step:>+9.3f}{d_bon:>+8.3f}")
    print("\nORDER transfers if the signs agree; MAGNITUDE transfers if the sizes "
          "do. They are different claims and a benchmark can pass the first while "
          "failing the second.")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid_root", required=True, type=Path)
    p.add_argument("--scores_name", default="onpolicy_verifier")
    p.add_argument("--outcomes", required=True, type=Path)
    p.add_argument("--processbench", required=True, type=Path,
                   help="offpolicy_f1pb.json: calib-20 F1_PB per cell.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    global PB
    PB = json.loads(args.processbench.read_text())
    outcomes = load_outcomes(args.outcomes)
    per = load_cells(args.grid_root, args.scores_name, outcomes)
    if len(per) < 4:
        raise SystemExit(f"only {len(per)} dense cells scored on {args.scores_name}")
    problems = sorted({o["problem_id"] for o in outcomes.values()})

    cells = sorted(per)
    family = [c for c in cells if c[1] == "linear"] + \
             [c for c in cells if c == ("step_tokens", "attn_query")]
    rep = {"n_problems": len(problems), "scores_name": args.scores_name,
           "units": [report_unit("cell (19 verifier cells)", cells, per, problems,
                                 args.bootstrap, args.seed)]}
    if len(family) >= 4:
        rep["units"].append(report_unit("family (linear learner held fixed)",
                                        family, per, problems, args.bootstrap,
                                        args.seed))
    rep["effect_sizes"] = report_effect_sizes(per)
    rep["cells"] = {f"{r} x {l}": {
        "prm800k_auroc": mean_metric(per[(r, l)], "prm800k_auroc"),
        "processbench_f1pb": PB.get(f"{r} x {l}"),
        "onpolicy_step_auroc": mean_metric(per[(r, l)], "onpolicy_step_auroc"),
        "onpolicy_within_auroc": mean_metric(per[(r, l)], "onpolicy_within_auroc"),
        "best_of_n": mean_metric(per[(r, l)], "bon"),
        "n_seeds": len(per[(r, l)])} for (r, l) in cells}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2))
        print(f"\n[transfer] wrote {args.out}")


if __name__ == "__main__":
    main()
