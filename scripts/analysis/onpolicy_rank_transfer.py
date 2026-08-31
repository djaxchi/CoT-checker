#!/usr/bin/env python3
"""Does the representation ranking survive the change of policy?

The claim this arm can make is about **rank**. Absolute F1 is not comparable
across the two arms: different text distribution, different prevalence, different
number of traces, and the project rule is that F1 is never compared across sets
of differing prevalence. So every number here is a rank statistic or a
within-arm contrast.

Four things it reports, and why each is needed for the claim to mean anything.

**The correlation.** Spearman and Kendall between the two arms over the shared
cells, on seed-averaged F1_PB at calib-20, recomputed from saved per-trace scores
under exactly the protocol the leaderboard uses (imported, not reimplemented).

**A reliability ceiling.** A correlation of 0.7 is a different claim depending on
whether the measurement can reach 0.95 or only 0.75. Within the off-policy arm
alone, one seed's ranking already disagrees with another's; the split-half
correlation across seeds is what the transfer correlation should be read against,
and it is the thing the earlier backbone-transfer result (Spearman 0.919) never
reported. A transfer correlation at the ceiling means the ranking survived as
well as it survives being remeasured.

**Two uncertainties, because there are two.** Resampling cells says how much the
correlation depends on which cells happen to be in the grid, which matters at
n=19. Resampling seeds says how much it depends on training noise. Neither
subsumes the other and one number alone would be misleading.

**The load-bearing contrasts, one at a time.** A high correlation can hide the
particular comparison the argument rests on flipping, since one pair out of 19
barely moves Spearman. Each contrast is reported with its seed spread in both
arms, plus a sign test over the set.

Optionally restricts to traces whose steps are of comparable length in both arms
(`--length_meta`), because on-policy steps are shorter by construction and step
length was already flagged as a minor confound in the probe-anatomy work.
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

from merge_rep_grid_leaderboard import (  # noqa: E402
    CALIB_SIZE, PB_SUBSETS, calib20_subset, load_traces,
)

# The comparisons the leaderboard's argument rests on. Each is (name, worse,
# better, off-policy gap), taken from experiments/unified_harness_7b/leaderboard.md.
CONTRASTS = [
    ("last_token -> step_mean (same dim, same params)",
     ("last_token", "linear"), ("step_mean", "linear"), 0.050),
    ("step_delta -> last_token",
     ("step_delta", "linear"), ("last_token", "linear"), 0.024),
    ("step_mean -> step_stats (5x wider input)",
     ("step_mean", "linear"), ("step_stats", "linear"), 0.042),
    ("fixed pooling -> learned pooling",
     ("step_mean", "linear"), ("step_tokens", "attn_query"), 0.089),
]


# ---------------------------------------------------------------------------
# Rank statistics, written out because the cluster venv has no scipy
# ---------------------------------------------------------------------------

def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not invent an ordering."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    xs = np.asarray(x)[order]
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(np.asarray(a)), rankdata(np.asarray(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def kendall(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else float("nan")


def sign_test_p(successes: int, n: int) -> float:
    """Two-sided exact binomial p at q=0.5. n is small here, so exact is free."""
    if n == 0:
        return float("nan")
    def tail(k):
        return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    k = max(successes, n - successes)
    return min(1.0, 2 * tail(k))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def cell_key(res: dict) -> tuple[str, str]:
    return res["rep"], res["learner"]


def restrict(traces_path: Path, keep_ids: set[str] | None):
    """Traces from a scores file, optionally restricted to a set of ids."""
    if keep_ids is None:
        return load_traces(traces_path)
    out = []
    for line in traces_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in keep_ids:
            out.append((int(r["label"]), [float(x) for x in r["scores"]]))
    return out


def offpolicy_score(cell_dir: Path, keep_ids: set[str] | None) -> float | None:
    """F1_PB at calib-20, averaged over the four ProcessBench subsets."""
    vals = []
    for sub in PB_SUBSETS:
        f = cell_dir / f"pb_step_scores_{sub}.jsonl"
        if not f.exists():
            return None
        tr = restrict(f, keep_ids)
        if len(tr) <= CALIB_SIZE:
            return None
        vals.append(calib20_subset(tr))
    return float(np.mean(vals))


def onpolicy_score(cell_dir: Path, name: str, keep_ids: set[str] | None) -> float | None:
    f = cell_dir / f"pb_step_scores_{name}.jsonl"
    if not f.exists():
        return None
    tr = restrict(f, keep_ids)
    if len(tr) <= CALIB_SIZE:
        return None
    return calib20_subset(tr)


def trace_mean_step_tokens(split_dir: Path) -> dict[str, float]:
    """Mean step length in tokens per trace, from the store's own meta."""
    from src.repstore.store import ShardedRepSplit
    meta = ShardedRepSplit(split_dir).meta()
    acc: dict[str, list[int]] = defaultdict(list)
    for m in meta:
        # span-only stores rewrite n_tokens to the kept span; orig_* holds the
        # full sequence, so the step's own length is n_tokens minus the boundary
        acc[m["id"]].append(int(m["n_tokens"]) - int(m["step_start_idx"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def length_matched_ids(on_len: dict[str, float], off_len: dict[str, float],
                       trim: float = 0.10) -> tuple[set[str], set[str], dict]:
    """Keep the traces of both arms whose mean step length lies in the overlap of
    their central ranges. Crude on purpose: it answers whether the ranking holds
    where the arms look alike, not how length acts on the score.

    `trim` sets how much of each tail is cut before intersecting. Interquartile
    would be tidier but the arms are offset by construction (on-policy steps are
    shorter), and two ranges offset by more than half an interquartile range have
    an empty intersection even when the distributions overlap substantially. A
    10% trim keeps the comparison possible; when even that intersection is empty
    the overlap really is negligible, which is reported rather than papered
    over."""
    on_v = np.array(list(on_len.values()))
    off_v = np.array(list(off_len.values()))
    lo = max(np.quantile(on_v, trim), np.quantile(off_v, trim))
    hi = min(np.quantile(on_v, 1 - trim), np.quantile(off_v, 1 - trim))
    info = {"window": [float(lo), float(hi)], "trim": trim,
            "on_median": float(np.median(on_v)), "off_median": float(np.median(off_v)),
            "on_iqr": [float(np.quantile(on_v, 0.25)), float(np.quantile(on_v, 0.75))],
            "off_iqr": [float(np.quantile(off_v, 0.25)), float(np.quantile(off_v, 0.75))]}
    if lo >= hi:
        info["degenerate"] = True
        return set(on_len), set(off_len), info
    keep_on = {k for k, v in on_len.items() if lo <= v <= hi}
    keep_off = {k for k, v in off_len.items() if lo <= v <= hi}
    info.update({"kept_on": len(keep_on), "kept_off": len(keep_off),
                 "total_on": len(on_len), "total_off": len(off_len)})
    return keep_on, keep_off, info


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def per_cell_table(root: Path, name: str, keep_on, keep_off) -> dict:
    """{(rep, learner): {"off": [per seed], "on": [per seed], "seeds": [...]}}"""
    cells: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"off": [], "on": [], "seeds": []})
    for d in sorted(root.iterdir()):
        rj = d / "results.json"
        if not rj.exists():
            continue
        res = json.loads(rj.read_text())
        off = offpolicy_score(d, keep_off)
        on = onpolicy_score(d, name, keep_on)
        if off is None or on is None:
            continue
        k = cell_key(res)
        cells[k]["off"].append(off)
        cells[k]["on"].append(on)
        cells[k]["seeds"].append(int(res["seed"]))
    return dict(cells)


def bootstrap_over_cells(off: np.ndarray, on: np.ndarray, b: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(off)
    vals = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        if len(np.unique(off[idx])) < 3:
            continue
        vals.append(spearman(off[idx], on[idx]))
    v = np.array(vals)
    return {"mean": float(v.mean()), "ci95": [float(np.quantile(v, 0.025)),
                                              float(np.quantile(v, 0.975))],
            "n_resamples": len(v)}


def bootstrap_over_seeds(cells: dict, b: int, seed: int) -> dict:
    """Draw one seed per cell, per arm, and recompute. Says how much of the
    correlation is training noise rather than representation."""
    rng = np.random.default_rng(seed)
    keys = list(cells)
    vals = []
    for _ in range(b):
        o = np.array([rng.choice(cells[k]["off"]) for k in keys])
        n = np.array([rng.choice(cells[k]["on"]) for k in keys])
        vals.append(spearman(o, n))
    v = np.array(vals)
    return {"mean": float(v.mean()), "ci95": [float(np.quantile(v, 0.025)),
                                              float(np.quantile(v, 0.975))],
            "n_resamples": len(v)}


def reliability_ceiling(cells: dict, arm: str) -> dict:
    """Split-half over seeds within one arm: how well does this measurement
    reproduce its own ranking? The transfer correlation is read against this."""
    keys = [k for k in cells if len(cells[k][arm]) >= 2]
    if len(keys) < 3:
        return {"note": "not enough cells with two seeds"}
    n_seeds = min(len(cells[k][arm]) for k in keys)
    pairs = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            a = np.array([cells[k][arm][i] for k in keys])
            b = np.array([cells[k][arm][j] for k in keys])
            pairs.append(spearman(a, b))
    return {"pairwise_spearman": [float(x) for x in pairs],
            "mean": float(np.mean(pairs)), "n_cells": len(keys)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid_root", required=True, type=Path,
                   help="Directory of cell dirs, each with results.json and both "
                        "arms' pb_step_scores files.")
    p.add_argument("--onpolicy_name", default="onpolicy_verifier",
                   help="Suffix of the on-policy scores file.")
    p.add_argument("--length_meta_on", type=Path, default=None,
                   help="On-policy store split, for the length-matched rerun.")
    p.add_argument("--length_meta_off", type=Path, default=None,
                   help="ProcessBench store split, likewise.")
    p.add_argument("--length_trim", type=float, default=0.10,
                   help="Tail fraction cut from each arm before intersecting the "
                        "length ranges for the matched rerun.")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    length_info = None
    keep_on = keep_off = None
    if args.length_meta_on and args.length_meta_off:
        on_len = trace_mean_step_tokens(args.length_meta_on)
        off_len = trace_mean_step_tokens(args.length_meta_off)
        keep_on, keep_off, length_info = length_matched_ids(on_len, off_len,
                                                            args.length_trim)

    cells = per_cell_table(args.grid_root, args.onpolicy_name, None, None)
    if len(cells) < 3:
        raise SystemExit(f"only {len(cells)} cells have both arms scored")

    keys = sorted(cells)
    off = np.array([np.mean(cells[k]["off"]) for k in keys])
    on = np.array([np.mean(cells[k]["on"]) for k in keys])

    rep = {
        "n_cells": len(keys),
        "onpolicy_name": args.onpolicy_name,
        "spearman": spearman(off, on),
        "kendall": kendall(off, on),
        "bootstrap_over_cells": bootstrap_over_cells(off, on, args.bootstrap, args.seed),
        "bootstrap_over_seeds": bootstrap_over_seeds(cells, args.bootstrap, args.seed),
        "reliability_ceiling_offpolicy": reliability_ceiling(cells, "off"),
        "reliability_ceiling_onpolicy": reliability_ceiling(cells, "on"),
        "cells": {f"{r} x {l}": {"off_mean": float(np.mean(cells[(r, l)]["off"])),
                                 "off_sd": float(np.std(cells[(r, l)]["off"])),
                                 "on_mean": float(np.mean(cells[(r, l)]["on"])),
                                 "on_sd": float(np.std(cells[(r, l)]["on"])),
                                 "n_seeds": len(cells[(r, l)]["off"])}
                  for (r, l) in keys},
    }

    off_rank = {k: r for k, r in zip(keys, rankdata(-off))}
    on_rank = {k: r for k, r in zip(keys, rankdata(-on))}

    print(f"{len(keys)} cells, seed-averaged F1_PB at calib-20\n")
    print(f"{'cell':<44}{'off-policy':>18}{'on-policy':>18}{'rank':>12}")
    for k in sorted(keys, key=lambda k: off_rank[k]):
        c = cells[k]
        print(f"{k[0] + ' x ' + k[1]:<44}"
              f"{np.mean(c['off']):>11.3f} +-{np.std(c['off']):<5.3f}"
              f"{np.mean(c['on']):>11.3f} +-{np.std(c['on']):<5.3f}"
              f"{int(off_rank[k]):>6} ->{int(on_rank[k]):>4}")

    bc, bs = rep["bootstrap_over_cells"], rep["bootstrap_over_seeds"]
    ceil = rep["reliability_ceiling_offpolicy"].get("mean")
    print(f"\nSpearman {rep['spearman']:+.3f}   Kendall {rep['kendall']:+.3f}")
    print(f"  resampling cells  95% CI [{bc['ci95'][0]:+.3f}, {bc['ci95'][1]:+.3f}]")
    print(f"  resampling seeds  95% CI [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}]")
    if ceil is not None:
        print(f"  reliability ceiling (off-policy split-half over seeds) {ceil:+.3f}")
        print(f"  -> the transfer correlation is {rep['spearman']/ceil:.2f} of what "
              f"remeasuring the same arm achieves" if ceil > 0 else "")

    print("\nThe contrasts the argument rests on:")
    survived = 0
    tested = 0
    contrast_rows = []
    for name, worse, better, off_gap in CONTRASTS:
        if worse not in cells or better not in cells:
            print(f"  [missing] {name}")
            continue
        w, b = cells[worse], cells[better]
        d_off = float(np.mean(b["off"]) - np.mean(w["off"]))
        d_on = float(np.mean(b["on"]) - np.mean(w["on"]))
        # A gap is only "kept" if it stays positive by more than the noise in it.
        spread = float(np.hypot(np.std(b["on"]), np.std(w["on"])))
        keeps = d_on > spread
        tested += 1
        survived += int(keeps)
        contrast_rows.append({"name": name, "off_gap_published": off_gap,
                              "off_gap": d_off, "on_gap": d_on,
                              "on_seed_spread": spread, "survives": bool(keeps)})
        print(f"  [{'keeps' if keeps else 'BREAKS'}] {name}")
        print(f"           off-policy {d_off:+.3f} (published {off_gap:+.3f})   "
              f"on-policy {d_on:+.3f} +-{spread:.3f}")
    rep["contrasts"] = contrast_rows
    rep["contrast_sign_test_p"] = sign_test_p(survived, tested)
    print(f"\n  {survived}/{tested} contrasts keep their sign beyond the seed "
          f"spread; sign test p = {rep['contrast_sign_test_p']:.3f}")

    if length_info is not None:
        print(f"\nLength-matched rerun: on-policy median step {length_info['on_median']:.0f} "
              f"tokens, off-policy {length_info['off_median']:.0f}, "
              f"window {length_info['window'][0]:.0f}-{length_info['window'][1]:.0f}")
        if length_info.get("degenerate"):
            print("  the two interquartile ranges do not overlap; no matched rerun")
        else:
            cells_m = per_cell_table(args.grid_root, args.onpolicy_name, keep_on, keep_off)
            km = sorted(cells_m)
            if len(km) >= 3:
                om = np.array([np.mean(cells_m[k]["off"]) for k in km])
                nm = np.array([np.mean(cells_m[k]["on"]) for k in km])
                rep["length_matched"] = {"spearman": spearman(om, nm),
                                         "n_cells": len(km), **length_info}
                print(f"  kept {length_info['kept_on']}/{length_info['total_on']} "
                      f"on-policy and {length_info['kept_off']}/{length_info['total_off']} "
                      f"off-policy traces")
                print(f"  Spearman within the window {rep['length_matched']['spearman']:+.3f} "
                      f"(against {rep['spearman']:+.3f} overall)")
        rep["length"] = length_info

    print("\nF1 is not compared across the two arms anywhere above; only ranks and "
          "within-arm gaps are.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2))
        print(f"[rank] wrote {args.out}")


if __name__ == "__main__":
    main()
