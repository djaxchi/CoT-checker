#!/usr/bin/env python3
"""Calibrate the ProcessBench first-error threshold on a small held-out subset.

The unified harness selects its deployable threshold on the balanced PRM800K
val split, which picks ~0.5 and under-tunes for ProcessBench's correct-skewed
prevalence (hence the wide val-selected vs oracle F1_PB gap). This asks a
deployable question instead: if we are allowed a SMALL labeled ProcessBench
calibration set, pick the threshold there (grid-max F1_PB), apply it to the
held-out remainder, and measure how small that calibration set can be while
recovering most of the oracle.

Pure offline analysis over the per-step scores the harness already saved
(pb_step_scores_<name>.jsonl); no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

Trace = tuple[int, list[float]]  # (first_error_label, per_step_scores)


# ---------------------------------------------------------------------------
# Core metric (mirrors evaluate_processbench in train_easy_probe_method.py)
# ---------------------------------------------------------------------------

def trace_f1_pb(traces: list[Trace], threshold: float) -> tuple[float, float, float]:
    """ProcessBench F1 at a threshold. Prediction = first step whose score > t
    (else -1). Returns (F1_PB, Acc_error, Acc_correct)."""
    n_err = n_cor = err_hit = cor_hit = 0
    for label, scores in traces:
        pred = -1
        for i, s in enumerate(scores):
            if s > threshold:
                pred = i
                break
        if label == -1:
            n_cor += 1
            cor_hit += int(pred == -1)
        else:
            n_err += 1
            err_hit += int(pred == label)
    acc_err = err_hit / n_err if n_err else 0.0
    acc_cor = cor_hit / n_cor if n_cor else 0.0
    denom = acc_err + acc_cor
    f1 = (2 * acc_err * acc_cor / denom) if denom > 0 else 0.0
    return f1, acc_err, acc_cor


def select_threshold(traces: list[Trace], grid: np.ndarray) -> tuple[float, float]:
    """Threshold on `grid` maximizing F1_PB over `traces`. Ties -> lowest t."""
    best_t, best_f1 = float(grid[0]), -1.0
    for t in grid:
        f1, _, _ = trace_f1_pb(traces, float(t))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def stratified_calib_split(
    traces: list[Trace], calib_size: int, rng: np.random.Generator
) -> tuple[list[Trace], list[Trace]]:
    """Split into (calib, eval), stratified by error/correct so both classes
    appear in calib. calib_size is the total number of calibration traces."""
    err = [t for t in traces if t[0] != -1]
    cor = [t for t in traces if t[0] == -1]
    frac = calib_size / len(traces)
    n_err_c = min(len(err), round(len(err) * frac))
    n_cor_c = min(len(cor), calib_size - n_err_c)
    err_idx = rng.permutation(len(err))
    cor_idx = rng.permutation(len(cor))
    calib = [err[i] for i in err_idx[:n_err_c]] + [cor[i] for i in cor_idx[:n_cor_c]]
    ev = [err[i] for i in err_idx[n_err_c:]] + [cor[i] for i in cor_idx[n_cor_c:]]
    return calib, ev


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def load_traces(path: Path) -> list[Trace]:
    traces: list[Trace] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        traces.append((int(r["label"]), [float(x) for x in r["scores"]]))
    return traces


def sweep(
    traces: list[Trace], calib_sizes: list[int], seeds: list[int], grid: np.ndarray
) -> dict:
    full_oracle_t, full_oracle_f1 = select_threshold(traces, grid)
    val_f1, _, _ = trace_f1_pb(traces, 0.5)  # the harness's deployable baseline

    rows = []
    for cs in calib_sizes:
        evals, thrs = [], []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            calib, ev = stratified_calib_split(traces, cs, rng)
            t_star, _ = select_threshold(calib, grid)
            f1_eval, _, _ = trace_f1_pb(ev, t_star)
            evals.append(f1_eval)
            thrs.append(t_star)
        rows.append({
            "calib_size": cs,
            "eval_f1_mean": float(np.mean(evals)),
            "eval_f1_std": float(np.std(evals)),
            "t_star_median": float(np.median(thrs)),
            "frac_of_oracle": float(np.mean(evals) / full_oracle_f1) if full_oracle_f1 else 0.0,
        })
    return {
        "n_traces": len(traces),
        "val_selected_t0.5_f1": val_f1,
        "full_oracle_t": full_oracle_t,
        "full_oracle_f1": full_oracle_f1,
        "calibration_sweep": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True, type=Path, nargs="+",
                   help="One or more pb_step_scores_<name>.jsonl files.")
    p.add_argument("--calib_sizes", type=int, nargs="+",
                   default=[10, 20, 40, 80, 160])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--grid_step", type=float, default=0.01)
    p.add_argument("--out_json", type=Path, default=None)
    args = p.parse_args()

    grid = np.arange(args.grid_step, 1.0, args.grid_step)
    out = {}
    for sp in args.scores:
        name = sp.stem.replace("pb_step_scores_", "").replace("pb_scores_", "")
        res = sweep(load_traces(sp), args.calib_sizes, args.seeds, grid)
        out[name] = res
        print(f"\n=== {name}  (n={res['n_traces']}) ===")
        print(f"  val-selected (t=0.5): F1_PB={res['val_selected_t0.5_f1']:.4f}")
        print(f"  full oracle (t={res['full_oracle_t']:.2f}): F1_PB={res['full_oracle_f1']:.4f}")
        print(f"  {'calib':>6} {'t*_med':>7} {'eval_F1':>9} {'std':>6} {'%oracle':>8}")
        for r in res["calibration_sweep"]:
            print(f"  {r['calib_size']:>6} {r['t_star_median']:>7.2f} "
                  f"{r['eval_f1_mean']:>9.4f} {r['eval_f1_std']:>6.3f} "
                  f"{100*r['frac_of_oracle']:>7.1f}%")
    if args.out_json:
        args.out_json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
