"""Shared data loading and metrics for step_bert_probe_v1 fits (both arms).

Every fit writes the same files as v0's finetune.py, so one aggregator reads
both: probs_onpolicy.npy and probs_pb_<subset>.npy (one probability per step, in
steps.npz order) and metrics.json (val threshold, test F1 at it and at the
oracle, AUROC, always-positive F1, ProcessBench F1_PB per subset at the val
threshold, calib-20 and oracle, known-step AUROC, and their means).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import best_threshold, f1_pb, f1_pb_calib20, f1_pb_oracle  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def trace_runs(trace_id: np.ndarray) -> dict[str, tuple[int, int]]:
    """trace -> (first row, n steps); rows must be grouped by trace."""
    starts = np.flatnonzero(np.r_[True, trace_id[1:] != trace_id[:-1]])
    ends = np.r_[starts[1:], len(trace_id)]
    out = {str(trace_id[s]): (int(s), int(e - s)) for s, e in zip(starts, ends)}
    if len(out) != len(starts):
        raise ValueError("trace rows are not contiguous")
    return out


def split_runs(steps, split_path: Path):
    runs = trace_runs(steps["trace_id"])
    split = json.loads(Path(split_path).read_text())
    return runs, {k: [runs[t] for t in split[k] if t in runs] for k in ("train", "val", "test")}


def rows_of(rs) -> np.ndarray:
    return np.concatenate([np.arange(s, s + n) for s, n in rs])


def step_weights(rs) -> np.ndarray:
    """Per-step loss weight 1/n_steps(trace), normalized to mean 1 over the given traces,
    indexed by global row (rows outside rs get 0)."""
    n_rows = max(s + n for s, n in rs)
    w = np.zeros(n_rows, dtype=np.float32)
    for s, n in rs:
        w[s:s + n] = 1.0 / n
    return w / w[rows_of(rs)].mean()


def finalize(out_dir: Path, args, steps, p_all: np.ndarray, parts: dict, log: list, best: float,
             pb: list[tuple[str, dict, np.ndarray]]) -> dict:
    y = steps["y"].astype(int)
    rv, rt = rows_of(parts["val"]), rows_of(parts["test"])
    th, _ = best_threshold(y[rv], p_all[rv])
    yt, pt = y[rt], p_all[rt]
    prev = float(yt.mean())
    metrics = {"args": {k: str(v) for k, v in vars(args).items()}, "val_threshold": th, "best_val_f1": best,
               "test": {"n": int(len(rt)), "prevalence": prev, "auroc": float(roc_auc_score(yt, pt)),
                        "f1_val_threshold": float(f1_score(yt, pt >= th)),
                        "f1_oracle": best_threshold(yt, pt)[1],
                        "always_positive_f1": 2 * prev / (1 + prev)},
               "processbench": {}, "log": log}
    np.save(out_dir / "probs_onpolicy.npy", p_all)
    for name, zp, pp in pb:
        np.save(out_dir / f"probs_{name}.npy", pp)
        known = (zp["first_error_idx"] < 0) | (zp["step_idx"] <= zp["first_error_idx"])
        a = (zp["trace_id"], zp["step_idx"], zp["first_error_idx"])
        metrics["processbench"][name] = {
            "f1_pb_val_threshold": f1_pb(pp, *a, th),
            "f1_pb_oracle": f1_pb_oracle(pp, *a),
            "f1_pb_calib20": f1_pb_calib20(pp, *a),
            "known_step_auroc": float(roc_auc_score(zp["y"][known], pp[known])),
        }
    if metrics["processbench"]:
        for key in ("f1_pb_val_threshold", "f1_pb_calib20", "f1_pb_oracle", "known_step_auroc"):
            metrics[f"processbench_mean_{key}"] = float(np.mean([v[key] for v in metrics["processbench"].values()]))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("log", "args")}, indent=2), flush=True)
    return metrics


def val_f1(steps, p_all: np.ndarray, parts: dict) -> tuple[float, float]:
    rv = rows_of(parts["val"])
    yv = steps["y"][rv].astype(int)
    return best_threshold(yv, p_all[rv])[1], float(roc_auc_score(yv, p_all[rv]))
