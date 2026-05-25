"""Evaluate a saved linear probe on a single ProcessBench cache (one subset).

Score / threshold convention -- IDENTICAL to train_easy_probe_method.py and
train_eval_ssae_probe.py:

  * Probe outputs a logit; we apply sigmoid -> score in [0,1].
    Score == P(step is the first-error step).  Higher score = more
    suspicious / more "error".
  * For each trace we sort steps by step_idx and predict the FIRST step
    whose score > threshold; if none, prediction = -1 (trace is correct).
  * ProcessBench label: -1 means trace is fully correct; non-negative
    is the index of the first-error step (repeated on every row of the
    trace).
  * F1_PB = harmonic mean of Acc_error and Acc_correct.

The evaluator NEVER inverts scores.  If you pass a probe trained with a
different convention you must apply your own transform upstream.

Inputs come in three flavors:

  1. dense_linear / random:
       --pb_latents <pb_step_h.npy>
       (no representation; probe is fc(h_dim, 1))

  2. SAE (sae_positive / sae_mixed / sae_contrastive):
       --pb_latents <pb_step_h.npy>
       --sae_repr <run_dir>/representation.pt
       (we ReLU(W_enc h + b_enc) on the fly to match training)

  3. SSAE (ssae_*):
       --pb_latents <run_dir>/latents_full_pb/<subset>/pb_step_z.npy
       (no SAE step; latents already produced by extract_ssae_pb_all.py)

Threshold is either:
  --threshold <float>                       (single threshold)
  --threshold_json <path/to/threshold.json> (use 'selected_threshold')

If --random is set, the probe state dict is ignored and uniform random
scores are produced (matches the 'random' method in
train_easy_probe_method.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Mirror SAE + Probe definitions verbatim from train_easy_probe_method.py
# (We do not import from that module because it pulls heavyweight deps and we
# want this evaluator to be lean and runnable on CPU.)

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)


class SAE(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, hidden_dim, bias=True)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(h))


def evaluate_processbench(scores: np.ndarray, meta: list[dict], threshold: float
                          ) -> tuple[list[dict], dict]:
    groups: dict[str, list[tuple[int, float]]] = {}
    labels: dict[str, int] = {}
    n_steps_map: dict[str, int] = {}
    for i, row in enumerate(meta):
        tid = row["id"]
        groups.setdefault(tid, []).append((int(row["step_idx"]), float(scores[i])))
        labels[tid] = int(row["label"])
        n_steps_map[tid] = int(row["n_steps"])

    rows: list[dict] = []
    n_error = n_correct = 0
    acc_error_hits = acc_correct_hits = 0
    exact_all = 0
    for tid, items in groups.items():
        items.sort(key=lambda x: x[0])
        step_scores = [s for _, s in items]
        pred = -1
        for t_idx, s in items:
            if s > threshold:
                pred = t_idx
                break
        label = labels[tid]
        rows.append({
            "id": tid,
            "label": label,
            "n_steps": n_steps_map[tid],
            "scores": step_scores,
            "threshold": threshold,
            "prediction": pred,
        })
        if label == -1:
            n_correct += 1
            if pred == -1:
                acc_correct_hits += 1
        else:
            n_error += 1
            if pred == label:
                acc_error_hits += 1
        if pred == label:
            exact_all += 1

    acc_error = acc_error_hits / max(n_error, 1) if n_error else 0.0
    acc_correct = acc_correct_hits / max(n_correct, 1) if n_correct else 0.0
    denom = acc_error + acc_correct
    f1_pb = (2 * acc_error * acc_correct / denom) if denom > 0 else 0.0
    metrics = {
        "n_traces": len(groups),
        "n_error_traces": n_error,
        "n_correct_traces": n_correct,
        "Acc_error": acc_error,
        "Acc_correct": acc_correct,
        "F1_PB": f1_pb,
        "Exact_match_all": exact_all / max(len(groups), 1),
    }
    return rows, metrics


# Coarse 0.1 grid kept ONLY for back-compat with old val-selection code that
# imports this module. The oracle sweep uses a much finer grid by default
# (see DEFAULT_ORACLE_STEP below) -- previous SSAE runs had optima at e.g.
# 0.555, which a 0.1 grid would miss.
THRESHOLD_GRID = [round(0.1 * i, 1) for i in range(1, 11)]

DEFAULT_ORACLE_STEP = 0.005


def build_oracle_grid(step: float) -> list[float]:
    """Return [step, 2*step, ..., 1-step] rounded to 6 decimals.

    step must be in (0, 1). Endpoints 0.0 and 1.0 are excluded because
    score>0 and score>1 are degenerate boundary cases for sigmoid scores.
    """
    if not (0.0 < step < 1.0):
        raise ValueError(f"oracle threshold step must be in (0,1), got {step}")
    n = int(round(1.0 / step))
    grid = [round(step * i, 6) for i in range(1, n)]
    # Guarantee at least one point.
    return grid or [0.5]


def find_oracle_threshold(scores: np.ndarray, meta: list[dict],
                          grid: list[float]) -> tuple[float, float]:
    best_t = grid[0]
    best_f1 = -1.0
    for t in grid:
        _, m = evaluate_processbench(scores, meta, t)
        if m["F1_PB"] > best_f1:
            best_f1 = m["F1_PB"]
            best_t = t
    return best_t, best_f1


def load_meta(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, default=None,
                   help="Path to linear_probe.pt (state_dict). Omit only with --random.")
    p.add_argument("--pb_latents", type=Path, required=True,
                   help="pb_step_h.npy (dense / SAE input) or pb_step_z.npy (SSAE).")
    p.add_argument("--pb_meta", type=Path, required=True,
                   help="pb_step_meta.jsonl matching --pb_latents row-for-row.")
    p.add_argument("--sae_repr", type=Path, default=None,
                   help="Optional SAE representation.pt; when set we encode "
                        "h -> ReLU(W_enc h + b_enc) before scoring.")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--threshold_json", type=Path, default=None,
                   help="JSON with 'selected_threshold' field (output of "
                        "train_easy_probe_method.py).")
    p.add_argument("--also_oracle", action="store_true",
                   help="Also compute oracle threshold over PB and report.")
    p.add_argument("--oracle_threshold_step", type=float, default=DEFAULT_ORACLE_STEP,
                   help=f"Step for the oracle sweep grid in (0,1). "
                        f"Default {DEFAULT_ORACLE_STEP} -> 199 points; matches "
                        "fine-grained sweeps where optima land at e.g. 0.555.")
    p.add_argument("--method_name", default="unknown")
    p.add_argument("--pb_subset", default=None,
                   help="Optional subset tag stored in the output metrics.")
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_scores_jsonl", type=Path, default=None)
    p.add_argument("--out_predictions_jsonl", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--random", action="store_true",
                   help="Generate uniform random scores instead of running probe.")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default=None,
                   help="cuda|cpu (default: auto).")
    return p.parse_args()


def resolve_threshold(args) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    if args.threshold_json is not None:
        meta = json.loads(args.threshold_json.read_text())
        return float(meta["selected_threshold"])
    sys.exit("Provide --threshold or --threshold_json.")


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    h = np.load(args.pb_latents).astype(np.float32)
    if not np.all(np.isfinite(h)):
        sys.exit(f"[eval_probe] pb_latents has NaN/Inf: {args.pb_latents}")
    meta = load_meta(args.pb_meta)
    if h.shape[0] != len(meta):
        sys.exit(f"[eval_probe] rows({h.shape[0]}) != meta rows({len(meta)})")

    # ---- Optional SAE encode (matches train_easy_probe_method.py exactly)
    if args.sae_repr is not None:
        # Infer dims from the state_dict.
        sd = torch.load(args.sae_repr, map_location="cpu")
        enc_w = sd["encoder.weight"]
        latent_dim, hidden_dim = enc_w.shape
        if h.shape[1] != hidden_dim:
            sys.exit(f"[eval_probe] PB h dim ({h.shape[1]}) != SAE in_dim ({hidden_dim})")
        sae = SAE(hidden_dim, latent_dim).to(device)
        sae.load_state_dict(sd)
        sae.eval()
        zs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, h.shape[0], args.batch_size):
                chunk = torch.from_numpy(h[i : i + args.batch_size]).to(device)
                zs.append(sae.encode(chunk).cpu().numpy())
        z = np.concatenate(zs, axis=0).astype(np.float32)
    else:
        z = h  # dense or already-SSAE latents

    # ---- Score
    t0 = time.time()
    if args.random:
        rng = np.random.default_rng(args.seed)
        scores = rng.uniform(0.0, 1.0, size=z.shape[0]).astype(np.float32)
    else:
        if args.probe is None:
            sys.exit("--probe is required unless --random is set.")
        probe_sd = torch.load(args.probe, map_location="cpu")
        in_dim = probe_sd["fc.weight"].shape[1]
        if z.shape[1] != in_dim:
            sys.exit(
                f"[eval_probe] probe in_dim ({in_dim}) != input dim ({z.shape[1]}). "
                "Did you forget --sae_repr for SAE methods, or pass the wrong cache?"
            )
        probe = LinearProbe(in_dim).to(device)
        probe.load_state_dict(probe_sd)
        probe.eval()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, z.shape[0], args.batch_size):
                chunk = torch.from_numpy(z[i : i + args.batch_size]).to(device)
                outs.append(torch.sigmoid(probe(chunk)).cpu().numpy())
        scores = np.concatenate(outs, axis=0).astype(np.float32)
    score_time = time.time() - t0
    if not np.all(np.isfinite(scores)):
        sys.exit(f"[eval_probe] scores have NaN/Inf for {args.method_name}")

    # ---- Threshold(s)
    val_t = resolve_threshold(args)
    t0 = time.time()
    rows_val, m_val = evaluate_processbench(scores, meta, val_t)
    eval_time_val = time.time() - t0

    oracle_block = None
    if args.also_oracle:
        oracle_grid = build_oracle_grid(args.oracle_threshold_step)
        best_t, _ = find_oracle_threshold(scores, meta, oracle_grid)
        rows_oracle, m_oracle = evaluate_processbench(scores, meta, best_t)
        oracle_block = {
            "threshold": best_t,
            "threshold_step": args.oracle_threshold_step,
            "n_grid_points": len(oracle_grid),
            "metrics": m_oracle,
        }

    n_steps = z.shape[0]
    n_traces = m_val["n_traces"]
    total_eval = score_time + eval_time_val
    metrics = {
        "method": args.method_name,
        "pb_subset": args.pb_subset,
        "threshold_type": "val_selected",
        "threshold": val_t,
        **m_val,
        "eval_time_sec": total_eval,
        "score_time_sec": score_time,
        "mean_step_latency_ms": total_eval * 1000.0 / max(n_steps, 1),
        "mean_trace_latency_ms": total_eval * 1000.0 / max(n_traces, 1),
        "device": device.type,
        "n_steps_total": int(n_steps),
        "scoring_convention": (
            "probe -> sigmoid(logit); s > threshold predicts first-error; "
            "trace pred = first step exceeding threshold or -1 if none."
        ),
    }
    if oracle_block is not None:
        metrics["oracle"] = oracle_block

    args.out_json.write_text(json.dumps(metrics, indent=2))

    if args.out_scores_jsonl is not None:
        args.out_scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_scores_jsonl.open("w") as f:
            for row in rows_val:
                f.write(json.dumps(row) + "\n")

    if args.out_predictions_jsonl is not None:
        args.out_predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_predictions_jsonl.open("w") as f:
            for row in rows_val:
                f.write(json.dumps({
                    "id": row["id"], "label": row["label"],
                    "prediction": row["prediction"], "threshold": row["threshold"],
                }) + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
