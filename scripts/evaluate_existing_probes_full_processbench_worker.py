"""Worker that processes a slice of the full ProcessBench evaluation job list.

Reads a JSON file emitted by ``build_full_processbench_eval_jobs.py`` and
runs each job sequentially on its assigned device. Each job writes:

  per_job/<method>/<subset>/val_selected_metrics.json
  per_job/<method>/<subset>/oracle_metrics.json
  per_job/<method>/<subset>/pb_step_scores.jsonl
  per_job/<method>/<subset>/pb_predictions.jsonl

Score / threshold convention is IDENTICAL to
``train_easy_probe_method.py`` and ``train_eval_ssae_probe.py``:

  * probe -> sigmoid(logit) = P(step is the first error). Higher = more
    "error". No silent inversion.
  * trace prediction = first step whose score > threshold; -1 otherwise.
  * F1_PB = harmonic mean of Acc_error and Acc_correct.

Validation before scoring (hard exit on failure):
  * probe + latents + meta files exist
  * latents row count == meta row count
  * no NaN/Inf in latents
  * if family == 'sae': SAE representation.pt exists; in_dim matches PB h
  * probe in_dim matches latent dim
  * for val-selected runs: threshold.json present (else fall back to 0.5
    for 'random'; for other methods this is an error).

Oracle threshold uses a fine grid (default step 0.005 -> 199 points).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_EVAL_PATH = ROOT / "scripts" / "evaluate_saved_probe_on_processbench.py"
_spec = importlib.util.spec_from_file_location("pb_eval_mod", _EVAL_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

LinearProbe = _mod.LinearProbe
SAE = _mod.SAE
evaluate_processbench = _mod.evaluate_processbench
find_oracle_threshold = _mod.find_oracle_threshold
build_oracle_grid = _mod.build_oracle_grid
DEFAULT_ORACLE_STEP = _mod.DEFAULT_ORACLE_STEP
load_meta = _mod.load_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs_json", type=Path, required=True)
    p.add_argument("--worker_id", type=int, required=True)
    p.add_argument("--device", default=None,
                   help="cuda|cpu (default: auto). With one CUDA device "
                        "visible (via CUDA_VISIBLE_DEVICES) this is cuda:0.")
    p.add_argument("--oracle_threshold_step", type=float, default=DEFAULT_ORACLE_STEP)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing per-job outputs.")
    return p.parse_args()


def score_one(job: dict, h_or_z: np.ndarray, device: torch.device,
              batch_size: int, seed: int) -> np.ndarray:
    family = job["representation_type"]
    z = h_or_z

    if family == "sae":
        sd = torch.load(job["sae_repr"], map_location="cpu")
        enc_w = sd["encoder.weight"]
        latent_dim, hidden_dim = enc_w.shape
        if z.shape[1] != hidden_dim:
            sys.exit(
                f"[{job['method']}/{job['pb_subset']}] h dim {z.shape[1]} "
                f"!= SAE in_dim {hidden_dim}"
            )
        sae = SAE(hidden_dim, latent_dim).to(device)
        sae.load_state_dict(sd)
        sae.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, z.shape[0], batch_size):
                chunk = torch.from_numpy(z[i:i + batch_size]).to(device)
                outs.append(sae.encode(chunk).cpu().numpy())
        z = np.concatenate(outs, axis=0).astype(np.float32)

    if job["is_random"]:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, size=z.shape[0]).astype(np.float32)

    probe_sd = torch.load(job["probe"], map_location="cpu")
    in_dim = probe_sd["fc.weight"].shape[1]
    if z.shape[1] != in_dim:
        sys.exit(
            f"[{job['method']}/{job['pb_subset']}] probe in_dim {in_dim} "
            f"!= input dim {z.shape[1]}"
        )
    probe = LinearProbe(in_dim).to(device)
    probe.load_state_dict(probe_sd)
    probe.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, z.shape[0], batch_size):
            chunk = torch.from_numpy(z[i:i + batch_size]).to(device)
            outs.append(torch.sigmoid(probe(chunk)).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    args = parse_args()
    pid = os.getpid()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[worker-{args.worker_id}] pid={pid} device={device.type} "
          f"jobs_json={args.jobs_json}", flush=True)

    jobs = json.loads(args.jobs_json.read_text())
    if not isinstance(jobs, list):
        sys.exit(f"[worker-{args.worker_id}] jobs_json must be a JSON list")
    print(f"[worker-{args.worker_id}] {len(jobs)} job(s) to process", flush=True)

    oracle_grid = build_oracle_grid(args.oracle_threshold_step)
    n_fail = 0
    for i, job in enumerate(jobs):
        method = job["method"]; subset = job["pb_subset"]
        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        val_out = out_dir / "val_selected_metrics.json"
        ora_out = out_dir / "oracle_metrics.json"
        if val_out.exists() and ora_out.exists() and not args.force:
            print(f"[worker-{args.worker_id}] [{i+1}/{len(jobs)}] "
                  f"{method}/{subset}: SKIP (exists)")
            continue

        try:
            # ---- Validation ------------------------------------------------
            lat_path = Path(job["pb_latents"])
            meta_path = Path(job["pb_meta"])
            if not lat_path.exists() or not meta_path.exists():
                raise FileNotFoundError(
                    f"missing latents/meta: {lat_path} / {meta_path}"
                )
            h_or_z = np.load(lat_path).astype(np.float32)
            if not np.all(np.isfinite(h_or_z)):
                raise ValueError(f"NaN/Inf in {lat_path}")
            meta = load_meta(meta_path)
            if h_or_z.shape[0] != len(meta):
                raise ValueError(
                    f"rows mismatch: array={h_or_z.shape[0]} meta={len(meta)}"
                )

            # ---- Threshold -------------------------------------------------
            if job["threshold_json"] is None:
                if job["is_random"]:
                    val_t = 0.5
                else:
                    raise FileNotFoundError(
                        f"missing threshold.json for {method}"
                    )
            else:
                tj = json.loads(Path(job["threshold_json"]).read_text())
                val_t = float(tj["selected_threshold"])

            # ---- Score -----------------------------------------------------
            t_score = time.time()
            scores = score_one(job, h_or_z, device, args.batch_size, args.seed)
            t_score = time.time() - t_score
            if not np.all(np.isfinite(scores)):
                raise ValueError("NaN/Inf in produced scores")

            # ---- Evaluate at val and oracle --------------------------------
            t0 = time.time()
            rows_val, m_val = evaluate_processbench(scores, meta, val_t)
            t_val = (time.time() - t0) + t_score

            best_t, _ = find_oracle_threshold(scores, meta, oracle_grid)
            t0 = time.time()
            rows_ora, m_ora = evaluate_processbench(scores, meta, best_t)
            t_ora = (time.time() - t0) + t_score

            base = {
                "method": method,
                "representation_type": job["representation_type"],
                "pb_subset": subset,
                "n_steps_total": int(scores.shape[0]),
                "scoring_convention":
                    "sigmoid(probe_logit) = P(step is first-error); "
                    "trace pred = first step with score > threshold or -1.",
                "device": device.type,
            }
            (val_out).write_text(json.dumps({
                **base,
                "threshold_type": "val_selected",
                "threshold": val_t,
                "threshold_source": "threshold.json:selected_threshold",
                **m_val,
                "eval_time_sec": t_val,
                "mean_step_latency_ms": t_val * 1000.0 / max(scores.shape[0], 1),
                "mean_trace_latency_ms": t_val * 1000.0 / max(m_val["n_traces"], 1),
            }, indent=2))
            (ora_out).write_text(json.dumps({
                **base,
                "threshold_type": "oracle",
                "threshold": best_t,
                "threshold_source": (
                    f"oracle_grid_max_F1_PB step={args.oracle_threshold_step} "
                    "(NOT DEPLOYABLE)"
                ),
                "oracle_grid_step": args.oracle_threshold_step,
                "oracle_grid_n_points": len(oracle_grid),
                **m_ora,
                "eval_time_sec": t_ora,
                "mean_step_latency_ms": t_ora * 1000.0 / max(scores.shape[0], 1),
                "mean_trace_latency_ms": t_ora * 1000.0 / max(m_ora["n_traces"], 1),
            }, indent=2))

            write_jsonl(out_dir / "pb_step_scores.jsonl", rows_val)
            write_jsonl(out_dir / "pb_predictions.jsonl",
                        [{"id": r["id"], "label": r["label"],
                          "prediction": r["prediction"],
                          "threshold": r["threshold"]} for r in rows_val])

            print(f"[worker-{args.worker_id}] [{i+1}/{len(jobs)}] "
                  f"{method}/{subset}: val t={val_t} F1={m_val['F1_PB']:.4f} "
                  f"oracle t={best_t} F1={m_ora['F1_PB']:.4f}", flush=True)
        except Exception as e:
            n_fail += 1
            err_path = out_dir / "ERROR.txt"
            err_path.write_text(f"{type(e).__name__}: {e}")
            print(f"[worker-{args.worker_id}] [{i+1}/{len(jobs)}] "
                  f"{method}/{subset}: FAIL {type(e).__name__}: {e}",
                  flush=True)

    if n_fail:
        print(f"[worker-{args.worker_id}] FAILED {n_fail}/{len(jobs)} job(s)")
        sys.exit(1)
    print(f"[worker-{args.worker_id}] OK ({len(jobs)} jobs)")


if __name__ == "__main__":
    main()
