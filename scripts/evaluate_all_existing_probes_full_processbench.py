"""Evaluate every trained probe on every ProcessBench subset and emit the
full leaderboard (val-selected + oracle thresholds).

Methods covered (probe directory layout expected under --runs_root):

    dense_linear                <run>/linear_probe.pt + threshold.json
    random                      <run>/threshold.json (no probe)
    sae_positive                <run>/linear_probe.pt + representation.pt + threshold.json
    sae_mixed                   ditto
    sae_contrastive             ditto
    ssae_positive               <run>/linear_probe.pt + threshold.json
                                + <run>/latents_full_pb/<subset>/pb_step_z.npy
    ssae_mixed                  ditto
    ssae_contrastive            ditto (--contrastive_ckpt was set at extract time)
    ssae_contrastive_auxlr1e-3_full   ditto

Inputs needed at runtime:
  --runs_root             $SCRATCH/cot_mech/prestudy_v1/runs
  --dense_pb_cache_root   $SCRATCH/cot_mech/prestudy_v1/cache/qwen2_5_1_5b_processbench_full
                          (must contain <subset>/pb_step_h.npy + meta + combined/)
  --out_dir               $SCRATCH/cot_mech/prestudy_v1/runs/full_processbench_eval

For each method x subset (and the 'combined' pooled view):
  * load PB latents (dense h for dense/random/sae, run-local z for ssae);
  * load probe + threshold;
  * compute F1_PB + Acc_error/Acc_correct/Exact_match_all using the
    method's val-selected threshold (from threshold.json);
  * separately compute oracle threshold on PB itself (NOT DEPLOYABLE);
  * write per-method/subset metrics JSON + aggregate CSV / MD.

Aggregate rows include:
  pooled_combined         F1 on the concatenated PB across subsets
  macro_avg_per_subset    arithmetic mean of per-subset F1, Acc_error,
                          Acc_correct, Exact_match_all (equal weight per
                          subset).

The script does NOT retrain. It does NOT touch existing artifacts. It only
writes under --out_dir.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the generic evaluator's helpers in-process (no import side effects).
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

import torch  # noqa: E402  (after import path setup)


# Method -> (representation_family, requires_sae_repr, requires_run_local_pb)
METHOD_SPECS: dict[str, tuple[str, bool, bool]] = {
    "dense_linear":                       ("dense",  False, False),
    "random":                             ("dense",  False, False),
    "sae_positive":                       ("sae",    True,  False),
    "sae_mixed":                          ("sae",    True,  False),
    "sae_contrastive":                    ("sae",    True,  False),
    "ssae_positive":                      ("ssae",   False, True),
    "ssae_mixed":                         ("ssae",   False, True),
    "ssae_contrastive":                   ("ssae",   False, True),
    "ssae_contrastive_auxlr1e-3_full":    ("ssae",   False, True),
}


def find_subsets(dense_pb_root: Path) -> list[str]:
    """All <subset> directories under dense_pb_root that contain pb_step_h.npy.
    'combined' is included but emitted as its own pooled row.
    """
    out: list[str] = []
    for child in sorted(dense_pb_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "pb_step_h.npy").exists() and (child / "pb_step_meta.jsonl").exists():
            out.append(child.name)
    if not out:
        sys.exit(f"[full_pb_eval] no <subset>/pb_step_h.npy under {dense_pb_root}")
    return out


def pb_inputs_for(method: str, subset: str, runs_root: Path,
                  dense_pb_root: Path) -> tuple[Path, Path]:
    """Return (latents_npy, meta_jsonl) for this method/subset pair."""
    family, _, run_local = METHOD_SPECS[method]
    sub_dir = dense_pb_root / subset
    if family == "ssae":
        run_dir = runs_root / method
        z = run_dir / "latents_full_pb" / subset / "pb_step_z.npy"
        m = run_dir / "latents_full_pb" / subset / "pb_step_meta.jsonl"
        return z, m
    # dense / random / sae all use the dense cache for h.
    return sub_dir / "pb_step_h.npy", sub_dir / "pb_step_meta.jsonl"


def score_one(method: str, run_dir: Path, h_or_z: np.ndarray,
              device: torch.device, batch_size: int, seed: int) -> np.ndarray:
    family, needs_sae, _ = METHOD_SPECS[method]
    z = h_or_z

    if family == "sae":
        sae_path = run_dir / "representation.pt"
        if not sae_path.exists():
            sys.exit(f"[full_pb_eval] missing SAE rep for {method}: {sae_path}")
        sd = torch.load(sae_path, map_location="cpu")
        enc_w = sd["encoder.weight"]
        latent_dim, hidden_dim = enc_w.shape
        if z.shape[1] != hidden_dim:
            sys.exit(f"[{method}] h dim {z.shape[1]} != SAE in_dim {hidden_dim}")
        sae = SAE(hidden_dim, latent_dim).to(device)
        sae.load_state_dict(sd)
        sae.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, z.shape[0], batch_size):
                chunk = torch.from_numpy(z[i:i + batch_size]).to(device)
                outs.append(sae.encode(chunk).cpu().numpy())
        z = np.concatenate(outs, axis=0).astype(np.float32)

    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, size=z.shape[0]).astype(np.float32)

    probe_path = run_dir / "linear_probe.pt"
    if not probe_path.exists():
        sys.exit(f"[full_pb_eval] missing probe for {method}: {probe_path}")
    probe_sd = torch.load(probe_path, map_location="cpu")
    in_dim = probe_sd["fc.weight"].shape[1]
    if z.shape[1] != in_dim:
        sys.exit(
            f"[{method}] probe in_dim {in_dim} != input dim {z.shape[1]} "
            f"(family={family}, subset-dependent latent path)."
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


def load_val_threshold(run_dir: Path, method: str) -> float:
    p = run_dir / "threshold.json"
    if not p.exists():
        sys.exit(f"[full_pb_eval] missing threshold.json for {method}: {p}")
    return float(json.loads(p.read_text())["selected_threshold"])


def make_row(method: str, family: str, subset: str, threshold_type: str,
             threshold: float, metrics: dict, eval_time: float) -> dict:
    return {
        "method": method,
        "representation_type": family,
        "pb_subset": subset,
        "threshold_type": threshold_type,
        "threshold": threshold,
        "n_traces": metrics["n_traces"],
        "n_error_traces": metrics["n_error_traces"],
        "n_correct_traces": metrics["n_correct_traces"],
        "F1_PB": metrics["F1_PB"],
        "Acc_error": metrics["Acc_error"],
        "Acc_correct": metrics["Acc_correct"],
        "Exact_match_all": metrics["Exact_match_all"],
        "eval_time_sec": eval_time,
    }


def write_table(rows: list[dict], csv_path: Path, md_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # Markdown
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        cells: list[str] = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n")


def macro_avg(rows: list[dict], drop_combined: bool = True) -> dict:
    keys = ["F1_PB", "Acc_error", "Acc_correct", "Exact_match_all"]
    use = [r for r in rows if (not drop_combined or r["pb_subset"] != "combined")]
    if not use:
        return {f"macro_avg_{k}": None for k in keys}
    out = {}
    for k in keys:
        vals = [r[k] for r in use]
        out[f"macro_avg_{k}"] = sum(vals) / len(vals) if vals else None
    out["macro_avg_subsets"] = sorted({r["pb_subset"] for r in use})
    return out


def pooled_combined(rows: list[dict]) -> dict:
    cmb = [r for r in rows if r["pb_subset"] == "combined"]
    if not cmb:
        return {"pooled_F1_PB_combined": None}
    return {
        "pooled_F1_PB_combined": cmb[0]["F1_PB"],
        "pooled_Acc_error": cmb[0]["Acc_error"],
        "pooled_Acc_correct": cmb[0]["Acc_correct"],
        "pooled_Exact_match_all": cmb[0]["Exact_match_all"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--dense_pb_cache_root", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=list(METHOD_SPECS.keys()),
                   help="Subset of methods to evaluate. Default: all known.")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--oracle_threshold_step", type=float, default=DEFAULT_ORACLE_STEP,
                   help=f"Oracle sweep step in (0,1). Default {DEFAULT_ORACLE_STEP}.")
    p.add_argument("--skip_missing", action="store_true",
                   help="Skip method/subset combos whose artifacts are missing "
                        "instead of erroring.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    subsets = find_subsets(args.dense_pb_cache_root)
    oracle_grid = build_oracle_grid(args.oracle_threshold_step)
    print(f"[full_pb_eval] subsets detected: {subsets}")
    print(f"[full_pb_eval] methods: {args.methods}")
    print(f"[full_pb_eval] device: {device.type}")
    print(f"[full_pb_eval] oracle grid: step={args.oracle_threshold_step} "
          f"n_points={len(oracle_grid)} "
          f"[{oracle_grid[0]:.3f}..{oracle_grid[-1]:.3f}]")

    val_rows: list[dict] = []
    oracle_rows: list[dict] = []
    per_method_summary: list[dict] = []

    for method in args.methods:
        if method not in METHOD_SPECS:
            print(f"[full_pb_eval] WARNING: unknown method {method}; skipping.")
            continue
        family, _, _ = METHOD_SPECS[method]
        run_dir = args.runs_root / method
        if not run_dir.is_dir():
            msg = f"[full_pb_eval] missing run dir: {run_dir}"
            if args.skip_missing:
                print(msg + " -> skip")
                continue
            sys.exit(msg)

        try:
            val_t = load_val_threshold(run_dir, method) if method != "random" else 0.5
        except SystemExit:
            if not args.skip_missing:
                raise
            val_t = 0.5

        method_subset_rows_val: list[dict] = []
        method_subset_rows_oracle: list[dict] = []

        for subset in subsets:
            lat_path, meta_path = pb_inputs_for(method, subset, args.runs_root,
                                                args.dense_pb_cache_root)
            if not lat_path.exists() or not meta_path.exists():
                msg = (f"[{method}|{subset}] missing latents/meta: "
                       f"{lat_path} / {meta_path}")
                if args.skip_missing:
                    print(msg + " -> skip")
                    continue
                sys.exit(msg)

            h_or_z = np.load(lat_path).astype(np.float32)
            if not np.all(np.isfinite(h_or_z)):
                sys.exit(f"[{method}|{subset}] NaN/Inf in {lat_path}")
            meta = load_meta(meta_path)
            if h_or_z.shape[0] != len(meta):
                sys.exit(f"[{method}|{subset}] rows mismatch: "
                         f"{h_or_z.shape[0]} vs {len(meta)}")

            t0 = time.time()
            scores = score_one(method, run_dir, h_or_z, device,
                               args.batch_size, args.seed)
            score_t = time.time() - t0

            t0 = time.time()
            _, m_val = evaluate_processbench(scores, meta, val_t)
            eval_t_val = (time.time() - t0) + score_t

            best_t, _ = find_oracle_threshold(scores, meta, oracle_grid)
            t0 = time.time()
            _, m_oracle = evaluate_processbench(scores, meta, best_t)
            eval_t_oracle = (time.time() - t0) + score_t

            r_val = make_row(method, family, subset, "val_selected",
                             val_t, m_val, eval_t_val)
            r_or = make_row(method, family, subset, "oracle",
                            best_t, m_oracle, eval_t_oracle)
            val_rows.append(r_val)
            oracle_rows.append(r_or)
            method_subset_rows_val.append(r_val)
            method_subset_rows_oracle.append(r_or)

            (args.out_dir / f"{method}__{subset}__val.json").write_text(
                json.dumps(r_val, indent=2))
            (args.out_dir / f"{method}__{subset}__oracle.json").write_text(
                json.dumps(r_or, indent=2))
            print(f"[{method}|{subset}] val t={val_t} F1={m_val['F1_PB']:.4f} "
                  f"oracle t={best_t} F1={m_oracle['F1_PB']:.4f}")

        # Per-method aggregate (val + oracle)
        agg = {
            "method": method,
            "val_selected_threshold": val_t,
            "val": {
                **macro_avg(method_subset_rows_val),
                **pooled_combined(method_subset_rows_val),
            },
            "oracle": {
                **macro_avg(method_subset_rows_oracle),
                **pooled_combined(method_subset_rows_oracle),
            },
        }
        per_method_summary.append(agg)
        (args.out_dir / f"{method}__summary.json").write_text(json.dumps(agg, indent=2))

    # ---- Leaderboards
    write_table(val_rows,
                args.out_dir / "leaderboard_full_pb_val_threshold.csv",
                args.out_dir / "leaderboard_full_pb_val_threshold.md")
    write_table(oracle_rows,
                args.out_dir / "leaderboard_full_pb_oracle_threshold.csv",
                args.out_dir / "leaderboard_full_pb_oracle_threshold.md")

    # ---- Method-level averages CSV
    avg_rows: list[dict] = []
    for s in per_method_summary:
        for tag in ("val", "oracle"):
            agg = s[tag]
            avg_rows.append({
                "method": s["method"],
                "threshold_type": tag,
                "threshold": s["val_selected_threshold"] if tag == "val" else None,
                "macro_avg_F1_PB": agg.get("macro_avg_F1_PB"),
                "macro_avg_Acc_error": agg.get("macro_avg_Acc_error"),
                "macro_avg_Acc_correct": agg.get("macro_avg_Acc_correct"),
                "macro_avg_Exact_match_all": agg.get("macro_avg_Exact_match_all"),
                "pooled_F1_PB_combined": agg.get("pooled_F1_PB_combined"),
                "subsets_in_macro": ",".join(agg.get("macro_avg_subsets", []) or []),
            })
    write_table(avg_rows,
                args.out_dir / "leaderboard_full_pb_method_averages.csv",
                args.out_dir / "leaderboard_full_pb_method_averages.md")

    print(f"\n[full_pb_eval] wrote leaderboards under {args.out_dir}")
    print("[full_pb_eval] NOTE: oracle threshold is grid-searched on PB itself; "
          "treat as analysis only, NOT DEPLOYABLE.")


if __name__ == "__main__":
    main()
