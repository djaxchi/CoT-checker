#!/usr/bin/env python3
"""Merge per-model rows into the model-size DenseLinear leaderboard.

Scans <runs_root>/<tag>/ for each model that has a per_subset_metrics.json and
emits:

    leaderboard_model_size.csv                 (all columns, machine-readable)
    leaderboard_model_size_val_threshold.md    (ranked by macro_f1_val_threshold)
    leaderboard_model_size_oracle_threshold.md (ranked by macro_f1_oracle)

Every field is read from the per-model JSON artifacts (model_config.json,
train_metrics.json, val_threshold_metrics.json, per_subset_metrics.json, and the
encode manifests); nothing is hardcoded. Missing optional fields render blank.

Usage:
    python scripts/s1ms_merge_leaderboard.py --runs_root runs/s1_model_size_dense
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SUBSETS = ["gsm8k", "math", "olympiadbench", "omnimath"]

COLUMNS = [
    "model", "params_label", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "train_examples",
    "val_examples", "val_selected_threshold", "macro_f1_val_threshold",
    "macro_f1_oracle",
    "gsm8k_f1_val_threshold", "math_f1_val_threshold",
    "olympiadbench_f1_val_threshold", "omnimath_f1_val_threshold",
    "gsm8k_f1_oracle", "math_f1_oracle",
    "olympiadbench_f1_oracle", "omnimath_f1_oracle",
    "encode_walltime", "train_walltime", "eval_walltime",
]


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def encode_walltime(model_dir: Path) -> float | None:
    """Wall estimate: slowest PRM shard + slowest PB subset (parallel within stage)."""
    prm_times: list[float] = []
    for mani in (model_dir / "prm800k_encode_shards").glob("shard_*/encoding_manifest.json"):
        d = load_json(mani) or {}
        t = (d.get("timing") or {}).get("total_encoding_time_sec")
        if t is not None:
            prm_times.append(float(t))
    pb_times: list[float] = []
    for mani in (model_dir / "processbench_eval_shards").glob("*/encoding_manifest_pb.json"):
        d = load_json(mani) or {}
        t = (d.get("timing") or {}).get("total_encoding_time_sec")
        if t is not None:
            pb_times.append(float(t))
    if not prm_times and not pb_times:
        return None
    return (max(prm_times) if prm_times else 0.0) + (max(pb_times) if pb_times else 0.0)


def build_row(model_dir: Path) -> dict | None:
    per_subset = load_json(model_dir / "per_subset_metrics.json")
    if per_subset is None:
        return None
    cfg = load_json(model_dir / "model_config.json") or {}
    train_m = load_json(model_dir / "train_metrics.json") or {}
    val_thr = load_json(model_dir / "val_threshold_metrics.json") or {}

    ps = per_subset.get("per_subset", {})

    def f1(subset: str, kind: str):
        node = ps.get(subset, {}).get(kind, {})
        return node.get("F1_PB")

    row = {
        "model": cfg.get("model_name", model_dir.name),
        "params_label": cfg.get("params_label"),
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "train_examples": train_m.get("probe_train_n"),
        "val_examples": train_m.get("val_n"),
        "val_selected_threshold": val_thr.get("selected_threshold",
                                              train_m.get("selected_threshold")),
        "macro_f1_val_threshold": per_subset.get("macro_f1_val_threshold"),
        "macro_f1_oracle": per_subset.get("macro_f1_oracle"),
        "encode_walltime": encode_walltime(model_dir),
        "train_walltime": train_m.get("probe_train_time_sec"),
        "eval_walltime": per_subset.get("eval_walltime_sec"),
    }
    for s in SUBSETS:
        row[f"{s}_f1_val_threshold"] = f1(s, "val_selected")
        row[f"{s}_f1_oracle"] = f1(s, "oracle")
    return row


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_md(path: Path, rows: list[dict], sort_key: str, title: str, headline_cols: list[str]) -> None:
    ranked = sorted(rows, key=lambda r: (r.get(sort_key) is None, -(r.get(sort_key) or 0.0)))
    lines = [f"# {title}", "",
             "Macro F1_PB across the 4 ProcessBench subsets (gsm8k, math, "
             "olympiadbench, omnimath). DenseLinear probe on the final-layer "
             "last-token hidden state; PRM800K 40K train / 1K val; no context "
             "truncation (sequences conditioned on question + all previous steps "
             "+ current step).", ""]
    lines.append("| " + " | ".join(headline_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(headline_cols)) + " |")
    for r in ranked:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in headline_cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, default=Path("runs/s1_model_size_dense"))
    args = p.parse_args()

    rows: list[dict] = []
    for child in sorted(args.runs_root.iterdir()):
        if child.is_dir():
            row = build_row(child)
            if row is not None:
                rows.append(row)

    csv_path = args.runs_root / "leaderboard_model_size.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in COLUMNS})

    val_cols = ["params_label", "model", "hidden_size", "num_hidden_layers",
                "val_selected_threshold", "macro_f1_val_threshold",
                "gsm8k_f1_val_threshold", "math_f1_val_threshold",
                "olympiadbench_f1_val_threshold", "omnimath_f1_val_threshold"]
    oracle_cols = ["params_label", "model", "hidden_size", "num_hidden_layers",
                   "macro_f1_oracle", "gsm8k_f1_oracle", "math_f1_oracle",
                   "olympiadbench_f1_oracle", "omnimath_f1_oracle"]

    write_md(args.runs_root / "leaderboard_model_size_val_threshold.md", rows,
             "macro_f1_val_threshold",
             "Model-Size DenseLinear Leaderboard (val-selected threshold)", val_cols)
    write_md(args.runs_root / "leaderboard_model_size_oracle_threshold.md", rows,
             "macro_f1_oracle",
             "Model-Size DenseLinear Leaderboard (oracle threshold)", oracle_cols)

    print(f"[leaderboard] {len(rows)} model row(s) -> {csv_path}")
    for r in rows:
        print(f"  {fmt(r.get('params_label')):>6}  "
              f"val={fmt(r.get('macro_f1_val_threshold'))}  "
              f"oracle={fmt(r.get('macro_f1_oracle'))}")


if __name__ == "__main__":
    main()
