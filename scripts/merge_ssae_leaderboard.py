"""Merge SSAE per-method outputs into leaderboard_ssae.csv and .md.

Reads from runs/<method>/{eval_metrics.json, threshold.json,
probe_train_metrics.json, ssae_train_metrics.json, latents/latent_manifest.json}.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METHODS = ("ssae_positive", "ssae_mixed", "ssae_contrastive")

COLUMNS = [
    "method",
    "threshold",
    "F1_PB",
    "Acc_error",
    "Acc_correct",
    "Exact_match_all",
    "best_val_balanced_accuracy",
    "val_f1_binary",
    "reconstruction_ce",
    "l1_mean",
    "aux_bce",
    "representation_train_time_sec",
    "latent_extraction_time_sec",
    "probe_train_time_sec",
    "eval_time_sec",
    "mean_step_latency_ms",
    "mean_trace_latency_ms",
    "n_latents",
    "representation_train_n",
    "probe_train_n",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", required=True, type=Path)
    p.add_argument("--out_csv", required=True, type=Path)
    p.add_argument("--out_md", required=True, type=Path)
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for method in METHODS:
        d = args.runs_dir / method
        if not d.is_dir():
            continue
        ev = load_json(d / "eval_metrics.json")
        th = load_json(d / "threshold.json")
        pr = load_json(d / "probe_train_metrics.json")
        tr = load_json(d / "ssae_train_metrics.json")
        manifest = load_json(d / "latents" / "latent_manifest.json")
        latent_ext_time = None
        if manifest.get("extraction_time_sec"):
            latent_ext_time = sum(manifest["extraction_time_sec"].values())
        # rep_train_n: count rows in the train JSONL named in ssae_train_metrics.
        rep_train_n: int | None = None
        train_jsonl = tr.get("train_jsonl")
        if train_jsonl and Path(train_jsonl).exists():
            with open(train_jsonl) as f:
                rep_train_n = sum(1 for line in f if line.strip())
        rows.append({
            "method": method,
            "threshold": ev.get("threshold"),
            "F1_PB": ev.get("F1_PB"),
            "Acc_error": ev.get("Acc_error"),
            "Acc_correct": ev.get("Acc_correct"),
            "Exact_match_all": ev.get("Exact_match_all"),
            "best_val_balanced_accuracy": th.get("best_val_balanced_accuracy"),
            "val_f1_binary": th.get("val_f1_binary"),
            "reconstruction_ce": tr.get("reconstruction_ce"),
            "l1_mean": tr.get("l1_mean"),
            "aux_bce": tr.get("aux_bce"),
            "representation_train_time_sec": tr.get("train_time_sec"),
            "latent_extraction_time_sec": latent_ext_time,
            "probe_train_time_sec": pr.get("probe_train_time_sec"),
            "eval_time_sec": ev.get("eval_time_sec"),
            "mean_step_latency_ms": ev.get("mean_step_latency_ms"),
            "mean_trace_latency_ms": ev.get("mean_trace_latency_ms"),
            "n_latents": tr.get("n_latents") or pr.get("n_latents"),
            "representation_train_n": rep_train_n,
            "probe_train_n": pr.get("probe_train_n"),
        })

    rows.sort(key=lambda r: (-(r["F1_PB"] or 0.0), r["method"]))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    def fmt(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join(["---"] * len(COLUMNS)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in COLUMNS) + " |")
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_csv} and {args.out_md} ({len(rows)} methods).")


if __name__ == "__main__":
    main()
