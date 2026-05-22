"""Merge per-method outputs into a single easy-probe leaderboard (CSV + MD)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = [
    "method",
    "threshold",
    "F1_PB",
    "Acc_error",
    "Acc_correct",
    "Exact_match_all",
    "best_val_balanced_accuracy",
    "val_f1_binary",
    "representation_train_time_sec",
    "probe_train_time_sec",
    "eval_time_sec",
    "mean_step_latency_ms",
    "mean_trace_latency_ms",
    "latent_dim",
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
    for method_dir in sorted(args.runs_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        eval_p = method_dir / "eval_metrics.json"
        train_p = method_dir / "train_metrics.json"
        thr_p = method_dir / "threshold.json"
        if not eval_p.exists() or not train_p.exists() or not thr_p.exists():
            continue
        ev = load_json(eval_p)
        tr = load_json(train_p)
        th = load_json(thr_p)
        rows.append(
            {
                "method": tr.get("method", method_dir.name),
                "threshold": ev.get("threshold"),
                "F1_PB": ev.get("F1_PB"),
                "Acc_error": ev.get("Acc_error"),
                "Acc_correct": ev.get("Acc_correct"),
                "Exact_match_all": ev.get("Exact_match_all"),
                "best_val_balanced_accuracy": th.get("best_val_balanced_accuracy"),
                "val_f1_binary": th.get("val_f1_binary"),
                "representation_train_time_sec": tr.get("representation_train_time_sec"),
                "probe_train_time_sec": tr.get("probe_train_time_sec"),
                "eval_time_sec": ev.get("eval_time_sec"),
                "mean_step_latency_ms": ev.get("mean_step_latency_ms"),
                "mean_trace_latency_ms": ev.get("mean_trace_latency_ms"),
                "latent_dim": tr.get("latent_dim"),
                "representation_train_n": tr.get("representation_train_n"),
                "probe_train_n": tr.get("probe_train_n"),
            }
        )

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
