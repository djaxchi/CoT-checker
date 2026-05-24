"""
Collect eval_summary.json from each dense_full run and emit a leaderboard.

Reads <runs_dir>/<method>/eval_summary.json (and train_metrics.json + the
PRM800K + PB encoding manifests when available) and writes:
    <out_csv>
    <out_md>

Row fields:
    train_stem, train_n, val_stem, val_n, pb_name, threshold_type, threshold,
    F1_PB, Acc_error, Acc_correct, Exact_match_all,
    encoding_time_sec, probe_train_time_sec, eval_time_sec
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    ap.add_argument("--prm_manifest", type=Path, default=None,
                    help="Optional encoding_manifest.json for PRM800K cache.")
    ap.add_argument("--pb_manifest", type=Path, default=None,
                    help="Optional combined PB encoding_manifest_pb.json.")
    ap.add_argument("--train_stem", type=str, default="probe_train_400k")
    ap.add_argument("--val_stem", type=str, default="val_10k")
    args = ap.parse_args()

    rows: list[dict] = []
    prm_manifest = load_json(args.prm_manifest) if args.prm_manifest else None
    pb_manifest = load_json(args.pb_manifest) if args.pb_manifest else None

    def n_from_manifest(stem: str) -> int | None:
        if not prm_manifest:
            return None
        files = prm_manifest.get("files", {})
        if stem in files:
            return files[stem].get("n_examples")
        return None

    encoding_time_sec = None
    if prm_manifest:
        encoding_time_sec = prm_manifest.get("timing", {}).get("total_encoding_time_sec")

    for method_dir in sorted(args.runs_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        summary = load_json(method_dir / "eval_summary.json")
        train_m = load_json(method_dir / "train_metrics.json")
        if summary is None or train_m is None:
            continue

        train_n = train_m.get("probe_train_n") or n_from_manifest(args.train_stem)
        val_n = train_m.get("val_n") or n_from_manifest(args.val_stem)
        probe_train_time = train_m.get("probe_train_time_sec")

        for entry in summary["runs"]:
            rows.append({
                "method": train_m.get("method"),
                "run_dir": str(method_dir),
                "train_stem": args.train_stem,
                "train_n": train_n,
                "val_stem": args.val_stem,
                "val_n": val_n,
                "pb_name": entry.get("pb_name"),
                "threshold_type": entry.get("threshold_type"),
                "threshold": entry.get("threshold"),
                "F1_PB": entry.get("F1_PB"),
                "Acc_error": entry.get("Acc_error"),
                "Acc_correct": entry.get("Acc_correct"),
                "Exact_match_all": entry.get("Exact_match_all"),
                "n_traces": entry.get("n_traces"),
                "encoding_time_sec": encoding_time_sec,
                "probe_train_time_sec": probe_train_time,
                "eval_time_sec": entry.get("eval_time_sec"),
            })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        args.out_csv.write_text("")

    # Markdown
    md_lines = ["# Dense Full Leaderboard", ""]
    if not rows:
        md_lines.append("(no runs found)")
    else:
        keys = ["method", "pb_name", "threshold_type", "threshold", "F1_PB",
                "Acc_error", "Acc_correct", "Exact_match_all", "n_traces",
                "train_n", "val_n", "probe_train_time_sec", "eval_time_sec"]
        md_lines.append("| " + " | ".join(keys) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for r in rows:
            md_lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    args.out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {args.out_csv} and {args.out_md} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
