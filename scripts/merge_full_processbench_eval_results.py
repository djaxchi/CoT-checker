"""Aggregate per-job ProcessBench metrics into final leaderboards.

Scans ``<out_dir>/per_job/<method>/<subset>/{val_selected,oracle}_metrics.json``
and writes:

  leaderboard_full_pb_val_threshold.csv
  leaderboard_full_pb_val_threshold.md
  leaderboard_full_pb_oracle_threshold.csv
  leaderboard_full_pb_oracle_threshold.md
  leaderboard_full_pb_method_averages.csv
  leaderboard_full_pb_method_averages.md

Averages computed per method:
  * macro_avg_F1_PB_across_subsets (equal weight per subset, excludes
    the pooled "combined" pseudo-subset if present)
  * macro_avg_Acc_error
  * macro_avg_Acc_correct
  * macro_avg_Exact_match_all
  * pooled_F1_PB_combined (only when the 'combined' subset was evaluated)

Refuses to overwrite existing leaderboards unless --force.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


SUMMARY_COLS = [
    "method", "representation_type", "pb_subset", "threshold_type",
    "threshold", "n_traces", "n_error_traces", "n_correct_traces",
    "F1_PB", "Acc_error", "Acc_correct", "Exact_match_all", "eval_time_sec",
]
AVG_COLS = [
    "method", "threshold_type", "threshold",
    "macro_avg_F1_PB", "macro_avg_Acc_error", "macro_avg_Acc_correct",
    "macro_avg_Exact_match_all", "pooled_F1_PB_combined",
    "n_subsets_in_macro", "subsets_in_macro",
]


def write_table(rows: list[dict], cols: list[str], csv_path: Path, md_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if v is not None else "")
            elif v is None:
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n")


def macro_avg(rows: list[dict], keys=("F1_PB", "Acc_error",
                                      "Acc_correct", "Exact_match_all")
              ) -> dict:
    use = [r for r in rows if r.get("pb_subset") != "combined"]
    out: dict = {}
    if not use:
        for k in keys:
            out[f"macro_avg_{k}"] = None
        out["subsets_in_macro"] = []
        out["n_subsets_in_macro"] = 0
        return out
    for k in keys:
        vals = [r[k] for r in use if r.get(k) is not None]
        out[f"macro_avg_{k}"] = (sum(vals) / len(vals)) if vals else None
    out["subsets_in_macro"] = sorted({r["pb_subset"] for r in use})
    out["n_subsets_in_macro"] = len(out["subsets_in_macro"])
    return out


def pooled(rows: list[dict]) -> float | None:
    for r in rows:
        if r.get("pb_subset") == "combined":
            return r.get("F1_PB")
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Root that contains per_job/ and is the leaderboard target.")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Optional method allow-list. When set, ignore any "
                        "stale per_job/<method>/ directories not listed here.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    per_job = args.out_dir / "per_job"
    if not per_job.is_dir():
        sys.exit(f"[merge_eval] missing {per_job}")

    leaderboard_paths = [
        args.out_dir / "leaderboard_full_pb_val_threshold.csv",
        args.out_dir / "leaderboard_full_pb_val_threshold.md",
        args.out_dir / "leaderboard_full_pb_oracle_threshold.csv",
        args.out_dir / "leaderboard_full_pb_oracle_threshold.md",
        args.out_dir / "leaderboard_full_pb_method_averages.csv",
        args.out_dir / "leaderboard_full_pb_method_averages.md",
    ]
    if not args.force:
        for p in leaderboard_paths:
            if p.exists():
                sys.exit(f"[merge_eval] refusing to overwrite {p}; --force")

    val_rows: list[dict] = []
    ora_rows: list[dict] = []
    errors: list[dict] = []
    method_filter = set(args.methods) if args.methods else None
    for method_dir in sorted(per_job.iterdir()):
        if not method_dir.is_dir():
            continue
        if method_filter is not None and method_dir.name not in method_filter:
            continue
        for sub_dir in sorted(method_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            err_path = sub_dir / "ERROR.txt"
            if err_path.exists():
                errors.append({"method": method_dir.name,
                               "pb_subset": sub_dir.name,
                               "error": err_path.read_text().strip()})
                continue
            val_p = sub_dir / "val_selected_metrics.json"
            ora_p = sub_dir / "oracle_metrics.json"
            if val_p.exists():
                val_rows.append(json.loads(val_p.read_text()))
            if ora_p.exists():
                ora_rows.append(json.loads(ora_p.read_text()))

    if errors:
        (args.out_dir / "eval_failures.json").write_text(json.dumps(errors, indent=2))
        print(f"[merge_eval] {len(errors)} job(s) failed; see eval_failures.json")

    write_table(val_rows, SUMMARY_COLS,
                args.out_dir / "leaderboard_full_pb_val_threshold.csv",
                args.out_dir / "leaderboard_full_pb_val_threshold.md")
    write_table(ora_rows, SUMMARY_COLS,
                args.out_dir / "leaderboard_full_pb_oracle_threshold.csv",
                args.out_dir / "leaderboard_full_pb_oracle_threshold.md")

    # Per-method averages
    methods = sorted({r["method"] for r in val_rows + ora_rows})
    avg_rows: list[dict] = []
    for m in methods:
        val_for_m = [r for r in val_rows if r["method"] == m]
        ora_for_m = [r for r in ora_rows if r["method"] == m]
        if val_for_m:
            macro = macro_avg(val_for_m)
            avg_rows.append({
                "method": m,
                "threshold_type": "val_selected",
                "threshold": val_for_m[0].get("threshold"),
                "macro_avg_F1_PB": macro["macro_avg_F1_PB"],
                "macro_avg_Acc_error": macro["macro_avg_Acc_error"],
                "macro_avg_Acc_correct": macro["macro_avg_Acc_correct"],
                "macro_avg_Exact_match_all": macro["macro_avg_Exact_match_all"],
                "pooled_F1_PB_combined": pooled(val_for_m),
                "n_subsets_in_macro": macro["n_subsets_in_macro"],
                "subsets_in_macro": ",".join(macro["subsets_in_macro"]),
            })
        if ora_for_m:
            macro = macro_avg(ora_for_m)
            avg_rows.append({
                "method": m,
                "threshold_type": "oracle",
                "threshold": None,  # per-subset oracle differs
                "macro_avg_F1_PB": macro["macro_avg_F1_PB"],
                "macro_avg_Acc_error": macro["macro_avg_Acc_error"],
                "macro_avg_Acc_correct": macro["macro_avg_Acc_correct"],
                "macro_avg_Exact_match_all": macro["macro_avg_Exact_match_all"],
                "pooled_F1_PB_combined": pooled(ora_for_m),
                "n_subsets_in_macro": macro["n_subsets_in_macro"],
                "subsets_in_macro": ",".join(macro["subsets_in_macro"]),
            })

    write_table(avg_rows, AVG_COLS,
                args.out_dir / "leaderboard_full_pb_method_averages.csv",
                args.out_dir / "leaderboard_full_pb_method_averages.md")

    print(f"[merge_eval] wrote leaderboards under {args.out_dir}")
    if errors:
        print(f"[merge_eval] WARNING: {len(errors)} jobs failed; "
              "see eval_failures.json")


if __name__ == "__main__":
    main()
