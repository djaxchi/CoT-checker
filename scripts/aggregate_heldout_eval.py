"""Aggregate per-size eval JSONs from eval_prm800k_heldout_probe.py into one table.

The all-sizes encode+eval array job writes results/prm800k_test_full_eval/<label>.json
(one per model size). This collects them, orders by parameter count, and writes a
single CSV (+ prints a compact table) for the cross-size F1 comparison.

Usage:
    python scripts/aggregate_heldout_eval.py \
        --in_dir results/prm800k_test_full_eval --out results/prm800k_test_full_eval/table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

COLS = ["tag", "n", "deployed_threshold", "auc", "test_f1",
        "test_balanced_accuracy", "test_accuracy", "oracle_balanced_accuracy",
        "val_balanced_accuracy", "val_to_test_gap", "encoding"]


def size_key(tag: str) -> float:
    """Sort 1.5B < 3B < 7B < 14B < 32B by numeric param count."""
    m = re.search(r"([\d.]+)\s*[bB]", str(tag))
    return float(m.group(1)) if m else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, default=Path("results/prm800k_test_full_eval"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(p for p in args.in_dir.glob("*.json") if p.name != "table.json")
    rows = [json.loads(p.read_text()) for p in files]
    rows.sort(key=lambda r: size_key(r.get("tag", "")))
    if not rows:
        raise SystemExit(f"no eval JSONs found in {args.in_dir}")

    out = args.out or (args.in_dir / "table.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r.get(c) for c in COLS])

    hdr = f"{'size':>6} {'n':>6} {'AUC':>6} {'F1':>6} {'bal-acc':>8} {'acc':>6} {'oracle':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{str(r.get('tag','')):>6} {r.get('n',0):>6} "
              f"{r.get('auc',float('nan')):>6.3f} {r.get('test_f1',float('nan')):>6.3f} "
              f"{r.get('test_balanced_accuracy',float('nan')):>8.3f} "
              f"{r.get('test_accuracy',float('nan')):>6.3f} "
              f"{r.get('oracle_balanced_accuracy',float('nan')):>7.3f}")
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
