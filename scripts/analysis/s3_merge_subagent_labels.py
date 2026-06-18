"""Merge subagent failure-mode labels, validate, join detection, analyse.

Reads results/s3_first_error/chunks/labels_*.jsonl (written by the Haiku
labelling subagents), dedupes on sample_id, validates against the taxonomy and
the full sample, joins probe detection/score back in, and writes:
  - failure_labels.jsonl           one validated label per sampled step
  - failure_labels_review.csv      human-verification table
  - failure_mode_by_detection.csv  failure-mode x detected/missed contingency
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.eval.failure_taxonomy import FAILURE_MODES

ROOT = Path("results/s3_first_error")


def main() -> None:
    sample = {json.loads(l)["sample_id"]: json.loads(l)
              for l in (ROOT / "first_error_sample.jsonl").read_text().splitlines() if l}

    labels: dict[str, dict] = {}
    bad_mode = 0
    for p in sorted((ROOT / "chunks").glob("labels_*.jsonl")):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = r.get("sample_id")
            if sid not in sample:  # drops stray/duplicate/malformed rows
                continue
            if r.get("failure_mode") not in FAILURE_MODES:
                bad_mode += 1
                continue
            labels[sid] = r  # last write wins; dedupes the off-by-one

    missing = [s for s in sample if s not in labels]
    print(f"[merge] sample={len(sample)} | labelled={len(labels)} | "
          f"missing={len(missing)} | invalid_mode_rows={bad_mode}")
    if missing:
        print(f"[merge] MISSING sample_ids ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    rows = []
    for sid, lab in labels.items():
        s = sample[sid]
        rows.append({
            "sample_id": sid, "subset": s["subset"], "trace_id": s["trace_id"],
            "step_idx": s["step_idx"], "detected": s["detected"],
            "probe_score": s["probe_score"], "gold_first_error": s["gold_first_error"],
            "failure_mode": lab["failure_mode"], "confidence": lab.get("confidence", ""),
            "rationale": lab.get("rationale", ""),
        })

    with (ROOT / "failure_labels.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (ROOT / "failure_labels_review.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "subset", "detected", "probe_score",
                    "failure_mode", "confidence", "rationale"])
        for r in sorted(rows, key=lambda x: (x["subset"], x["failure_mode"])):
            w.writerow([r["sample_id"], r["subset"], r["detected"], r["probe_score"],
                        r["failure_mode"], r["confidence"], r["rationale"]])

    # ---- overall distribution ----
    dist = Counter(r["failure_mode"] for r in rows)
    print(f"\n[dist] overall failure-mode distribution (n={len(rows)}):")
    for m in FAILURE_MODES:
        if dist[m]:
            print(f"  {m:34s} {dist[m]:3d}  ({dist[m]/len(rows)*100:4.1f}%)")

    # ---- failure mode x detection ----
    by_mode = defaultdict(Counter)
    by_score = defaultdict(list)
    for r in rows:
        by_mode[r["failure_mode"]]["detected" if r["detected"] else "missed"] += 1
        by_score[r["failure_mode"]].append(r["probe_score"])
    overall_rate = sum(1 for r in rows if r["detected"]) / len(rows)
    with (ROOT / "failure_mode_by_detection.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["failure_mode", "n", "detected", "missed", "detection_rate", "mean_probe_score"])
        print(f"\n[contingency] failure mode x probe detection "
              f"(overall detection rate among sampled = {overall_rate:.2f}):")
        for m in FAILURE_MODES:
            det = by_mode[m]["detected"]; mis = by_mode[m]["missed"]; n = det + mis
            if n == 0:
                continue
            rate = det / n
            ms = sum(by_score[m]) / n
            w.writerow([m, n, det, mis, round(rate, 3), round(ms, 3)])
            print(f"  {m:34s} n={n:3d}  det={det:3d} mis={mis:3d}  "
                  f"rate={rate:.2f}  mean_score={ms:.2f}")
    print(f"\n[merge] wrote failure_labels.jsonl, failure_labels_review.csv, "
          f"failure_mode_by_detection.csv -> {ROOT}/")


if __name__ == "__main__":
    main()
