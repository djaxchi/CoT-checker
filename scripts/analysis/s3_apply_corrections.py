"""Apply human corrections to the Opus labels and recompute the contingency.

Reads failure_labels_opus.jsonl + label_corrections.json (audit trail of
{sample_id: {from, to, reason}}), writes failure_labels_final.jsonl with a
`corrected_from` field on changed rows, and rebuilds the failure-mode x detection
contingency from the corrected labels.

Outputs (results/s3_first_error/):
  - failure_labels_final.jsonl
  - failure_mode_by_detection_final.csv
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.eval.failure_taxonomy import FAILURE_MODES

ROOT = Path("results/s3_first_error")


def main() -> None:
    labels = [json.loads(l) for l in (ROOT / "failure_labels_opus.jsonl").read_text().splitlines() if l]
    corr = json.loads((ROOT / "label_corrections.json").read_text())

    applied = 0
    for r in labels:
        c = corr.get(r["sample_id"])
        if not c:
            continue
        if r["failure_mode"] != c["from"]:
            print(f"[warn] {r['sample_id']} current={r['failure_mode']} != correction.from={c['from']} (skipping)")
            continue
        if c["to"] not in FAILURE_MODES:
            print(f"[warn] {r['sample_id']} target mode {c['to']} not in taxonomy (skipping)")
            continue
        r["corrected_from"] = r["failure_mode"]
        r["failure_mode"] = c["to"]
        applied += 1
    print(f"[apply] {applied}/{len(corr)} corrections applied over {len(labels)} labels")

    with (ROOT / "failure_labels_final.jsonl").open("w") as f:
        for r in labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- recompute contingency ----
    by_mode = defaultdict(Counter); by_score = defaultdict(list)
    for r in labels:
        by_mode[r["failure_mode"]]["det" if r["detected"] else "mis"] += 1
        by_score[r["failure_mode"]].append(r["probe_score"])
    dist = Counter(r["failure_mode"] for r in labels)

    print(f"\n[final dist] n={len(labels)}:")
    for m in FAILURE_MODES:
        if dist[m]:
            print(f"  {m:34s} {dist[m]:3d}  ({dist[m]/len(labels)*100:4.1f}%)")

    with (ROOT / "failure_mode_by_detection_final.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["failure_mode", "n", "detected", "missed", "detection_rate", "mean_probe_score"])
        print("\n[final contingency] failure mode x detection (sample is 50/50 detected/missed):")
        for m in FAILURE_MODES:
            det = by_mode[m]["det"]; mis = by_mode[m]["mis"]; n = det + mis
            if n == 0:
                continue
            rate = det / n; ms = sum(by_score[m]) / n
            w.writerow([m, n, det, mis, round(rate, 3), round(ms, 3)])
            print(f"  {m:34s} n={n:3d}  det={det:3d} mis={mis:3d}  rate={rate:.2f}  mean_score={ms:.2f}")
    print(f"\n[apply] wrote failure_labels_final.jsonl + failure_mode_by_detection_final.csv")


if __name__ == "__main__":
    main()
