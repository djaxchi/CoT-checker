"""Merge Opus relabels, compare to Haiku, rebuild the contingency.

Reads results/s3_first_error/chunks/labels_opus_*.jsonl, validates, joins probe
detection/score, and writes the Opus-labelled outputs. Also reports Haiku-vs-Opus
agreement (overall + where they disagree on arithmetic) so we know how much the
relabel moved things.

Outputs (results/s3_first_error/):
  - failure_labels_opus.jsonl
  - failure_labels_opus_review.csv
  - failure_mode_by_detection_opus.csv
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.eval.failure_taxonomy import FAILURE_MODES

ROOT = Path("results/s3_first_error")


def load_jsonl_map(path: Path, key="sample_id") -> dict[str, dict]:
    return {json.loads(l)[key]: json.loads(l)
            for l in path.read_text().splitlines() if l.strip()}


def main() -> None:
    sample = load_jsonl_map(ROOT / "first_error_sample.jsonl")
    haiku = (load_jsonl_map(ROOT / "failure_labels.jsonl")
             if (ROOT / "failure_labels.jsonl").exists() else {})

    opus: dict[str, dict] = {}
    bad = 0
    for p in sorted((ROOT / "chunks").glob("labels_opus_*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = r.get("sample_id")
            if sid not in sample:
                continue
            if r.get("failure_mode") not in FAILURE_MODES:
                bad += 1
                continue
            opus[sid] = r

    missing = [s for s in sample if s not in opus]
    print(f"[opus] sample={len(sample)} | labelled={len(opus)} | missing={len(missing)} | invalid={bad}")
    if missing:
        print(f"[opus] MISSING ({len(missing)}): {missing[:10]}")

    rows = []
    for sid, lab in opus.items():
        s = sample[sid]
        rows.append({
            "sample_id": sid, "subset": s["subset"], "detected": s["detected"],
            "probe_score": s["probe_score"], "failure_mode": lab["failure_mode"],
            "confidence": lab.get("confidence", ""), "rationale": lab.get("rationale", ""),
        })
    with (ROOT / "failure_labels_opus.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (ROOT / "failure_labels_opus_review.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "subset", "detected", "probe_score",
                    "failure_mode", "confidence", "rationale"])
        for r in sorted(rows, key=lambda x: (x["subset"], x["failure_mode"])):
            w.writerow([r[k] for k in ("sample_id", "subset", "detected", "probe_score",
                                       "failure_mode", "confidence", "rationale")])

    # ---- distribution ----
    dist = Counter(r["failure_mode"] for r in rows)
    print(f"\n[opus dist] n={len(rows)}:")
    for m in FAILURE_MODES:
        if dist[m]:
            print(f"  {m:34s} {dist[m]:3d}  ({dist[m]/len(rows)*100:4.1f}%)")

    # ---- contingency ----
    by_mode = defaultdict(Counter); by_score = defaultdict(list)
    for r in rows:
        by_mode[r["failure_mode"]]["det" if r["detected"] else "mis"] += 1
        by_score[r["failure_mode"]].append(r["probe_score"])
    with (ROOT / "failure_mode_by_detection_opus.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["failure_mode", "n", "detected", "missed", "detection_rate", "mean_probe_score"])
        print("\n[opus contingency] failure mode x detection (sample is 50/50 detected/missed):")
        for m in FAILURE_MODES:
            det = by_mode[m]["det"]; mis = by_mode[m]["mis"]; n = det + mis
            if n == 0:
                continue
            rate = det / n; ms = sum(by_score[m]) / n
            w.writerow([m, n, det, mis, round(rate, 3), round(ms, 3)])
            print(f"  {m:34s} n={n:3d}  det={det:3d} mis={mis:3d}  rate={rate:.2f}  mean_score={ms:.2f}")

    # ---- Haiku vs Opus agreement ----
    if haiku:
        both = [s for s in opus if s in haiku]
        agree = sum(1 for s in both if opus[s]["failure_mode"] == haiku[s]["failure_mode"])
        print(f"\n[agreement] Haiku vs Opus: {agree}/{len(both)} = {agree/len(both)*100:.1f}% same label")
        # where Haiku said arithmetic, what did Opus say?
        moved = Counter(opus[s]["failure_mode"] for s in both
                        if haiku[s]["failure_mode"] == "arithmetic_error")
        print("  Haiku 'arithmetic_error' -> Opus:")
        for m, c in moved.most_common():
            print(f"    {m:34s} {c}")
    print(f"\n[opus] wrote failure_labels_opus.jsonl + review + contingency -> {ROOT}/")


if __name__ == "__main__":
    main()
