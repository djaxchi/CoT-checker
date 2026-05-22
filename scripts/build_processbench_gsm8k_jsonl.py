"""Materialize processbench_gsm8k.jsonl from the same raw source used by the
dense encoder (encode_processbench_hidden_states.py).

Schema (one row per step):
    {"id": "gsm8k-5", "step_idx": 2, "label": 2, "n_steps": 4,
     "problem": "...", "prefix": "...", "candidate_step": "..."}

`label` is the trace-level first-error index (-1 means trace is fully correct),
and is repeated on every row of the trace (matching pb_gsm8k_step_meta.jsonl).

Cross-check:
    Row count and (id, step_idx, label, n_steps) tuples must match
    pb_gsm8k_step_meta.jsonl exactly, in the same order. This script raises
    if they do not.

Usage:
    python scripts/build_processbench_gsm8k_jsonl.py \
      --raw_file       <path to ProcessBench gsm8k.json> \
      --pb_meta        $SCRATCH/cot_mech/prestudy_v1/cache/qwen2_5_1_5b_processbench/pb_gsm8k_step_meta.jsonl \
      --out_jsonl      $SCRATCH/cot_mech/prestudy_v1/data/processbench_gsm8k.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_file", type=Path, required=True,
                   help="Raw ProcessBench GSM8K source (gsm8k.json) — the SAME "
                        "file passed to encode_processbench_hidden_states.py.")
    p.add_argument("--pb_meta", type=Path, required=True,
                   help="pb_gsm8k_step_meta.jsonl from the dense encoder. "
                        "Used as the authoritative ordering and row-count check.")
    p.add_argument("--out_jsonl", type=Path, required=True,
                   help="Output JSONL path (typically "
                        "$SCRATCH/cot_mech/prestudy_v1/data/processbench_gsm8k.jsonl).")
    return p.parse_args()


def expand_traces(traces: list[dict]) -> list[dict]:
    """Mirror encode_all_steps' flattening logic verbatim."""
    flat: list[dict] = []
    for trace in traces:
        required = {"id", "problem", "steps", "label"}
        missing = required - set(trace.keys())
        if missing:
            raise KeyError(f"ProcessBench trace missing fields {missing}: {trace.get('id')}")
        problem = trace["problem"]
        steps = trace["steps"]
        trace_label = int(trace["label"])
        n_steps = len(steps)
        for k, step_text in enumerate(steps):
            prefix = "\n\n".join(steps[:k])  # empty for k=0
            flat.append({
                "id": trace["id"],
                "step_idx": k,
                "label": trace_label,
                "n_steps": n_steps,
                "problem": problem,
                "prefix": prefix,
                "candidate_step": step_text,
            })
    return flat


def load_pb_meta(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()

    if not args.raw_file.exists():
        sys.exit(
            f"[build_pb_jsonl] Raw ProcessBench source missing: {args.raw_file}\n"
            f"This script does not download data. Provide the SAME file used to "
            f"produce pb_gsm8k_step_h.npy (typically ProcessBench gsm8k.json)."
        )
    if not args.pb_meta.exists():
        sys.exit(
            f"[build_pb_jsonl] pb_gsm8k_step_meta.jsonl missing: {args.pb_meta}\n"
            f"This script requires the dense encoder's meta file as the "
            f"authoritative row-order reference."
        )

    print(f"[build_pb_jsonl] Loading raw traces from {args.raw_file}")
    traces = json.loads(args.raw_file.read_text(encoding="utf-8"))
    if not isinstance(traces, list) or not traces:
        sys.exit(f"[build_pb_jsonl] Raw file is not a non-empty JSON list: {args.raw_file}")

    flat = expand_traces(traces)
    print(f"[build_pb_jsonl] Flattened to {len(flat)} step rows from {len(traces)} traces")

    pb_meta = load_pb_meta(args.pb_meta)
    print(f"[build_pb_jsonl] Loaded {len(pb_meta)} reference rows from {args.pb_meta}")

    # ---- Row-count check -------------------------------------------------
    if len(flat) != len(pb_meta):
        sys.exit(
            f"[build_pb_jsonl] FATAL: row count mismatch.\n"
            f"  raw expansion: {len(flat)} steps\n"
            f"  pb_meta:       {len(pb_meta)} rows\n"
            f"The raw ProcessBench source you passed in does NOT match the "
            f"source that built pb_gsm8k_step_meta.jsonl. Refusing to write "
            f"a misaligned JSONL."
        )

    # ---- Per-row alignment check ----------------------------------------
    keys = ("id", "step_idx", "label", "n_steps")
    mismatches: list[tuple[int, dict, dict]] = []
    for i, (a, b) in enumerate(zip(flat, pb_meta)):
        for k in keys:
            if a[k] != b[k]:
                mismatches.append((i, a, b))
                break
        if len(mismatches) >= 5:
            break
    if mismatches:
        msg_lines = ["[build_pb_jsonl] FATAL: (id, step_idx, label, n_steps) mismatch:"]
        for i, a, b in mismatches:
            msg_lines.append(
                f"  row {i}: raw={ {k:a[k] for k in keys} }  pb_meta={ {k:b[k] for k in keys} }"
            )
        sys.exit("\n".join(msg_lines))

    # ---- Write output ---------------------------------------------------
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in flat:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- Trace-level label consistency (defensive) -----------------------
    seen: dict[str, int] = {}
    for row in flat:
        tid = row["id"]
        lbl = int(row["label"])
        if tid in seen and seen[tid] != lbl:
            sys.exit(
                f"[build_pb_jsonl] FATAL: inconsistent labels within trace "
                f"{tid}: {seen[tid]} vs {lbl}"
            )
        seen[tid] = lbl

    print(
        f"[build_pb_jsonl] OK: wrote {len(flat)} rows ({len(seen)} traces) "
        f"to {args.out_jsonl}"
    )


if __name__ == "__main__":
    main()
