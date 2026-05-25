"""Discover and inspect all ProcessBench subset files under a root directory.

For every *.jsonl (or *.json) file under --pb_root that looks like a
ProcessBench dump (filename matching ``processbench_<subset>.jsonl`` /
``processbench_<subset>.json``), the script:

* infers the subset name from the filename;
* loads the file (supports JSON list or JSONL);
* verifies the trace schema (id, problem, steps, label);
* counts traces, error traces (label != -1), correct traces (label == -1);
* counts steps and reports avg/min/max steps per trace;
* writes a single manifest JSON describing every subset found.

The manifest path defaults to::

    $SCRATCH/cot_mech/prestudy_v1/processbench_full_manifest.json

This script is read-only with respect to the PB files; it never downloads
data and never assumes there are only two subsets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


SUBSET_RE = re.compile(r"^processbench[_-](?P<name>[A-Za-z0-9._-]+)\.(?:jsonl|json)$",
                       re.IGNORECASE)


def infer_subset_name(path: Path) -> str | None:
    m = SUBSET_RE.match(path.name)
    if m:
        return m.group("name").lower()
    # Fallback: strip suffix
    stem = path.stem
    if stem.lower().startswith("processbench"):
        rest = stem[len("processbench"):].lstrip("_-")
        return rest.lower() or None
    return None


def load_traces(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    s = text.lstrip()
    if s.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: top-level JSON is not a list")
        return data
    out: list[dict] = []
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{ln}: invalid JSON line ({e})") from e
    return out


def inspect_subset(name: str, path: Path) -> dict:
    traces = load_traces(path)
    if not traces:
        return {
            "subset": name,
            "path": str(path),
            "n_traces": 0,
            "schema_ok": False,
            "error": "empty file",
        }

    keys_first = sorted(traces[0].keys())
    required = {"id", "problem", "steps", "label"}
    missing = required - set(traces[0].keys())
    schema_ok = not missing

    n_traces = len(traces)
    n_error = 0
    n_correct = 0
    label_hist: dict[str, int] = {}
    step_counts: list[int] = []
    n_total_steps = 0
    bad_rows = 0
    for t in traces:
        if not isinstance(t, dict):
            bad_rows += 1
            continue
        lbl = t.get("label", None)
        try:
            lbl_int = int(lbl)
        except (TypeError, ValueError):
            bad_rows += 1
            continue
        if lbl_int == -1:
            n_correct += 1
        else:
            n_error += 1
        label_hist[str(lbl_int)] = label_hist.get(str(lbl_int), 0) + 1
        steps = t.get("steps") or []
        step_counts.append(len(steps))
        n_total_steps += len(steps)

    steps_min = min(step_counts) if step_counts else 0
    steps_max = max(step_counts) if step_counts else 0
    steps_avg = (sum(step_counts) / len(step_counts)) if step_counts else 0.0

    return {
        "subset": name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "n_traces": n_traces,
        "n_error_traces": n_error,
        "n_correct_traces": n_correct,
        "n_steps_total": n_total_steps,
        "steps_per_trace": {
            "min": steps_min,
            "max": steps_max,
            "avg": round(steps_avg, 3),
        },
        "label_histogram": label_hist,
        "first_row_keys": keys_first,
        "schema_ok": schema_ok,
        "schema_missing_fields": sorted(missing),
        "bad_rows": bad_rows,
    }


def discover(pb_root: Path) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    # Iterate files (recursive=False is enough; PB dumps live flat).
    candidates = sorted(list(pb_root.glob("*.jsonl")) + list(pb_root.glob("*.json")))
    for p in candidates:
        name = infer_subset_name(p)
        if not name:
            continue
        # Prefer .jsonl over .json if both exist for the same subset.
        existing = found.get(name)
        if existing is None or (existing.suffix == ".json" and p.suffix == ".jsonl"):
            found[name] = p
    return [(n, found[n]) for n in sorted(found.keys())]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pb_root", type=Path, required=True,
                   help="Directory holding processbench_<subset>.jsonl files.")
    p.add_argument(
        "--out_manifest", type=Path, default=None,
        help="Output manifest JSON. Defaults to "
             "$SCRATCH/cot_mech/prestudy_v1/processbench_full_manifest.json",
    )
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-subset stdout summary lines.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pb_root.is_dir():
        sys.exit(f"[list_pb] --pb_root not a directory: {args.pb_root}")

    out_manifest = args.out_manifest
    if out_manifest is None:
        scratch = os.environ.get("SCRATCH")
        if not scratch:
            sys.exit(
                "[list_pb] --out_manifest unset and $SCRATCH not in env. "
                "Pass --out_manifest explicitly when running off-cluster."
            )
        out_manifest = Path(scratch) / "cot_mech" / "prestudy_v1" / "processbench_full_manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    pairs = discover(args.pb_root)
    if not pairs:
        sys.exit(f"[list_pb] No processbench_*.jsonl|json under {args.pb_root}")

    subsets_info: list[dict] = []
    for name, path in pairs:
        info = inspect_subset(name, path)
        subsets_info.append(info)
        if not args.quiet:
            print(
                f"[list_pb] subset={name:<14} path={path}\n"
                f"           n_traces={info['n_traces']} "
                f"err={info.get('n_error_traces','?')} "
                f"cor={info.get('n_correct_traces','?')} "
                f"steps_total={info.get('n_steps_total','?')} "
                f"schema_ok={info.get('schema_ok','?')}"
            )

    manifest = {
        "pb_root": str(args.pb_root),
        "n_subsets": len(subsets_info),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subsets": subsets_info,
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[list_pb] wrote manifest -> {out_manifest}")


if __name__ == "__main__":
    main()
