#!/usr/bin/env python3
"""reprobe_comparison_v1: test problems that also occur in a training source.

Problems are compared after `normalize_problem` (NFKC, lowercase, whitespace collapsed).
Writes <data>/overlap_audit.json (counts per set and source) and
<data>/overlap_<source>.json (the overlapping test trace ids, for analyze.py --exclude_traces).

    python scripts/reprobe_cmp/overlap_audit.py --data <R>/data \\
        --source prm800k_human:<dense_full_7b_v1>/data/prm800k_probe_train_full.jsonl:problem \\
        --source reprobe_train:<R>/data/rt/ds_train.items.jsonl:problem
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reprobe_cmp.analyze import TEST  # noqa: E402
from src.data.prm_trajectories import normalize_problem  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--source", nargs="+", required=True, help="name:path.jsonl:field")
    a = p.parse_args()
    audit = {}
    for spec in a.source:
        name, path, field = spec.split(":")
        probs = {normalize_problem(json.loads(l)[field]) for l in open(path) if l.strip()}
        ids, audit[name] = [], {"problems": len(probs)}
        for s in TEST:
            ts = [json.loads(l) for l in open(a.data / "test" / f"{s}.traces.jsonl") if l.strip()]
            hit = [t["trace_id"] for t in ts if normalize_problem(t["question"]) in probs]
            ids += hit
            audit[name][s] = {"traces": len(ts), "overlap": len(hit)}
        (a.data / f"overlap_{name}.json").write_text(json.dumps(ids))
        print(name, json.dumps(audit[name]), flush=True)
    (a.data / "overlap_audit.json").write_text(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
