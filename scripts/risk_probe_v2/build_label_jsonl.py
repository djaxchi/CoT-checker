#!/usr/bin/env python3
"""The risk_probe_v2 label set in the span encoder's input format (plan section 3).

One row per risk_probe_v1 label step (views/lab_views.jsonl), with the problem and
step texts of its aligned `full` view (views/inputs/*.jsonl): problem, prefix (the
earlier steps joined by blank lines, as in the PRM800K and ProcessBench stores) and
candidate_step, plus the fields scripts/encode_prm800k_token_store.py reads (uid,
problem_id, solution_id, step_idx, label) and the targets: label = hard `harmful`,
soft = P(err_after > err_before) under Beta posteriors. Writes <out> and
<out stem>.audit.json (counts, sha256 of inputs and output).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.risk_probe_v2 import parse_header, soft_harmful  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", type=Path, required=True, help="risk_probe_v1 views dir")
    ap.add_argument("--step_values", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lab = [json.loads(l) for l in open(a.views / "lab_views.jsonl") if l.strip()]
    want = {r["view_id"] for r in lab}
    inputs = {}
    for f in sorted((a.views / "inputs").glob("*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r["trace_id"] in want:
                inputs[r["trace_id"]] = r
    sv = {(r["trace_id"], r["step"]): r for r in map(json.loads, open(a.step_values)) }
    missing = [r["view_id"] for r in lab if r["view_id"] not in inputs or (r["trace_id"], r["step"]) not in sv]
    if missing:
        sys.exit(f"[label] {len(missing)} label steps without a view or label, e.g. {missing[:3]}")
    v = [sv[(r["trace_id"], r["step"])] for r in lab]
    q = soft_harmful(np.array([round(x["err_before"] * x["k_before"]) for x in v]), np.array([x["k_before"] for x in v]),
                     np.array([round(x["err_after"] * x["k_after"]) for x in v]), np.array([x["k_after"] for x in v]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r, x, qq in zip(lab, v, q):
            inp = inputs[r["view_id"]]
            f.write(json.dumps({
                "uid": r["view_id"], "problem_id": r["problem_key"], "solution_id": r["trace_id"],
                "step_idx": r["step"], "label": int(x["harmful"]), "soft": float(qq), "fold": r["fold"],
                "risk": x["risk"], "se": x["se"], "y_ds": x["y_ds"] if x["mask_ds"] else -1, "mask_ds": x["mask_ds"],
                "phase": x.get("phase"), "problem": parse_header(inp["header"]),
                "prefix": "\n\n".join(inp["steps"][:-1]), "candidate_step": inp["steps"][-1]}, ensure_ascii=False) + "\n")
    audit = {"n_rows": len(lab), "n_harmful": int(sum(int(x["harmful"]) for x in v)), "soft_mean": float(np.mean(q)),
             "sha256": {"lab_views": sha(a.views / "lab_views.jsonl"), "step_values": sha(a.step_values), "out": sha(a.out)}}
    a.out.with_name(a.out.stem + ".audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
