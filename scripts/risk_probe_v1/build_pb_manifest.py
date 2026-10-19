#!/usr/bin/env python3
"""ProcessBench step views for the risk_probe_v1 readouts (post hoc).

Copies the four ProcessBench splits of context_ablation_v3's `full` view manifest
(the same view format as the label-step and fork views) into a manifest of their
own, keeping each view's shard: encode_manifest.jsonl rows, inputs/ and meta/
files. Writes build_audit.json with view counts, the shards to encode and the
sha256 of every source file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PB = ("pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="context_ablation_v3/views/full")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if (a.out / "encode_manifest.jsonl").exists():
        raise SystemExit(f"[pb] {a.out} already holds a manifest")
    for d in ("inputs", "meta"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    rows = [l for l in open(a.src / "encode_manifest.jsonl") if json.loads(l)["split"] in PB]
    (a.out / "encode_manifest.jsonl").write_text("".join(rows))
    counts, shards, sources = {}, set(), {"encode_manifest.jsonl": sha(a.src / "encode_manifest.jsonl")}
    for l in rows:
        r = json.loads(l)
        counts[r["split"]] = counts.get(r["split"], 0) + 1
        shards.add(r["shard"])
    for sp in PB:
        for d in ("inputs", "meta"):
            shutil.copyfile(a.src / d / f"{sp}.jsonl", a.out / d / f"{sp}.jsonl")
            sources[f"{d}/{sp}.jsonl"] = sha(a.src / d / f"{sp}.jsonl")
        n_in = sum(1 for _ in open(a.out / "inputs" / f"{sp}.jsonl"))
        if n_in != counts.get(sp, 0):
            raise SystemExit(f"[pb] {sp}: {n_in} inputs against {counts.get(sp, 0)} manifest rows")
    audit = {"source": str(a.src), "views": counts, "n_views": len(rows), "shards": sorted(shards), "sources_sha256": sources}
    (a.out / "build_audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps({k: audit[k] for k in ("views", "n_views", "shards")}))


if __name__ == "__main__":
    main()
