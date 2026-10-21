#!/usr/bin/env python3
"""Save the last row of every item of node-local stores as plain arrays on shared storage.

Each argument is name=prm_root,pb_root; every split under either root is written
to <out>/<name>/<split>/{h.npy, y.npy, meta.jsonl}, in global_index order (the
ProcessBench first-error order), float16. For a vector store the last row is the
vector; for rt_layers it is the read token's final state; for a step-span store
it is the last step token.

    python scripts/analysis/save_split_vectors.py --out <mech>/vectors \
        rt_L35=<L>/prm/rt_append,<L>/pb/rt_append prm_L27=<L>/prm_states/step_spans,<L>/prm_states/pb_step_spans
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.derive_delta_from_token_store import derive_split  # noqa: E402


def save(split: Path, out: Path) -> tuple[int, int]:
    X, y, meta = derive_split(split, "last", sort=True)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "h.npy", X)
    np.save(out / "y.npy", y)
    (out / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    return X.shape


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("specs", nargs="+", help="name=prm_root,pb_root")
    a = p.parse_args()
    for spec in a.specs:
        name, roots = spec.split("=", 1)
        for root in (Path(r) for r in roots.split(",")):
            for split in sorted(x for x in root.iterdir() if x.is_dir()):
                print(f"[vectors] {name}/{split.name} {save(split, a.out / name / split.name)}", flush=True)


if __name__ == "__main__":
    main()
