#!/usr/bin/env python3
"""LDA on the inline read token and on the passive text states of the same pass (Stage H).

`read_token_inline.py encode` stores, for every step, the read token's final state
(`sep`) and the text states at the separator token it follows (`passive_block<l>`,
`passive_final`). The passive rows are what a free probe on the generation states
would read, so they are the control the read token has to beat for its extra token
to pay. Each row gets the same LDA as M1 (scripts/analysis/rth_layer_lda.py: fit on
the encode's train sample, shrinkage on val AUROC, scored on test and ProcessBench).

    python scripts/analysis/inline_passive_lda.py --enc <local>/h1_dq_correct --out passive_lda.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.rth_layer_lda import PB_SUBSETS, layer_lda  # noqa: E402


def load(enc: Path, name: str) -> tuple[dict, np.ndarray, list[dict]]:
    z = np.load(enc / f"{name}.shard0.npz")
    if not z["kept"].all():
        raise SystemExit(f"{enc}/{name}: {int((~z['kept']).sum())} items were not encoded")
    meta = [json.loads(l) for l in (enc / f"{name}.shard0.meta.jsonl").read_text().splitlines() if l.strip()]
    reps = {"read": z["sep"][:, 0].astype(np.float32)}
    for k in sorted(z.files):
        if k.startswith("passive_"):
            reps[k] = z[k][:, 0].astype(np.float32)
    return reps, z["y"].astype(np.int64), meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enc", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = (load(a.enc, n) for n in ("train", "val", "test"))
    pb = {s: load(a.enc, f"pb_{s}") for s in PB_SUBSETS}
    # prm_geometry's ProcessBench code reads the trace's first error as `label`
    pb_meta = {}
    for s, v in pb.items():
        n = Counter(m["id"] for m in v[2])
        pb_meta[s] = [{"id": m["id"], "step_idx": m["step_idx"], "label": m["first_error"],
                       "n_steps": n[m["id"]]} for m in v[2]]
    out = {"enc": str(a.enc), "n_train": int(len(tr[1])), "reps": {}}
    for rep in tr[0]:
        r = layer_lda(tr[0][rep], tr[1], va[0][rep], va[1], te[0][rep], te[1],
                      {s: (pb[s][0][rep], pb_meta[s]) for s in PB_SUBSETS}, dev)
        out["reps"][rep] = r
        print(f"[lda] {rep}: test AUROC {r['test_auroc']:.4f} PB AUROC {r['pb_auroc_mean']:.4f} "
              f"calib-20 {r['pb_calib20']:.4f} oracle {r['pb_oracle']:.4f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
