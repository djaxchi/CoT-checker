#!/usr/bin/env python3
"""Checks for slurm/rep_separation_tamia.sh, printed before anything is read off its outputs.

1. The re-encoded PRM states and the Instruct store give the same LDA test AUROC
   as the prm_backbone_v1 geometry run (within 0.002).
2. The mechanistic bundle agrees with the stores it should reproduce, on test_2k:
   Qwen3-8B last step token row 34 with the Instruct store's layer 35, the read
   token row 34 with rt_L35, the PRM row 26 with the PRM store's layer 27
   (median cosine at least 0.999; bf16 batches of different composition differ
   slightly).
Exits non-zero if any check fails.
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
from scripts.read_token_mech_bundle import load_bundle  # noqa: E402

OLD_GEO = Path("/scratch/d/dchikhi/cot_mech/prm_backbone_v1/geometry")


def median_cos(u: np.ndarray, v: np.ndarray) -> float:
    u, v = np.asarray(u, np.float32), np.asarray(v, np.float32)
    return float(np.median((u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geo", type=Path, required=True)
    p.add_argument("--mech", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True, help="Instruct repstore root")
    p.add_argument("--old_geo", type=Path, default=OLD_GEO)
    a = p.parse_args()
    ok = True
    for bb in ("prm", "instruct"):
        old = json.loads((a.old_geo / bb / "last_token/metrics.json").read_text())["methods"]["lda"]["auroc_test"]
        new = json.loads((a.geo / bb / "last_token/metrics.json").read_text())["methods"]["lda"]["auroc_test"]
        good = abs(old - new) <= 0.002
        ok &= good
        print(f"[repro] {bb} last_token LDA test AUROC: prm_backbone_v1 {old:.4f}, this run {new:.4f} "
              f"{'ok' if good else 'MISMATCH'}")
    b = load_bundle(a.mech / "bundle" / "test_2k")
    inst, _, meta = derive_split(a.ref / "step_spans" / "test_2k", "last", sort=True)
    if [m["uid"] for m in meta] != [m["uid"] for m in b["meta"]]:
        raise SystemExit("bundle test_2k is not in the Instruct store's order")
    uid = [m["uid"] for m in b["meta"]]

    def vectors(name):
        d = a.mech / "vectors" / name / "test_2k"
        pos = {json.loads(l)["uid"]: k for k, l in enumerate((d / "meta.jsonl").read_text().splitlines())}
        keep = [k for k, u in enumerate(uid) if u in pos]
        return keep, np.load(d / "h.npy")[[pos[uid[k]] for k in keep]]

    k_rt, rt = vectors("rt_L35")
    k_prm, prm = vectors("prm_L27")
    for name, rows, ref in (
            ("last_step row 34 vs Instruct store L35", b["last_step"][:, 34], inst),
            ("read row 34 vs rt_L35", b["read"][k_rt, 34], rt),
            (f"prm row 26 vs PRM store L27 ({len(k_prm)} items in both)", b["prm"][k_prm, 26], prm)):
        c = median_cos(rows, ref)
        ok &= c >= 0.999
        print(f"[bundle check] {name}: median cosine {c:.5f} {'ok' if c >= 0.999 else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
