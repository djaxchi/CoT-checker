#!/usr/bin/env python3
"""Pick C* and m_all from read_token_v0 by PRM800K val AUROC (read_token_v1).

Selection rule (docs/read_token_v1_plan.md): C* is the v0 configuration whose
trained head has the highest val AUROC; within 0.002 of the best, the smallest m
wins. m_all is C* if its m is above 1, otherwise the best val AUROC among the
configurations with m in {4, 16}. ProcessBench is never used to select; v0's
best ProcessBench calib-20 (head or LDA, from geometry.json) is only the gate
that sends the run to Stage B (> 0.568, passive Qwen3-8B L35 LDA) or Stage D.

Writes selection.json and prints shell assignments for the slurm script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

V0 = {"m1": (1, "vocab"), "m4": (4, "vocab"), "m16": (16, "vocab"), "m4text": (4, "text")}
TIE = 0.002
PASSIVE_LDA = 0.568


def select(val_auroc: dict[str, float], configs: dict[str, tuple[int, str]] = V0) -> tuple[str, str]:
    best = max(val_auroc.values())
    near = [n for n in val_auroc if val_auroc[n] >= best - TIE]
    cstar = min(near, key=lambda n: (configs[n][0], -val_auroc[n]))
    if configs[cstar][0] > 1:
        return cstar, cstar
    multi = [n for n in val_auroc if configs[n][0] in (4, 16)]
    return cstar, max(multi, key=lambda n: val_auroc[n])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v0", type=Path, required=True, help="read_token_v0 OUT (geometry_<name>/)")
    p.add_argument("--val", nargs="+", required=True, help="name:dir with val.shard0.npz")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    val = {}
    for spec in a.val:
        name, d = spec.split(":", 1)
        z = np.load(Path(d) / "val.shard0.npz")
        kept = z["kept"].astype(bool)
        val[name] = float(roc_auc_score(z["y"][kept], 1 - z["head_p_correct"][kept]))
    cstar, m_all = select(val)
    pb = {}
    for name in val:
        rows = json.loads((a.v0 / f"geometry_{name}" / "geometry.json").read_text())
        pb[name] = max(r["pb_calib20"] for r in rows if r["method"] in ("prm_head", "lda"))
    v0_best = max(pb.values())
    out = {"val_auroc": val, "v0_pb_calib20": pb, "v0_best_pb_calib20": v0_best,
           "cstar": cstar, "m_all": m_all, "stage": "B" if v0_best > PASSIVE_LDA else "D",
           "configs": {n: {"m": V0[n][0], "init": V0[n][1]} for n in val}}
    a.out.write_text(json.dumps(out, indent=1))
    print(f"CSTAR={cstar} CSTAR_M={V0[cstar][0]} CSTAR_INIT={V0[cstar][1]} "
          f"MALL={m_all} MALL_M={V0[m_all][0]} MALL_INIT={V0[m_all][1]} GATE_STAGE={out['stage']}")


if __name__ == "__main__":
    main()
