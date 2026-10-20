#!/usr/bin/env python3
"""Pick the read token Stage F encodes, by PRM800K val AUROC (read_token_v1 Stage F).

Candidates are the deep read tokens trained with seed 0 (Stage D's m4text and
the four Stage D2 start points). The rule is read_token_select's: the highest
val AUROC of the trained head wins, and within 0.002 of it the smallest m.
ProcessBench is never read. Prints `FTOKEN=<name> FTOKEN_M=<m>` for the slurm script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.read_token_select import select  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val", nargs="+", required=True, help="name:m:dir with val.shard0.npz")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    val, cfg = {}, {}
    for spec in a.val:
        name, m, d = spec.split(":", 2)
        z = np.load(Path(d) / "val.shard0.npz")
        kept = z["kept"].astype(bool)
        val[name] = float(roc_auc_score(z["y"][kept], 1 - z["head_p_correct"][kept]))
        cfg[name] = (int(m), "text")
    pick, _ = select(val, cfg)
    a.out.write_text(json.dumps({"val_auroc": val, "m": {n: c[0] for n, c in cfg.items()},
                                 "pick": pick}, indent=1))
    print(f"FTOKEN={pick} FTOKEN_M={cfg[pick][0]}")


if __name__ == "__main__":
    main()
