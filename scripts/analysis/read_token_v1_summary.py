#!/usr/bin/env python3
"""One table over read_token_v0 and read_token_v1 outputs (docs/read_token_v1_plan.md).

For every geometry_<run>/geometry.json under the given roots: RTH, the token's own
jointly trained head (method `prm_head` in backbone_geometry's naming), LDA, the raw class-mean axis
and the best single PC with its index, as PRM800K test AUROC and ProcessBench
calib-20 F1_PB. Then the B6 trained readouts, if present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    a = p.parse_args()
    print(f"{'run':<26}{'rep':<10}{'RTH head':>14}{'lda':>14}{'mean_axis':>14}{'best_pc':>18}")
    for root in a.roots:
        for g in sorted(root.glob("geometry_*/geometry.json")):
            rows = json.loads(g.read_text())
            head = {r["config"]: r for r in rows if r["method"] == "prm_head"}
            keys = sorted({(r["config"], r["rep"]) for r in rows if r["method"] != "prm_head"})
            for cfg, rep in keys:
                by = {r["method"]: r for r in rows if r["config"] == cfg and r["rep"] == rep}
                cell = lambda r: f"{r['test_auroc']:.3f}/{r['pb_calib20']:.3f}" if r else "-"  # noqa: E731
                pc = by.get("best_pc")
                pc_cell = f"{cell(pc)} (PC{pc['pc']})" if pc else "-"
                print(f"{cfg:<26}{rep:<10}{cell(head.get(cfg)):>14}{cell(by.get('lda')):>14}"
                      f"{cell(by.get('mean_axis')):>14}  {pc_cell:<18}")
        for f in sorted(root.glob("readouts_*/readouts.json")):
            for r in json.loads(f.read_text()):
                print(f"{r['config']:<26}{r['rep']:<10}{r['method']:>14} test {r['test_auroc']:.3f} "
                      f"PB calib-20 {r['pb_calib20']:.3f} (val {r['val_auroc']:.3f})")
    print("cells: PRM800K test AUROC / ProcessBench calib-20 F1_PB (four-subset mean)")


if __name__ == "__main__":
    main()
