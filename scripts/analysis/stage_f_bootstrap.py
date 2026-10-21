#!/usr/bin/env python3
"""Paired trace bootstrap of the read_token_v1 Stage F comparisons (calib-20 F1_PB).

Each comparison is calib-20(a) - calib-20(b), four-subset mean, both sides
recomputed on the same resampled ProcessBench traces
(scripts/analysis/judge_prompt_report.py's boot_delta); a side with several
seeds is their mean. Comparisons run in parallel processes.

    python scripts/analysis/stage_f_bootstrap.py --stage_f <stage_f> \
        --instruct <rep_grid_q3> --prm <prm_backbone_v1/cells> --out <stage_f>/bootstrap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.judge_prompt_report import boot_delta  # noqa: E402

D512 = "step_tokens__transformer_d512_l2_f2048_h8"


def comparisons(F: Path, I: Path, P: Path) -> list[tuple[str, list[str], list[str]]]:
    fa_lin, fa_mlp = [str(F / "cells_fab/last_token__linear__seed42")], [str(F / "cells_fab/last_token__mlp_h1024__seed42")]
    drv, ft, fc = ([str(F / f"{d}/{D512}__seed42")] for d in ("cells_fab", "cells_traj", "cells_fc"))
    inst = [str(I / f"{D512}__seed{s}") for s in (42, 43, 44)]
    prm = [str(P / f"boundary_stats__mlp_h1024x2__seed{s}") for s in (42, 43, 44)]
    return [("DRV - Instruct step_tokens d512 (3 seeds)", drv, inst),
            ("DRV - Instruct step_tokens d512 seed 42", drv, inst[:1]),
            ("DRV - PRM boundary_stats mlp h1024x2 (3 seeds)", drv, prm),
            ("DRV - F-A mlp h1024", drv, fa_mlp),
            ("F-A mlp h1024 - F-A linear", fa_mlp, fa_lin),
            ("F-T - F-A mlp h1024", ft, fa_mlp),
            ("F-T - DRV", ft, drv),
            ("F-C - Instruct step_tokens d512 (3 seeds)", fc, inst)]


def one(job):
    name, a, b, n_boot = job
    return {"comparison": name, "a": a, "b": b, **boot_delta(a, b, n_boot)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage_f", type=Path, required=True)
    p.add_argument("--instruct", type=Path, required=True)
    p.add_argument("--prm", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_boot", type=int, default=200)
    a = p.parse_args()
    jobs = [(n, x, y, a.n_boot) for n, x, y in comparisons(a.stage_f, a.instruct, a.prm)]
    with ProcessPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(one, jobs))
    a.out.write_text(json.dumps(res, indent=1))
    for r in res:
        lo, hi = r["ci95_normal"]
        print(f"{r['comparison']:48s} delta {r['delta']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(delta<=0) {r['frac_boot_le0']:.3f}", flush=True)


if __name__ == "__main__":
    main()
