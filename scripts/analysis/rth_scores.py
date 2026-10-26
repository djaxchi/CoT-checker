#!/usr/bin/env python3
"""ProcessBench step scores of RTH, in the leaderboard cells' file format.

RTH (Read-Token Head) is the deep read token classified by its own linear head,
trained jointly with the token (docs/read_token_v1_plan.md). Its head logit for
every ProcessBench step is in the mechanistic bundle
(scripts/read_token_mech_bundle.py). This writes pb_step_scores_<subset>.jsonl
(one trace per line: id, first-error label, n_steps, P(incorrect) per step) in
the same trace order as the leaderboard cells, so the calib-20 code and the paired
trace bootstrap (scripts/analysis/stage_f_bootstrap.py) read it like any cell.

    python scripts/analysis/rth_scores.py --bundle <mech>/bundle --out <stage_f>/cells_rth/rth_dq_correct \
        --like <stage_f>/cells_fab/step_tokens__transformer_d512_l2_f2048_h8__seed42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.read_token_mech_bundle import load_bundle  # noqa: E402
from scripts.train_easy_probe_method import evaluate_processbench  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def traces(logit: np.ndarray, meta: list[dict]) -> list[dict]:
    p = 1.0 / (1.0 + np.exp(-np.asarray(logit, np.float64)))
    rows, _ = evaluate_processbench(p, meta, 0.5)
    return [{"id": r["id"], "label": int(r["label"]), "n_steps": len(r["scores"]),
             "scores": [float(x) for x in r["scores"]]} for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--like", type=Path, help="a leaderboard cell whose trace order must be matched")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for sub in PB_SUBSETS:
        b = load_bundle(a.bundle / sub, arrays=("head_logit",))
        rows = traces(b["head_logit"], b["meta"])
        if a.like is not None:
            ref = [json.loads(l)["id"] for l in (a.like / f"pb_step_scores_{sub}.jsonl").read_text().splitlines()
                   if l.strip()]
            if ref != [r["id"] for r in rows]:
                raise SystemExit(f"{sub}: trace order differs from {a.like}")
        with (a.out / f"pb_step_scores_{sub}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[rth] {sub}: {len(rows)} traces", flush=True)


if __name__ == "__main__":
    main()
