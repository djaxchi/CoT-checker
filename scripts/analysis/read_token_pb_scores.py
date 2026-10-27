#!/usr/bin/env python3
"""A read token's ProcessBench head scores, from its encode outputs, in the leaderboard format.

`read_token_probe.py encode` writes pb_<subset>.shard0.npz (head P(correct) per step)
and pb_<subset>.shard0.meta.jsonl on node-local disk. This turns them into
pb_step_scores_<subset>.jsonl (one trace per line: id, first-error label, n_steps,
P(incorrect) per step), the files the leaderboard's calib-20 code and the paired
trace bootstrap (scripts/analysis/stage_f_bootstrap.py) read. With --like, traces
are written in that cell's trace order, which the paired bootstrap needs, and a
trace missing on either side stops the run.

    python scripts/analysis/read_token_pb_scores.py --enc <local>/bl1_correct_seed0 \
        --out <OUT>/pb_scores/bl1_correct_seed0 --like <stage_f>/cells_rth/rth_dq_correct
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train_easy_probe_method import evaluate_processbench  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def traces(enc: Path, sub: str) -> list[dict]:
    z = np.load(enc / f"pb_{sub}.shard0.npz")
    meta = [json.loads(l) for l in (enc / f"pb_{sub}.shard0.meta.jsonl").read_text().splitlines() if l.strip()]
    kept = z["kept"].astype(bool)
    if not kept.all():
        raise SystemExit(f"{enc}/pb_{sub}: {int((~kept).sum())} steps were not encoded")
    n = Counter(m["id"] for m in meta)
    rows = [{"id": m["id"], "step_idx": m["step_idx"], "label": m["first_error"], "n_steps": n[m["id"]]}
            for m in meta]
    out, _ = evaluate_processbench(1.0 - z["head_p_correct"].astype(np.float64), rows, 0.5)
    return [{"id": r["id"], "label": int(r["label"]), "n_steps": r["n_steps"],
             "scores": [float(x) for x in r["scores"]]} for r in out]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enc", type=Path, required=True, help="encode output dir with pb_<subset>.shard0.*")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--like", type=Path, help="a cell whose trace order to follow")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for sub in PB_SUBSETS:
        rows = traces(a.enc, sub)
        if a.like is not None:
            order = [json.loads(l)["id"] for l in (a.like / f"pb_step_scores_{sub}.jsonl").read_text().splitlines()
                     if l.strip()]
            by = {r["id"]: r for r in rows}
            if set(order) != set(by):
                raise SystemExit(f"{sub}: traces differ from {a.like} "
                                 f"({len(set(order) ^ set(by))} not in both)")
            rows = [by[i] for i in order]
        with (a.out / f"pb_step_scores_{sub}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[pb_scores] {sub}: {len(rows)} traces", flush=True)


if __name__ == "__main__":
    main()
