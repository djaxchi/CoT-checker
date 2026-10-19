#!/usr/bin/env python3
"""Score step-span stores with saved leaderboard cells (risk_probe_v2 plan, sections 6-7).

For each `name=path/to/cell_dir` (a train_rep_learner_cell.py output with
results.json and model.pt), rebuild the learner from results.json, load model.pt
and write raw logits:
  * <out_dir>/<name>__label.jsonl   uid, logit for every item of --label_store/<stem>
  * <out_dir>/<name>__pb_<sub>.jsonl  id, step_idx, logit for --pb_store/<sub>
With --check_saved the ProcessBench probabilities are compared with the cell's
saved pb_step_scores_<sub>.jsonl (gate 2: median |dP| <= 0.002, max <= 0.02);
the comparison goes to <out_dir>/<name>__gate.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "risk_probe_v2"))
from train_rep_learner_cell import build_handles  # noqa: E402
from train_risk_cell import score_logits  # noqa: E402
from src.harness.learners import build_learner  # noqa: E402
from src.harness.spanloader import SpanLoader  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402


def load_cell(cell: Path, device):
    res = json.loads((cell / "results.json").read_text())
    proto = res.get("protocol", {})
    m = build_learner(res["learner"], int(res["dim"]), t_max=int(proto.get("t_max", 512)),
                      dropout=float(proto.get("dropout", 0.1))).to(device)
    m.load_state_dict(torch.load(cell / "model.pt", map_location=device))
    return m, int(proto.get("t_max", 512))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", required=True, help="name=cell_dir")
    ap.add_argument("--label_store", type=Path, default=None)
    ap.add_argument("--stem", default="label")
    ap.add_argument("--pb_store", type=Path, default=None)
    ap.add_argument("--pb_subsets", nargs="+", default=["gsm8k", "math", "olympiadbench", "omnimath"])
    ap.add_argument("--check_saved", action="store_true")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in a.cells:
        name, cell = spec.split("=", 1)
        model, t_max = load_cell(Path(cell), device)
        if a.label_store:
            h, meta = build_handles(ShardedRepSplit(a.label_store / a.stem))
            s = score_logits(model, SpanLoader(h, t_max, device, preload=True).collate, np.arange(len(h)), a.batch_size)
            with open(a.out_dir / f"{name}__label.jsonl", "w") as f:
                for m, v in zip(meta, s):
                    f.write(json.dumps({"uid": m["uid"], "logit": float(v)}) + "\n")
            print(f"[score] {name}: {len(h)} label items", flush=True)
        if a.pb_store:
            dp = []
            for sub in a.pb_subsets:
                h, meta = build_handles(ShardedRepSplit(a.pb_store / sub))
                s = score_logits(model, SpanLoader(h, t_max, device, preload=True).collate, np.arange(len(h)), a.batch_size)
                with open(a.out_dir / f"{name}__pb_{sub}.jsonl", "w") as f:
                    for m, v in zip(meta, s):
                        f.write(json.dumps({"id": m["id"], "step_idx": int(m["step_idx"]), "logit": float(v)}) + "\n")
                if a.check_saved:
                    new: dict = {}
                    for m, v in zip(meta, s):
                        new.setdefault(m["id"], []).append((int(m["step_idx"]), float(v)))
                    for l in open(Path(cell) / f"pb_step_scores_{sub}.jsonl"):
                        r = json.loads(l)
                        mine = [v for _, v in sorted(new[r["id"]])]
                        if len(mine) != len(r["scores"]):
                            sys.exit(f"[score] {name} {sub} {r['id']}: {len(mine)} steps against {len(r['scores'])} saved")
                        dp += [abs(1 / (1 + np.exp(-v)) - p) for v, p in zip(mine, r["scores"])]
            if a.check_saved:
                gate = {"n": len(dp), "median_abs_dP": float(np.median(dp)), "max_abs_dP": float(np.max(dp)),
                        "pass": bool(np.median(dp) <= 0.002 and np.max(dp) <= 0.02)}
                (a.out_dir / f"{name}__gate.json").write_text(json.dumps(gate))
                print(f"[score] {name} reproduction: {gate}", flush=True)


if __name__ == "__main__":
    main()
