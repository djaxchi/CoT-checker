#!/usr/bin/env python3
"""InStep (PRM800K) without ProcessBench problems ("no-overlap" protocol): data and stability inputs.

  filter  for each seed, InStep (PRM800K)'s 100,000-item training draw exactly as
          read_token_inline.py train draws it (prm_review_data.draw_indices), minus the items whose
          normalised problem is one of ProcessBench's (--pb_problems, the --exclude_problems file):
          <out>/draw_seed<s>.uids.json (the filtered draw, for STEP and ReProbe's head) and
          <out>/filter.json (per seed: drawn, excluded, kept, share incorrect of the kept labels,
          and the constant predictor's loss on them, as score_val computes it).
  sweep   a sweep.json in read_token_inline.score_val's format for a run trained without --val:
          the running training loss of its log ("[train inline] step k/K loss <tag> v", every 50
          steps; score_val keeps every 10th), its step count and final loss, the constant
          predictor's loss from `filter`, and the validation AUROC / AP of an encode of the
          validation items (P(incorrect) = 1 - head_p_correct, labels from the npz), so
          scripts/analysis/instep_lr_select.py can apply the stability rule to it.

    python scripts/instep_v1/nopb_tools.py filter --full $PRMD/prm800k_probe_train_full.jsonl \\
        --pb_problems $W/pb_problems.json --seeds 0,1,2,3,4 --out $RV/nopb/data
    python scripts/instep_v1/nopb_tools.py sweep --log $LOGS/instep_prm_nopb_seed0-701011.log --seed 0 \\
        --filter $RV/nopb/data/filter.json --val_npz $RV/nopb/val5k/seed0/val5k.shard0.npz \\
        --train $PRMD/prm800k_probe_train_full.jsonl --val $PRMD/prm800k_val_5k.jsonl --out $RV/nopb/sweep/seed0
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.instep_v1.prm_review_data import draw_indices  # noqa: E402

LINE = re.compile(r"^\[train inline\] step (\d+)/(\d+) loss (\S+) (\S+)")


def const_loss(p: float) -> float:
    """score_val's constant predictor: binary entropy of the training prevalence."""
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p))) if 0 < p < 1 else 0.0


def filter_draws(lines: list[str], pb_norm: set[str], seeds: list[int], n_train: int) -> tuple[dict, dict]:
    """({seed: kept uids}, {seed: counts}) for each seed's draw minus ProcessBench problems."""
    from src.data.prm_trajectories import normalize_problem
    uids, info = {}, {}
    for s in seeds:
        rows = [json.loads(lines[i]) for i in draw_indices(len(lines), n_train, s)]
        keep = [r for r in rows if normalize_problem(r["problem"]) not in pb_norm]
        y = np.array([int(r["label"]) for r in keep])
        uids[s] = [r["uid"] for r in keep]
        info[s] = {"drawn": len(rows), "excluded": len(rows) - len(keep), "kept": len(keep),
                   "share_incorrect": float(y.mean()), "const_loss": const_loss(float(y.mean()))}
    return uids, info


def parse_log(text: str) -> dict:
    """Running-loss trace, step count, final loss and whether the run died, from a training log."""
    trace, steps, tag, dead = [], None, None, None
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        k, K, tag, v = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        steps = K
        if v == "dead":
            dead = k if dead is None else dead
            continue
        trace.append([k, float(v)])
    if "non-finite loss" in text and dead is None:
        dead = trace[-1][0] if trace else 0
    if steps is None:
        raise SystemExit("no '[train inline] step' line in the log")
    return {"trace": trace, "steps": steps, "tag": tag, "dead_at": dead,
            "final_loss": trace[-1][1] if trace else None, "finished": bool(trace) and trace[-1][0] == steps}


def val_metrics(npz: Path) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    with np.load(npz) as z:
        y, kept, pc = z["y"].astype(int), z["kept"].astype(bool), z["head_p_correct"].astype(np.float64)
    s = 1 - pc
    if not kept.all() or not np.isfinite(s).all():
        return {"val_auroc": None, "val_ap": None, "val_n": int(len(y)), "val_prev": float(y.mean())}
    return {"val_auroc": float(roc_auc_score(y, s)), "val_ap": float(average_precision_score(y, s)),
            "val_n": int(len(y)), "val_prev": float(y.mean())}


def build_sweep(log: dict, seed: int, counts: dict, val: dict, train: str, val_path: str, lr: float,
                warmup: float) -> dict:
    return {"seed": seed, "train": train, "val": val_path, "layout": "chat", "n_train_items": counts["kept"],
            "train_prev": counts["share_incorrect"], "const_loss": counts["const_loss"], "steps": log["steps"],
            "val_n": val["val_n"], "val_prev": val["val_prev"], "reconstructed": True,
            "trace_every": 50, "configs": [{"tag": log["tag"], "lr": lr, "warmup": warmup, "dead_at": log["dead_at"],
                                            "final_loss": log["final_loss"], "trace": log["trace"],
                                            "val_auroc": val["val_auroc"], "val_ap": val["val_ap"]}]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["filter", "sweep"])
    p.add_argument("--full", type=Path, help="filter: prm800k_probe_train_full.jsonl")
    p.add_argument("--pb_problems", type=Path, help="filter: ProcessBench problems (json list)")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n_train", type=int, default=100000)
    p.add_argument("--log", type=Path, help="sweep: training log")
    p.add_argument("--seed", type=int, help="sweep")
    p.add_argument("--filter", type=Path, help="sweep: filter.json")
    p.add_argument("--val_npz", type=Path, help="sweep: encode of the validation items")
    p.add_argument("--train", default="", help="sweep: training file (recorded)")
    p.add_argument("--val", default="", help="sweep: validation file (recorded)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.phase == "filter":
        from src.data.prm_trajectories import normalize_problem
        lines = [l for l in open(a.full) if l.strip()]
        pb_norm = {normalize_problem(q) for q in json.loads(a.pb_problems.read_text())}
        uids, info = filter_draws(lines, pb_norm, [int(x) for x in a.seeds.split(",")], a.n_train)
        for s, u in uids.items():
            (a.out / f"draw_seed{s}.uids.json").write_text(json.dumps(u))
            print(f"[filter] seed {s}: {info[s]}", flush=True)
        (a.out / "filter.json").write_text(json.dumps({"pb_problems": len(pb_norm), "seeds": info}, indent=1))
        return
    counts = json.loads(a.filter.read_text())["seeds"][str(a.seed)]
    log = parse_log(a.log.read_text())
    if not log["finished"]:
        raise SystemExit(f"{a.log}: training did not reach its last step")
    sw = build_sweep(log, a.seed, counts, val_metrics(a.val_npz), a.train, a.val, a.lr, a.warmup)
    (a.out / "sweep.json").write_text(json.dumps(sw))
    c = sw["configs"][0]
    print(f"[sweep] seed {a.seed}: {log['steps']} steps, final loss {c['final_loss']}, constant {sw['const_loss']:.4f}, "
          f"val AUROC {c['val_auroc']}", flush=True)


if __name__ == "__main__":
    main()
