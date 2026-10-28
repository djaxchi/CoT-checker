#!/usr/bin/env python3
"""Does a scorer's decision threshold transfer to ProcessBench without its labels?

calib-20 picks the threshold on 20 labelled traces of each ProcessBench subset; a user
without ProcessBench labels has to bring a threshold from elsewhere. For each cell
(ProcessBench step scores in the leaderboard format, P(incorrect) per step), F1_PB
at thresholds of increasing access to ProcessBench labels:

  p05        0.5, the classifier's own decision point (no labels at all)
  val_bacc   picked on PRM800K val step labels, best balanced accuracy (the
             leaderboard's "val-selected" rule, on raw probabilities); cells given --val
  val_f1     the same, best step F1 on the incorrect class
  em         no labels: the subset's error prior estimated from its own unlabelled step
             scores by EM (Saerens et al. 2002), assuming the scorer was trained at a
             50 % prior (PRM800K train and val are balanced); a posterior above 0.5
             after the prior shift is a threshold of 1 - prior on the raw scores
             (read-token cells only: the PRM's training prior is unknown)
  nat        no labels: the same prior shift to the natural PRM800K step-error rate
             (--natural_prior, the unbalanced count before the 50 % balancing; 0.278 =
             261,511 / 939,415 in build_full_manifest.json), threshold 1 - prior
  xsub       picked on the other three ProcessBench subsets (best mean F1_PB there),
             applied to the held-out one: another domain's labels, none of this one's
  calib20    20 labelled traces of the subset (the leaderboard's rule)
  oracle     the best threshold on the subset itself (ceiling)

plus, at 0.5, the fraction of traces the scorer calls erroneous against the true
fraction (over-flagging shows up here), and the oracle threshold per subset (a
threshold that moves across subsets cannot transfer).

    python scripts/analysis/rth_threshold_transfer.py \
        --cells rth_s0=<pb_scores>/dq_correct prm=<prm_head_scores> \
        --val rth_s0=<bundle>/val_5k --out results/read_token_v1/mech/threshold_transfer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.merge_rep_grid_leaderboard import calib20_subset, f1_pb_from_preds, pred_matrix  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
RULES = ("p05", "em", "nat", "val_bacc", "val_f1", "xsub", "calib20", "oracle")


def load_cell(d: Path) -> dict[str, list[tuple[int, list[float]]]]:
    out = {}
    for sub in PB_SUBSETS:
        rows = [json.loads(l) for l in (d / f"pb_step_scores_{sub}.jsonl").read_text().splitlines() if l.strip()]
        out[sub] = [(int(r["label"]), [float(x) for x in r["scores"]]) for r in rows]
    return out


def grid_of(traces) -> np.ndarray:
    s = np.concatenate([np.asarray(sc, float) for _, sc in traces])
    return np.unique(np.concatenate([np.quantile(s, np.linspace(0.001, 0.999, 999)), [0.5]]))


def f1_curve(traces, grid: np.ndarray) -> np.ndarray:
    return f1_pb_from_preds(*pred_matrix(traces, grid))


def f1_at(traces, t: float) -> float:
    return float(f1_curve(traces, np.array([t]))[0])


def plateau_mid(curve: np.ndarray) -> int:
    """Index of the middle of the best plateau (not its lowest edge)."""
    best = np.flatnonzero(curve >= curve.max() - 1e-12)
    return int(best[len(best) // 2])


def em_prior(p: np.ndarray, prior_train: float = 0.5, iters: int = 200) -> float:
    """The target prior of the positive class from unlabelled posteriors (Saerens et al. 2002)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    pi = prior_train
    for _ in range(iters):
        a = pi / prior_train * p
        b = (1 - pi) / (1 - prior_train) * (1 - p)
        new = float((a / (a + b)).mean())
        if abs(new - pi) < 1e-9:
            break
        pi = new
    return pi


def val_threshold(p: np.ndarray, y: np.ndarray, crit: str) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.001, 0.999, 999)))
    pred = p[None, :] > grid[:, None]
    tp, fp = (pred & (y == 1)).sum(1), (pred & (y == 0)).sum(1)
    fn, tn = (~pred & (y == 1)).sum(1), (~pred & (y == 0)).sum(1)
    tpr, tnr = tp / max((y == 1).sum(), 1), tn / max((y == 0).sum(), 1)
    if crit == "bacc":
        score = 0.5 * (tpr + tnr)
    else:
        prec = tp / np.maximum(tp + fp, 1)
        score = 2 * prec * tpr / np.maximum(prec + tpr, 1e-12)
    return float(grid[plateau_mid(score)])


def cell_report(cell: dict, val: tuple[np.ndarray, np.ndarray] | None, em: bool = False,
                natural_prior: float | None = None) -> dict:
    grids = {s: grid_of(cell[s]) for s in PB_SUBSETS}
    union = np.unique(np.concatenate(list(grids.values())))
    curves = {s: f1_curve(cell[s], union) for s in PB_SUBSETS}
    thr = {"p05": 0.5}
    if val is not None:
        thr["val_bacc"] = val_threshold(*val, "bacc")
        thr["val_f1"] = val_threshold(*val, "f1")
    if em and natural_prior is not None:
        thr["nat"] = 1 - natural_prior
    out = {"thresholds": thr, "by_subset": {}}
    for s in PB_SUBSETS:
        tr = cell[s]
        others = np.mean([curves[o] for o in PB_SUBSETS if o != s], 0)
        t_x = float(union[plateau_mid(others)])
        oi = plateau_mid(curves[s])
        preds, labels = pred_matrix(tr, np.array([0.5]))
        r = {k: f1_at(tr, t) for k, t in thr.items()}
        if em:
            pi = em_prior(np.concatenate([sc for _, sc in tr]))
            r |= {"em": f1_at(tr, 1 - pi), "em_prior": pi}
        r |= {"xsub": f1_at(tr, t_x), "xsub_threshold": t_x, "calib20": float(calib20_subset(tr)),
              "oracle": float(curves[s][oi]), "oracle_threshold": float(union[oi]),
              "pred_error_rate_p05": float((preds[:, 0] != -1).mean()),
              "true_error_rate": float((labels != -1).mean())}
        out["by_subset"][s] = r
    out["mean"] = {k: float(np.mean([out["by_subset"][s][k] for s in PB_SUBSETS]))
                   for k in RULES + ("pred_error_rate_p05", "true_error_rate") if k in out["by_subset"]["math"]}
    return out


def load_val(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    parts = sorted(split_dir.glob("part_*"))
    lg = np.concatenate([np.load(p / "head_logit.npy") for p in parts]).astype(np.float64)
    y = np.concatenate([np.load(p / "y.npy") for p in parts]).astype(np.int64)
    return 1 / (1 + np.exp(-lg)), y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", required=True, help="name=dir with pb_step_scores_<subset>.jsonl")
    p.add_argument("--val", nargs="*", default=[], help="name=bundle split dir (head_logit, y) for that cell")
    p.add_argument("--no_em", nargs="*", default=[], help="cells without the 50 %% training prior")
    p.add_argument("--natural_prior", type=float, default=None,
                   help="natural PRM800K step-error rate for the nat rule (em cells only)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    vals = {k: load_val(Path(v)) for k, v in (s.split("=", 1) for s in a.val)}
    res = {}
    for spec in a.cells:
        name, d = spec.split("=", 1)
        res[name] = cell_report(load_cell(Path(d)), vals.get(name), em=name not in a.no_em,
                                 natural_prior=a.natural_prior)
        m = res[name]["mean"]
        print(f"{name:22s} " + " ".join(f"{k} {m[k]:.3f}" for k in RULES if k in m)
              + f" | flagged@0.5 {m['pred_error_rate_p05']:.2f} vs true {m['true_error_rate']:.2f}"
              + " | oracle t " + " ".join(f"{res[name]['by_subset'][s]['oracle_threshold']:.3f}"
                                         for s in PB_SUBSETS), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
