#!/usr/bin/env python3
"""Can a density model of CORRECT steps alone detect incorrect ones?

Background (project memory, conicity study 2026-08-19, Qwen2.5-7B dense_last):
centered correct steps sit in a tight cone against a permuted-label null
(z = 5.7) while incorrect steps do not (z = 1.0). If incorrectness is "anything
off the correct manifold", a Gaussian fitted to correct steps only should rank
incorrect steps nearly as well as two-class LDA, and might transfer better,
since it never fits the source dataset's particular errors.

Scorers, all closed form, fit on the training split, shrinkage picked on val:
  oc_correct   Mahalanobis distance to the correct-step Gaussian (one-class)
  oc_incorrect minus the distance to the incorrect-step Gaussian (mirror control)
  lda          two-class LDA with shrunk pooled covariance (the §21.15 baseline)
  euclid       Euclidean distance to the correct centroid (no whitening)

Metrics: PRM800K test AUROC; ProcessBench known-step AUROC and calib-20 F1_PB,
four-subset means. Prespecified GO (docs/sae_localization_v0_plan.md, one-class
section): oc_correct within 0.02 of lda on test AUROC and not more than 0.02
below it on ProcessBench calib-20; or oc_correct above lda on calib-20 by 0.02.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.sae_feature_localization import SUBSETS, load_dense, pb_traces  # noqa: E402
from scripts.merge_rep_grid_leaderboard import calib20_subset  # noqa: E402

SHRINK = (1e-4, 1e-3, 1e-2, 0.1, 0.5)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Gauss:
    """Mean and eigendecomposed covariance; shrinkage applied at scoring time."""

    def __init__(self, x: np.ndarray):
        self.mu = x.mean(0)
        c = np.cov(x - self.mu, rowvar=False)
        self.evals, self.evecs = np.linalg.eigh(c)
        self.evals = np.clip(self.evals, 0, None)
        self.tr = float(self.evals.mean())

    def maha(self, x: np.ndarray, lam: float) -> np.ndarray:
        w = (1 - lam) * self.evals + lam * self.tr
        z = (x - self.mu) @ self.evecs
        return (z ** 2 / w).sum(1)


def lda_score(g1: Gauss, g0: Gauss, pooled: Gauss, x: np.ndarray, lam: float) -> np.ndarray:
    w = (1 - lam) * pooled.evals + lam * pooled.tr
    dmu = (g1.mu - g0.mu) @ pooled.evecs
    return ((x - pooled.mu) @ pooled.evecs) @ (dmu / w)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense_cache", type=Path, required=True)
    p.add_argument("--reps", nargs="+", default=["step_mean", "last_token"])
    p.add_argument("--n_fit", type=int, default=150_000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results = {}
    for rep in a.reps:
        tr_x, tr_y, _ = load_dense(a.dense_cache, rep, "probe_train_full")
        idx = np.sort(rng.choice(len(tr_y), min(a.n_fit, len(tr_y)), replace=False))
        x = np.asarray(tr_x[idx], dtype=np.float64); y = tr_y[idx]
        log(f"{rep}: fit on {len(y)} rows, dim {x.shape[1]}")
        g1, g0 = Gauss(x[y == 1]), Gauss(x[y == 0])
        xc = np.concatenate([x[y == 1] - g1.mu, x[y == 0] - g0.mu])
        pooled = Gauss(xc); pooled.mu = x.mean(0)
        del xc
        va_x, va_y, _ = load_dense(a.dense_cache, rep, "val_5k")
        te_x, te_y, _ = load_dense(a.dense_cache, rep, "test_2k")
        pb = {s: load_dense(a.dense_cache, rep, s) for s in SUBSETS}
        scorers = {
            "oc_correct": lambda X, l: g0.maha(X, l),
            "oc_incorrect": lambda X, l: -g1.maha(X, l),
            "lda": lambda X, l: lda_score(g1, g0, pooled, X, l),
            "euclid": lambda X, l: ((X - g0.mu) ** 2).sum(1),
        }
        res = {}
        for name, f in scorers.items():
            lams = (0.0,) if name == "euclid" else SHRINK
            va = {l: roc_auc_score(va_y, f(np.asarray(va_x, np.float64), l)) for l in lams}
            lam = max(va, key=va.get)
            r = {"shrink": lam, "val_auroc": va[lam],
                 "test_auroc": float(roc_auc_score(te_y, f(np.asarray(te_x, np.float64), lam)))}
            aus, cs = [], []
            for s, (px, py, meta) in pb.items():
                sc = f(np.asarray(px, np.float64), lam)
                # calib-20 thresholds on raw scores via a rank-preserving map to (0,1)
                rk = (np.argsort(np.argsort(sc)) + 0.5) / len(sc)
                aus.append(roc_auc_score(py, sc)); cs.append(calib20_subset(pb_traces(meta, rk)))
                r[f"pb_{s}_auroc"] = float(aus[-1]); r[f"pb_{s}_calib20"] = float(cs[-1])
            r["pb_auroc"] = float(np.mean(aus)); r["pb_calib20"] = float(np.mean(cs))
            res[name] = r
            log(f"{rep} {name}: shrink={lam} test={r['test_auroc']:.4f} "
                f"pb_auroc={r['pb_auroc']:.4f} pb_calib20={r['pb_calib20']:.4f}")
        oc, lda = res["oc_correct"], res["lda"]
        res["go"] = bool((lda["test_auroc"] - oc["test_auroc"] <= 0.02 and lda["pb_calib20"] - oc["pb_calib20"] <= 0.02)
                         or oc["pb_calib20"] - lda["pb_calib20"] >= 0.02)
        results[rep] = res
        (a.out / "summary.json").write_text(json.dumps(results, indent=1))
    log("done")


if __name__ == "__main__":
    main()
