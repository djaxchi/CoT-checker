#!/usr/bin/env python3
"""Does the dense correctness readout survive within a step type?

§21.29 found the strongest Qwen-Scope features separating incorrect from correct
steps encode the kind of step (asserting a computed value vs planning), not an
error. If the dense LDA readout mostly reads step type, its AUROC should collapse
inside a type. This fits LDA on dense `step_mean` (Qwen3-8B-Base,
hidden_states[35]) on the training split, then scores test_2k and ProcessBench
known steps overall and within strata defined by SAE features chosen on the
training split only (the top positive- and negative-d features of
results/sae_localization_v0/loc/top_features.json). A logistic model on those
features alone is the "type only" reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.one_class_correct import Gauss  # noqa: E402
from scripts.analysis.sae_feature_localization import (  # noqa: E402
    SUBSETS, known_mask, load_csr, load_dense)


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae_dir", type=Path, required=True)
    p.add_argument("--dense_cache", type=Path, required=True)
    p.add_argument("--top_features", type=Path, required=True)
    p.add_argument("--n_type", type=int, default=6, help="features per sign")
    p.add_argument("--n_fit", type=int, default=150_000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    feats = json.load(open(a.top_features))
    pos = [f["feature"] for f in feats if f["cohen_d"] > 0][:a.n_type]
    neg = [f["feature"] for f in feats if f["cohen_d"] < 0][:a.n_type]
    type_cols = np.array(pos + neg)
    rng = np.random.default_rng(0)

    tr_x, tr_y, _ = load_dense(a.dense_cache, "step_mean", "probe_train_full")
    idx = np.sort(rng.choice(len(tr_y), min(a.n_fit, len(tr_y)), replace=False))
    x = np.asarray(tr_x[idx], np.float64); y = tr_y[idx]
    mu1, mu0 = x[y == 1].mean(0), x[y == 0].mean(0)
    pooled = Gauss(np.concatenate([x[y == 1] - mu1, x[y == 0] - mu0])); pooled.mu = x.mean(0)
    w = (1 - 0.1) * pooled.evals + 0.1 * pooled.tr
    dmu_e = (mu1 - mu0) @ pooled.evecs
    lda = lambda X: ((np.asarray(X, np.float64) - pooled.mu) @ pooled.evecs) @ (dmu_e / w)  # noqa: E731

    sae_tr, sae_ytr = load_csr(a.sae_dir / "sae_mean__probe_train_full.npz")
    t_tr = np.asarray(sae_tr[idx][:, type_cols].todense())
    type_clf = LogisticRegression(max_iter=2000).fit(t_tr, sae_ytr[idx])

    sets = {"test": (load_dense(a.dense_cache, "step_mean", "test_2k"),
                     load_csr(a.sae_dir / "sae_mean__test_2k.npz"), None)}
    for s in SUBSETS:
        dx = load_dense(a.dense_cache, "step_mean", s)
        sets[s] = (dx, load_csr(a.sae_dir / s / "sae_mean.npz"), known_mask(dx[2]))
    out = {"pos_features": pos, "neg_features": neg, "sets": {}}
    for name, ((dx, dy, _), (sx, sy), known) in sets.items():
        assert np.array_equal(dy, sy)
        k = np.ones(len(dy), bool) if known is None else known
        score = lda(dx)[k]; yy = dy[k].astype(int)
        t = np.asarray(sx[:, type_cols].todense())[k]
        comp = (t[:, :len(pos)] > 0).any(1)
        plan = (t[:, len(pos):] > 0).any(1)
        strata = {"all": np.ones(len(yy), bool), "computational": comp & ~plan,
                  "planning": plan & ~comp, "both": comp & plan, "neither": ~comp & ~plan}
        r = {"type_only_auroc": auc(yy, type_clf.decision_function(t))}
        for sn, m in strata.items():
            r[sn] = {"n": int(m.sum()), "prev_incorrect": float(yy[m].mean()) if m.any() else float("nan"),
                     "lda_auroc": auc(yy[m], score[m])}
        out["sets"][name] = r
        print(name, json.dumps(r))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
