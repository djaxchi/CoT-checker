#!/usr/bin/env python3
"""Did process supervision create the PRM's correctness direction, or amplify the parent's?

Qwen2.5-Math-PRM-7B is fine-tuned from Qwen2.5-Math-7B-Instruct, so their
residual coordinates are comparable. For a (layer, readout) block this fits LDA
on each model's own training states and then:

  * cosine between the two LDA directions and between the two class-mean axes;
  * PRM direction applied to parent states (and the reverse), scored by test
    AUROC and ProcessBench calib-20, against each model's own direction.

Rows are aligned by uid, since the two encodes were sharded differently.
Prespecified reading (docs/backbone_geometry_v0_plan.md, cross-model addendum):
"amplified" if the PRM direction on parent states reaches the parent's own LDA
calib-20 within 0.02; "new" if it falls more than 0.05 below.
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
from scripts.analysis.backbone_geometry import SHRINK, SUBSETS, evaluate, load_split  # noqa: E402
from scripts.analysis.one_class_correct import Gauss  # noqa: E402


def aligned(da: Path, db: Path, name: str):
    A, ma = load_split(da, name)
    B, mb = load_split(db, name)
    ib = {m["uid"]: i for i, m in enumerate(mb)}
    ia = [i for i, m in enumerate(ma) if m["uid"] in ib]
    jb = [ib[ma[i]["uid"]] for i in ia]
    return A, B, ia, jb, [ma[i] for i in ia]


def lda_fit(x, y, xv, yv):
    mu1, mu0 = x[y == 1].mean(0), x[y == 0].mean(0)
    g = Gauss(np.concatenate([x[y == 1] - mu1, x[y == 0] - mu0])); g.mu = x.mean(0)
    best = None
    for lam in SHRINK:
        w = (1 - lam) * g.evals + lam * g.tr
        dirn = g.evecs @ (((mu1 - mu0) @ g.evecs) / w)
        a = roc_auc_score(yv, (xv - g.mu) @ dirn)
        if best is None or a > best[0]:
            best = (a, dirn)
    return best[1] / np.linalg.norm(best[1]), g.mu, (mu1 - mu0) / np.linalg.norm(mu1 - mu0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prm", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--blocks", nargs="+", default=["27:mean", "28:last", "20:last"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = {k: aligned(a.prm, a.parent, k) for k in ["train", "val", "test"] + [f"pb_{s}" for s in SUBSETS]}
    rows = []
    for spec in a.blocks:
        layer, rep = int(spec.split(":")[0]), spec.split(":")[1]
        X = {}
        for k, (A, B, ia, jb, meta) in data.items():
            li = [int(l) for l in A["layers"]].index(layer)
            y = A["y"][ia]
            assert np.array_equal(y, B["y"][jb])
            X[k] = (A[rep][ia, li].astype(np.float64), B[rep][jb, li].astype(np.float64), y, meta)
        d_prm, mu_prm, m_prm = lda_fit(X["train"][0], X["train"][2], X["val"][0], X["val"][2])
        d_par, mu_par, m_par = lda_fit(X["train"][1], X["train"][2], X["val"][1], X["val"][2])
        res = {"block": spec, "cos_lda": float(d_prm @ d_par), "cos_mean_axis": float(m_prm @ m_par)}
        for src, dirn in (("prm_dir", d_prm), ("parent_dir", d_par)):
            for tgt, ti, mu in (("on_prm", 0, mu_prm), ("on_parent", 1, mu_par)):
                ev = {"test": ((X["test"][ti] - mu) @ dirn, X["test"][2])}
                for s in SUBSETS:
                    k = f"pb_{s}"
                    ev[s] = ((X[k][ti] - mu) @ dirn, X[k][2], X[k][3])
                r = evaluate(lambda z: z, ev)
                res[f"{src}_{tgt}"] = {"test_auroc": r["test_auroc"], "pb_auroc": r["pb_auroc"],
                                       "pb_calib20": r["pb_calib20"]}
        rows.append(res)
        print(json.dumps(res), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
