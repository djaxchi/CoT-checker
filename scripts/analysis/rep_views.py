#!/usr/bin/env python3
"""How each candidate represents correct and incorrect steps (read_token_v1 Stage K).

One vector per PRM800K step for every candidate (an encode directory in the layout of
scripts/encode_pooled_multilayer.py, read by backbone_geometry.load_split). For each:

    pca   the top two principal components of the val vectors (no labels)
    lda   shrunk LDA (Ledoit-Wolf) fit on val, scored on test

Every projection is fit on val_5k and only applied to test_2k, so no candidate sees
the test labels. Reported per candidate: test AUROC of the LDA score, of PC1 and PC2
alone (max of AUROC and 1 - AUROC, the sign of a PC being arbitrary), of the
candidate's own head where it has one, and d' along the LDA axis. Written for the
page: the same 500 correct and 500 incorrect test steps for every candidate (uids in
all candidates), their PC1, PC2 and LDA values in units of the val standard deviation,
and LDA histograms over every test step.

    python scripts/analysis/rep_views.py --out views.json \
        --cand "Online read token=<dir>:sep:-1" "PRM-7B=<dir>:sep:28" ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.backbone_geometry import load_split  # noqa: E402


def vectors(d: Path, split: str, rep: str, layer: int):
    S, meta = load_split(d, split)
    li = [int(x) for x in S["layers"]].index(layer)
    hp = S["head_p_correct"] if "head_p_correct" in S else np.full(len(S["y"]), np.nan)
    return S[rep][:, li].astype(np.float64), S["y"].astype(int), [m["uid"] for m in meta], hp


def sym_auc(y, s) -> float:
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cand", nargs="+", required=True, help="label=dir:rep:layer")
    p.add_argument("--fit", default="val")
    p.add_argument("--test", default="test")
    p.add_argument("--n_points", type=int, default=500, help="per class")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    cands = []
    for spec in a.cand:
        label, rest = spec.split("=", 1)
        d, rep, layer = rest.rsplit(":", 2)
        cands.append((label, Path(d), rep, int(layer)))
    data = {}
    for label, d, rep, layer in cands:
        data[label] = (vectors(d, a.fit, rep, layer), vectors(d, a.test, rep, layer))
        print(f"[views] {label}: fit {len(data[label][0][1])}, test {len(data[label][1][1])}, "
              f"dim {data[label][0][0].shape[1]}", flush=True)

    # the same test steps for every candidate
    common = set.intersection(*(set(t[2]) for _, t in data.values()))
    if len(common) < 2 * a.n_points:
        raise SystemExit(f"only {len(common)} test steps are in every candidate")
    first = next(iter(data.values()))[1]
    lab = {u: int(y) for u, y in zip(first[2], first[1])}
    rng = np.random.default_rng(a.seed)
    pick = []
    for c in (0, 1):
        pool = sorted(u for u in common if lab[u] == c)
        pick += list(rng.choice(pool, min(a.n_points, len(pool)), replace=False))
    print(f"[views] {len(common)} test steps in every candidate; plotting {len(pick)}", flush=True)

    out = {"fit": a.fit, "test": a.test, "points_label": [lab[u] for u in pick], "candidates": []}
    for label, d, rep, layer in cands:
        (xf, yf, _, _), (xt, yt, ut, hpt) = data[label]
        pca = PCA(2, random_state=a.seed).fit(xf)
        pf, pt = pca.transform(xf), pca.transform(xt)
        sd = pf.std(0)
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(xf, yf)
        lf, lt = lda.decision_function(xf), lda.decision_function(xt)
        # one unit = the val pooled within-class standard deviation along the axis
        wsd = np.sqrt(((lf[yf == 1] - lf[yf == 1].mean()) ** 2).sum() / len(lf)
                      + ((lf[yf == 0] - lf[yf == 0].mean()) ** 2).sum() / len(lf))
        c0 = lf[yf == 0].mean()
        lz = (lt - c0) / wsd
        dprime = float((lt[yt == 1].mean() - lt[yt == 0].mean())
                       / np.sqrt(0.5 * (lt[yt == 1].var() + lt[yt == 0].var())))
        idx = {u: i for i, u in enumerate(ut)}
        rows = [idx[u] for u in pick]
        edges = np.linspace(np.quantile(lz, 0.005), np.quantile(lz, 0.995), a.bins + 1)
        h1, _ = np.histogram(np.clip(lz[yt == 1], edges[0], edges[-1]), edges)
        h0, _ = np.histogram(np.clip(lz[yt == 0], edges[0], edges[-1]), edges)
        r = {"label": label, "rep": rep, "layer": layer, "dim": int(xf.shape[1]),
             "lda_test_auroc": float(roc_auc_score(yt, lt)), "dprime": dprime,
             "pc1_auroc": sym_auc(yt, pt[:, 0]), "pc2_auroc": sym_auc(yt, pt[:, 1]),
             "pc_var_share": [float(v) for v in pca.explained_variance_ratio_],
             "head_test_auroc": (float(roc_auc_score(yt, 1 - hpt)) if np.isfinite(hpt).all() else None),
             "points": [[round(float(pt[i, 0] / sd[0]), 3), round(float(pt[i, 1] / sd[1]), 3),
                         round(float(lz[i]), 3)] for i in rows],
             "hist": {"edges": [round(float(e), 3) for e in edges],
                      "correct": (h0 / max(1, h0.sum())).round(4).tolist(),
                      "incorrect": (h1 / max(1, h1.sum())).round(4).tolist()}}
        out["candidates"].append(r)
        print(f"[views] {label}: LDA test AUROC {r['lda_test_auroc']:.4f}, d' {dprime:.2f}, "
              f"PC1 {r['pc1_auroc']:.3f}, PC2 {r['pc2_auroc']:.3f}, head {r['head_test_auroc']}", flush=True)
    a.out.write_text(json.dumps(out))
    print(f"[views] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
