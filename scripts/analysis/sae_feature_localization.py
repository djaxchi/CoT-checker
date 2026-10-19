#!/usr/bin/env python3
"""Does step correctness localize to a few Qwen-Scope SAE features?

The rep grid (cot-checker-results/rep_grid_q3) already showed that a probe on
the full 65,536-wide `sae_mean` code ties its dense twin `step_mean` on
PRM800K AUROC (0.867 against 0.869, linear). That says the code keeps the
signal. It does not say whether the signal sits in a handful of features, which
is the interpretability claim an SAE would earn. This script measures that.

Both bases are ranked by the same univariate statistic on the training split
only (|Cohen's d| between incorrect and correct steps), and a logistic probe is
refit on the top-k columns for k in a fixed ladder. C is chosen on val_5k. The
same is done for the dense step_mean coordinates, so "localizes" has a
baseline: dense coordinates are not a privileged basis, and if the SAE's top-k
does no better than the dense top-k, the dictionary has not isolated anything.

Reported per k and basis: PRM800K test AUROC, ProcessBench known-step AUROC and
calib-20 F1_PB (four-subset means; AUROC on known steps only). Prespecified GO (docs/sae_localization_v0_plan.md):
at k=16 the SAE probe reaches test AUROC >= 0.80 and beats the dense top-16 by
at least 0.05. The top features are dumped with class statistics and their
most-activating training rows for reading.

Label convention: y = 1 is an incorrect step (ProcessBench prevalence ~0.09).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.merge_rep_grid_leaderboard import calib20_subset  # noqa: E402

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
CS = (0.001, 0.01, 0.1, 1.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_csr(path: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    z = np.load(path)
    n, d = (int(x) for x in z["shape"])
    x = sp.csr_matrix((z["values"].astype(np.float32), z["indices"], z["indptr"]), shape=(n, d))
    return x, z["y"].astype(np.int8)


def load_dense(cache: Path, rep: str, split: str) -> tuple[np.ndarray, np.ndarray, list | None]:
    h = glob.glob(str(cache / f"{rep}__{split}__*_h.npy"))
    if len(h) != 1:
        raise FileNotFoundError(f"{rep} {split}: {h}")
    x = np.load(h[0], mmap_mode="r")
    y = np.load(h[0].replace("_h.npy", "_y.npy")).astype(np.int8)
    m = h[0].replace("_h.npy", "_meta.jsonl")
    meta = [json.loads(l) for l in open(m)] if Path(m).exists() else None
    return x, y, meta


def cohen_d_sparse(x: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
    stats = []
    for c in (1, 0):
        xc = x[y == c]
        mu = np.asarray(xc.mean(axis=0)).ravel()
        m2 = np.asarray(xc.multiply(xc).mean(axis=0)).ravel()
        stats.append((mu, m2 - mu ** 2, xc.shape[0]))
    (m1, v1, n1), (m0, v0, n0) = stats
    sd = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
    return np.where(sd > 1e-8, (m1 - m0) / np.maximum(sd, 1e-8), 0.0)


def cohen_d_dense(x: np.ndarray, y: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    s = {0: 0.0, 1: 0.0}; s2 = {0: 0.0, 1: 0.0}; n = {0: 0, 1: 0}
    for i in range(0, len(y), chunk):
        xb = np.asarray(x[i:i + chunk], dtype=np.float64); yb = y[i:i + chunk]
        for c in (0, 1):
            xc = xb[yb == c]
            s[c] = s[c] + xc.sum(0); s2[c] = s2[c] + (xc ** 2).sum(0); n[c] += len(xc)
    mu = {c: s[c] / n[c] for c in (0, 1)}
    var = {c: s2[c] / n[c] - mu[c] ** 2 for c in (0, 1)}
    sd = np.sqrt(((n[1] - 1) * var[1] + (n[0] - 1) * var[0]) / (n[0] + n[1] - 2))
    return np.where(sd > 1e-8, (mu[1] - mu[0]) / np.maximum(sd, 1e-8), 0.0)


def cols(x, idx: np.ndarray) -> np.ndarray:
    if sp.issparse(x):
        return np.asarray(x[:, idx].todense(), dtype=np.float32)
    return np.asarray(x[:, idx], dtype=np.float32)


def pb_traces(meta: list[dict], scores: np.ndarray) -> list:
    """(first-error index or -1, scores in step order) per trace, the layout
    calib20_subset expects. ProcessBench meta carries the trace's first-error
    index in `label` on every step row; post-error steps are stored too."""
    by: dict[str, list] = {}
    fe: dict[str, int] = {}
    for m, s in zip(meta, scores):
        by.setdefault(m["id"], []).append((m["step_idx"], float(s)))
        fe[m["id"]] = int(m["label"])
    return [(fe[k], [s for _, s in sorted(v)]) for k, v in by.items()]


def known_mask(meta: list[dict]) -> np.ndarray:
    """Steps whose label is defined: every step of a correct trace, and steps up
    to and including the first error. Post-error steps are excluded from AUROC."""
    return np.array([m["label"] == -1 or m["step_idx"] <= m["label"] for m in meta])


def fit_eval(xtr, ytr, xva, yva, xte, yte, pb, rng_seed=0):
    mu = xtr.mean(0); sd = xtr.std(0) + 1e-6
    z = lambda a: (a - mu) / sd  # noqa: E731
    best = None
    for c in CS:
        clf = LogisticRegression(C=c, max_iter=2000)
        clf.fit(z(xtr), ytr)
        a = roc_auc_score(yva, clf.decision_function(z(xva)))
        if best is None or a > best[0]:
            best = (a, c, clf)
    _, c, clf = best
    res = {"C": c, "val_auroc": best[0],
           "test_auroc": float(roc_auc_score(yte, clf.decision_function(z(xte))))}
    pa, pc = [], []
    for sub, (xp, yp, meta) in pb.items():
        s = 1 / (1 + np.exp(-clf.decision_function(z(xp))))
        k = known_mask(meta)
        pa.append(roc_auc_score(yp[k], s[k]))
        pc.append(calib20_subset(pb_traces(meta, s)))
        res[f"pb_{sub}_auroc"] = float(pa[-1]); res[f"pb_{sub}_calib20"] = float(pc[-1])
    res["pb_auroc"] = float(np.mean(pa)); res["pb_calib20"] = float(np.mean(pc))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae_dir", type=Path, required=True)
    p.add_argument("--dense_cache", type=Path, required=True)
    p.add_argument("--sae_rep", default="sae_mean")
    p.add_argument("--dense_rep", default="step_mean")
    p.add_argument("--n_fit", type=int, default=200_000)
    p.add_argument("--n_top_rows", type=int, default=12)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    log("loading sae")
    S = {s: load_csr(a.sae_dir / f"{a.sae_rep}__{s}.npz") for s in ("probe_train_full", "val_5k", "test_2k")}
    ytr = S["probe_train_full"][1]
    fit_idx = np.sort(rng.choice(len(ytr), min(a.n_fit, len(ytr)), replace=False))
    dense_meta = {sub: load_dense(a.dense_cache, a.dense_rep, sub)[2] for sub in SUBSETS}
    S_pb = {}
    for sub in SUBSETS:
        x, y = load_csr(a.sae_dir / sub / f"{a.sae_rep}.npz")
        meta = [json.loads(l) for l in open(a.sae_dir / sub / f"{a.sae_rep}_meta.jsonl")]
        assert [m["id"] for m in meta] == [m["id"] for m in dense_meta[sub]], sub
        S_pb[sub] = (x, y, meta)

    log("cohen d sae")
    d_sae = cohen_d_sparse(S["probe_train_full"][0], ytr)
    order_sae = np.argsort(-np.abs(d_sae))

    log("loading dense")
    D = {s: load_dense(a.dense_cache, a.dense_rep, s) for s in ("probe_train_full", "val_5k", "test_2k")}
    assert np.array_equal(D["probe_train_full"][1], ytr), "dense/sae train labels differ"
    assert np.array_equal(D["test_2k"][1], S["test_2k"][1])
    D_pb = {sub: load_dense(a.dense_cache, a.dense_rep, sub) for sub in SUBSETS}
    log("cohen d dense")
    d_den = cohen_d_dense(D["probe_train_full"][0], ytr)
    order_den = np.argsort(-np.abs(d_den))

    # Rows used for fitting, materialized once per basis: sparse CSR for the code,
    # a float32 copy of the subsample for the dense states.
    fit_x = {"sae": S["probe_train_full"][0][fit_idx],
             "dense": np.asarray(D["probe_train_full"][0][fit_idx], dtype=np.float32)}
    rows = []
    for basis, X, XPB, order in (("sae", S, S_pb, order_sae), ("dense", D, D_pb, order_den)):
        for k in KS + ("all",):
            if k == "all":
                if basis == "sae":
                    continue  # the full-code probe is the rep grid's sae_mean cell
                idx = np.arange(X["test_2k"][0].shape[1])
            else:
                idx = np.sort(order[:k])
            xtr = cols(fit_x[basis], idx)
            pb = {sub: (cols(XPB[sub][0], idx), XPB[sub][1], XPB[sub][2] if basis == "sae" else dense_meta[sub])
                  for sub in SUBSETS}
            r = fit_eval(xtr, ytr[fit_idx], cols(X["val_5k"][0], idx), X["val_5k"][1],
                         cols(X["test_2k"][0], idx), X["test_2k"][1], pb)
            r.update(basis=basis, k=k)
            rows.append(r)
            log(f"{basis} k={k} test={r['test_auroc']:.4f} pb_auroc={r['pb_auroc']:.4f} "
                f"pb_calib20={r['pb_calib20']:.4f} C={r['C']}")
            (a.out / "ladder.json").write_text(json.dumps(rows, indent=1))

    xtr_sae = S["probe_train_full"][0]
    feats = []
    for f in order_sae[:40]:
        col = np.asarray(xtr_sae[:, int(f)].todense()).ravel()
        top = np.argsort(-col)[:a.n_top_rows]
        feats.append({"feature": int(f), "cohen_d": float(d_sae[f]),
                      "freq_incorrect": float((col[ytr == 1] > 0).mean()),
                      "freq_correct": float((col[ytr == 0] > 0).mean()),
                      "mean_incorrect": float(col[ytr == 1].mean()),
                      "mean_correct": float(col[ytr == 0].mean()),
                      "top_rows": [int(i) for i in top],
                      "top_rows_label": [int(ytr[i]) for i in top]})
    (a.out / "top_features.json").write_text(json.dumps(feats, indent=1))
    k16 = {r["basis"]: r for r in rows if r["k"] == 16}
    go = k16["sae"]["test_auroc"] >= 0.80 and k16["sae"]["test_auroc"] - k16["dense"]["test_auroc"] >= 0.05
    summary = {"go": bool(go), "k16": k16,
               "max_abs_d_sae": float(np.abs(d_sae).max()), "max_abs_d_dense": float(np.abs(d_den).max()),
               "n_fit": int(len(fit_idx))}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"GO={go}")


if __name__ == "__main__":
    main()
