#!/usr/bin/env python3
"""prm_geometry_v1: is the correct/incorrect signal visible in the raw geometry
of the PRM's activations, without a trained learner? Qwen3-8B (Instruct) as the
matched control.

Every score here is closed form: class means and covariances of the PRM800K
train split, nothing optimised by gradient descent. Val (val_5k) is used only to
pick a shrinkage level and a threshold; test (test_2k) and ProcessBench are
never touched by either.

Scorers (higher = more likely an incorrect step):
  mean_diff      (mu1 - mu0) . x                    the raw class axis
  centroid_cos   cos(x-mu, c1-mu) - cos(x-mu, c0-mu) nearest centroid, cosine
  lda            Sw^-1 (mu1 - mu0) . x               the same axis, whitened
  lda_pcK        LDA restricted to the top-K PCs of the train covariance
  pc_best        the single unsupervised PC whose val AUROC is furthest from 0.5
  qda            log N(x; mu1, S1) - log N(x; mu0, S0), shrunk covariances
  cov_only       qda with both means set to the pooled mean: second-order signal
                 only, i.e. do the classes differ in SHAPE, not just location?
  knn            share of incorrect steps among the 50 cosine-nearest train steps

Geometry summaries:
  mahalanobis D between class means (shrunk Sw) and the AUROC two Gaussians with
  that separation would give, Phi(D / sqrt 2);
  how D^2 spreads over the within-class eigenbasis: participation ratio and the
  number of directions holding 50% / 90% of it;
  per-PC test AUROC for the top 256 PCs (where in the variance spectrum the
  signal sits).

Outputs per (backbone, rep): metrics.json and arrays.npz (2-D views, score
histograms, curves) for scripts/analysis/plot_prm_geometry.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
SHRINKS = (1e-4, 1e-3, 1e-2, 1e-1, 0.5)
PC_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
N_PC_AUROC = 256
KNN_K = 50
KNN_BANK = 200_000


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from scripts.train_easy_probe_method import auroc_numpy
    return float(auroc_numpy(np.asarray(y), np.asarray(s, dtype=np.float64)))


# ---------------------------------------------------------------------------
# closed-form statistics
# ---------------------------------------------------------------------------

def moments(X: np.ndarray, y: np.ndarray, dev, chunk: int = 32768):
    """Class means, class covariances and total covariance in float64, streamed."""
    d = X.shape[1]
    s = {c: torch.zeros(d, dtype=torch.float64, device=dev) for c in (0, 1)}
    ss = {c: torch.zeros(d, d, dtype=torch.float64, device=dev) for c in (0, 1)}
    n = {c: 0 for c in (0, 1)}
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + chunk], dtype=np.float32)).to(dev).double()
        yb = torch.from_numpy(np.asarray(y[i:i + chunk])).to(dev)
        for c in (0, 1):
            xc = xb[yb == c]
            s[c] += xc.sum(0)
            ss[c] += xc.T @ xc
            n[c] += int(xc.shape[0])
    mu = {c: s[c] / n[c] for c in (0, 1)}
    cov = {c: ss[c] / n[c] - torch.outer(mu[c], mu[c]) for c in (0, 1)}
    N = n[0] + n[1]
    mu_all = (s[0] + s[1]) / N
    tot = (ss[0] + ss[1]) / N - torch.outer(mu_all, mu_all)
    sw = (n[0] * cov[0] + n[1] * cov[1]) / N
    return {"mu0": mu[0], "mu1": mu[1], "mu": mu_all, "cov0": cov[0], "cov1": cov[1],
            "sw": sw, "tot": tot, "n0": n[0], "n1": n[1]}


def shrink(S: torch.Tensor, a: float) -> torch.Tensor:
    d = S.shape[0]
    return (1 - a) * S + a * (torch.trace(S) / d) * torch.eye(d, dtype=S.dtype, device=S.device)


def project(X: np.ndarray, W: torch.Tensor, dev, center: torch.Tensor | None = None,
            chunk: int = 32768) -> np.ndarray:
    """X @ W (W: (d,) or (d,k)), optionally after subtracting `center`."""
    out = []
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + chunk], dtype=np.float32)).to(dev).double()
        if center is not None:
            xb = xb - center
        out.append((xb @ W).cpu().numpy())
    return np.concatenate(out)


def gauss_loglik(X: np.ndarray, mu: torch.Tensor, S: torch.Tensor, dev,
                 chunk: int = 16384) -> np.ndarray:
    L = torch.linalg.cholesky(S)
    logdet = 2 * torch.log(torch.diagonal(L)).sum()
    out = []
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + chunk], dtype=np.float32)).to(dev).double()
        z = torch.linalg.solve_triangular(L, (xb - mu).T, upper=False)
        out.append((-0.5 * (z * z).sum(0) - 0.5 * logdet).cpu().numpy())
    return np.concatenate(out)


def knn_scores(bank: np.ndarray, ybank: np.ndarray, X: np.ndarray, mu: torch.Tensor,
               dev, k: int = KNN_K, chunk: int = 2048) -> np.ndarray:
    # built chunk by chunk in half precision: a 200k x 24,576 float32 bank alone
    # is 20 GB, and it used to coexist with the float64 covariances (OOM on the
    # Instruct boundary_stats run, job 502727)
    B = torch.empty(len(bank), bank.shape[1], dtype=torch.float16, device=dev)
    for i in range(0, len(bank), 16384):
        b = torch.from_numpy(np.ascontiguousarray(bank[i:i + 16384], dtype=np.float32)).to(dev)
        B[i:i + 16384] = torch.nn.functional.normalize(b - mu.float(), dim=1).half()
    yb = torch.from_numpy(np.asarray(ybank, dtype=np.float32)).to(dev)
    out = []
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + chunk], dtype=np.float32)).to(dev) - mu.float()
        xb = torch.nn.functional.normalize(xb, dim=1).half()
        idx = (xb @ B.T).topk(k, dim=1).indices
        out.append(yb[idx].mean(1).cpu().numpy())
    del B
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# evaluation: AUROC, in-domain F1, ProcessBench F1 (val-selected, oracle, calib-20)
# ---------------------------------------------------------------------------

def to_unit(ref: np.ndarray):
    """Monotone map of raw scores into [0, 1] by the val scores' empirical CDF.

    Rank-preserving, so AUROC is untouched; it only puts every scorer on the
    probability-like scale the shared threshold and ProcessBench code expects.
    """
    srt = np.sort(np.asarray(ref, dtype=np.float64))
    return lambda s: np.searchsorted(srt, np.asarray(s, dtype=np.float64), side="right") / len(srt)


def evaluate(name: str, s_va, s_te, s_pb: dict, yva, yte, pb_meta: dict) -> dict:
    from scripts.train_easy_probe_method import (THRESHOLD_GRID, evaluate_processbench,
                                                 select_threshold, step_binary_metrics)
    from scripts.merge_rep_grid_leaderboard import (calib20_subset, f1_pb_from_preds,
                                                    pred_matrix, quantile_grid)
    u = to_unit(s_va)
    va, te = u(s_va), u(s_te)
    t, _, _ = select_threshold(va, yva)
    res = {"auroc_val": auroc(yva, s_va), "auroc_test": auroc(yte, s_te),
           "id_f1_incorrect_val_selected": step_binary_metrics(yte, te, t)["f1_incorrect"],
           "threshold": float(t), "pb": {}}
    for sub, s in s_pb.items():
        rows, m_val = evaluate_processbench(u(s), pb_meta[sub], t)
        traces = [(r["label"], r["scores"]) for r in rows]
        grid = np.unique(np.concatenate([quantile_grid(traces), np.asarray(THRESHOLD_GRID)]))
        preds, labels = pred_matrix(traces, grid)
        res["pb"][sub] = {"F1_PB_val_selected": float(m_val["F1_PB"]),
                          "F1_PB_oracle": float(f1_pb_from_preds(preds, labels).max()),
                          "F1_PB_calib20": float(calib20_subset(traces))}
    for key in ("F1_PB_val_selected", "F1_PB_oracle", "F1_PB_calib20"):
        vals = [v[key] for v in res["pb"].values()]
        res[f"pb_avg_{key}"] = float(np.mean(vals)) if len(vals) == len(PB_SUBSETS) else None
    print(f"  {name:14s} test AUROC {res['auroc_test']:.4f}  ID F1 "
          f"{res['id_f1_incorrect_val_selected']:.4f}  PB calib20 "
          f"{res.get('pb_avg_F1_PB_calib20') or float('nan'):.4f}  oracle "
          f"{res.get('pb_avg_F1_PB_oracle') or float('nan'):.4f}", flush=True)
    return res


# ---------------------------------------------------------------------------
# the study, on arrays
# ---------------------------------------------------------------------------

def study(Xtr, ytr, Xva, yva, Xte, yte, pb: dict, dev, knn: bool = True,
          seed: int = 0) -> tuple[dict, dict]:
    """pb: subset -> (X, meta). Returns (metrics, arrays)."""
    t0 = time.time()
    m = moments(Xtr, ytr, dev)
    d = Xtr.shape[1]
    dmu = m["mu1"] - m["mu0"]
    metrics: dict = {"dim": d, "n_train": [m["n0"], m["n1"]], "methods": {}}
    arrays: dict = {}
    pb_meta = {k: v[1] for k, v in pb.items()}

    def run(name, f):
        s_va, s_te = f(Xva), f(Xte)
        s_pb = {k: f(v[0]) for k, v in pb.items()}
        metrics["methods"][name] = evaluate(name, s_va, s_te, s_pb, yva, yte, pb_meta)
        arrays[f"score_test__{name}"] = s_te.astype(np.float32)
        for k, v in s_pb.items():
            arrays[f"score_pb_{k}__{name}"] = np.asarray(v, dtype=np.float32)
        return s_va

    # --- first order, raw metric
    run("mean_diff", lambda X: project(X, dmu, dev))
    c0, c1 = m["mu0"] - m["mu"], m["mu1"] - m["mu"]
    C = torch.stack([c0 / c0.norm(), c1 / c1.norm()], 1)

    def centroid(X):
        P = project(X, C, dev, center=m["mu"])
        mu_np = m["mu"].float().cpu().numpy()
        nrm = np.concatenate([np.linalg.norm(np.asarray(X[i:i + 32768], dtype=np.float32) - mu_np, axis=1)
                              for i in range(0, len(X), 32768)])
        return (P[:, 1] - P[:, 0]) / np.maximum(nrm, 1e-12)
    run("centroid_cos", centroid)

    # --- whitened (LDA), shrinkage picked on val
    best = (-1.0, None, None)
    for a in SHRINKS:
        w = torch.linalg.solve(shrink(m["sw"], a), dmu)
        av = auroc(yva, project(Xva, w, dev))
        metrics.setdefault("lda_shrink_val_auroc", {})[str(a)] = av
        if av > best[0]:
            best = (av, a, w)
    a_star, w_lda = best[1], best[2]
    metrics["lda_shrink"] = a_star
    run("lda", lambda X: project(X, w_lda, dev))

    # --- Mahalanobis separation and how it spreads over directions
    Ssw = shrink(m["sw"], a_star)
    lam, U = torch.linalg.eigh(Ssw)
    proj = U.T @ dmu
    contrib = (proj ** 2 / lam).cpu().numpy()[::-1]          # descending variance order
    lam_np = lam.cpu().numpy()[::-1]
    D2 = float(contrib.sum())
    c_sorted = np.sort(contrib)[::-1]
    cum = np.cumsum(c_sorted) / D2
    metrics["mahalanobis"] = {
        "D": math.sqrt(D2),
        "gaussian_auroc": 0.5 * (1 + math.erf(math.sqrt(D2) / 2)),   # Phi(D/sqrt2)
        "participation_ratio": float(D2 ** 2 / (contrib ** 2).sum()),
        "dirs_for_50pct": int(np.searchsorted(cum, 0.5) + 1),
        "dirs_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
        "share_in_top10pct_variance_dirs": float(contrib[: max(1, d // 10)].sum() / D2),
        "share_in_bottom50pct_variance_dirs": float(contrib[d // 2:].sum() / D2),
    }
    arrays["d2_contrib_by_variance_rank"] = contrib.astype(np.float32)
    arrays["sw_eigvals_desc"] = lam_np.astype(np.float32)
    print(f"  mahalanobis D {math.sqrt(D2):.3f} -> Gaussian AUROC "
          f"{metrics['mahalanobis']['gaussian_auroc']:.4f}; PR "
          f"{metrics['mahalanobis']['participation_ratio']:.1f}", flush=True)
    del U

    # --- unsupervised PCA of the train covariance
    lt, Ut = torch.linalg.eigh(m["tot"])
    lt, Ut = lt.flip(0), Ut.flip(1)
    arrays["pca_eigvals_desc"] = lt.cpu().numpy().astype(np.float32)
    k_pc = min(N_PC_AUROC, d)
    Pva = project(Xva, Ut[:, :k_pc], dev, center=m["mu"])
    Pte = project(Xte, Ut[:, :k_pc], dev, center=m["mu"])
    pc_au_val = np.array([auroc(yva, Pva[:, j]) for j in range(k_pc)])
    pc_au_te = np.array([auroc(yte, Pte[:, j]) for j in range(k_pc)])
    arrays["pc_auroc_val"], arrays["pc_auroc_test"] = pc_au_val, pc_au_te
    j_best = int(np.argmax(np.abs(pc_au_val - 0.5)))
    sign = 1.0 if pc_au_val[j_best] >= 0.5 else -1.0
    metrics["pc_best_index"] = j_best
    u_best = Ut[:, j_best] * sign
    run("pc_best", lambda X: project(X, u_best, dev, center=m["mu"]))
    arrays["pca2_test"] = Pte[:, :2].astype(np.float32)
    explained = (lt / lt.sum()).cpu().numpy()
    metrics["pca_explained_top"] = {str(k): float(explained[:k].sum()) for k in (1, 2, 10, 100)}

    # --- LDA restricted to the top-K PCs: how many dimensions does separation need?
    kcurve = {}
    for K in [k for k in PC_KS if k < d] + [d]:
        UK = Ut[:, :K]
        SwK = UK.T @ m["sw"] @ UK
        wK = UK @ torch.linalg.solve(shrink(SwK, a_star), UK.T @ dmu)
        kcurve[K] = {"auroc_val": auroc(yva, project(Xva, wK, dev)),
                     "auroc_test": auroc(yte, project(Xte, wK, dev))}
    metrics["lda_topk_pcs"] = {str(k): v for k, v in kcurve.items()}
    print("  LDA on top-K PCs (test AUROC): " + ", ".join(
        f"{k}:{v['auroc_test']:.3f}" for k, v in kcurve.items()), flush=True)

    # --- 2-D view: LDA axis x the top PC orthogonal to it
    e1 = w_lda / w_lda.norm()
    v = Ut[:, 0] - (Ut[:, 0] @ e1) * e1
    e2 = v / v.norm()
    arrays["lda2_test"] = project(Xte, torch.stack([e1, e2], 1), dev, center=m["mu"]).astype(np.float32)
    arrays["y_test"] = np.asarray(yte, dtype=np.int8)
    del Ut, lt

    # --- second order: QDA and covariance-only
    S0, S1 = shrink(m["cov0"], a_star), shrink(m["cov1"], a_star)
    run("qda", lambda X: gauss_loglik(X, m["mu1"], S1, dev) - gauss_loglik(X, m["mu0"], S0, dev))
    run("cov_only", lambda X: gauss_loglik(X, m["mu"], S1, dev) - gauss_loglik(X, m["mu"], S0, dev))
    e0, e1v = torch.linalg.eigvalsh(S0), torch.linalg.eigvalsh(S1)
    metrics["class_cov"] = {
        "trace_ratio_incorrect_over_correct": float(torch.trace(m["cov1"]) / torch.trace(m["cov0"])),
        "logdet_shrunk_incorrect_minus_correct": float(torch.log(e1v).sum() - torch.log(e0).sum()),
        "rel_frobenius_diff": float((m["cov1"] - m["cov0"]).norm() / m["sw"].norm()),
    }
    del S0, S1

    # --- non-parametric: kNN vote over a train subsample
    for key in ("cov0", "cov1", "sw", "tot"):
        m.pop(key)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if knn:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(Xtr), size=min(KNN_BANK, len(Xtr)), replace=False))
        bank, ybank = np.asarray(Xtr[idx], dtype=np.float32), np.asarray(ytr)[idx]
        run("knn", lambda X: knn_scores(bank, ybank, X, m["mu"], dev))
        del bank

    metrics["seconds"] = round(time.time() - t0, 1)
    return metrics, arrays


# ---------------------------------------------------------------------------
# store I/O
# ---------------------------------------------------------------------------

def load_backbone(prm_store: Path, pb_store: Path, rep: str, prederived: bool = False):
    """prederived: the stores hold one subdirectory per rep, already derived
    (scripts/derive_vector_store.py), read as one row per item."""
    from scripts.train_rep_learner_cell import load_vectors
    if prederived:
        prm_store, pb_store = prm_store / rep, pb_store / rep
    out = {}
    for stem in ("probe_train_full", "val_5k", "test_2k"):
        X, y, _ = load_vectors(prm_store, stem, rep, None, sort=False, prederived=prederived)
        out[stem] = (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64))
    pb = {}
    for sub in PB_SUBSETS:
        X, _, meta = load_vectors(pb_store, sub, rep, None, sort=True, prederived=prederived)
        pb[sub] = (np.asarray(X, dtype=np.float32), meta)
    return out, pb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", nargs="+", required=True,
                   help="name=prm_store_dir,pb_store_dir")
    p.add_argument("--reps", nargs="+", default=["last_token", "step_mean", "boundary_stats"])
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--no_knn", action="store_true")
    p.add_argument("--prederived", action="store_true",
                   help="each store dir holds <rep>/<split> vector stores")
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for spec in a.backbone:
        name, paths = spec.split("=", 1)
        prm_store, pb_store = (Path(x) for x in paths.split(","))
        for rep in a.reps:
            out = a.out_dir / name / rep
            if (out / "metrics.json").exists():
                print(f"[skip] {name}/{rep}")
                continue
            print(f"=== {name} / {rep}", flush=True)
            t0 = time.time()
            splits, pb = load_backbone(prm_store, pb_store, rep, a.prederived)
            print(f"  loaded in {time.time()-t0:.0f}s: train {splits['probe_train_full'][0].shape}",
                  flush=True)
            metrics, arrays = study(*splits["probe_train_full"], *splits["val_5k"],
                                    *splits["test_2k"], pb, dev, knn=not a.no_knn)
            metrics.update({"backbone": name, "rep": rep, "prm_store": str(prm_store),
                            "pb_store": str(pb_store)})
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "arrays.npz", **arrays)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
            del splits, pb
            torch.cuda.empty_cache() if dev.type == "cuda" else None


if __name__ == "__main__":
    main()
