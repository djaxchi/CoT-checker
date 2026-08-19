"""Conicity / class-cone geometry of the unified-harness step representations.

Motivation (a lab suggestion): instead of fitting a probe, take each step's vector,
compute the mean vector of the correct steps and of the incorrect steps, and look at
the cone geometry of the two classes.

Conicity (Mohankumar et al., ACL 2020): for a set V,
    ATM(v, V) = cos(v, mean(V))          "alignment to mean"
    conicity(V) = mean_{v in V} ATM(v, V)
High conicity = the set lies in a narrow cone.

This script separates the two things that suggestion bundles together:

  (a) a GEOMETRY DIAGNOSTIC: how tight is each class cone, and how far apart are the
      two class centroids;
  (b) a CLASSIFIER: nearest-centroid / mean-difference decision rule, which is LDA
      with the covariance forced to identity.

Two controls decide whether any of it means anything:

  ANISOTROPY. LLM residual streams have a large shared mean component, so raw
  conicity is ~0.9+ for *any* subset and measures Qwen, not the labels. Every
  conicity here is therefore reported raw AND after subtracting the global mean;
  only the centered number is interpretable.

  LABEL NULL. Any set of N points has positive ATM. So each class conicity is
  compared against the same statistic under randomly permuted labels
  (--n_shuffle repeats), giving a z-score against the size-matched null.

One row is a SANITY CHECK, not a result. Projecting out the mean-difference
direction makes the two class centroids identical by construction (mu_1 - mu_0 is
parallel to w_md), so ALL first-order separation is gone and any linear probe must
fall to chance. It is reported because it verifies the deflation is implemented
correctly; it is NOT evidence about where the signal lives. The corollary matters
though: for a linear reader the class mean difference IS the whole signal, so the
gap between the raw centroid rule and LDA is entirely a question of METRIC
(whitening), never of which direction to look along.

Reported on the frozen spine (fit on train, threshold on val, report on test), with
project metric conventions: F1 at val-selected + oracle threshold against the
trivial always-positive baseline, AUROC secondary.

Outputs (results/conicity/<tag>/):
  conicity_<tag>.json    every number
  conicity_<tag>.png     4-panel summary

Usage (local smoke, small cache):
    python scripts/analysis/conicity_class_geometry.py \
        --cache_dir runs/s1_model_size_dense/qwen2_5_7b/merged \
        --train_stem val_1k --val_stem val_1k --test_stem val_1k --tag smoke

Usage (TamIA, unified harness dense_last):
    python scripts/analysis/conicity_class_geometry.py \
        --cache_dir $SCRATCH/cot_mech/dense_full_7b_v1/cache \
        --train_stem probe_train_full --val_stem val_5k --test_stem test_2k \
        --tag dense_last_7b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

CHUNK = 8192


# --------------------------------------------------------------------------- io


def load_cache(cache_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Memory-mapped h plus in-memory y, harness cache convention."""
    h = np.load(cache_dir / f"{stem}_h.npy", mmap_mode="r")
    y = np.load(cache_dir / f"{stem}_y.npy").astype(np.int64)
    if h.shape[0] != y.shape[0]:
        raise ValueError(f"{stem}: h/y row mismatch {h.shape[0]} vs {y.shape[0]}")
    return h, y


def chunked_mean(h: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Mean over a row subset of a (possibly mmapped) matrix, without loading it all."""
    d = h.shape[1]
    acc = np.zeros(d, dtype=np.float64)
    for i in range(0, len(rows), CHUNK):
        acc += np.asarray(h[np.sort(rows[i : i + CHUNK])], dtype=np.float64).sum(0)
    return (acc / max(len(rows), 1)).astype(np.float32)


def subsample(rng: np.random.Generator, n: int, cap: int | None) -> np.ndarray:
    idx = np.arange(n)
    if cap is not None and n > cap:
        idx = rng.choice(n, size=cap, replace=False)
        idx.sort()
    return idx


# ---------------------------------------------------------------- cone geometry


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def atm(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """cos(x_i, c) for every row."""
    nx = np.linalg.norm(X, axis=1)
    nx[nx == 0] = 1.0
    return (X @ unit(c)) / nx


def conicity_stats(X: np.ndarray, y: np.ndarray, n_shuffle: int, seed: int) -> dict:
    """Conicity per class, raw and global-mean-centered, with a permuted-label null."""
    mu = X.mean(0)
    Xc = X - mu
    out: dict = {"global_mean_norm": float(np.linalg.norm(mu)),
                 "mean_row_norm": float(np.linalg.norm(X, axis=1).mean())}

    for name, M in (("raw", X), ("centered", Xc)):
        per_class = {}
        for k, lab in ((0, "correct"), (1, "incorrect")):
            Mk = M[y == k]
            ck = Mk.mean(0)
            per_class[lab] = {
                "conicity": float(atm(Mk, ck).mean()),
                "n": int(len(Mk)),
                "centroid_norm": float(np.linalg.norm(ck)),
            }
        c0 = M[y == 0].mean(0)
        c1 = M[y == 1].mean(0)
        cos_cc = float(np.dot(unit(c0), unit(c1)))
        per_class["between_centroid_cos"] = cos_cc
        per_class["between_centroid_deg"] = float(np.degrees(np.arccos(np.clip(cos_cc, -1, 1))))
        per_class["pooled_conicity"] = float(atm(M, M.mean(0)).mean())
        # NOTE: after global-mean centering the two class centroids satisfy
        # p0*c0 + p1*c1 = 0, so the centered angle is ~180 deg BY CONSTRUCTION and
        # carries no information. Only the raw angle, and the Cohen's d below, do.
        per_class["angle_is_degenerate"] = (name == "centered")
        out[name] = per_class

    # size-matched null: same class sizes, labels permuted
    rng = np.random.default_rng(seed)
    null = {"correct": [], "incorrect": []}
    for _ in range(n_shuffle):
        yp = rng.permutation(y)
        c0 = Xc[yp == 0].mean(0)
        c1 = Xc[yp == 1].mean(0)
        null["correct"].append(float(atm(Xc[yp == 0], c0).mean()))
        null["incorrect"].append(float(atm(Xc[yp == 1], c1).mean()))

    # between-class separation that is NOT degenerate: standardized effect size of the
    # mean-difference direction (Cohen's d), plus the Fisher ratio.
    w = unit(Xc[y == 1].mean(0) - Xc[y == 0].mean(0))
    proj = Xc @ w
    p0, p1 = proj[y == 0], proj[y == 1]
    sd_pool = float(np.sqrt(((len(p0) - 1) * p0.var(ddof=1) + (len(p1) - 1) * p1.var(ddof=1))
                            / max(len(p0) + len(p1) - 2, 1)))
    out["separation"] = {
        "mean_diff_norm": float(np.linalg.norm(Xc[y == 1].mean(0) - Xc[y == 0].mean(0))),
        "cohens_d_along_mean_diff": float((p1.mean() - p0.mean()) / sd_pool) if sd_pool > 0 else float("nan"),
        "pooled_sd_along_mean_diff": sd_pool,
    }

    out["null_centered"] = {}
    for key, obs in (("correct", out["centered"]["correct"]["conicity"]),
                     ("incorrect", out["centered"]["incorrect"]["conicity"])):
        arr = np.asarray(null[key])
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out["null_centered"][key] = {
            "observed": float(obs),
            "null_mean": float(arr.mean()),
            "null_sd": sd,
            "z": float((obs - arr.mean()) / sd) if sd > 0 else float("nan"),
            "n_shuffle": int(n_shuffle),
        }
    return out


# -------------------------------------------------------------------- metrics


def trivial_f1(y: np.ndarray) -> float:
    p = float(y.mean())
    return 2 * p / (1 + p) if p > 0 else 0.0


def score_report(name: str, s_val: np.ndarray, y_val: np.ndarray,
                 s_test: np.ndarray, y_test: np.ndarray, grid: int = 201) -> dict:
    """AUROC + F1 at val-selected threshold + oracle-threshold F1 ceiling."""
    lo, hi = float(min(s_val.min(), s_test.min())), float(max(s_val.max(), s_test.max()))
    ts = np.linspace(lo, hi, grid)
    f1_val = [f1_score(y_val, (s_val >= t).astype(int), zero_division=0) for t in ts]
    t_star = float(ts[int(np.argmax(f1_val))])
    f1_test_grid = [f1_score(y_test, (s_test >= t).astype(int), zero_division=0) for t in ts]
    return {
        "name": name,
        "auroc": float(roc_auc_score(y_test, s_test)),
        "f1_val_selected": float(f1_score(y_test, (s_test >= t_star).astype(int), zero_division=0)),
        "f1_oracle": float(max(f1_test_grid)),
        "threshold_val_selected": t_star,
    }


# ----------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True, type=Path)
    p.add_argument("--train_stem", default="probe_train_full")
    p.add_argument("--val_stem", default="val_5k")
    p.add_argument("--test_stem", default="test_2k")
    p.add_argument("--tag", default="dense_last_7b")
    p.add_argument("--out_root", type=Path, default=Path("results/conicity"))
    p.add_argument("--max_train", type=int, default=100_000,
                   help="Subsample cap for the covariance/probe fits (centroids use ALL rows).")
    p.add_argument("--max_geom", type=int, default=50_000,
                   help="Subsample cap for the in-memory conicity/null block.")
    p.add_argument("--n_shuffle", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    h_tr, y_tr = load_cache(args.cache_dir, args.train_stem)
    h_va, y_va = load_cache(args.cache_dir, args.val_stem)
    h_te, y_te = load_cache(args.cache_dir, args.test_stem)
    d = h_tr.shape[1]
    print(f"train {h_tr.shape}  val {h_va.shape}  test {h_te.shape}  dim {d}")

    # --- centroids from ALL training rows (streamed, mmap-safe) --------------
    rows_all = np.arange(h_tr.shape[0])
    mu_global = chunked_mean(h_tr, rows_all)
    c_cor = chunked_mean(h_tr, rows_all[y_tr == 0]) - mu_global
    c_inc = chunked_mean(h_tr, rows_all[y_tr == 1]) - mu_global
    w_md = c_inc - c_cor                      # mean-difference direction
    w_md_u = unit(w_md)

    # --- cone geometry on an in-memory block --------------------------------
    gi = subsample(rng, h_tr.shape[0], args.max_geom)
    Xg = np.asarray(h_tr[gi], dtype=np.float32)
    geom = conicity_stats(Xg, y_tr[gi], args.n_shuffle, args.seed)
    geom["train_centroid_gap_norm"] = float(np.linalg.norm(w_md))
    del Xg

    # --- fits on a capped training subsample --------------------------------
    ti = subsample(rng, h_tr.shape[0], args.max_train)
    Xt = np.asarray(h_tr[ti], dtype=np.float32)
    yt = y_tr[ti]
    Xv = np.asarray(h_va, dtype=np.float32)
    Xs = np.asarray(h_te, dtype=np.float32)

    scaler = StandardScaler().fit(Xt)
    Zt, Zv, Zs = scaler.transform(Xt), scaler.transform(Xv), scaler.transform(Xs)

    results = []

    # (1) nearest-centroid on DIRECTION only: cos to incorrect minus cos to correct
    def cone_score(X: np.ndarray) -> np.ndarray:
        Xc = X - mu_global
        return atm(Xc, c_inc) - atm(Xc, c_cor)

    results.append(score_report("nearest_centroid_cosine", cone_score(Xv), y_va,
                                cone_score(Xs), y_te))

    # (2) plain mean-difference projection (magnitude kept)
    results.append(score_report("mean_diff_projection", (Xv - mu_global) @ w_md_u, y_va,
                                (Xs - mu_global) @ w_md_u, y_te))

    # (3) whitened mean difference (LDA) -- covariance the identity assumption throws away
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Zt, yt)
    results.append(score_report("lda_whitened_mean_diff", lda.decision_function(Zv), y_va,
                                lda.decision_function(Zs), y_te))

    # (4) reference: trained logistic probe
    logit = LogisticRegression(max_iter=2000, C=1.0).fit(Zt, yt)
    results.append(score_report("logistic_probe", logit.decision_function(Zv), y_va,
                                logit.decision_function(Zs), y_te))

    # (5) SANITY: deflating w_md zeroes mu_1 - mu_0 by construction, so this MUST
    #     land at chance. A value away from ~0.5 means the deflation is buggy.
    def deflate(X: np.ndarray) -> np.ndarray:
        Xc = X - mu_global
        return Xc - np.outer(Xc @ w_md_u, w_md_u)

    Dt, Dv, Ds = deflate(Xt), deflate(Xv), deflate(Xs)
    dsc = StandardScaler().fit(Dt)
    logit_d = LogisticRegression(max_iter=2000, C=1.0).fit(dsc.transform(Dt), yt)
    results.append(score_report("SANITY_meandiff_removed_must_be_chance",
                                logit_d.decision_function(dsc.transform(Dv)), y_va,
                                logit_d.decision_function(dsc.transform(Ds)), y_te))

    # how much of the trained probe IS the mean difference?
    w_log = logit.coef_.ravel() / scaler.scale_          # back to raw feature space
    geom["cos_logistic_vs_mean_diff"] = float(np.dot(unit(w_log), w_md_u))
    w_lda = lda.coef_.ravel() / scaler.scale_
    geom["cos_lda_vs_mean_diff"] = float(np.dot(unit(w_lda), w_md_u))

    n_fit, warn = len(ti), None
    sanity = next(r for r in results if r["name"].startswith("SANITY"))
    if abs(sanity["auroc"] - 0.5) > 0.05:
        warn = (f"SANITY row AUROC={sanity['auroc']:.4f} is not ~0.5; deflating the "
                "mean-difference direction should destroy all first-order separation. "
                "Check the deflation, or suspect n_fit < 10*d overfitting "
                f"(n_fit={n_fit}, 10*d={10 * d}).")
        print(f"\n[WARN] {warn}")

    payload = {
        "tag": args.tag,
        "cache_dir": str(args.cache_dir),
        "stems": {"train": args.train_stem, "val": args.val_stem, "test": args.test_stem},
        "n": {"train": int(h_tr.shape[0]), "val": int(h_va.shape[0]), "test": int(h_te.shape[0]),
              "train_fit_subsample": int(len(ti)), "geom_subsample": int(len(gi))},
        "dim": int(d),
        "trivial_always_positive_f1_test": trivial_f1(y_te),
        "test_prevalence": float(y_te.mean()),
        "geometry": geom,
        "scores": results,
        "seed": args.seed,
        "warning": warn,
    }
    jpath = out_dir / f"conicity_{args.tag}.json"
    jpath.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------ plots
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    gi2 = subsample(rng, h_te.shape[0], 20_000)
    Xp = np.asarray(h_te[gi2], dtype=np.float32) - mu_global
    yp = y_te[gi2]
    a_inc, a_cor = atm(Xp, c_inc), atm(Xp, c_cor)

    for lab, k, col in (("correct", 0, "tab:blue"), ("incorrect", 1, "tab:red")):
        ax[0, 0].hist(a_inc[yp == k], bins=60, alpha=0.55, label=lab, color=col, density=True)
    ax[0, 0].set_title("ATM to the INCORRECT centroid (test, global-mean centered)")
    ax[0, 0].set_xlabel("cos(x - mu, c_incorrect)")
    ax[0, 0].legend()

    s = a_inc - a_cor
    for lab, k, col in (("correct", 0, "tab:blue"), ("incorrect", 1, "tab:red")):
        ax[0, 1].hist(s[yp == k], bins=60, alpha=0.55, label=lab, color=col, density=True)
    ax[0, 1].set_title("nearest-centroid score: cos(.,c_inc) - cos(.,c_cor)")
    ax[0, 1].legend()

    names = ["raw\ncorrect", "raw\nincorrect", "centered\ncorrect", "centered\nincorrect",
             "null\ncorrect", "null\nincorrect"]
    vals = [geom["raw"]["correct"]["conicity"], geom["raw"]["incorrect"]["conicity"],
            geom["centered"]["correct"]["conicity"], geom["centered"]["incorrect"]["conicity"],
            geom["null_centered"]["correct"]["null_mean"],
            geom["null_centered"]["incorrect"]["null_mean"]]
    ax[1, 0].bar(names, vals, color=["0.6", "0.6", "tab:blue", "tab:red", "0.85", "0.85"])
    ax[1, 0].set_title("conicity: raw is anisotropy, centered vs permuted-label null is signal")
    ax[1, 0].set_ylabel("conicity")
    ax[1, 0].tick_params(axis="x", labelsize=8)

    rn = [r["name"].replace("_", "\n") for r in results]
    ax[1, 1].bar(rn, [r["auroc"] for r in results], color="tab:green", alpha=0.75, label="AUROC")
    ax[1, 1].plot(rn, [r["f1_val_selected"] for r in results], "ko-", label="F1 (val-sel)")
    ax[1, 1].axhline(payload["trivial_always_positive_f1_test"], ls="--", c="r",
                     label="trivial always-positive F1")
    ax[1, 1].set_ylim(0.4, 1.0)
    ax[1, 1].tick_params(axis="x", labelsize=7)
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title(f"test={args.test_stem}")

    fig.suptitle(f"Class-cone geometry of step representations [{args.tag}]")
    fig.tight_layout()
    ppath = out_dir / f"conicity_{args.tag}.png"
    fig.savefig(ppath, dpi=150)

    # ----------------------------------------------------------------- stdout
    print("\n--- cone geometry (train) ---")
    print(f"  raw conicity      correct={geom['raw']['correct']['conicity']:.4f}  "
          f"incorrect={geom['raw']['incorrect']['conicity']:.4f}   (anisotropy-dominated)")
    print(f"  centered conicity correct={geom['centered']['correct']['conicity']:.4f}  "
          f"incorrect={geom['centered']['incorrect']['conicity']:.4f}")
    for k, v in geom["null_centered"].items():
        print(f"  null[{k}] obs={v['observed']:.4f} null={v['null_mean']:.4f}"
              f" +/- {v['null_sd']:.4f}  z={v['z']:.1f}")
    print(f"  centroid angle (raw, meaningful) = {geom['raw']['between_centroid_deg']:.2f} deg"
          f"   [centered angle ~180 by construction, ignored]")
    print(f"  Cohen's d along mean-diff = {geom['separation']['cohens_d_along_mean_diff']:.3f}"
          f"   |mu_inc - mu_cor| = {geom['separation']['mean_diff_norm']:.2f}")
    print(f"  cos(logistic w, mean-diff) = {geom['cos_logistic_vs_mean_diff']:.4f}")
    print(f"  cos(LDA w, mean-diff)      = {geom['cos_lda_vs_mean_diff']:.4f}")
    print(f"\n--- decoding (test={args.test_stem}, trivial F1="
          f"{payload['trivial_always_positive_f1_test']:.3f}) ---")
    for r in results:
        print(f"  {r['name']:36s} AUROC={r['auroc']:.4f}  "
              f"F1(val-sel)={r['f1_val_selected']:.4f}  F1(oracle)={r['f1_oracle']:.4f}")
    print(f"\nwrote {jpath}\nwrote {ppath}")


if __name__ == "__main__":
    main()
