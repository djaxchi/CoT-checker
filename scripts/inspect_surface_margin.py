#!/usr/bin/env python3
"""Inspect which surface features explain the supervised fork probe margin.

Target (scalar explanation problem):
  margin_i = w . (h_neg_i - h_pos_i)      column `probe_margin` in pair_signatures.csv

We answer "how much of the margin is surface?" with four lenses:
  1. univariate Pearson/Spearman correlations
  2. standardized OLS / RidgeCV coefficients + cross-validated R^2
  3. permutation importance and leave-one-feature-out delta-R^2
  4. high- vs low-margin extreme-case surface summaries

CRITICAL distinction (kept separate, never conflated):
  SURFACE features  = length_diff, token_overlap, char_dissim, number_diff,
                      numbers_changed, operator_changed, neg/pos_has_final
  GEOMETRY features = delta_norm, cos_delta_mu, pc1..pc3   (representation-derived;
                      NOT surface). Including these answers a different question.
The script runs the regression twice: SURFACE-ONLY (the headline "surface explains
margin" claim) and SURFACE+GEOMETRY (exploration). The headline R^2 is surface-only.

Input : runs/fork_rep_audit/<tag>/tables/pair_signatures.csv
Output: <out>/surface_margin_report.md, surface_margin_summary.json, tables/, plots/

Usage:
  python scripts/inspect_surface_margin.py \
    --pair_signatures runs/fork_rep_audit/qwen2_5_7b/tables/pair_signatures.csv \
    --out runs/fork_rep_audit/qwen2_5_7b/surface_margin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_CANDIDATES = ["probe_margin", "margin", "w_margin", "w_dot_delta", "score_margin"]

# True surface features (the clean "surface explains margin" claim).
SURFACE_FEATURES = ["length_diff", "abs_length_diff", "token_overlap", "char_dissim",
                    "number_diff", "numbers_changed", "operator_changed",
                    "neg_has_final", "pos_has_final"]
# Representation-derived diagnostics (NOT surface; exploration only).
GEOMETRY_FEATURES = ["delta_norm", "cos_delta_mu", "pc1", "pc2", "pc3"]

RIDGE_ALPHAS = np.logspace(-4, 4, 41)


def find_target(df: pd.DataFrame, override: str | None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"target {override!r} not in columns {list(df.columns)}")
        return override
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"no target column among {TARGET_CANDIDATES}; have {list(df.columns)}")


def present_numeric(df: pd.DataFrame, names: list[str]) -> list[str]:
    return [c for c in names if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan"), float("nan")
    r, p = (pearsonr if kind == "pearson" else spearmanr)(x[mask], y[mask])
    return float(r), float(p)


def univariate(df: pd.DataFrame, feats: list[str], target: str) -> pd.DataFrame:
    y = df[target].to_numpy(float)
    rows = []
    for f in feats:
        x = df[f].to_numpy(float)
        pr, pp = safe_corr(x, y, "pearson")
        sr, sp = safe_corr(x, y, "spearman")
        rows.append({"feature": f, "pearson_r": pr, "pearson_p": pp,
                     "spearman_r": sr, "spearman_p": sp,
                     "abs_spearman_r": abs(sr) if np.isfinite(sr) else np.nan})
    return pd.DataFrame(rows).sort_values("abs_spearman_r", ascending=False)


def _ridge() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()),
                     ("model", RidgeCV(alphas=RIDGE_ALPHAS))])


def _ols() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])


def cv_r2(model: Pipeline, X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[float, float]:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    s = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return float(s.mean()), float(s.std())


def regression_block(df: pd.DataFrame, feats: list[str], target: str, seed: int) -> dict:
    X = df[feats].astype(float).fillna(df[feats].astype(float).median())
    y = df[target].to_numpy(float)
    ols, ridge = _ols(), _ridge()
    ols.fit(X, y)
    ridge.fit(X, y)
    ols_cv = cv_r2(ols, X, y, seed)
    ridge_cv = cv_r2(ridge, X, y, seed)

    coef = pd.DataFrame({"feature": feats,
                         "ols_coef": ols.named_steps["model"].coef_,
                         "ridge_coef": ridge.named_steps["model"].coef_})
    coef["abs_ridge_coef"] = coef["ridge_coef"].abs()
    coef = coef.sort_values("abs_ridge_coef", ascending=False)

    perm = permutation_importance(ridge, X, y, scoring="r2", n_repeats=50, random_state=seed)
    perm_df = pd.DataFrame({"feature": feats,
                            "perm_importance_mean": perm.importances_mean,
                            "perm_importance_std": perm.importances_std}
                           ).sort_values("perm_importance_mean", ascending=False)

    # leave-one-feature-out CV R^2.
    loo = []
    for f in feats:
        kept = [c for c in feats if c != f]
        if not kept:
            continue
        m, _ = cv_r2(_ridge(), X[kept], y, seed)
        loo.append({"removed_feature": f, "cv_r2_without": m,
                    "delta_r2_when_removed": ridge_cv[0] - m})
    loo_df = pd.DataFrame(loo).sort_values("delta_r2_when_removed", ascending=False)

    return {"ols_train_r2": float(r2_score(y, ols.predict(X))),
            "ridge_train_r2": float(r2_score(y, ridge.predict(X))),
            "ols_cv_r2_mean": ols_cv[0], "ols_cv_r2_std": ols_cv[1],
            "ridge_cv_r2_mean": ridge_cv[0], "ridge_cv_r2_std": ridge_cv[1],
            "ridge_alpha": float(ridge.named_steps["model"].alpha_),
            "coef_df": coef, "perm_df": perm_df, "loo_df": loo_df}


def high_low_summary(df: pd.DataFrame, feats: list[str], target: str, q: float) -> pd.DataFrame:
    lo = df[df[target] <= df[target].quantile(q)]
    hi = df[df[target] >= df[target].quantile(1 - q)]
    rows = [{"feature": f, "low_margin_mean": float(lo[f].mean()),
             "high_margin_mean": float(hi[f].mean()),
             "diff_high_minus_low": float(hi[f].mean() - lo[f].mean())} for f in feats]
    return pd.DataFrame(rows).sort_values(
        "diff_high_minus_low", key=lambda s: s.abs(), ascending=False)


def scatter(df, target, feat, out):
    x, y = df[feat].to_numpy(float), df[target].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    plt.figure(figsize=(6, 4.2)); plt.scatter(x[m], y[m], s=10, alpha=0.4)
    plt.xlabel(feat); plt.ylabel(target); plt.title(f"{target} vs {feat}")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()


def binned(df, target, feat, out, bins=10):
    t = df[[feat, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(t) < bins * 3:
        return
    try:
        t["bin"] = pd.qcut(t[feat], q=bins, duplicates="drop")
    except ValueError:
        return
    agg = t.groupby("bin", observed=True).agg(fm=(feat, "mean"), mm=(target, "mean"),
                                              ms=(target, "std"), n=(target, "size"))
    sem = agg["ms"] / np.sqrt(agg["n"].clip(lower=1))
    plt.figure(figsize=(6.5, 4.2))
    plt.errorbar(agg["fm"], agg["mm"], yerr=sem, marker="o", capsize=3)
    plt.xlabel(feat); plt.ylabel(f"mean {target}"); plt.title(f"binned {target} by {feat}")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()


def _md(df: pd.DataFrame, n: int = 12) -> str:
    """to_markdown if tabulate is present, else a plain fixed-width fallback."""
    try:
        return df.head(n).to_markdown(index=False)
    except Exception:
        return "```\n" + df.head(n).to_string(index=False) + "\n```"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_signatures", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k_plots", type=int, default=6)
    ap.add_argument("--extreme_q", type=float, default=0.10)
    args = ap.parse_args()

    (args.out / "plots").mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pair_signatures)
    if "length_diff" in df.columns and "abs_length_diff" not in df.columns:
        df["abs_length_diff"] = df["length_diff"].abs()

    target = find_target(df, args.target)
    surf = present_numeric(df, SURFACE_FEATURES)
    geom = present_numeric(df, GEOMETRY_FEATURES)
    if not surf:
        raise ValueError(f"no surface columns found; have {list(df.columns)}")

    # ---- univariate (surface + geometry, for transparency)
    corr = univariate(df, surf + geom, target)
    corr.to_csv(args.out / "tables" / "margin_surface_correlations.csv", index=False)

    # ---- two regressions: SURFACE-ONLY (headline) and SURFACE+GEOMETRY (exploration)
    surf_reg = regression_block(df, surf, target, args.seed)
    full_reg = regression_block(df, surf + geom, target, args.seed)
    surf_reg["coef_df"].to_csv(args.out / "tables" / "margin_surface_coefficients.csv", index=False)
    surf_reg["perm_df"].to_csv(args.out / "tables" / "margin_surface_permutation_importance.csv", index=False)
    surf_reg["loo_df"].to_csv(args.out / "tables" / "margin_surface_leave_one_out_r2.csv", index=False)
    full_reg["coef_df"].to_csv(args.out / "tables" / "margin_surface_plus_geometry_coefficients.csv", index=False)

    extremes = high_low_summary(df, surf, target, args.extreme_q)
    extremes.to_csv(args.out / "tables" / "margin_high_low_surface_summary.csv", index=False)
    df.sort_values(target, ascending=False).head(50).to_csv(
        args.out / "tables" / "top_high_margin_pairs.csv", index=False)
    df.sort_values(target, ascending=True).head(50).to_csv(
        args.out / "tables" / "top_low_margin_pairs.csv", index=False)

    for f in corr["feature"].head(args.top_k_plots).tolist():
        scatter(df, target, f, args.out / "plots" / f"scatter_{target}_vs_{f}.png")
        binned(df, target, f, args.out / "plots" / f"binned_{target}_by_{f}.png")

    summary = {
        "n_pairs": int(len(df)), "target": target,
        "surface_features": surf, "geometry_features": geom,
        "surface_only": {k: surf_reg[k] for k in
                         ("ols_train_r2", "ridge_train_r2", "ols_cv_r2_mean", "ols_cv_r2_std",
                          "ridge_cv_r2_mean", "ridge_cv_r2_std", "ridge_alpha")},
        "surface_plus_geometry": {k: full_reg[k] for k in
                                  ("ridge_cv_r2_mean", "ridge_cv_r2_std")},
        "top_correlations": corr.head(10).to_dict("records"),
        "surface_permutation_importance": surf_reg["perm_df"].head(10).to_dict("records"),
        "surface_leave_one_out": surf_reg["loo_df"].head(10).to_dict("records"),
    }
    (args.out / "surface_margin_summary.json").write_text(json.dumps(summary, indent=2))

    r = surf_reg["ridge_cv_r2_mean"]
    band = ("< 0.05 surface barely explains margin" if r < 0.05 else
            "0.05-0.15 weak but real surface dependence" if r < 0.15 else
            "0.15-0.30 substantial surface dependence" if r < 0.30 else
            "> 0.30 margin is heavily surface-driven")
    with open(args.out / "surface_margin_report.md", "w") as f:
        f.write("# Surface explanation of fork probe margin\n\n")
        f.write(f"- pairs: `{len(df)}`  target: `{target}`\n")
        f.write(f"- surface features: `{surf}`\n")
        f.write(f"- geometry features (kept separate): `{geom}`\n\n")
        f.write("## Headline: SURFACE-ONLY -> margin\n\n")
        f.write(f"- **Ridge CV R^2 = {r:.4f} +/- {surf_reg['ridge_cv_r2_std']:.4f}**  ({band})\n")
        f.write(f"- OLS CV R^2 = {surf_reg['ols_cv_r2_mean']:.4f} +/- {surf_reg['ols_cv_r2_std']:.4f}\n")
        f.write(f"- surface train R^2 (ridge) = {surf_reg['ridge_train_r2']:.4f}\n")
        f.write(f"- for contrast, SURFACE+GEOMETRY ridge CV R^2 = "
                f"{full_reg['ridge_cv_r2_mean']:.4f} +/- {full_reg['ridge_cv_r2_std']:.4f} "
                f"(adds representation diagnostics; not a surface claim)\n\n")
        f.write("## Univariate correlations (surface + geometry)\n\n" + _md(corr) + "\n\n")
        f.write("## Surface-only standardized ridge coefficients\n\n" + _md(surf_reg["coef_df"]) + "\n\n")
        f.write("## Surface-only permutation importance\n\n" + _md(surf_reg["perm_df"]) + "\n\n")
        f.write("## Surface-only leave-one-feature-out delta-R^2\n\n" + _md(surf_reg["loo_df"]) + "\n\n")
        f.write("## High vs low margin surface differences\n\n" + _md(extremes) + "\n")

    print(f"[ok] {args.out}")
    print(f"[headline] surface-only ridge CV R^2 = {r:.4f} +/- {surf_reg['ridge_cv_r2_std']:.4f}  ({band})")
    print(f"[contrast] surface+geometry ridge CV R^2 = {full_reg['ridge_cv_r2_mean']:.4f}")
    print("[top univariate]"); print(corr.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
