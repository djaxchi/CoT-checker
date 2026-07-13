"""attention_routing_v0 paired fork analysis (stages 2-4 of the brief).

Reads the extraction output (metadata.parquet + features.npy) and runs the
paired correct-minus-wrong comparison. Inference protocol:

  confirmatory  SIX FROZEN METRICS at the FIRST candidate token, full data:
                sink-corrected grounding ratio, previous-step mass,
                external-semantic mass, recency ratio, entropy, mean
                distance. The first token is the cleanest test: the
                candidate has no internal context yet, so any difference is
                in how the two candidate queries access the identical
                prefix. (table_confirmatory.csv)
  global        every feature x read for context (table_global.csv), plus a
                cluster-robust paired regression of each first-token delta
                on surface deltas (length, logprob, predictive entropy,
                number count): beta0 = routing difference after surface
                controls (table_regression.csv), plus the length-matched
                subset rerun (table_global_lengthmatched.csv)
  layer         mean over heads per layer, reads first and mean
                (table_layer.csv + layer curve plots)
  head          DISCOVERY/CONFIRMATION problem split (50/50 by question
                hash): heads are selected on the discovery half (BH q<0.05
                within feature, ranked by |dz|, capped) and re-tested with
                cluster-aware paired stats on the confirmation half
                (table_head_confirmed.csv; heatmaps show the discovery half
                only). Read = first.

All tests are cluster-aware at the problem level (see paired_stats). Derived
quantities: question_nosink = question - sink (token-0 attention sink is not
question-content retrieval), external_semantic = prev_all + question_nosink
(saved because the grounding RATIO alone is unstable when both masses are
small), prefix_mass = 1 - self_mass (total reliance on context outside the
candidate), grounding_ratio = prev_all / external_semantic, recency_ratio =
prev1 / prev_all. Ratios are computed from aggregated masses (ratio of
means). No classifier is trained here.

Usage:
  python scripts/analysis/ar_fork_analysis.py \
      --run_dir runs/attention_routing/forks_attn \
      --out_dir runs/attention_routing/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.attention_routing import (  # noqa: E402
    FEATURES,
    READS,
    grounding_ratio,
    paired_regression,
    paired_stats,
    recency_ratio,
)

# validated CVD-safe pair (dataviz skill): correct=blue, wrong=orange
C_CORRECT = "#2563eb"
C_WRONG = "#ea580c"
C_NEUTRAL = "#6b7280"

EXT_FEATURES = FEATURES + ["question_nosink_mass", "external_semantic_mass",
                           "prefix_mass", "grounding_ratio", "recency_ratio"]
CONFIRMATORY = ["grounding_ratio", "prev_all_mass", "external_semantic_mass",
                "recency_ratio", "entropy", "mean_distance"]
REGRESSION_FEATURES = CONFIRMATORY + ["prev1_mass", "question_nosink_mass",
                                      "sink_mass", "self_mass"]
KEY_FEATURES = ["prev_all_mass", "question_nosink_mass",
                "external_semantic_mass", "self_mass", "sink_mass",
                "entropy", "mean_distance", "grounding_ratio"]
HEAD_FEATURES = ["prev_all_mass", "prev1_mass", "question_nosink_mass",
                 "sink_mass", "entropy", "mean_distance", "recency_ratio"]
HEATMAP_FEATURES = ["prev_all_mass", "question_nosink_mass", "entropy",
                    "mean_distance"]
COVARIATES = ["cand_token_len", "mean_logprob", "mean_pred_entropy",
              "n_numbers"]
MAX_HEADS_PER_FEATURE = 20


def add_ratios(feats: np.ndarray) -> np.ndarray:
    """Append the derived quantities along the feature axis (EXT_FEATURES
    order). feats: (..., n_features, n_reads) masses in FEATURES order."""
    prev_all = feats[..., FEATURES.index("prev_all_mass"), :]
    q = feats[..., FEATURES.index("question_mass"), :]
    sink = feats[..., FEATURES.index("sink_mass"), :]
    prev1 = feats[..., FEATURES.index("prev1_mass"), :]
    self_m = feats[..., FEATURES.index("self_mass"), :]
    q_nosink = np.clip(q - sink, 0.0, None)
    ext_sem = prev_all + q_nosink
    prefix = np.clip(1.0 - self_m, 0.0, None)
    g = grounding_ratio(prev_all, q_nosink)
    r = recency_ratio(prev1, prev_all)
    return np.concatenate(
        [feats, q_nosink[..., None, :], ext_sem[..., None, :],
         prefix[..., None, :], g[..., None, :], r[..., None, :]], axis=-2)


def boot_ci_matrix(delta: np.ndarray, groups: np.ndarray, n_boot: int,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster-bootstrap 95% CI of the mean for many cells at once.

    delta: (n_pairs, n_cells); groups: (n_pairs,) problem ids.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    n_groups = len(uniq)
    sums = np.zeros((n_groups, delta.shape[1]))
    np.add.at(sums, inv, delta)
    counts = np.bincount(inv, minlength=n_groups).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot, delta.shape[1]))
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        means[b] = sums[pick].sum(axis=0) / counts[pick].sum()
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return lo, hi


def dz_and_pgt(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """delta: (n_pairs, ...) -> (dz, P(f+ > f-)) along axis 0."""
    sd = delta.std(axis=0, ddof=1)
    dz = np.divide(delta.mean(axis=0), sd, out=np.zeros_like(sd),
                   where=sd > 0)
    n = delta.shape[0]
    p_gt = ((delta > 0).sum(axis=0) + 0.5 * (delta == 0).sum(axis=0)) / n
    return dz, p_gt


def pair_rows(meta: pd.DataFrame):
    """Row indices of correct/wrong per fork plus pair-level metadata with
    the wrong role's surface covariates attached."""
    idx = meta.reset_index().pivot(index="fork_id", columns="role",
                                   values="index")
    idx = idx.dropna().astype(int)
    ic, iw = idx["correct"].to_numpy(), idx["wrong"].to_numpy()
    pm = meta.iloc[ic].reset_index(drop=True).copy()
    for col in COVARIATES:
        pm[f"wrong_{col}"] = meta.iloc[iw][col].to_numpy()
    return ic, iw, pm


def split_problems(groups: np.ndarray, seed: int):
    """Deterministic 50/50 problem split -> (discovery_mask, confirm_mask)."""
    uniq = np.sort(np.unique(groups))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    disc_set = set(uniq[perm[: len(uniq) // 2]])
    disc = np.array([g in disc_set for g in groups])
    return disc, ~disc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path,
                    default=Path("runs/attention_routing/forks_attn"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/attention_routing/analysis"))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(args.run_dir / "metadata.parquet")
    feats = np.load(args.run_dir / "features.npy").astype(np.float32)
    assert len(meta) == feats.shape[0]
    n_layers, n_heads = feats.shape[1], feats.shape[2]
    r_first, r_mean = READS.index("first"), READS.index("mean")

    ic, iw, pm = pair_rows(meta)

    # drop pairs with any non-finite feature or covariate (loud, never fatal)
    finite = np.isfinite(feats[ic]).all(axis=(1, 2, 3, 4)) \
        & np.isfinite(feats[iw]).all(axis=(1, 2, 3, 4))
    for c in COVARIATES:
        finite &= np.isfinite(pm[c].to_numpy(dtype=np.float64))
        finite &= np.isfinite(pm[f"wrong_{c}"].to_numpy(dtype=np.float64))
    n_dropped = int((~finite).sum())
    if n_dropped:
        print(f"[ar-analysis] WARNING: dropping {n_dropped}/{len(pm)} pairs "
              "with non-finite features or covariates", flush=True)
        ic, iw = ic[finite], iw[finite]
        pm = pm[finite].reset_index(drop=True)

    groups = pm["question_hash"].to_numpy()
    n_pairs = len(pm)
    print(f"[ar-analysis] {n_pairs} pairs, {len(np.unique(groups))} problems, "
          f"{n_layers} layers x {n_heads} heads", flush=True)

    # ---- global level (mean over layers x heads, derived appended) --------
    g_cor = add_ratios(feats[ic].mean(axis=(1, 2)))
    g_wro = add_ratios(feats[iw].mean(axis=(1, 2)))
    g_delta = g_cor - g_wro  # (n_pairs, n_ext, n_reads)

    rows = []
    for fi, feat in enumerate(EXT_FEATURES):
        for ri, read_name in enumerate(READS):
            s = paired_stats(g_delta[:, fi, ri], groups,
                             n_boot=args.n_boot, seed=args.seed)
            rows.append({"feature": feat, "read": read_name, **s})
    table_global = pd.DataFrame(rows)
    table_global.to_csv(args.out_dir / "table_global.csv", index=False)

    # ---- confirmatory: frozen metrics, first token, full data -------------
    conf_rows = []
    for feat in CONFIRMATORY:
        fi = EXT_FEATURES.index(feat)
        s = paired_stats(g_delta[:, fi, r_first], groups,
                         n_boot=args.n_boot, seed=args.seed)
        conf_rows.append({"feature": feat, "read": "first", **s})
    table_conf = pd.DataFrame(conf_rows)
    table_conf.to_csv(args.out_dir / "table_confirmatory.csv", index=False)

    # ---- surface-control regression (first token) -------------------------
    dcov = np.column_stack([
        (pm[c] - pm[f"wrong_{c}"]).to_numpy(dtype=np.float64)
        for c in COVARIATES])
    reg_rows = []
    for feat in REGRESSION_FEATURES:
        fi = EXT_FEATURES.index(feat)
        r = paired_regression(g_delta[:, fi, r_first], dcov, groups)
        reg_rows.append({"feature": feat, "read": "first",
                         "covariates": "+".join(COVARIATES), **r})
    pd.DataFrame(reg_rows).to_csv(args.out_dir / "table_regression.csv",
                                  index=False)

    # ---- length-matched subset rerun (first token) -------------------------
    dlen = dcov[:, COVARIATES.index("cand_token_len")]
    rel = np.abs(dlen) / np.maximum(
        pm["cand_token_len"].to_numpy(),
        pm["wrong_cand_token_len"].to_numpy())
    matched = rel <= 0.10
    ctrl_rows = []
    for fi, feat in enumerate(EXT_FEATURES):
        d = g_delta[:, fi, r_first]
        rho = stats.spearmanr(d, dlen).statistic
        s = paired_stats(d[matched], groups[matched],
                         n_boot=args.n_boot, seed=args.seed)
        ctrl_rows.append({"feature": feat, "read": "first",
                          "spearman_delta_vs_dlen": float(rho),
                          "n_matched": int(matched.sum()), **s})
    pd.DataFrame(ctrl_rows).to_csv(
        args.out_dir / "table_global_lengthmatched.csv", index=False)

    # ---- layer level (mean over heads), reads first and mean --------------
    l_cor = add_ratios(feats[ic].mean(axis=2))
    l_wro = add_ratios(feats[iw].mean(axis=2))
    n_ext = len(EXT_FEATURES)
    layer_rows = []
    for ri, read_name in [(r_first, "first"), (r_mean, "mean")]:
        flat = (l_cor[..., ri] - l_wro[..., ri]).reshape(n_pairs, -1)
        dz_l, pgt_l = dz_and_pgt(flat)
        lo_l, hi_l = boot_ci_matrix(flat, groups, args.n_boot, args.seed)
        mean_l = flat.mean(axis=0)
        for li in range(n_layers):
            for fi, feat in enumerate(EXT_FEATURES):
                j = li * n_ext + fi
                layer_rows.append({
                    "layer": li, "feature": feat, "read": read_name,
                    "mean": mean_l[j], "ci_lo": lo_l[j], "ci_hi": hi_l[j],
                    "dz": dz_l.reshape(n_layers, n_ext)[li, fi],
                    "p_gt": pgt_l.reshape(n_layers, n_ext)[li, fi],
                })
    tl = pd.DataFrame(layer_rows)
    tl.to_csv(args.out_dir / "table_layer.csv", index=False)
    del l_cor, l_wro

    # ---- head level: discovery/confirmation split, read = first -----------
    h_ext_cor = add_ratios(feats[ic][..., [r_first]])[..., 0]
    h_ext_wro = add_ratios(feats[iw][..., [r_first]])[..., 0]
    h_delta = h_ext_cor - h_ext_wro  # (n_pairs, L, H, n_ext)
    del h_ext_cor, h_ext_wro
    hf_idx = [EXT_FEATURES.index(f) for f in HEAD_FEATURES]
    h_delta = h_delta[:, :, :, hf_idx]

    disc, conf = split_problems(groups, args.seed)
    n_disc_groups = len(np.unique(groups[disc]))
    n_conf_groups = len(np.unique(groups[conf]))
    print(f"[ar-analysis] head split: {disc.sum()} discovery pairs "
          f"({n_disc_groups} problems), {conf.sum()} confirmation pairs "
          f"({n_conf_groups} problems)", flush=True)

    dz_h, pgt_h = dz_and_pgt(h_delta[disc])
    print("[ar-analysis] head-level Wilcoxon (discovery) ...", flush=True)
    p_wx = stats.wilcoxon(h_delta[disc], axis=0, zero_method="wilcox",
                          nan_policy="omit").pvalue
    p_bh = np.full_like(p_wx, np.nan)
    for fi in range(len(HEAD_FEATURES)):  # BH within each feature family
        p_flat = p_wx[:, :, fi].reshape(-1)
        ok = np.isfinite(p_flat)
        q = np.full_like(p_flat, np.nan)
        q[ok] = stats.false_discovery_control(p_flat[ok])
        p_bh[:, :, fi] = q.reshape(n_layers, n_heads)
    np.savez(args.out_dir / "head_effects_discovery.npz",
             dz=dz_h, p_gt=pgt_h, p_bh=p_bh,
             features=np.array(HEAD_FEATURES), read=np.array(["first"]))

    confirm_rows = []
    can_confirm = n_disc_groups >= 4 and n_conf_groups >= 4
    if not can_confirm:
        print("[ar-analysis] too few problems for a split; skipping head "
              "confirmation", flush=True)
    else:
        for fi, feat in enumerate(HEAD_FEATURES):
            sel = np.argwhere(p_bh[:, :, fi] < 0.05)
            sel = sorted(sel, key=lambda lh: -abs(dz_h[lh[0], lh[1], fi]))
            for li, hi_ in sel[:MAX_HEADS_PER_FEATURE]:
                d_conf = h_delta[conf, li, hi_, fi]
                s = paired_stats(d_conf, groups[conf],
                                 n_boot=min(args.n_boot, 1000),
                                 seed=args.seed)
                disc_dz = float(dz_h[li, hi_, fi])
                replicated = (np.sign(s["mean"]) == np.sign(disc_dz)
                              and (s["ci_lo"] > 0 or s["ci_hi"] < 0))
                confirm_rows.append({
                    "feature": feat, "layer": int(li), "head": int(hi_),
                    "discovery_dz": disc_dz,
                    "discovery_p_bh": float(p_bh[li, hi_, fi]),
                    "replicated": bool(replicated),
                    **{f"confirm_{k}": v for k, v in s.items()},
                })
        pd.DataFrame(confirm_rows).to_csv(
            args.out_dir / "table_head_confirmed.csv", index=False)

    # ---- plots -------------------------------------------------------------
    made = []

    def save(fig, name):
        p = plots_dir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(p)

    # ratio and mass distributions, first token (primary) and mean
    for feat in ["grounding_ratio", "recency_ratio",
                 "external_semantic_mass"]:
        fi = EXT_FEATURES.index(feat)
        for ri, read_name in [(r_first, "first"), (r_mean, "mean")]:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
            hi_edge = 1.0 if "ratio" in feat else max(
                g_cor[:, fi, ri].max(), g_wro[:, fi, ri].max())
            bins = np.linspace(0, hi_edge, 41)
            axes[0].hist(g_cor[:, fi, ri], bins=bins, alpha=0.6,
                         color=C_CORRECT, label="correct")
            axes[0].hist(g_wro[:, fi, ri], bins=bins, alpha=0.6,
                         color=C_WRONG, label="wrong")
            axes[0].set_xlabel(feat)
            axes[0].set_ylabel("forks")
            axes[0].legend(frameon=False)
            d = g_delta[:, fi, ri]
            axes[1].hist(d, bins=40, color=C_NEUTRAL, alpha=0.8)
            axes[1].axvline(0, color="black", lw=1)
            axes[1].axvline(d.mean(), color=C_CORRECT, lw=1.5,
                            label=f"mean Δ = {d.mean():+.4f}")
            axes[1].set_xlabel(f"Δ {feat} (correct − wrong)")
            axes[1].legend(frameon=False)
            fig.suptitle(f"{feat} (mean over layers and heads, "
                         f"read={read_name})", fontsize=10)
            save(fig, f"{feat}_dist_{read_name}.png")

    # paired delta histograms, all ext features, first token
    n_cols = 4
    n_rows_grid = int(np.ceil(len(EXT_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows_grid, n_cols,
                             figsize=(3.6 * n_cols, 2.7 * n_rows_grid))
    for ax in axes.ravel()[len(EXT_FEATURES):]:
        ax.axis("off")
    for fi, (feat, ax) in enumerate(zip(EXT_FEATURES, axes.ravel())):
        d = g_delta[:, fi, r_first]
        ax.hist(d, bins=40, color=C_NEUTRAL, alpha=0.8)
        ax.axvline(0, color="black", lw=1)
        s = table_global[(table_global.feature == feat)
                         & (table_global.read == "first")].iloc[0]
        ax.set_title(f"{feat}\ndz={s['dz']:+.3f}  P(+>−)={s['p_gt']:.3f}",
                     fontsize=9)
    fig.suptitle("Paired Δ (correct − wrong), global mean, read=first",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "feature_delta_hists_first.png")

    # layer curves per read
    for read_name in ["first", "mean"]:
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
        for feat, ax in zip(KEY_FEATURES, axes.ravel()):
            sub = tl[(tl.feature == feat) & (tl.read == read_name)]
            ax.plot(sub.layer, sub["mean"], color=C_CORRECT, lw=1.5)
            ax.fill_between(sub.layer, sub.ci_lo, sub.ci_hi,
                            color=C_CORRECT, alpha=0.25, lw=0)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(feat, fontsize=9)
            ax.set_xlabel("layer")
        axes[0, 0].set_ylabel("mean Δ (correct − wrong)")
        axes[1, 0].set_ylabel("mean Δ (correct − wrong)")
        fig.suptitle(f"Layer-wise paired difference, head-mean, "
                     f"read={read_name}, 95% cluster-bootstrap CI",
                     fontsize=11)
        fig.tight_layout()
        save(fig, f"layer_curves_{read_name}.png")

    # layer x head heatmaps (discovery half, read=first)
    for feat in HEATMAP_FEATURES:
        fi = HEAD_FEATURES.index(feat)
        fig, ax = plt.subplots(figsize=(8, 6))
        v = np.abs(dz_h[:, :, fi]).max()
        im = ax.imshow(dz_h[:, :, fi].T, cmap="RdBu_r", vmin=-v, vmax=v,
                       aspect="auto", origin="lower")
        sig = p_bh[:, :, fi] < 0.05
        ys, xs = np.where(sig.T)
        ax.scatter(xs, ys, s=2, c="black", alpha=0.5)
        ax.set_xlabel("layer")
        ax.set_ylabel("head")
        ax.set_title(f"paired dz, Δ {feat} (read=first, DISCOVERY half); "
                     "dots: BH q<0.05")
        fig.colorbar(im, ax=ax, label="dz (correct − wrong)")
        save(fig, f"head_heatmap_{feat}_first.png")

    # ---- summary -----------------------------------------------------------
    n_replicated = sum(r["replicated"] for r in confirm_rows)
    lines = [
        "# attention_routing_v0 paired fork analysis",
        f"generated {datetime.now(timezone.utc).isoformat()}",
        f"pairs: {n_pairs}, problems: {len(np.unique(groups))}",
        "",
        "## Confirmatory metrics (frozen, first token, full data)",
        "",
        "| feature | mean Δ | 95% CI | dz | P(f+>f−) | P group | p_wilcoxon |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, s in table_conf.iterrows():
        lines.append(
            f"| {s['feature']} | {s['mean']:+.4f} | [{s['ci_lo']:+.4f}, "
            f"{s['ci_hi']:+.4f}] | {s['dz']:+.3f} | {s['p_gt']:.3f} | "
            f"{s['p_gt_group']:.3f} | {s['p_wilcoxon']:.2e} |")
    lines += [
        "",
        "## Surface-control regression (beta0 after length/logprob/entropy/"
        "number controls)",
        "",
        "| feature | raw mean Δ | beta0 | se | p |",
        "|---|---|---|---|---|",
    ]
    for r in reg_rows:
        lines.append(f"| {r['feature']} | {r['raw_mean']:+.4f} | "
                     f"{r['beta0']:+.4f} | {r['se0']:.4f} | {r['p0']:.2e} |")
    lines += [
        "",
        f"head confirmation: {n_replicated}/{len(confirm_rows)} selected "
        "heads replicated on the confirmation half"
        if can_confirm else "head confirmation: skipped (too few problems)",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    with open(args.out_dir / "analysis_manifest.json", "w") as f:
        json.dump({"created_utc": datetime.now(timezone.utc).isoformat(),
                   "run_dir": str(args.run_dir), "n_pairs": int(n_pairs),
                   "n_dropped_nonfinite": n_dropped,
                   "n_boot": args.n_boot, "seed": args.seed,
                   "confirmatory": CONFIRMATORY,
                   "head_features": HEAD_FEATURES,
                   "n_discovery_pairs": int(disc.sum()),
                   "n_confirmation_pairs": int(conf.sum())}, f, indent=2)

    print("\n".join(lines), flush=True)
    print("[ar-analysis] plots:", flush=True)
    for p in made:
        print(f"  {p}", flush=True)


if __name__ == "__main__":
    main()
