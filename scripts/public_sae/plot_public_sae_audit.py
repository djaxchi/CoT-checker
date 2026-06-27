"""Figures (A-F) + auto-assembled report for the public-SAE audit.

Consumes the outputs of probe_public_sae.py (metrics.json, scores_{L}.npz,
localization_{L}.json, direction_{L}.json), the encode manifests, and the dense /
sae_z arrays, and writes per-layer figure sets plus the final report:

  figures/A_metrics_{L}.png       dense vs z vs h_hat vs resid : AUROC/balacc/d'
  figures/B_score_hist_{L}.png    probe-score histograms (correct vs incorrect) x4
  figures/C_supervised_proj_{L}.png  probe-axis score vs 1st orthogonal PC (dense | z)
  figures/D_unsup_geom_{L}.png    PCA(2) of dense h and of SAE z, colour=label
  figures/E_localization_{L}.png  dense-vs-z score scatter; (w_i,dz_i); cum-AUROC vs top-k
  runs/public_sae_audit/public_sae_audit_report.md

Split is reproduced identically to the probe (SEED=42, stratified, same test_size).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split

SEED = 42
REPS = [("dense", "dense h"), ("sae_z", "SAE z"), ("hhat", "h_hat"), ("resid", "h - h_hat")]


def load_z(npz):
    d = np.load(npz)
    n, dim = int(d["shape"][0]), int(d["shape"][1])
    return sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=(n, dim))


def fig_A(rows, L, out):
    sub = {r["representation"]: r for r in rows if r["layer"] == L}
    order = ["dense h", "SAE z", "SAE h_hat", "SAE residual h-h_hat"]
    labels = ["dense", "z", "h_hat", "resid"]
    auroc = [sub[o]["auroc"] for o in order]
    bal = [sub[o]["balacc_oracle"] for o in order]
    dp = [sub[o]["dprime"] for o in order]
    x = np.arange(len(order)); w = 0.27
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, auroc, w, label="AUROC")
    ax.bar(x, bal, w, label="bal-acc (oracle)")
    ax2 = ax.twinx(); ax2.plot(x + w, dp, "ko-", label="d'")
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC / bal-acc"); ax2.set_ylabel("d'")
    ax.set_title(f"{L}: dense vs SAE representations"); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_B(scores, L, out):
    y = scores["y_test"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    for ax, (key, name) in zip(axes, REPS):
        s = scores[key]
        ax.hist(s[y == 0], bins=30, alpha=0.6, label="correct", density=True)
        ax.hist(s[y == 1], bins=30, alpha=0.6, label="incorrect", density=True)
        ax.set_title(name); ax.set_xlabel("P(incorrect)")
    axes[0].legend(); axes[0].set_ylabel("density")
    fig.suptitle(f"{L}: probe-score class separation")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def _orth_pc(X, score):
    """1st PC of X (n,d, dense) after regressing out the per-example probe score."""
    Xc = X - X.mean(0)
    s = np.asarray(score, float) - float(np.mean(score))
    ss = float(s @ s) or 1.0
    beta = (Xc.T @ s) / ss                 # per-feature slope on the score
    resid = Xc - np.outer(s, beta)         # component orthogonal to the probe axis
    return PCA(n_components=1, random_state=SEED).fit_transform(resid).ravel()


def fig_C(h_te, Z_te, scores, L, out):
    y = scores["y_test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # dense: x=probe score, y=1st PC of h orthogonal to the probe axis
    yax = _orth_pc(h_te, np.asarray(scores["dense"]))
    _scatter(axes[0], scores["dense"], yax, y, "dense h", "probe score")
    # sae: reduce z to a cheap dense subspace (SVD-50) then same orthogonal PC
    k = min(50, Z_te.shape[1] - 1, Z_te.shape[0] - 1)
    zr = TruncatedSVD(n_components=max(k, 2), random_state=SEED).fit_transform(Z_te)
    yax_z = _orth_pc(zr, np.asarray(scores["sae_z"]))
    _scatter(axes[1], scores["sae_z"], yax_z, y, "SAE z", "probe score")
    fig.suptitle(f"{L}: supervised projection (probe axis vs orthogonal structure)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_D(h_all, Z_all, y, L, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    pc = PCA(n_components=2, random_state=SEED).fit_transform(h_all - h_all.mean(0))
    _scatter(axes[0], pc[:, 0], pc[:, 1], y, "dense h  (PCA)", "PC1", "PC2")
    sv = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(Z_all)
    _scatter(axes[1], sv[:, 0], sv[:, 1], y, "SAE z  (SVD)", "SV1", "SV2")
    fig.suptitle(f"{L}: unsupervised geometry, colour = correctness")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_E(scores, loc, direction, L, out):
    y = scores["y_test"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _scatter(axes[0], scores["dense"], scores["sae_z"], y,
             "dense vs SAE-z probe score", "dense P(inc)", "z P(inc)")
    w = np.array(loc["top20_probe_weight"]); dz = np.array(loc["top20_class_delta"])
    axes[1].scatter(dz, w, c="tab:purple"); axes[1].axhline(0, c="grey", lw=.6)
    axes[1].axvline(0, c="grey", lw=.6); axes[1].set_xlabel("class delta dz_i")
    axes[1].set_ylabel("probe weight w_i"); axes[1].set_title("top-20 SAE feats")
    ks, cum = loc["top_k_grid"], loc["cumulative_auroc"]
    axes[2].plot(ks, cum, "o-"); axes[2].axhline(loc["full_z_auroc"], ls="--", c="k",
                                                 label=f"full z AUROC={loc['full_z_auroc']:.3f}")
    axes[2].set_xscale("log"); axes[2].set_xlabel("# top SAE features")
    axes[2].set_ylabel("AUROC"); axes[2].set_title("cumulative AUROC vs top-k"); axes[2].legend()
    fig.suptitle(f"{L}: is correctness localized in a few SAE features?")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def _scatter(ax, x, y_axis, label, title, xl="", yl=""):
    x = np.asarray(x); y_axis = np.asarray(y_axis)
    ax.scatter(x[label == 0], y_axis[label == 0], s=6, alpha=.4, label="correct")
    ax.scatter(x[label == 1], y_axis[label == 1], s=6, alpha=.4, label="incorrect")
    ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(markerscale=2)


def build_report(args, rows, layers):
    L0 = layers[0]
    lines = ["# Public-SAE representation audit (Qwen2.5-7B-Instruct)", ""]
    lines += ["## 1. Model and SAE", "",
              "* **Case: Instruct-matched** (no public base-7B residual SAE exists; see "
              "`scripts/public_sae/download_public_sae.md`).",
              "* Backbone: `Qwen/Qwen2.5-7B-Instruct`. SAE: "
              "`andyrdt/saes-qwen2.5-7b-instruct`, BatchTopK, dict_size 131072.",
              ""]
    for L in layers:
        man = json.loads((args.sae_dir / f"{L}_t{args.trainer}_encode_manifest.json").read_text())
        lines.append(f"  * {L} <- `{man['sae_folder']}/trainer_{args.trainer}` "
                     f"(k={man['k']}); recon FVU={man['recon_fvu']} [{man['fvu_gate']}]; "
                     f"active/row median={man['active_per_row_median']:.0f}; "
                     f"feats ever active={man['n_features_ever_active']}.")
    em = json.loads((args.enc_dir / "extract_manifest.json").read_text())
    lines += ["", "> **Caveat:** Instruct SAE + Instruct backbone (matched). The S3 base-7B "
              "dense caches are NOT used here except for the example text/labels. The "
              "prompt is the S3 base-style prompt (no chat template), mildly OOD vs the "
              "SAE's chat/pile training - the FVU gate above is the empirical check.", ""]
    lines += ["## 2. Dataset and split", "",
              f"* PRM800K held-out steps, n={em['n']} "
              f"(correct={em['n_correct']}, incorrect={em['n_incorrect']}).",
              f"* Fixed stratified split, seed {SEED}, test_size={args.test_size}. "
              "PRM800K only (no ProcessBench transfer in this pilot).", ""]
    lines += ["## 3. Main metrics table", "",
              "| layer | representation | dim | used-dim | AUROC | bal-acc | d' | overlap | F1-inc | notes |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['layer']} | {r['representation']} | {r['dim']} | {r['used_dim']} "
                     f"| {r['auroc']} | {r['balacc_oracle']} | {r['dprime']} | {r['overlap']} "
                     f"| {r['f1_incorrect_oracle']} | {r['notes']} |")
    lines += ["", "## 4. Feature localization", ""]
    for L in layers:
        loc = json.loads((args.probe_dir / f"localization_{L}.json").read_text())
        lines.append(f"* {L}: full SAE-z AUROC={loc['full_z_auroc']:.3f}; "
                     f"top features to reach 95% = {loc['n_feats_for_95pct']}, "
                     f"99% = {loc['n_feats_for_99pct']} (of {loc['top_k_grid'][-1]}+ ranked).")
    lines += ["", "## 5. Probe-direction comparison", ""]
    for L in layers:
        ddirec = json.loads((args.probe_dir / f"direction_{L}.json").read_text())
        seg = (f"cos(SAE-mapped dir, dense dir)={ddirec['cos_saemap_vs_dense']:.3f}, "
               f"cos(SAE-mapped, mean-diff)={ddirec['cos_saemap_vs_meandiff']:.3f}, "
               if "cos_saemap_vs_dense" in ddirec else "")
        lines.append(f"* {L}: {seg}cos(dense dir, mean-diff)={ddirec['cos_dense_vs_meandiff']:.3f}, "
                     f"pearson(dense score, z score)={ddirec['pearson_dense_vs_z_score']:.3f}.")
    lines += ["", "## 5b. Top SAE feature table (G)", ""]
    for L in layers:
        fe = args.probe_dir / f"feature_examples_{L}.md"
        if fe.exists():
            lines.append(f"* {L}: top features + max-activating steps -> `{fe}` "
                         "(inspect for: step-validity / contradiction / correction / "
                         "arithmetic vs. confounds like length / position / formatting).")
    lines += ["", "## 6. Figures", ""]
    for L in layers:
        for tag in ["A_metrics", "B_score_hist", "C_supervised_proj", "D_unsup_geom", "E_localization"]:
            lines.append(f"![{tag} {L}](figures/{tag}_{L}.png)")
    lines += ["", "## 7. Conclusion", "",
              "_Auto-table above is the evidence; fill the verdict after inspecting it._",
              "Decision rule: SAE **helps** if z beats dense AUROC, or matches dense with "
              "few features (sec 4), or sec D shows visible clusters. SAE **does not help** "
              "if z<=dense and the signal still needs many features and geometry stays one blob. "
              "Outcome **D** (recon loses signal but residual keeps it) = compare rows "
              "`SAE h_hat` vs `SAE residual` against `dense h`.", ""]
    (args.report).write_text("\n".join(lines))
    print(f"[plot] wrote {args.report}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enc_dir", type=Path, required=True)
    ap.add_argument("--sae_dir", type=Path, required=True)
    ap.add_argument("--probe_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True, help="figures dir")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--layers", nargs="+", default=["L20", "L28"])
    ap.add_argument("--trainer", type=int, default=1)
    ap.add_argument("--test_size", type=float, default=0.3)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads((args.probe_dir / "metrics.json").read_text())
    y = np.load(args.enc_dir / "heldout_y.npy").astype(int)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=args.test_size,
                                      random_state=SEED, stratify=y)

    for L in args.layers:
        t = args.trainer
        scores = dict(np.load(args.probe_dir / f"scores_{L}.npz"))
        loc = json.loads((args.probe_dir / f"localization_{L}.json").read_text())
        direction = json.loads((args.probe_dir / f"direction_{L}.json").read_text())
        h = np.load(args.enc_dir / f"heldout_{L}_h.npy").astype(np.float32)
        Z = load_z(args.sae_dir / f"{L}_t{t}_z.npz")
        for name, fn in [
            ("A_metrics", lambda o: fig_A(rows, L, o)),
            ("B_score_hist", lambda o: fig_B(scores, L, o)),
            ("C_supervised_proj", lambda o: fig_C(h[idx_te], Z[idx_te], scores, L, o)),
            ("D_unsup_geom", lambda o: fig_D(h, Z, y, L, o)),
            ("E_localization", lambda o: fig_E(scores, loc, direction, L, o)),
        ]:
            out = args.out_dir / f"{name}_{L}.png"
            try:
                fn(out); print(f"[plot] {out.name}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[plot] WARN {name}_{L} failed: {e}", flush=True)

    build_report(args, rows, args.layers)


if __name__ == "__main__":
    main()
