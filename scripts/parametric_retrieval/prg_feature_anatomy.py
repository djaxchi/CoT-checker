"""parametric_retrieval_geometry_v0: anatomy of the top retrieval-signal SAE
features (default: hs24 direct_prompt latents, the FVU-healthy arm).

Stage A (this script, local, no GPU): for the top-N discriminative latents,
using the STORED final-prompt-token states:
  - activation distribution per retrieval class (mean, quartiles)
  - Spearman correlation with log gbc (popularity-feature check), with
    question/subject/object surface lengths, and partial AUROC within gbc bins
  - top / bottom activating questions (what content drives the feature)
  - pairwise feature correlation (are the ~10 L1-picked latents one signal?)

Also writes feature_pack_layer{B}.npz (encoder rows, biases, threshold, b_dec
for the selected features) so the per-token TamIA job
(prg_feature_tokens.py) does not need the 3.5 GB ae.pt.

Outputs under <out_dir>/sae/:
  feature_anatomy_hs{K}.csv, feature_examples_hs{K}.md,
  feature_pack_layer{K-1}.npz, plots/feature_anatomy_hs{K}.png

  python scripts/parametric_retrieval/prg_feature_anatomy.py \
      --out_dir runs/parametric_retrieval_geometry_v0 \
      --sae_root data/public_sae/andyrdt-qwen2.5-7b-instruct --hs 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLASSES = ["direct_retrieval", "reasoning_unlocked", "unstable_retrieval",
           "non_retrieved"]
COLORS = {"direct_retrieval": "#3987e5", "reasoning_unlocked": "#cc7a00",
          "unstable_retrieval": "#9467bd", "non_retrieved": "#e34948"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--sae_root", type=Path,
                    default=Path("data/public_sae/andyrdt-qwen2.5-7b-instruct"))
    ap.add_argument("--hs", type=int, default=24)
    ap.add_argument("--trainer", type=int, default=1)
    ap.add_argument("--rep", default="direct_prompt",
                    choices=["direct_prompt", "direct_answer"])
    ap.add_argument("--n_features", type=int, default=12)
    args = ap.parse_args()

    import torch
    from safetensors.numpy import load_file
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    K = args.hs
    block = K - 1
    sae_dir = args.out_dir / "sae"
    top = pd.read_csv(sae_dir / f"top_features_hs{K}_{args.rep}.csv")
    top = top.reindex(top.auc_dr_vs_nr.sub(0.5).abs()
                      .sort_values(ascending=False).index)
    feats = top.feature.head(args.n_features).astype(int).tolist()
    print(f"[anat] hs{K} {args.rep}: features {feats}", flush=True)

    # ---- feature pack (also consumed by the TamIA per-token job) ----------
    sd = torch.load(args.sae_root / f"resid_post_layer_{block}"
                    / f"trainer_{args.trainer}" / "ae.pt", map_location="cpu")
    pack = {
        "features": np.asarray(feats, np.int64),
        "W_enc": sd["encoder.weight"][feats].numpy().astype(np.float32),
        "b_enc": sd["encoder.bias"][feats].numpy().astype(np.float32),
        "b_dec": sd["b_dec"].numpy().astype(np.float32),
        "threshold": np.float32(sd["threshold"].item()),
        "block": np.int64(block), "hs_idx": np.int64(K),
    }
    pack_path = sae_dir / f"feature_pack_layer{block}.npz"
    np.savez(pack_path, **pack)
    del sd
    print(f"[anat] wrote {pack_path}", flush=True)

    # ---- gather stored states + compute selected-feature activations ------
    hs_dir = args.out_dir / "hidden_states"
    meta = pd.read_parquet(hs_dir / "hs_meta.parquet") \
        .reset_index().rename(columns={"index": "row_pos"})
    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    md = pd.read_parquet(args.out_dir / "metadata.parquet")
    qa = grading[~grading.is_control
                 & grading.retrieval_class.isin(CLASSES)].copy()
    qa = qa.merge(md[["question_id", "question", "gbc", "subject", "object"]],
                  on="question_id", how="left").sort_values("question_id") \
        .reset_index(drop=True)
    mode, pos = (("direct", "final_prompt_token")
                 if args.rep == "direct_prompt"
                 else ("direct", "final_answer_token"))
    g = meta[(meta.prompt_mode == mode) & (meta.position_name == pos)
             & meta.question_id.isin(set(qa.question_id))]
    g = g.set_index("question_id").reindex(qa.question_id)
    H = load_file(hs_dir / f"layer_{K:02d}.safetensors")["h"]
    X = H[g.row_pos.to_numpy()].astype(np.float32)
    pre = (X - pack["b_dec"]) @ pack["W_enc"].T + pack["b_enc"]
    Z = np.maximum(pre, 0.0)
    Z = Z * (Z > pack["threshold"])

    y = qa.retrieval_class.to_numpy()
    yb_mask = np.isin(y, ["direct_retrieval", "non_retrieved"])
    yb = (y[yb_mask] == "direct_retrieval").astype(int)
    lg = np.log1p(qa.gbc.to_numpy(float))
    qlen = qa.question.str.len().to_numpy(float)

    rows, md_lines = [], [f"# Feature anatomy hs{K} {args.rep}\n"]
    for j, f in enumerate(feats):
        v = Z[:, j]
        auc = roc_auc_score(yb, v[yb_mask])
        rho_gbc = spearmanr(v, lg).statistic
        rho_len = spearmanr(v, qlen).statistic
        within = []
        gbc_bin = md.set_index("question_id").gbc_bin \
            .reindex(qa.question_id).to_numpy()
        for b in ["low", "mid", "high", "very_high"]:
            m = yb_mask & (gbc_bin == b)
            ybb = (y[m] == "direct_retrieval").astype(int)
            if 0 < ybb.sum() < len(ybb):
                within.append(roc_auc_score(ybb, v[m]))
        row = {"feature": f, "auc": round(auc, 3),
               "auc_within_bin_mean": round(float(np.mean(within)), 3),
               "spearman_log_gbc": round(float(rho_gbc), 3),
               "spearman_qlen": round(float(rho_len), 3),
               "freq": round(float((v > 0).mean()), 3)}
        for c in CLASSES:
            row[f"mean_{c.split('_')[0]}"] = round(float(v[y == c].mean()), 2)
        rows.append(row)

        order = np.argsort(-v)
        md_lines.append(f"\n## feature {f}  (AUROC {auc:.3f}, "
                        f"rho(log gbc)={rho_gbc:.2f}, freq {(v>0).mean():.2f})")
        md_lines.append("\n**top activating questions:**")
        for i in order[:6]:
            md_lines.append(f"- [{v[i]:.1f}] ({y[i]}, gbc {qa.gbc.iloc[i]}) "
                            f"{qa.question.iloc[i]}")
        md_lines.append("\n**bottom activating questions:**")
        for i in order[::-1][:6]:
            md_lines.append(f"- [{v[i]:.1f}] ({y[i]}, gbc {qa.gbc.iloc[i]}) "
                            f"{qa.question.iloc[i]}")

    anat = pd.DataFrame(rows)
    anat.to_csv(sae_dir / f"feature_anatomy_hs{K}.csv", index=False)
    (sae_dir / f"feature_examples_hs{K}.md").write_text(
        "\n".join(md_lines))
    print(anat.to_string(index=False))

    corr = np.corrcoef(Z.T)
    print("\n[anat] |pairwise feature correlation| mean off-diag: "
          f"{np.abs(corr[~np.eye(len(feats), dtype=bool)]).mean():.2f}")

    # ---- figure: per-class violin-ish strip for top 6 ----------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), dpi=140)
    rng = np.random.default_rng(0)
    for ax, (j, f) in zip(axes.ravel(), enumerate(feats[:6])):
        for ci, c in enumerate(CLASSES):
            vv = Z[y == c, j]
            xs = ci + rng.uniform(-0.16, 0.16, len(vv))
            ax.scatter(xs, vv, s=2.5, alpha=0.28, color=COLORS[c],
                       linewidths=0)
            ax.hlines(np.median(vv), ci - 0.26, ci + 0.26,
                      color=COLORS[c], linewidth=2.4)
        ax.set_title(f"feature {f} (AUROC {anat.auc.iloc[j]:.3f})",
                     fontsize=9.5, loc="left")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["dr", "ru", "ur", "nr"], fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Top SAE latents at hs{K} ({args.rep}), activation by "
                 "retrieval class (bar = median)", fontsize=11)
    fig.tight_layout()
    out_png = sae_dir / "plots" / f"feature_anatomy_hs{K}.png"
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    print(f"[anat] wrote {out_png}")


if __name__ == "__main__":
    main()
