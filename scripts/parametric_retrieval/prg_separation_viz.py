"""parametric_retrieval_geometry_v0: visualize the separation-search winners.

Honest held-out pictures (fit on one half of FACTS, plot the other half):

  panel A  direct_answer_hs20 (l2): held-out logistic score distributions,
           direct_retrieval vs non_retrieved (best overall, answer-side)
  panel B  delta_cot-direct_prompt_hs20: same, pre-generation only
           (the instruction-sensitivity displacement; both states are prompt
           tokens, no answer content)
  panel C  4-class LDA (fit on train facts) of delta_cot-direct_prompt_hs20,
           held-out facts on the first two discriminants

Output: geometry/plots/separation_winners.png

  python scripts/parametric_retrieval/prg_separation_viz.py \
      --out_dir runs/parametric_retrieval_geometry_v0
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
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from safetensors.numpy import load_file
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    hs_dir = args.out_dir / "hidden_states"
    meta = pd.read_parquet(hs_dir / "hs_meta.parquet")
    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    qa = grading[~grading.is_control
                 & grading.retrieval_class.isin(CLASSES)].copy()
    qa = qa.sort_values("question_id").reset_index(drop=True)
    qid_i = {q: i for i, q in enumerate(qa.question_id)}
    n = len(qa)
    y = qa.retrieval_class.to_numpy()

    meta = meta.reset_index().rename(columns={"index": "row_pos"})
    meta_qa = meta[meta.question_id.isin(qid_i)]

    def gather(mode, pos, H):
        g = meta_qa[(meta_qa.prompt_mode == mode)
                    & (meta_qa.position_name == pos)]
        rows = np.full(n, -1, dtype=int)
        rows[[qid_i[q] for q in g.question_id]] = g.row_pos.to_numpy()
        ok = rows >= 0
        X = np.zeros((n, H.shape[1]), dtype=np.float32)
        X[ok] = H[rows[ok]].astype(np.float32)
        return X, ok

    H20 = load_file(hs_dir / "layer_20.safetensors")["h"]
    Xans, ok_a = gather("direct", "final_answer_token", H20)
    Xdp, ok_d = gather("direct", "final_prompt_token", H20)
    Xcp, ok_c = gather("cot", "final_prompt_token", H20)
    Xans = Xans / np.maximum(np.linalg.norm(Xans, axis=1, keepdims=True), 1e-8)
    Xdelta = Xcp - Xdp
    ok_delta = ok_d & ok_c

    # ---- fact-level 50/50 split --------------------------------------------
    rng = np.random.default_rng(args.seed)
    facts = np.array(sorted(qa.fact_id.unique()))
    rng.shuffle(facts)
    train_facts = set(facts[: len(facts) // 2])
    is_tr = qa.fact_id.isin(train_facts).to_numpy()

    def pipe():
        return Pipeline([("sc", StandardScaler()),
                         ("pca", PCA(n_components=256, random_state=0)),
                         ("lr", LogisticRegression(max_iter=3000))])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, (X, ok, title) in zip(axes[:2], [
            (Xans, ok_a, "A  direct_answer_hs20 (answer token, L2)"),
            (Xdelta, ok_delta,
             "B  delta cot-direct prompt hs20 (pre-generation)")]):
        mb = ok & np.isin(y, ["direct_retrieval", "non_retrieved"])
        yb = (y == "direct_retrieval").astype(int)
        tr, te = mb & is_tr, mb & ~is_tr
        p = pipe().fit(X[tr], yb[tr])
        s = p.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(yb[te], s)
        bins = np.linspace(0, 1, 36)
        for cls, val in [("non_retrieved", 0), ("direct_retrieval", 1)]:
            ax.hist(s[yb[te] == val], bins=bins, density=True, alpha=0.55,
                    color=COLORS[cls], label=cls, edgecolor="white",
                    linewidth=0.4)
        ax.set_title(f"{title}\nheld-out facts, AUROC {auc:.3f}", fontsize=10,
                     loc="left")
        ax.set_xlabel("logistic score (P retrieved)", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    # ---- panel C: held-out 4-class LDA of the delta rep --------------------
    ax = axes[2]
    tr, te = ok_delta & is_tr, ok_delta & ~is_tr
    red = Pipeline([("sc", StandardScaler()),
                    ("pca", PCA(n_components=256, random_state=0))])
    Ztr = red.fit_transform(Xdelta[tr])
    Zte = red.transform(Xdelta[te])
    lda = LinearDiscriminantAnalysis(n_components=2).fit(Ztr, y[tr])
    E = lda.transform(Zte)
    for cls in CLASSES:
        m = y[te] == cls
        ax.scatter(E[m, 0], E[m, 1], s=6, alpha=0.5, color=COLORS[cls],
                   label=f"{cls} ({m.sum()})", linewidths=0)
    for cls in CLASSES:
        m = y[te] == cls
        ax.scatter(E[m, 0].mean(), E[m, 1].mean(), s=130, marker="X",
                   color=COLORS[cls], edgecolor="white", linewidth=1.6,
                   zorder=5)
    ax.set_title("C  4-class LDA of delta cot-direct prompt hs20\n"
                 "held-out facts (X = class centroid)", fontsize=10,
                 loc="left")
    ax.set_xlabel("LD1", fontsize=9)
    ax.set_ylabel("LD2", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, markerscale=1.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    out = args.out_dir / "geometry" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = out / "separation_winners.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"[viz] wrote {path}")


if __name__ == "__main__":
    main()
