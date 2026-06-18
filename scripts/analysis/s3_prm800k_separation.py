"""What separates correct from incorrect PRM800K steps, and is it real?

The val_1k projection showed step-correctness is linearly decodable (~0.74) but
does not form clusters. This script characterises the *direction* that separates
the classes and stress-tests what drives it:

1. Direction. Fit a linear classifier (correct=0 vs incorrect=1) on the full
   3,584 standardized dims. Compare it to the difference-of-class-means vector and
   to the deployed 7B probe direction (cosine). Report honest out-of-fold AUC and
   the class separation (Cohen's d) on the 1D projection.

2. Visualise the difference. Project every step onto the correctness axis
   (out-of-fold decision scores) and plot the correct vs incorrect distributions
   (the picture of the separation), plus a 2D view of the axis against step length.

3. What causes it (confound decomposition). Do incorrect steps differ trivially in
   token length or step position? Decode the label from those two scalars alone;
   then residualise the hidden states against them and re-measure AUC. If AUC
   survives, the separation is not a length/position artifact.

Outputs (results/prm800k_val/):
  - separation_hist.png        correct vs incorrect along the correctness axis
  - separation_2d.html         correctness axis vs step length, coloured by label
  - separation_stats.json      AUC, Cohen's d, direction cosines, confound table

Usage:
    python scripts/analysis/s3_prm800k_separation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.data.prm800k_val_data import DEFAULT_MERGED_DIR, load_prm800k_val
from src.data.processbench_probe_data import DEFAULT_RUN_DIR, load_probe

ROOT = Path("results/prm800k_val")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    s = cross_val_predict(clf, X, y, cv=5, method="decision_function")
    return float(roc_auc_score(y, s)), s


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", type=Path, default=DEFAULT_MERGED_DIR)
    ap.add_argument("--stem", type=str, default="val_1k",
                    help="encoding stem to load, e.g. val_1k or prm800k_heldout_test_6k")
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = load_prm800k_val(args.merged_dir, stem=args.stem)
    y = d.label  # 0 = correct, 1 = incorrect
    n = len(d)
    print(f"[load] {n} steps | correct={(y==0).sum()} incorrect={(y==1).sum()}")

    scaler = StandardScaler().fit(d.hidden)
    Xs = scaler.transform(d.hidden)

    # ---- 1. direction + honest separation -------------------------------
    auc, cv_score = cv_auc(Xs, y, args.seed)
    dcorr, dinc = cv_score[y == 0], cv_score[y == 1]
    dval = cohens_d(dinc, dcorr)

    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xs, y)
    w = clf.coef_[0]
    dmu = Xs[y == 1].mean(0) - Xs[y == 0].mean(0)         # incorrect - correct
    cos_w_dmu = float(np.dot(unit(w), unit(dmu)))

    cos_probe = None
    try:
        pw, _ = load_probe(args.run_dir)
        if pw.shape[0] == d.hidden.shape[1]:
            w_raw = w / scaler.scale_  # map standardized weights back to raw space
            cos_probe = float(np.dot(unit(w_raw), unit(pw)))
    except Exception as e:  # noqa: BLE001
        print(f"[probe] cosine skipped ({e})")

    print(f"\n[separation] out-of-fold AUC = {auc:.3f} | Cohen's d = {dval:.2f}")
    print(f"[direction] cos(logreg, diff-of-means) = {cos_w_dmu:.3f}")
    if cos_probe is not None:
        print(f"[direction] cos(logreg, deployed 7B probe) = {cos_probe:.3f}")

    # ---- 2. confounds: length & position --------------------------------
    length = d.n_tokens.astype(float)
    pos = d.step_idx.astype(float)
    corr_len = float(np.corrcoef(y, length)[0, 1])
    corr_pos = float(np.corrcoef(y, pos)[0, 1])

    Z = StandardScaler().fit_transform(np.column_stack([length, pos]))
    auc_trivial, _ = cv_auc(Z, y, args.seed)

    # residualise hidden on [1, length, pos], re-measure AUC
    Zc = np.column_stack([np.ones(n), Z])
    beta, *_ = np.linalg.lstsq(Zc, Xs, rcond=None)
    resid = Xs - Zc @ beta
    auc_resid, _ = cv_auc(resid, y, args.seed)

    print(f"\n[confound] corr(label, n_tokens) = {corr_len:+.3f} | "
          f"corr(label, step_idx) = {corr_pos:+.3f}")
    print(f"[confound] AUC from (length, position) alone = {auc_trivial:.3f}")
    print(f"[confound] AUC after residualising out length+position = {auc_resid:.3f} "
          f"(full = {auc:.3f})")

    stats = {
        "n": n, "n_correct": int((y == 0).sum()), "n_incorrect": int((y == 1).sum()),
        "auc_full": round(auc, 4), "cohens_d": round(dval, 4),
        "cos_logreg_diffmeans": round(cos_w_dmu, 4),
        "cos_logreg_deployed_probe": None if cos_probe is None else round(cos_probe, 4),
        "corr_label_n_tokens": round(corr_len, 4),
        "corr_label_step_idx": round(corr_pos, 4),
        "auc_length_position_only": round(auc_trivial, 4),
        "auc_residualised": round(auc_resid, 4),
    }
    (args.out_dir / "separation_stats.json").write_text(json.dumps(stats, indent=2))

    # ---- 3a. histogram of the correctness axis --------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(cv_score.min(), cv_score.max(), 40)
    ax.hist(dcorr, bins=bins, alpha=0.6, color="#3cb44b", label=f"correct (n={len(dcorr)})")
    ax.hist(dinc, bins=bins, alpha=0.6, color="#e6194B", label=f"incorrect (n={len(dinc)})")
    ax.axvline(0, color="#555", lw=0.8, ls="--")
    ax.set_xlabel("correctness axis  (out-of-fold logreg decision score; >0 -> incorrect)")
    ax.set_ylabel("count")
    ax.set_title(f"PRM800K val_1k: correct vs incorrect along the learned direction\n"
                 f"AUC={auc:.3f}  Cohen's d={dval:.2f}  "
                 f"(residualised AUC={auc_resid:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "separation_hist.png", dpi=130)
    plt.close(fig)

    # ---- 3b. correctness axis vs length (confound view) -----------------
    fig2 = go.Figure()
    for lab, name, col in [(0, "correct", "#3cb44b"), (1, "incorrect", "#e6194B")]:
        m = y == lab
        fig2.add_trace(go.Scattergl(
            x=cv_score[m], y=length[m], mode="markers", name=name,
            marker=dict(size=5, color=col, opacity=0.6),
            text=[f"{d.uid[i]}<br>step={d.step_idx[i]} len={d.n_tokens[i]} "
                  f"rating={d.rating[i]:+d}" for i in np.where(m)[0]],
            hoverinfo="text"))
    fig2.update_layout(
        title=f"Correctness axis vs step length (corr(label,len)={corr_len:+.2f}; "
              f"length+pos-only AUC={auc_trivial:.2f})",
        xaxis_title="correctness axis (decision score; >0 -> incorrect)",
        yaxis_title="n_tokens (step length)",
        width=1000, height=700, template="plotly_white")
    fig2.write_html(args.out_dir / "separation_2d.html", include_plotlyjs="cdn")

    print(f"\n[done] wrote separation_hist.png + separation_2d.html + "
          f"separation_stats.json -> {args.out_dir}/")


if __name__ == "__main__":
    main()
