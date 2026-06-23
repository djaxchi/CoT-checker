#!/usr/bin/env python3
"""What drives the per-step correctness score? Visualize and attribute.

Companion to analyze_fork_representation_audit.py / inspect_surface_margin.py. The
fork audit showed the matched margin is real and non-surface, with length a minor
(~15-20% variance) confound. This script works at the level of individual steps
(pooled positives + negatives, anchors dropped) to (a) VISUALIZE the representation
colored by length / label / probe score, and (b) ATTRIBUTE the score: how much of
the correct/incorrect discrimination is length vs survives removing it.

Decisive number: pooled score AUC for label=incorrect, before and after regressing
length (and other cheap features) out of the score. If AUC barely drops, the signal
is not length.

Cheap per-step features (zero GPU): length = n_tokens, step_idx (from fork_id
prm800k::pid::sid::sidx), numeric_count and has_answer (from candidate_step text).

Inputs:
  --h runs/s1_model_size_dense/<tag>/forks/forks_val_items_h.npy
  --meta runs/s1_model_size_dense/<tag>/forks/forks_val_items_meta.jsonl
  --fork_items /scratch/d/dchikhi/cot_mech/s2_forks/data/forks_val_items.jsonl
  --probe runs/s1_model_size_dense/<tag>/linear_probe.pt
  --out runs/fork_rep_audit/<tag>/margin_drivers

Note: deeper "exactly what" attribution for a dense distributed direction is the SAE
stage; this rules candidates in/out and visualizes, it does not name features.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_NUM_RE = re.compile(r"-?\d+\.?\d*")
_FINAL_RE = re.compile(r"\\boxed|final answer|the answer is|=\s*\d+\s*$", re.IGNORECASE)
GREEN, RED = "#2c7fb8", "#e6550d"


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_probe(path: Path, hidden_dim: int) -> tuple[np.ndarray, float]:
    import torch
    obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    wkey = next(k for k in state if k.endswith("fc.weight") or k.endswith("weight"))
    bkey = next((k for k in state if k.endswith("fc.bias") or k.endswith("bias")), None)
    w = np.asarray(state[wkey], dtype=np.float64).reshape(-1)
    if w.shape[0] != hidden_dim:
        raise ValueError(f"probe dim {w.shape[0]} != hidden {hidden_dim}")
    b = float(np.asarray(state[bkey]).reshape(-1)[0]) if bkey else 0.0
    return w, b


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC of score for the positive class label==1 (incorrect)."""
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def residualize(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Residual of y after regressing on [1, standardized F] (keeps nothing of F)."""
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    X = np.column_stack([np.ones(len(y)), Fz])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_corr(y: np.ndarray, x: np.ndarray, others: np.ndarray | None) -> float:
    """Correlation of y and x after both are residualized on `others`."""
    if others is None or others.size == 0:
        ry, rx = y, x
    else:
        ry, rx = residualize(y, others), residualize(x, others)
    if np.std(ry) == 0 or np.std(rx) == 0:
        return 0.0
    return float(np.corrcoef(ry, rx)[0, 1])


def step_idx_of(fork_id: str) -> int:
    parts = fork_id.split("::")
    return int(parts[3]) if len(parts) > 3 and parts[3].lstrip("-").isdigit() else -1


def color_scatter(xy, c, title, out, clabel):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=8, c=c, cmap="viridis", alpha=0.7)
    fig.colorbar(sc, label=clabel)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--fork_items", required=True, type=Path)
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    (args.out / "plots").mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(parents=True, exist_ok=True)

    H = np.load(args.h).astype(np.float64)
    meta = read_jsonl(args.meta)
    items = {it["item_uid"]: it for it in read_jsonl(args.fork_items)}
    w, b = load_probe(args.probe, H.shape[1])

    # pooled steps: positives (label 0) and negatives (label 1); drop anchors.
    rows = [m for m in meta if m["role"] in ("positive", "negative")]
    idx = np.array([m["row"] for m in rows])
    label = np.array([1 if m["role"] == "negative" else 0 for m in rows])
    length = np.array([m["n_tokens"] for m in rows], dtype=np.float64)
    step_idx = np.array([step_idx_of(m["fork_id"]) for m in rows], dtype=np.float64)
    texts = [items.get(m["item_uid"], {}).get("candidate_step", "") for m in rows]
    numeric = np.array([len(_NUM_RE.findall(t)) for t in texts], dtype=np.float64)
    has_ans = np.array([1.0 if _FINAL_RE.search(t) else 0.0 for t in texts])
    Hs = H[idx]
    score = Hs @ w + b
    n = len(rows)
    print(f"[drivers] {n} steps ({int(label.sum())} incorrect / {int((1-label).sum())} correct)")

    # ---- decisive: does the score AUC survive removing length / all cheap features?
    feats = {"length": length, "step_idx": step_idx, "numeric": numeric, "has_answer": has_ans}
    raw_auc = auc(score, label)
    len_resid_auc = auc(residualize(score, length[:, None]), label)
    allf = np.column_stack([length, step_idx, numeric, has_ans])
    all_resid_auc = auc(residualize(score, allf), label)

    # ---- attribution: univariate + partial (controlling the other 3) corr with score
    attrib = []
    for k, v in feats.items():
        others = np.column_stack([feats[o] for o in feats if o != k])
        attrib.append({"feature": k,
                       "corr_with_score": round(partial_corr(score, v, None), 4),
                       "partial_corr_controlling_others": round(partial_corr(score, v, others), 4),
                       "corr_with_label": round(partial_corr(label.astype(float), v, None), 4)})
    attrib.sort(key=lambda r: -abs(r["partial_corr_controlling_others"]))

    metrics = {
        "n_steps": n, "base_rate_incorrect": float(label.mean()),
        "auc_raw": raw_auc,
        "auc_after_removing_length": len_resid_auc,
        "auc_after_removing_all_cheap_features": all_resid_auc,
        "auc_drop_from_length": round(raw_auc - len_resid_auc, 4),
        "frac_auc_lift_from_length": round((raw_auc - len_resid_auc) / max(raw_auc - 0.5, 1e-9), 3),
        "attribution": attrib,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- tables
    import csv
    with open(args.out / "tables" / "per_step.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["row", "label_incorrect", "score", "length", "step_idx", "numeric", "has_answer"])
        for i in range(n):
            wr.writerow([int(idx[i]), int(label[i]), f"{score[i]:.4f}", int(length[i]),
                         int(step_idx[i]), int(numeric[i]), int(has_ans[i])])

    # ---- plots
    # 1. score vs length, colored by label
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for lab, col, name in [(0, GREEN, "correct"), (1, RED, "incorrect")]:
        m = label == lab
        ax.scatter(length[m], score[m], s=7, c=col, alpha=0.4, label=name, linewidths=0)
    r = np.corrcoef(length, score)[0, 1]
    ax.set_xlabel("step length (tokens)"); ax.set_ylabel("probe score (higher=incorrect)")
    ax.set_title(f"score vs length  (r={r:.2f}; AUC raw {raw_auc:.3f} -> len-resid {len_resid_auc:.3f})")
    ax.legend(); fig.tight_layout()
    fig.savefig(args.out / "plots" / "score_vs_length.png", dpi=150); plt.close(fig)

    # 2. score distributions by label: raw vs length-residualized (separation persists?)
    sresid = residualize(score, length[:, None])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, s, ttl in [(axes[0], score, f"raw score (AUC {raw_auc:.3f})"),
                       (axes[1], sresid, f"length-residualized (AUC {len_resid_auc:.3f})")]:
        ax.hist(s[label == 0], bins=40, alpha=0.6, color=GREEN, label="correct")
        ax.hist(s[label == 1], bins=40, alpha=0.6, color=RED, label="incorrect")
        ax.set_title(ttl); ax.legend()
    fig.tight_layout(); fig.savefig(args.out / "plots" / "score_hist_raw_vs_lenresid.png", dpi=150); plt.close(fig)

    # 3. PCA of pooled representations colored by length / label / score
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=42).fit_transform(Hs)
    color_scatter(xy, length, "representation PCA colored by LENGTH",
                  args.out / "plots" / "rep_pca_by_length.png", "length (tokens)")
    color_scatter(xy, score, "representation PCA colored by PROBE SCORE",
                  args.out / "plots" / "rep_pca_by_score.png", "probe score")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for lab, col, name in [(0, GREEN, "correct"), (1, RED, "incorrect")]:
        m = label == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=8, c=col, alpha=0.5, label=name, linewidths=0)
    ax.set_title("representation PCA colored by LABEL"); ax.legend()
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(args.out / "plots" / "rep_pca_by_label.png", dpi=150); plt.close(fig)

    # ---- report
    band = ("length is NOT the driver" if metrics["frac_auc_lift_from_length"] < 0.25 else
            "length is a partial driver" if metrics["frac_auc_lift_from_length"] < 0.6 else
            "length is the main driver")
    L = ["# What drives the per-step correctness score\n",
         f"- steps: {n}  (incorrect base rate {label.mean():.3f})\n",
         "## Decisive: does the score survive removing length?\n",
         f"- raw pooled AUC = **{raw_auc:.3f}**",
         f"- after removing length = **{len_resid_auc:.3f}**  (drop {metrics['auc_drop_from_length']:+.3f})",
         f"- after removing length+step_idx+numeric+has_answer = {all_resid_auc:.3f}",
         f"- fraction of the AUC lift attributable to length = "
         f"**{metrics['frac_auc_lift_from_length']:.2f}** -> {band}\n",
         "## Attribution (correlation with score)\n",
         "| feature | corr w/ score | partial (controls others) | corr w/ label |",
         "|---|---|---|---|"]
    for a in attrib:
        L.append(f"| {a['feature']} | {a['corr_with_score']:+.3f} | "
                 f"{a['partial_corr_controlling_others']:+.3f} | {a['corr_with_label']:+.3f} |")
    L += ["\n## Plots",
          "- `score_vs_length.png` - if length were the driver this is a tight line; it is not.",
          "- `score_hist_raw_vs_lenresid.png` - correct/incorrect separation before vs after length removal.",
          "- `rep_pca_by_{length,label,score}.png` - length organizes raw variance, label does not (cf S15),",
          "  the probe score is the supervised axis that separates.\n",
          "## Caveat",
          "Cheap features only rule candidates in/out. Naming the non-length signal needs the SAE stage."]
    (args.out / "margin_drivers_report.md").write_text("\n".join(L))

    print(f"[drivers] AUC raw {raw_auc:.3f} -> length-removed {len_resid_auc:.3f} "
          f"({metrics['frac_auc_lift_from_length']:.2f} of lift is length) -> {band}")
    print(f"[drivers] wrote {args.out}")


if __name__ == "__main__":
    main()
