#!/usr/bin/env python3
"""Fork representation audit: geometry, probe margins, and coordinate-level
activation shifts between matched PRM800K fork siblings.

The core object is the matched triple (anchor_i, pos_i, neg_i): the same reasoning
prefix continued by a correct (positive) and an incorrect (negative) next step.
Because the prefix/problem is shared, differencing pos and neg cancels the dominant
topic/problem/length variance that buries the correctness direction in raw space
(REPORT.md S15: correctness is a ~0.01%-variance linear direction, invisible to any
variance- or neighborhood-based embedding). This script studies the triple through
several complementary lenses and validates every claim against controls.

Lenses
  1. Representation plots  - raw pos/neg/anchor; probe-axis; delta-space; anchor-relative.
  2. Coordinate-level      - per-dim paired shift, effect size, sign consistency, triggers.
  3. Common behavior       - mean displacement mu_D vs probe w vs uncentered PC1 of D.
  4. Pair comparison       - per-fork delta signatures, pair-pair cosine, typical/atypical.
  5. Controls              - fake unmatched deltas, sign-flip null, surface-confound audit.

Methodological note: dense residual-stream coordinates are NOT monosemantic. We report
"coordinate-level recurrent shifts", not "interpretable features". The clean
interpretable version repeats this audit on SAE/SSAE latents later.

Usage
  python scripts/analyze_fork_representation_audit.py \
    --fork_items /scratch/d/dchikhi/cot_mech/s2_forks/data/forks_val_items.jsonl \
    --h    runs/s1_model_size_dense/qwen2_5_7b/forks/forks_val_items_h.npy \
    --meta runs/s1_model_size_dense/qwen2_5_7b/forks/forks_val_items_meta.jsonl \
    --probe runs/s1_model_size_dense/qwen2_5_7b/probe/linear_probe.pt \
    --out  runs/fork_rep_audit/qwen2_5_7b
"""

from __future__ import annotations

import os

# Cap BLAS threads BEFORE importing numpy (shared login node; tiny eigh/gemm thrash).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# =========================================================================== IO

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_probe(path: Path | None, hidden_dim: int) -> tuple[np.ndarray | None, float]:
    """Return (w, b) for LinearProbe state_dict {fc.weight, fc.bias}. w is L2-unit."""
    if path is None:
        return None, 0.0
    import torch
    obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    wkey = next((k for k in state if k.endswith("fc.weight") or k.endswith("weight")), None)
    bkey = next((k for k in state if k.endswith("fc.bias") or k.endswith("bias")), None)
    if wkey is None:
        raise ValueError(f"No weight tensor in probe {path}; keys={list(state)[:8]}")
    w = np.asarray(state[wkey], dtype=np.float64).reshape(-1)
    if w.shape[0] != hidden_dim:
        raise ValueError(f"probe dim {w.shape[0]} != hidden_dim {hidden_dim}")
    b = float(np.asarray(state[bkey]).reshape(-1)[0]) if bkey is not None else 0.0
    return w, b


# =================================================================== pairing

def build_triples(meta: list[dict]) -> tuple[list[dict], int]:
    """Group encode-meta rows by fork_id into one matched triple per fork.

    Returns (triples, n_multi) where each triple is
      {fork_id, anchor_row|None, pos_row, neg_row, pos_uid, neg_uid,
       n_tokens_pos, n_tokens_neg}
    and n_multi counts forks that had >1 positive or >1 negative (we keep the first).
    Forks lacking either a positive or a negative are dropped.
    """
    by_fork: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for m in meta:
        by_fork[m["fork_id"]][m["role"]].append(m)
    triples, n_multi = [], 0
    for fid, roles in by_fork.items():
        pos, neg = roles.get("positive", []), roles.get("negative", [])
        if not pos or not neg:
            continue
        if len(pos) > 1 or len(neg) > 1:
            n_multi += 1
        anc = roles.get("anchor", [None])[0]
        p, n = pos[0], neg[0]
        triples.append({
            "fork_id": fid,
            "anchor_row": anc["row"] if anc else None,
            "pos_row": p["row"], "neg_row": n["row"],
            "pos_uid": p["item_uid"], "neg_uid": n["item_uid"],
            "n_tokens_pos": p.get("n_tokens"), "n_tokens_neg": n.get("n_tokens"),
        })
    return triples, n_multi


# =============================================================== core math (pure)

def probe_scores(H: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Raw probe logit w . h + b (w as stored, NOT unit-normalized)."""
    return H.astype(np.float64) @ w + b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def delta_stats(Dpos: np.ndarray, Dneg: np.ndarray) -> dict:
    """Per-dimension paired-difference statistics for delta = neg - pos.

    Returns dict of (H,) arrays: mean_shift, std, effect_size (mean/std),
    sign_consistency (P(sign(delta)==sign(mean_shift))), t (mean/(std/sqrt n)).
    """
    delta = Dneg.astype(np.float64) - Dpos.astype(np.float64)   # (n, H)
    n = delta.shape[0]
    mean_shift = delta.mean(0)
    std = delta.std(0, ddof=1)
    safe = np.where(std == 0, np.nan, std)
    effect = mean_shift / safe
    sign_target = np.sign(mean_shift)
    sign_consistency = (np.sign(delta) == sign_target[None, :]).mean(0)
    t = mean_shift / (safe / np.sqrt(n))
    return {"mean_shift": mean_shift, "std": std, "effect_size": effect,
            "sign_consistency": sign_consistency, "t": t, "delta": delta}


def common_energy(D: np.ndarray) -> float:
    """||mean(D)||^2 / mean_i ||D_i||^2  in [0,1]: fraction of energy in the shared mean."""
    mu = D.mean(0)
    denom = (D ** 2).sum(1).mean()
    return float((mu @ mu) / denom) if denom > 0 else 0.0


def sign_flip_null(D: np.ndarray, n_iter: int, rng: np.random.Generator) -> dict:
    """Null for the common displacement: randomly flip each delta's sign."""
    norms, energies = [], []
    for _ in range(n_iter):
        s = rng.choice([-1.0, 1.0], size=D.shape[0])[:, None]
        Dn = D * s
        norms.append(float(np.linalg.norm(Dn.mean(0))))
        energies.append(common_energy(Dn))
    return {"mu_norm_mean": float(np.mean(norms)), "mu_norm_p95": float(np.percentile(norms, 95)),
            "energy_mean": float(np.mean(energies)), "energy_p95": float(np.percentile(energies, 95))}


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation with no fixed points (perm[i] != i)."""
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    for k in fixed:
        swap = (k + 1) % n
        perm[k], perm[swap] = perm[swap], perm[k]
    return perm


def fake_unmatched_deltas(Dpos: np.ndarray, Dneg: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """delta_fake_i = neg_i - pos_j with j a derangement of i (j != i).

    Control: breaks the prefix match, so the shared-context cancellation that
    isolates the correctness direction in true matched deltas is destroyed.
    """
    perm = _derangement(Dpos.shape[0], rng)
    return Dneg.astype(np.float64) - Dpos[perm].astype(np.float64)


def within_role_deltas(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """delta_i = X_i - X_j with j a derangement of i, both rows the SAME role.

    Control: tests whether the true neg-pos displacement is special, or whether
    any random pair of same-role steps produces comparable energy/structure.
    """
    perm = _derangement(X.shape[0], rng)
    return X.astype(np.float64) - X[perm].astype(np.float64)


SURFACE_KEYS = ("length_diff", "char_dissim", "token_overlap", "number_diff",
                "numbers_changed", "operator_changed")


def surface_design_matrix(surf: list[dict]) -> np.ndarray:
    """(n, k+1) standardized surface features with an intercept column."""
    S = np.array([[s[k] for k in SURFACE_KEYS] for s in surf], dtype=np.float64)
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-9)
    return np.column_stack([np.ones(len(Sz)), Sz])


def surface_residualize(D: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Remove the linear component of each Delta dim explained by surface features.

    D_resid = D - X (X^+ D). X is the design matrix from surface_design_matrix.
    This is the matched-fork analogue of the REPORT S15.3 length+position
    residualization: if the correctness direction is genuinely non-surface, the
    probe ordering and mu_D / w alignment should mostly survive.
    """
    beta, *_ = np.linalg.lstsq(X, D, rcond=None)
    return D - X @ beta


def minimal_edit_mask(surf: list[dict], q: float = 0.5) -> np.ndarray:
    """Keep forks where pos/neg differ minimally: high token_overlap AND small
    |length_diff| (both past the q-quantile). Surface-controlled by construction."""
    tok = np.array([s["token_overlap"] for s in surf], dtype=np.float64)
    ld = np.abs(np.array([s["length_diff"] for s in surf], dtype=np.float64))
    return (tok >= np.quantile(tok, q)) & (ld <= np.quantile(ld, 1.0 - q))


def uncentered_pc1(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First UNCENTERED right singular vector of D, plus centered PCA scores (3 comps)."""
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    pc1_unc = Vt[0]
    Dc = D - D.mean(0)
    Uc, Sc, Vtc = np.linalg.svd(Dc, full_matrices=False)
    scores = Uc[:, :3] * Sc[:3]                       # (n, 3) centered PCA coords
    return pc1_unc, scores, Vtc[:3]


def var_along(direction: np.ndarray, X: np.ndarray) -> float:
    """Fraction of total variance of X captured along a unit direction."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    Xc = X - X.mean(0)
    proj_var = np.var(Xc @ d)
    total = Xc.var(0).sum()
    return float(proj_var / total) if total > 0 else 0.0


def trigger_rates(H_all: np.ndarray, idx_pos: np.ndarray, idx_neg: np.ndarray,
                  z_thresh: float = 2.0) -> dict:
    """Standardize each coord over ALL fork items, then per-dim trigger rate
    (|z|>z_thresh) for pos vs neg. Returns P(trig|neg)-P(trig|pos) etc."""
    mu = H_all.mean(0)
    sd = H_all.std(0)
    safe = np.where(sd == 0, np.nan, sd)
    Z = (H_all - mu) / safe
    trig = (np.abs(Z) > z_thresh)
    p_pos = trig[idx_pos].mean(0)
    p_neg = trig[idx_neg].mean(0)
    return {"p_trig_pos": p_pos, "p_trig_neg": p_neg, "diff_neg_minus_pos": p_neg - p_pos}


# ============================================================= surface features

_NUM_RE = re.compile(r"-?\d+\.?\d*")
_OP_RE = re.compile(r"[+\-*/=^%<>]|\\times|\\div|\\cdot|\\frac")
_FINAL_RE = re.compile(r"\\boxed|final answer|the answer is|=\s*\d+\s*$", re.IGNORECASE)


def surface_features(pos_text: str, neg_text: str,
                     n_tok_pos: int | None, n_tok_neg: int | None) -> dict:
    tp, tn = pos_text.split(), neg_text.split()
    sp, sn = set(tp), set(tn)
    union = sp | sn
    jacc = len(sp & sn) / len(union) if union else 1.0
    nums_p, nums_n = _NUM_RE.findall(pos_text), _NUM_RE.findall(neg_text)
    ops_p, ops_n = set(_OP_RE.findall(pos_text)), set(_OP_RE.findall(neg_text))
    return {
        "length_diff": (n_tok_neg - n_tok_pos) if (n_tok_pos and n_tok_neg) else (len(tn) - len(tp)),
        "token_overlap": jacc,
        "char_dissim": 1.0 - SequenceMatcher(None, pos_text, neg_text).ratio(),
        "number_diff": abs(len(nums_n) - len(nums_p)),
        "numbers_changed": len(set(nums_p) ^ set(nums_n)),
        "operator_changed": int(ops_p != ops_n),
        "neg_has_final": int(bool(_FINAL_RE.search(neg_text))),
        "pos_has_final": int(bool(_FINAL_RE.search(pos_text))),
    }


# ==================================================================== plotting

def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _project(X: np.ndarray, method: str, seed: int = 42) -> np.ndarray:
    """2D projection. method in {pca, tsne, umap}; falls back to pca on failure."""
    from sklearn.decomposition import PCA
    if method == "pca" or X.shape[0] < 6:
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
        except Exception:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        hp = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
        perp = max(5, min(30, X.shape[0] // 4))
        return TSNE(n_components=2, init="pca", perplexity=perp,
                    random_state=seed).fit_transform(hp)
    return PCA(n_components=2, random_state=seed).fit_transform(X)


GRAY, GREEN, RED = "#9e9e9e", "#2c7fb8", "#e6550d"


def plot_raw(Ha, Hp, Hn, method, out, max_lines=200):
    plt = _import_plt()
    X = np.concatenate([Ha, Hp, Hn], 0).astype(np.float64)
    xy = _project(X, method)
    na, npos = len(Ha), len(Hp)
    a_xy, p_xy, n_xy = xy[:na], xy[na:na + npos], xy[na + npos:]
    fig, ax = plt.subplots(figsize=(7, 6))
    for j in range(min(max_lines, len(p_xy))):
        ax.plot([p_xy[j, 0], n_xy[j, 0]], [p_xy[j, 1], n_xy[j, 1]],
                color="#bbbbbb", lw=0.3, alpha=0.5, zorder=1)
    ax.scatter(a_xy[:, 0], a_xy[:, 1], s=6, c=GRAY, alpha=0.5, label="anchor", linewidths=0)
    ax.scatter(p_xy[:, 0], p_xy[:, 1], s=8, c=GREEN, alpha=0.7, label="positive", linewidths=0)
    ax.scatter(n_xy[:, 0], n_xy[:, 1], s=8, c=RED, alpha=0.7, label="negative", linewidths=0)
    ax.set_title(f"raw representation ({method})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_probe_axis(sp, sn, out_dir):
    plt = _import_plt()
    d = sn - sp
    # histograms
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sp, bins=40, alpha=0.6, color=GREEN, label="positive")
    ax.hist(sn, bins=40, alpha=0.6, color=RED, label="negative")
    ax.set_xlabel("probe score  w . h + b   (higher = incorrect)"); ax.legend()
    ax.set_title("probe score distributions")
    fig.tight_layout(); fig.savefig(out_dir / "probe_score_hist.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d, bins=40, color="#555555")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("score_neg - score_pos"); ax.set_title(
        f"paired probe margin  (P(neg>pos)={np.mean(d > 0):.3f})")
    fig.tight_layout(); fig.savefig(out_dir / "probe_margin_hist.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo, hi = min(sp.min(), sn.min()), max(sp.max(), sn.max())
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--")
    ax.scatter(sp, sn, s=8, alpha=0.5, c=np.where(d > 0, RED, GREEN))
    ax.set_xlabel("pos score"); ax.set_ylabel("neg score")
    ax.set_title("pos vs neg probe score")
    fig.tight_layout(); fig.savefig(out_dir / "pos_vs_neg_probe_score.png", dpi=150); plt.close(fig)

    # paired line plot (subsample)
    k = min(120, len(sp))
    sel = np.linspace(0, len(sp) - 1, k).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    for j in sel:
        ax.plot([0, 1], [sp[j], sn[j]], color=(RED if sn[j] > sp[j] else GREEN),
                lw=0.5, alpha=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["positive", "negative"])
    ax.set_ylabel("probe score"); ax.set_title("paired pos -> neg")
    fig.tight_layout(); fig.savefig(out_dir / "paired_probe_scores.png", dpi=150); plt.close(fig)


def plot_delta_embedding(D, color_by: dict, method, out_dir):
    plt = _import_plt()
    xy = _project(D, method)
    for name, vals in color_by.items():
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=10, c=vals, cmap="coolwarm", alpha=0.8)
        fig.colorbar(sc, label=name)
        ax.set_title(f"delta space ({method}) colored by {name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"delta_{method}_by_{name}.png", dpi=150); plt.close(fig)


def plot_anchor_relative(Upos, Uneg, method, out_dir, max_lines=200):
    plt = _import_plt()
    X = np.concatenate([Upos, Uneg], 0)
    xy = _project(X, method)
    k = len(Upos)
    p_xy, n_xy = xy[:k], xy[k:]
    fig, ax = plt.subplots(figsize=(7, 6))
    for j in range(min(max_lines, k)):
        ax.plot([p_xy[j, 0], n_xy[j, 0]], [p_xy[j, 1], n_xy[j, 1]],
                color="#cccccc", lw=0.3, alpha=0.5)
    ax.scatter(p_xy[:, 0], p_xy[:, 1], s=8, c=GREEN, alpha=0.7, label="u_pos = pos - anchor")
    ax.scatter(n_xy[:, 0], n_xy[:, 1], s=8, c=RED, alpha=0.7, label="u_neg = neg - anchor")
    ax.set_title(f"anchor-relative displacement ({method})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / f"anchor_relative_{method}.png", dpi=150); plt.close(fig)


def plot_coordinate_level(stats, top_idx, Dpos, Dneg, trig, out_dir):
    plt = _import_plt()
    eff = stats["effect_size"]
    # effect-size histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(eff[np.isfinite(eff)], bins=80, color="#444444")
    ax.set_xlabel("per-dim effect size  mean(delta)/std(delta)")
    ax.set_title("distribution of coordinate effect sizes")
    fig.tight_layout(); fig.savefig(out_dir / "effect_size_hist.png", dpi=150); plt.close(fig)

    # top-dim shift heatmap (forks x top dims), subsample forks
    k = min(150, Dpos.shape[0])
    sel = np.linspace(0, Dpos.shape[0] - 1, k).astype(int)
    delta = (Dneg[sel] - Dpos[sel])[:, top_idx]
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.percentile(np.abs(delta), 98) or 1.0
    im = ax.imshow(delta, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, label="neg - pos")
    ax.set_xlabel(f"top {len(top_idx)} dims by |effect|"); ax.set_ylabel("forks (subsampled)")
    ax.set_title("coordinate-level shift fingerprint")
    fig.tight_layout(); fig.savefig(out_dir / "top_dim_shift_heatmap.png", dpi=150); plt.close(fig)

    # sign-consistency barplot for top dims
    sc = stats["sign_consistency"][top_idx]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(top_idx)), sc, color="#2c7fb8")
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_ylim(0, 1); ax.set_xlabel("top dims"); ax.set_ylabel("sign consistency")
    ax.set_title("per-fork sign agreement of top-shift dims")
    fig.tight_layout(); fig.savefig(out_dir / "sign_consistency_bar.png", dpi=150); plt.close(fig)

    # trigger-diff for top dims
    td = trig["diff_neg_minus_pos"][top_idx]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(top_idx)), td, color=np.where(td > 0, RED, GREEN))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("top dims"); ax.set_ylabel("P(trig|neg) - P(trig|pos)")
    ax.set_title("trigger-rate difference (|z|>2) on top-shift dims")
    fig.tight_layout(); fig.savefig(out_dir / "top_dim_trigger_diff.png", dpi=150); plt.close(fig)

    # paired slope (violin-ish) for the single strongest dim
    j = top_idx[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in sel:
        ax.plot([0, 1], [Dpos[i, j], Dneg[i, j]],
                color=(RED if Dneg[i, j] > Dpos[i, j] else GREEN), lw=0.4, alpha=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pos", "neg"])
    ax.set_title(f"dim {j}: per-fork activation (top effect)")
    fig.tight_layout(); fig.savefig(out_dir / "top_dim_paired_slope.png", dpi=150); plt.close(fig)


# ======================================================================== main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fork_items", type=Path, required=True)
    ap.add_argument("--h", type=Path, required=True, help="forks_*_h.npy (n_items, H)")
    ap.add_argument("--meta", type=Path, required=True, help="forks_*_meta.jsonl")
    ap.add_argument("--probe", type=Path, default=None, help="LinearProbe state_dict .pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_forks", type=int, default=None)
    ap.add_argument("--topk_dims", type=int, default=40)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--n_select", type=int, default=8, help="forks to print qualitatively")
    ap.add_argument("--embed", choices=["pca", "tsne", "umap"], default="pca")
    ap.add_argument("--no_plots", action="store_true",
                    help="Skip all matplotlib/sklearn plotting; emit metrics.json, "
                         "summary.md, and CSV tables only (pure numpy + torch). Use on "
                         "login nodes / offline envs without sklearn+matplotlib.")
    ap.add_argument("--null_iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # ---- load
    H = np.load(args.h).astype(np.float32)
    meta = read_jsonl(args.meta)
    items = {it["item_uid"]: it for it in read_jsonl(args.fork_items)}
    w, b = load_probe(args.probe, H.shape[1])
    print(f"[audit] H={H.shape}  meta={len(meta)}  items={len(items)}  probe={'yes' if w is not None else 'no'}")

    triples, n_multi = build_triples(meta)
    if args.max_forks and len(triples) > args.max_forks:
        sel = rng.choice(len(triples), size=args.max_forks, replace=False)
        triples = [triples[i] for i in sorted(sel)]
    n = len(triples)
    if n < 5:
        sys.exit(f"[audit] only {n} matched forks; need >=5.")
    print(f"[audit] {n} matched triples ({n_multi} forks had multiple pos/neg; kept first)")

    pos_rows = np.array([t["pos_row"] for t in triples])
    neg_rows = np.array([t["neg_row"] for t in triples])
    has_anc = all(t["anchor_row"] is not None for t in triples)
    Hp, Hn = H[pos_rows].astype(np.float64), H[neg_rows].astype(np.float64)
    Ha = H[np.array([t["anchor_row"] for t in triples])].astype(np.float64) if has_anc else None
    D = Hn - Hp                                              # (n, H) matched deltas

    metrics: dict = {"n_forks": n, "n_multi_pos_neg": n_multi, "hidden_dim": int(H.shape[1]),
                     "has_anchor": has_anc}

    # ---- 1. probe axis
    if w is not None:
        sp, sn = probe_scores(Hp, w, b), probe_scores(Hn, w, b)
        sa = probe_scores(Ha, w, b) if has_anc else None
        margin = sn - sp
        # Convention check: positives are known-correct (label 0), negatives
        # known-incorrect (label 1). If the trained probe puts HIGHER score on
        # incorrect, mean(sn) > mean(sp) and margin>0 means "incorrect scored
        # higher" (the script's assumption). If reversed, the probe was trained
        # higher=correct and every margin/plot must be read inverted.
        convention = "higher_score=incorrect" if sn.mean() > sp.mean() else "higher_score=correct"
        metrics["probe"] = {
            "convention": convention,
            "mean_score_pos_correct": float(sp.mean()),
            "mean_score_neg_incorrect": float(sn.mean()),
            "P_neg_gt_pos": float(np.mean(margin > 0)),
            "mean_margin": float(margin.mean()),
            "median_margin": float(np.median(margin)),
            "auc_pooled": _auc(sn, sp),   # neg = label-1 class, pos = label-0
        }
        if not args.no_plots:
            plot_probe_axis(sp, sn, out / "plots")
        flag = "" if convention == "higher_score=incorrect" else "  [!] PROBE INVERTED: read margins as pos>neg"
        print(f"[audit] probe convention: {convention}{flag}")
        print(f"[audit] P(score_neg>score_pos)={metrics['probe']['P_neg_gt_pos']:.3f}  "
              f"pooled AUC={metrics['probe']['auc_pooled']:.3f}")

    # ---- 2. coordinate-level
    stats = delta_stats(Hp, Hn)
    order = np.argsort(-np.abs(np.nan_to_num(stats["effect_size"])))
    top_idx = order[:args.topk_dims]
    all_idx = np.concatenate([pos_rows, neg_rows])
    idx_pos_in_all = np.arange(n)
    idx_neg_in_all = np.arange(n, 2 * n)
    H_all = H[all_idx].astype(np.float64)
    trig = trigger_rates(H_all, idx_pos_in_all, idx_neg_in_all)
    metrics["coordinate"] = {
        "median_abs_effect": float(np.nanmedian(np.abs(stats["effect_size"]))),
        "n_dims_effect_gt_0.2": int(np.sum(np.abs(np.nan_to_num(stats["effect_size"])) > 0.2)),
        "top_dim": int(top_idx[0]),
        "top_dim_effect": float(stats["effect_size"][top_idx[0]]),
        "top_dim_sign_consistency": float(stats["sign_consistency"][top_idx[0]]),
    }
    _write_dim_tables(stats, trig, out / "tables")
    if not args.no_plots:
        plot_coordinate_level(stats, top_idx, Hp, Hn, trig, out / "plots")

    # ---- 3. common behavior: mu_D vs w vs uncentered PC1
    mu_D = D.mean(0)
    pc1_unc, pca_scores, _ = uncentered_pc1(D)
    ce = common_energy(D)
    null = sign_flip_null(D, args.null_iter, rng)
    metrics["common"] = {
        "common_energy": ce,
        "null_energy_mean": null["energy_mean"], "null_energy_p95": null["energy_p95"],
        "mu_norm": float(np.linalg.norm(mu_D)), "null_mu_norm_p95": null["mu_norm_p95"],
        "cos_muD_pc1unc": cosine(mu_D, pc1_unc),
    }
    if w is not None:
        metrics["common"].update({
            "cos_muD_w": cosine(mu_D, w),
            "cos_pc1unc_w": cosine(pc1_unc, w),
            "w_var_frac_raw_pooled": var_along(w, np.concatenate([Hp, Hn], 0)),
            "w_var_frac_delta": var_along(w, D),
        })
    print(f"[audit] common_energy={ce:.4f} (null p95={null['energy_p95']:.4f})  "
          f"cos(muD,w)={metrics['common'].get('cos_muD_w', float('nan')):.3f}")

    # ---- 4. surface features + pair signatures
    surf = []
    for t in triples:
        pi, ni = items.get(t["pos_uid"]), items.get(t["neg_uid"])
        ptxt = pi["candidate_step"] if pi else ""
        ntxt = ni["candidate_step"] if ni else ""
        surf.append(surface_features(ptxt, ntxt, t["n_tokens_pos"], t["n_tokens_neg"]))
    delta_norm = np.linalg.norm(D, axis=1)
    cos_mu = np.array([cosine(D[i], mu_D) for i in range(n)])
    wmargin = (D @ w) if w is not None else np.full(n, np.nan)

    sig_rows = []
    for i, t in enumerate(triples):
        row = {"fork_id": t["fork_id"], "delta_norm": float(delta_norm[i]),
               "probe_margin": float(wmargin[i]), "cos_delta_mu": float(cos_mu[i]),
               "pc1": float(pca_scores[i, 0]), "pc2": float(pca_scores[i, 1]),
               "pc3": float(pca_scores[i, 2]),
               "top_shift_dims": ";".join(map(str, np.argsort(-np.abs(D[i]))[:5]))}
        row.update(surf[i])
        sig_rows.append(row)
    _write_csv(out / "tables" / "pair_signatures.csv", sig_rows)

    # pair-pair cosine + neighbors
    Dn_unit = D / (delta_norm[:, None] + 1e-12)
    sim = Dn_unit @ Dn_unit.T
    np.fill_diagonal(sim, -np.inf)
    nn_rows = []
    for i, t in enumerate(triples):
        nbr = np.argsort(-sim[i])[:args.knn]
        nn_rows.append({"fork_id": t["fork_id"],
                        "neighbors": ";".join(triples[j]["fork_id"] for j in nbr),
                        "neighbor_cos": ";".join(f"{sim[i, j]:.3f}" for j in nbr)})
    _write_csv(out / "tables" / "nearest_neighbors.csv", nn_rows)

    # selection tables
    _rank_table(out / "tables" / "most_typical_pairs.csv", triples, cos_mu, surf, wmargin, delta_norm, top=True)
    _rank_table(out / "tables" / "most_atypical_pairs.csv", triples, cos_mu, surf, wmargin, delta_norm, top=False)
    if w is not None:
        _rank_table(out / "tables" / "largest_probe_margin_pairs.csv", triples, wmargin, surf, wmargin, delta_norm, top=True)
        _rank_table(out / "tables" / "smallest_probe_margin_pairs.csv", triples, wmargin, surf, wmargin, delta_norm, top=False)
    _rank_table(out / "tables" / "largest_delta_norm_pairs.csv", triples, delta_norm, surf, wmargin, delta_norm, top=True)

    # ---- 5. controls
    Dfake = fake_unmatched_deltas(Hp, Hn, rng)
    ce_fake = common_energy(Dfake)
    # within-role random pairs: any two negatives / any two positives.
    D_negneg = within_role_deltas(Hn, rng)
    D_pospos = within_role_deltas(Hp, rng)
    # sign-flip null as a z-score on ||mu||.
    mu_norm = float(np.linalg.norm(mu_D))
    flip_norms = []
    for _ in range(args.null_iter):
        s = rng.choice([-1.0, 1.0], size=n)[:, None]
        flip_norms.append(float(np.linalg.norm((D * s).mean(0))))
    flip_norms = np.array(flip_norms)
    mu_z = float((mu_norm - flip_norms.mean()) / (flip_norms.std() + 1e-12))
    metrics["controls"] = {
        "fake_common_energy": ce_fake,
        "fake_mu_norm": float(np.linalg.norm(Dfake.mean(0))),
        "true_over_fake_energy": float(ce / ce_fake) if ce_fake > 0 else float("inf"),
        "within_neg_common_energy": common_energy(D_negneg),
        "within_pos_common_energy": common_energy(D_pospos),
        "true_over_within_neg_energy": float(ce / common_energy(D_negneg)) if common_energy(D_negneg) > 0 else float("inf"),
        "signflip_mu_z": mu_z,
        # label-swap (Delta_swap = pos - neg) must invert exactly; a non -1 cosine
        # would mean a pairing/sign bug upstream.
        "labelswap_cos_muD": cosine(-mu_D, mu_D),
    }
    if w is not None:
        metrics["controls"]["fake_cos_muD_w"] = cosine(Dfake.mean(0), w)
        metrics["controls"]["labelswap_margin_flips"] = bool(
            np.all(np.sign((-D) @ w) == -np.sign(D @ w + 1e-12)))

    # ---- 5b. surface residualization + minimal-edit subset (zero-cost, reuses D)
    # Does the matched-Delta signal survive removing the inter-sibling length/lexical
    # difference? If correctness is non-surface, mu_D/w alignment and the paired probe
    # ordering should mostly hold (cf. REPORT S15.3, where 87% of the lift survived).
    Xsurf = surface_design_matrix(surf)
    D_res = surface_residualize(D, Xsurf)
    mu_res = D_res.mean(0)
    res = {"common_energy": common_energy(D_res),
           "cos_muRes_w": cosine(mu_res, w) if w is not None else None,
           "true_over_fake_energy": (common_energy(D_res) /
               common_energy(fake_unmatched_deltas(Hp, Hn, rng)) if w is not None else None)}
    if w is not None:
        res["P_neg_gt_pos_resid"] = float(np.mean((D_res @ w) > 0))
        res["margin_retained_vs_raw"] = float(np.mean((D_res @ w) > 0) - np.mean((D @ w) > 0))
    metrics["residualized_surface"] = res

    me_mask = minimal_edit_mask(surf, q=0.5)
    n_me = int(me_mask.sum())
    me = {"n_kept": n_me, "frac_kept": float(me_mask.mean())}
    if n_me >= 20:
        Dme = D[me_mask]
        me["common_energy"] = common_energy(Dme)
        me["cos_muD_w"] = cosine(Dme.mean(0), w) if w is not None else None
        if w is not None:
            me["P_neg_gt_pos"] = float(np.mean((Dme @ w) > 0))
    metrics["minimal_edit"] = me
    print(f"[audit] residualized: cos(muRes,w)={res.get('cos_muRes_w') or float('nan'):.3f} "
          f"P(neg>pos)_resid={res.get('P_neg_gt_pos_resid', float('nan')):.3f}  |  "
          f"minimal-edit n={n_me} cos(muD,w)={me.get('cos_muD_w') or float('nan'):.3f}")

    # surface confound correlations
    conf = _surface_confounds(surf, wmargin, delta_norm, pca_scores[:, 0], cos_mu, w is not None)
    _write_csv(out / "tables" / "surface_confound_correlations.csv", conf)
    metrics["surface_confounds"] = {r["quantity"] + "_vs_" + r["feature"]: r["pearson_r"]
                                    for r in conf if r["feature"] in ("length_diff", "token_overlap")}
    main_q = ("probe_margin", "pc1_delta") if w is not None else ("pc1_delta",)
    worst = max((r for r in conf if r["quantity"] in main_q), key=lambda r: abs(r["pearson_r"]),
                default={"feature": "n/a", "pearson_r": 0.0})
    metrics["surface_confounds_max"] = {"feature": worst["feature"],
                                        "abs_r": abs(worst["pearson_r"])}

    # ---- plots needing the embedding
    if not args.no_plots:
        if has_anc:
            plot_raw(Ha, Hp, Hn, args.embed, out / "plots" / f"raw_{args.embed}_pos_neg_anchor.png")
            plot_anchor_relative(Hp - Ha, Hn - Ha, args.embed, out / "plots")
        color_by = {"probe_margin": wmargin, "delta_norm": delta_norm, "cos_delta_mu": cos_mu,
                    "length_diff": np.array([s["length_diff"] for s in surf], float)}
        plot_delta_embedding(D, color_by, args.embed, out / "plots")

    # ---- write outputs
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_summary(out, args, metrics, triples, sig_rows, sim, items, w is not None)
    print(f"[audit] wrote {out}/metrics.json, summary.md, {len(list((out/'plots').glob('*.png')))} plots, "
          f"{len(list((out/'tables').glob('*.csv')))} tables")


# ----------------------------------------------------------------- small utils

def _auc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    """AUC for separating (label1=neg-step, higher score) from positives."""
    y = np.r_[np.ones(len(neg_scores)), np.zeros(len(pos_scores))]
    s = np.r_[neg_scores, pos_scores]
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        path.write_text(""); return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)


def _write_dim_tables(stats, trig, tdir: Path) -> None:
    eff = np.nan_to_num(stats["effect_size"])
    order = np.argsort(-np.abs(eff))
    rows = [{"dim": int(j), "mean_shift": float(stats["mean_shift"][j]),
             "effect_size": float(stats["effect_size"][j]),
             "sign_consistency": float(stats["sign_consistency"][j]),
             "t": float(stats["t"][j]),
             "p_trig_neg": float(trig["p_trig_neg"][j]),
             "p_trig_pos": float(trig["p_trig_pos"][j]),
             "trig_diff": float(trig["diff_neg_minus_pos"][j])} for j in order[:200]]
    _write_csv(tdir / "top_shift_dims.csv", rows)
    torder = np.argsort(-np.abs(trig["diff_neg_minus_pos"]))
    trows = [{"dim": int(j), "trig_diff": float(trig["diff_neg_minus_pos"][j]),
              "p_trig_neg": float(trig["p_trig_neg"][j]),
              "p_trig_pos": float(trig["p_trig_pos"][j]),
              "effect_size": float(stats["effect_size"][j])} for j in torder[:200]]
    _write_csv(tdir / "top_trigger_dims.csv", trows)


def _rank_table(path, triples, key, surf, wmargin, dnorm, top: bool) -> None:
    order = np.argsort(-key if top else key)
    rows = []
    for i in order[:30]:
        rows.append({"fork_id": triples[i]["fork_id"], "key": float(key[i]),
                     "probe_margin": float(wmargin[i]), "delta_norm": float(dnorm[i]),
                     "length_diff": surf[i]["length_diff"],
                     "token_overlap": round(surf[i]["token_overlap"], 3),
                     "numbers_changed": surf[i]["numbers_changed"]})
    _write_csv(path, rows)


def _surface_confounds(surf, wmargin, dnorm, pc1, cos_mu, has_probe) -> list[dict]:
    feats = {k: np.array([s[k] for s in surf], float) for k in
             ("length_diff", "token_overlap", "char_dissim", "number_diff",
              "numbers_changed", "operator_changed")}
    quants = {"delta_norm": dnorm, "pc1_delta": pc1, "cos_delta_mu": cos_mu}
    if has_probe:
        quants["probe_margin"] = wmargin
    rows = []
    for qn, qv in quants.items():
        for fn, fv in feats.items():
            if np.std(qv) == 0 or np.std(fv) == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(qv, fv)[0, 1])
            rows.append({"quantity": qn, "feature": fn, "pearson_r": round(r, 4)})
    return rows


def _write_summary(out, args, metrics, triples, sig_rows, sim, items, has_probe) -> None:
    L = []
    L.append("# Fork representation audit\n")
    L.append(f"- model run: `{args.h}`")
    L.append(f"- matched forks: **{metrics['n_forks']}** (hidden_dim {metrics['hidden_dim']}, "
             f"anchor={'yes' if metrics['has_anchor'] else 'no'})\n")

    if has_probe:
        conv = metrics["probe"]["convention"]
        L.append(f"## Probe convention: **{conv}**\n")
        L.append(f"- mean score, correct (pos) siblings: {metrics['probe']['mean_score_pos_correct']:.3f}; "
                 f"incorrect (neg): {metrics['probe']['mean_score_neg_incorrect']:.3f}")
        if conv == "higher_score=incorrect":
            L.append("- Margins read as written: `score_neg - score_pos > 0` means the incorrect "
                     "sibling scored higher (expected).\n")
        else:
            L.append("- **[!] PROBE INVERTED** (trained higher=correct). Read every margin/plot "
                     "with the sign flipped: the relevant quantity is `score_pos - score_neg`.\n")

    L.append("## Interpretation chain\n")
    c = metrics["common"]
    chain = []
    if has_probe:
        p = metrics["probe"]
        chain.append(f"1. **Probe axis:** pos/neg consistently ordered - "
                     f"P(neg>pos)={p['P_neg_gt_pos']:.3f}, pooled AUC={p['auc_pooled']:.3f}.")
    chain.append(f"2. **Matched delta:** common displacement energy={c['common_energy']:.4f} "
                 f"vs sign-flip null p95={c['null_energy_p95']:.4f} "
                 f"({'SIGNIFICANT' if c['common_energy'] > c['null_energy_p95'] else 'not significant'}).")
    if has_probe:
        chain.append(f"3. **mu_D vs probe w:** cos(mu_D, w)={c['cos_muD_w']:.3f}; "
                     f"uncentered PC1 vs w cos={c['cos_pc1unc_w']:.3f}.")
        chain.append(f"4. **Buried signal:** var fraction along w in raw pooled space="
                     f"{c['w_var_frac_raw_pooled']:.5f} vs in matched-delta space="
                     f"{c['w_var_frac_delta']:.5f} "
                     f"({c['w_var_frac_delta'] / max(c['w_var_frac_raw_pooled'], 1e-9):.1f}x).")
    cc = metrics["controls"]
    chain.append(f"5. **Controls:** true/fake-unmatched energy ratio={cc['true_over_fake_energy']:.2f}; "
                 f"true/within-neg ratio={cc['true_over_within_neg_energy']:.2f}; "
                 f"sign-flip ||mu|| z={cc['signflip_mu_z']:.1f}; "
                 f"label-swap cos(-mu,mu)={cc['labelswap_cos_muD']:.2f} "
                 f"(must be -1.00).")
    co = metrics["coordinate"]
    chain.append(f"6. **Coordinate shifts:** top dim {co['top_dim']} effect={co['top_dim_effect']:.3f}, "
                 f"sign-consistency={co['top_dim_sign_consistency']:.3f}; "
                 f"{co['n_dims_effect_gt_0.2']} dims with |effect|>0.2.")
    sc = metrics.get("surface_confounds", {})
    if sc:
        worst = max(sc.items(), key=lambda kv: abs(kv[1]))
        chain.append(f"7. **Surface confound:** strongest |Pearson r| among length/overlap = "
                     f"{worst[0]} ({worst[1]:+.3f}).")
    L.extend(chain)
    L.append("\n> Claim is supported only if the chain holds end to end: pos/neg unordered in raw "
             "variance, ordered along w, with a significant matched displacement that aligns with w, "
             "is dominant in delta-space but invisible in raw PCA, beats the unmatched control, and is "
             "not explained by surface features.\n")

    # ---- automated green-light table for the L20 re-encode decision
    L.append("## Decision criteria (L20 re-encode green light)\n")
    rows = []
    if has_probe:
        p = metrics["probe"]
        rows.append(("P(score_neg>score_pos) > 0.65", p["P_neg_gt_pos"], p["P_neg_gt_pos"] > 0.65))
        rows.append(("cos(mu_D, w) > 0.15", c["cos_muD_w"], c["cos_muD_w"] > 0.15))
        ratio = c["w_var_frac_delta"] / max(c["w_var_frac_raw_pooled"], 1e-9)
        rows.append(("w_var_frac_delta / raw > 5x", ratio, ratio > 5))
    rows.append(("true/fake common energy > 2x", cc["true_over_fake_energy"], cc["true_over_fake_energy"] > 2))
    rows.append(("sign-flip ||mu|| z > 3", cc["signflip_mu_z"], cc["signflip_mu_z"] > 3))
    scm = metrics.get("surface_confounds_max", {"feature": "n/a", "abs_r": 0.0})
    rows.append((f"no surface confound dominates (|r|<0.5; worst={scm['feature']})",
                 scm["abs_r"], scm["abs_r"] < 0.5))
    L.append("| test | value | pass |")
    L.append("|---|---|---|")
    n_pass = 0
    for name, val, ok in rows:
        n_pass += int(ok)
        L.append(f"| {name} | {val:.3f} | {'PASS' if ok else 'fail'} |")
    verdict = "GREEN LIGHT" if n_pass >= max(3, len(rows) - 1) else "WEAK / HOLD"
    L.append(f"\n**{n_pass}/{len(rows)} criteria pass -> {verdict}** for the L20 multilayer re-encode. "
             "Cross-size margin correlation is evaluated separately by the aggregator.\n")

    rs = metrics.get("residualized_surface")
    me = metrics.get("minimal_edit")
    if rs:
        L.append("## Surface control: is the matched signal real or lexical?\n")
        L.append("Removing the inter-sibling length/lexical difference from every Delta "
                 "(REPORT S15.3 analogue). If correctness is non-surface, the paired ordering "
                 "and mu/w alignment mostly survive; if it collapses, the matched signal was "
                 "mostly surface.\n")
        if has_probe:
            L.append(f"- **P(neg>pos) after residualization: {rs['P_neg_gt_pos_resid']:.3f}** "
                     f"(raw {metrics['probe']['P_neg_gt_pos']:.3f}, "
                     f"delta {rs['margin_retained_vs_raw']:+.3f}).")
            L.append(f"- cos(mu_resid, w) = {rs['cos_muRes_w']:.3f} (raw {c.get('cos_muD_w', float('nan')):.3f}).")
        L.append(f"- residualized common energy = {rs['common_energy']:.4f}.")
        if me and "P_neg_gt_pos" in me:
            L.append(f"- **minimal-edit subset** (n={me['n_kept']}, high token-overlap + small "
                     f"|length_diff|): P(neg>pos)={me['P_neg_gt_pos']:.3f}, "
                     f"cos(mu_D,w)={me['cos_muD_w']:.3f}.")
        L.append("")

    L.append("## Selected forks (qualitative)\n")
    L.append("Most-typical (high cos(delta, mu_D)) and most-atypical (low) forks; "
             "inspect whether similar deltas map to similar failure modes.\n")
    cos_mu = np.array([r["cos_delta_mu"] for r in sig_rows])
    chosen = list(np.argsort(-cos_mu)[:args.n_select // 2]) + list(np.argsort(cos_mu)[:args.n_select // 2])
    for i in chosen:
        t = triples[i]; r = sig_rows[i]
        pi, ni = items.get(t["pos_uid"], {}), items.get(t["neg_uid"], {})
        nbr = np.argsort(-sim[i])[:3]
        L.append(f"### `{t['fork_id']}`  (cos_mu={r['cos_delta_mu']:.3f}, "
                 f"margin={r['probe_margin']:.3f}, ||delta||={r['delta_norm']:.2f})")
        L.append(f"- problem: {str(pi.get('problem',''))[:300]}")
        L.append(f"- prefix: ...{str(pi.get('prefix',''))[-300:]}")
        L.append(f"- **positive:** {str(pi.get('candidate_step',''))[:300]}")
        L.append(f"- **negative:** {str(ni.get('candidate_step',''))[:300]}")
        L.append(f"- top changed dims: {r['top_shift_dims']}  | "
                 f"length_diff={r['length_diff']} numbers_changed={r['numbers_changed']}")
        L.append(f"- nearest delta-neighbors: {', '.join(triples[j]['fork_id'] for j in nbr)}\n")

    L.append("## Caveat\n")
    L.append("Dense residual-stream coordinates are not monosemantic. The dim-level results are "
             "**recurrent coordinate-level shifts**, not interpretable features. Repeat on SAE/SSAE "
             "latents for semantic feature discovery.\n")
    (out / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
