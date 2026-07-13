"""parametric_retrieval_access_v1 Experiment B: localize successful retrieval.

Three analyses over stage-3 states, mixed-outcome groups only:

1. Accessibility probe: torch logistic regression success-vs-fail per
   (hs_idx x position) cell, grouped 5-fold CV by fact (a probe never sees
   its eval facts). Compared against a surface-confound-only probe
   (template/seed/direction/popularity/category/prompt-length) and a
   residualized-states probe at the best cell.
2. Low-rank geometry at the best cell: singular spectrum of TRAIN paired
   diffs + grouped-CV probe restricted to top-k singular directions,
   k in {1,2,4,8,16,32,64} (distributed vs low-rank).
3. Relation specificity: cosine similarity between per category x
   answer-type mean diffs and the global mean diff.

Outputs (in --out_dir/expB): probe_auc.csv, svd_spectrum.csv,
lowrank_probe.csv, relation_cosine.csv, manifest.json

  python scripts/parametric_retrieval/prga_expB_probe.py
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

from scripts.parametric_retrieval import prga_common as C  # noqa: E402
from src.analysis.parametric_retrieval_causal import (  # noqa: E402
    confound_features,
    paired_diffs,
    residualize,
)

PROBE_HS = [0, 4, 8, 12, 16, 20, 22, 24, 25, 26, 27, 28]
PROBE_POS = ["entity_last", "question_last", "answer_prefix",
             "final_prompt_token"]
TOPK = [1, 2, 4, 8, 16, 32, 64]


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def grouped_cv_auc(X: np.ndarray, y: np.ndarray, facts: np.ndarray,
                   n_folds: int = 5, seed: int = 42,
                   epochs: int = 60, lr: float = 0.05,
                   weight_decay: float = 1e-3) -> float:
    """Torch logistic regression with fact-grouped folds; mean eval AUC."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    uf = np.array(sorted(set(facts)))
    fold_of = dict(zip(uf, rng.integers(0, n_folds, len(uf))))
    folds = np.array([fold_of[f] for f in facts])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Xn = (X - mu) / sd
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y.astype(np.float32), device=device)
    aucs = []
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or len(set(y[te])) < 2:
            continue
        w = torch.zeros(X.shape[1], device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=weight_decay)
        tr_idx = torch.tensor(np.flatnonzero(tr), device=device)
        for _ in range(epochs):
            z = Xt[tr_idx] @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z, yt[tr_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            s = (Xt[torch.tensor(np.flatnonzero(te), device=device)] @ w
                 + b).cpu().numpy()
        aucs.append(auc(s, y[te]))
    return float(np.nanmean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    exp_dir = args.out_dir / "expB"
    exp_dir.mkdir(exist_ok=True)
    store = C.HSStore(args.out_dir)
    meta = store.meta
    groups = pd.read_parquet(args.out_dir / "group_outcomes.parquet")
    groups["fact_id"] = groups.fact_id.astype(str)
    mixed_keys = set(zip(groups[groups.is_mixed].fact_id,
                         groups[groups.is_mixed].direction))
    inst = pd.read_parquet(args.out_dir / "metadata.parquet")
    inst["fact_id"] = inst.fact_id.astype(str)
    extra = inst.set_index("instance_id")[["gbc_bin", "category",
                                           "subject_type", "object_type"]]

    # ---- 1. probe heatmap ---------------------------------------------------
    probe_rows = []
    best = (None, -1.0)
    for position in PROBE_POS:
        mask = (meta.position_name == position).to_numpy()
        rows = meta[mask].reset_index(drop=True).join(extra,
                                                      on="instance_id")
        sel = np.array([(f, d) in mixed_keys for f, d
                        in zip(rows.fact_id, rows.direction)])
        sel &= rows.is_correct.notna().to_numpy()
        rk = rows[sel].reset_index(drop=True)
        y = rk.is_correct.to_numpy(bool)
        facts = (rk.fact_id + "::" + rk.direction).to_numpy()
        F = confound_features(rk)
        if position == PROBE_POS[0]:
            conf_auc = grouped_cv_auc(F, y, facts, seed=args.seed)
            probe_rows.append({"hs_idx": -1, "position": "confounds_only",
                               "auc": conf_auc, "n": len(y)})
            print(f"[expB] confound-only probe AUC {conf_auc:.3f}",
                  flush=True)
        for hs_idx in PROBE_HS:
            H = store.layer(hs_idx)[mask][sel].astype(np.float32)
            a = grouped_cv_auc(H, y, facts, seed=args.seed)
            probe_rows.append({"hs_idx": hs_idx, "position": position,
                               "auc": a, "n": len(y)})
            if a > best[1]:
                best = ((hs_idx, position), a)
            print(f"[expB] probe hs{hs_idx:02d}/{position}: AUC {a:.3f}",
                  flush=True)
    pd.DataFrame(probe_rows).to_csv(exp_dir / "probe_auc.csv", index=False)

    # ---- 2/3 at the best cell ----------------------------------------------
    (hs_b, pos_b) = best[0]
    mask = (meta.position_name == pos_b).to_numpy()
    rows = meta[mask].reset_index(drop=True).join(extra, on="instance_id")
    sel = np.array([(f, d) in mixed_keys for f, d
                    in zip(rows.fact_id, rows.direction)])
    sel &= rows.is_correct.notna().to_numpy()
    rk = rows[sel].reset_index(drop=True)
    H = store.layer(hs_b)[mask][sel].astype(np.float32)
    y = rk.is_correct.to_numpy(bool)
    facts = (rk.fact_id + "::" + rk.direction).to_numpy()

    F = confound_features(rk)
    resid_auc = grouped_cv_auc(residualize(H, F), y, facts, seed=args.seed)
    print(f"[expB] best cell hs{hs_b}/{pos_b} AUC {best[1]:.3f}; "
          f"residualized {resid_auc:.3f}", flush=True)

    tr_mask = rk.split.to_numpy() == "train"
    diffs, _ = paired_diffs(H[tr_mask],
                            rk[tr_mask].reset_index(drop=True))
    _, S, Vt = np.linalg.svd(diffs, full_matrices=False)
    spec = pd.DataFrame({"rank": np.arange(1, min(65, len(S) + 1)),
                         "singular_value": S[:64],
                         "energy_frac": (S[:64] ** 2) / (S ** 2).sum()})
    spec.to_csv(exp_dir / "svd_spectrum.csv", index=False)

    lowrank = []
    for k in TOPK:
        P = Vt[:k]
        a = grouped_cv_auc(H @ P.T, y, facts, seed=args.seed)
        lowrank.append({"k": k, "auc": a})
        print(f"[expB] top-{k} subspace probe AUC {a:.3f}", flush=True)
    pd.DataFrame(lowrank).to_csv(exp_dir / "lowrank_probe.csv", index=False)

    rk = rk.assign(atype=np.where(rk.direction == "direct", rk.object_type,
                                  rk.subject_type))
    gmean = diffs.mean(axis=0)
    gmean /= np.linalg.norm(gmean)
    rel = []
    tr_rk = rk[tr_mask].reset_index(drop=True)
    Htr = H[tr_mask]
    for (cat, at), g in tr_rk.groupby(["category", "atype"]):
        dsub, _ = paired_diffs(Htr[g.index], g.reset_index(drop=True))
        if len(dsub) < 8:
            continue
        m = dsub.mean(axis=0)
        rel.append({"category": cat, "answer_type": at,
                    "n_groups": len(dsub),
                    "cos_to_global": float(m @ gmean / np.linalg.norm(m))})
    pd.DataFrame(rel).to_csv(exp_dir / "relation_cosine.csv", index=False)

    (exp_dir / "manifest.json").write_text(json.dumps(
        {"best_cell": {"hs_idx": hs_b, "position": pos_b, "auc": best[1]},
         "confound_only_auc": conf_auc, "residualized_auc": resid_auc,
         "probe_hs": PROBE_HS, "probe_positions": PROBE_POS,
         "seed": args.seed}, indent=2))
    print("[expB] done", flush=True)


if __name__ == "__main__":
    main()
