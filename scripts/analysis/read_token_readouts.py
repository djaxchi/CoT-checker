#!/usr/bin/env python3
"""Trained readouts on stored read-token states (read_token_v1, B6).

Closed-form LDA on the read state comes from scripts/analysis/backbone_geometry.py.
This fits, on the same PRM800K training subsample, the leaderboard learners that
LDA is compared against:

    linear      on `sep` (the state the read-token head reads)
    mlp:h1024   on `sep`
    attn_query  over `sep_all` (all m read-token states; only when m > 1)

under the leaderboard protocol (scripts/train_rep_learner_cell.py): AdamW + BCE,
batch 256, 30 epochs, early stopping on val loss with patience 3, lr x wd grid
{1e-3, 3e-4, 1e-4} x {0, 0.01} picked on val AUROC, seed 42, no rescaling.
Reports PRM800K test AUROC and ProcessBench known-step AUROC and calib-20 F1_PB
with backbone_geometry's scorer. Plan: docs/read_token_v1_plan.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.backbone_geometry import SUBSETS, evaluate, load_split, log  # noqa: E402
from src.harness.learners import build_learner  # noqa: E402

LR_GRID = (1e-3, 3e-4, 1e-4)
WD_GRID = (0.0, 0.01)


def fit(spec: str, Xtr, ytr, Xva, yva, dev, epochs=30, patience=3, batch=256, seed=42):
    best = None
    for lr in LR_GRID:
        for wd in WD_GRID:
            torch.manual_seed(seed)
            model = build_learner(spec, Xtr.shape[-1]).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            rng = np.random.default_rng(seed)
            best_loss, best_state, bad = float("inf"), None, 0
            for _ in range(epochs):
                model.train()
                for idx in np.array_split(rng.permutation(len(ytr)), max(1, len(ytr) // batch)):
                    xb, yb = Xtr[idx].to(dev), ytr[idx].to(dev)
                    mask = torch.ones(xb.shape[:2], device=dev) if xb.dim() == 3 else None
                    loss = F.binary_cross_entropy_with_logits(model(xb, mask), yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                vl = float(F.binary_cross_entropy_with_logits(score(model, Xva, dev), yva))
                if vl + 1e-8 < best_loss:
                    best_loss, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
            model.load_state_dict(best_state)
            va = roc_auc_score(yva.numpy(), score(model, Xva, dev).numpy())
            log(f"  {spec} lr={lr} wd={wd} val_loss={best_loss:.4f} val_auroc={va:.4f}")
            if best is None or va > best[0]:
                best = (va, lr, wd, model)
    return best


@torch.no_grad()
def score(model, X, dev, chunk=4096) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk].to(dev)
        mask = torch.ones(xb.shape[:2], device=dev) if xb.dim() == 3 else None
        out.append(model(xb, mask).float().cpu())
    return torch.cat(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", required=True, help="name:dir")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for spec in a.configs:
        cname, d = spec.split(":", 1)
        S = {k: load_split(Path(d), k) for k in ("train", "val", "test")}
        for sub in SUBSETS:
            S[sub] = load_split(Path(d), f"pb_{sub}")
        cells = [("sep", "linear"), ("sep", "mlp:h1024")]
        if "sep_all" in S["train"][0] and S["train"][0]["sep_all"].shape[1] > 1:
            cells.append(("sep_all", "attn_query"))
        for rep, learner in cells:
            def X(k):
                x = S[k][0][rep]
                return torch.from_numpy((x[:, 0] if rep == "sep" else x).astype(np.float32))
            ytr = torch.from_numpy(S["train"][0]["y"].astype(np.float32))
            yva = torch.from_numpy(S["val"][0]["y"].astype(np.float32))
            log(f"{cname} {rep} x {learner}: fitting on {len(ytr):,} rows")
            va, lr, wd, model = fit(learner, X("train"), ytr, X("val"), yva, dev)
            data = {"test": (X("test"), S["test"][0]["y"]), **{s: (X(s), S[s][0]["y"], S[s][1]) for s in SUBSETS}}
            r = evaluate(lambda x: score(model, x, dev).numpy(), data)
            rows.append(r | {"config": cname, "rep": rep, "method": learner, "val_auroc": va, "lr": lr, "wd": wd})
            log(f"{cname} {rep} x {learner}: test={r['test_auroc']:.4f} pb_calib20={r['pb_calib20']:.4f}")
            (a.out / "readouts.json").write_text(json.dumps(rows, indent=1))
    log("done")


if __name__ == "__main__":
    main()
