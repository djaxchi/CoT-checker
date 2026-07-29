#!/usr/bin/env python3
"""Attention-pool probe: learn from ALL of a step's last-layer token states.

A single learned query attends over every token of the step, producing one
pooled vector (a learned, permutation-invariant weighted sum over all tokens),
then a linear head predicts step incorrectness. The pooling weights are learned
over all tokens, so unlike mean/max pooling the model decides which states
matter; unlike a full transformer the detector stays minimal (query + linear
head), keeping the representation question in focus.

Consumes the token stores directly (no re-encoding). Same protocol as the linear
harness: train on PRM800K, select threshold on val, report in-domain PRM800K test
(AUROC + F1) and ProcessBench first-error F1 per subset; writes pb_step_scores
for offline calib-20.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_easy_probe_method import (  # noqa: E402
    auroc_numpy, evaluate_processbench, resolve_threshold_grid,
    select_threshold, step_binary_metrics,
)
from src.repstore.store import ShardedRepSplit  # noqa: E402


class AttnPoolProbe(nn.Module):
    """z = sum_i softmax(q . h_i) h_i ; logit = w . z + b."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.head = nn.Linear(d, 1)
        self.scale = d ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        att = (x @ self.q) * self.scale                 # (B, T)
        att = att.masked_fill(mask == 0, -1e9)
        w = torch.softmax(att, dim=-1).unsqueeze(-1)     # (B, T, 1)
        z = (w * x).sum(1)                               # (B, d)
        return self.head(z).squeeze(-1)                  # (B,)


def build_handles(view: ShardedRepSplit):
    """(rs, local_idx, step_start, n_tokens, label) per item in view order."""
    meta = view.meta()
    y = view.y
    out = []
    for k in range(len(view)):
        rs, li = view.item_handle(k)
        m = meta[k]
        out.append((rs, li, int(m["step_start_idx"]), int(m["n_tokens"]), int(y[k])))
    return out, meta


def read_span(rs, li, step_start, t_max):
    off = int(rs.offsets[li])
    end = int(rs.offsets[li + 1])
    span = np.asarray(rs.h[off + step_start:end], dtype=np.float32)  # step tokens only
    if span.shape[0] == 0:  # degenerate guard
        span = np.asarray(rs.h[end - 1:end], dtype=np.float32)
    if span.shape[0] > t_max:
        span = span[-t_max:]
    return span


def collate(handles, idx, t_max, device):
    spans = [read_span(*h[:3], t_max) for h in (handles[i] for i in idx)]
    T = max(s.shape[0] for s in spans)
    d = spans[0].shape[1]
    x = np.zeros((len(spans), T, d), dtype=np.float32)
    mask = np.zeros((len(spans), T), dtype=np.float32)
    for j, s in enumerate(spans):
        x[j, :s.shape[0]] = s
        mask[j, :s.shape[0]] = 1.0
    xb = torch.from_numpy(x).to(device)
    mb = torch.from_numpy(mask).to(device)
    yb = torch.tensor([handles[i][4] for i in idx], dtype=torch.float32, device=device)
    return xb, mb, yb


@torch.no_grad()
def score_split(model, handles, t_max, batch_size, device):
    model.eval()
    out = np.empty(len(handles), dtype=np.float32)
    for i in range(0, len(handles), batch_size):
        idx = list(range(i, min(i + batch_size, len(handles))))
        xb, mb, _ = collate(handles, idx, t_max, device)
        out[i:i + len(idx)] = torch.sigmoid(model(xb, mb)).cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prm_store", required=True, type=Path)
    p.add_argument("--pb_store", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--train_stem", default="probe_train_full")
    p.add_argument("--val_stem", default="val_5k")
    p.add_argument("--test_stem", default="test_2k")
    p.add_argument("--pb_subsets", nargs="+", default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--train_cap", type=int, default=150000)
    p.add_argument("--t_max", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold_grid", default="0.01")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[attn] loading stores", flush=True)
    train_h, _ = build_handles(ShardedRepSplit(args.prm_store / args.train_stem))
    val_view = ShardedRepSplit(args.prm_store / args.val_stem)
    val_h, _ = build_handles(val_view)
    test_view = ShardedRepSplit(args.prm_store / args.test_stem)
    test_h, _ = build_handles(test_view)
    d = ShardedRepSplit(args.prm_store / args.val_stem).spec.dim

    if len(train_h) > args.train_cap:
        keep = rng.choice(len(train_h), args.train_cap, replace=False)
        train_h = [train_h[i] for i in keep]
    print(f"[attn] train={len(train_h)} val={len(val_h)} test={len(test_h)} d={d}", flush=True)

    model = AttnPoolProbe(d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    val_y = np.array([h[4] for h in val_h])

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf"); bad = 0
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_h))
        t0 = time.time(); tot = 0.0; nb = 0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size].tolist()
            xb, mb, yb = collate(train_h, idx, args.t_max, device)
            loss = F.binary_cross_entropy_with_logits(model(xb, mb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        val_scores = score_split(model, val_h, args.t_max, args.batch_size, device)
        val_loss = float(F.binary_cross_entropy(
            torch.from_numpy(val_scores).clamp(1e-6, 1 - 1e-6),
            torch.from_numpy(val_y.astype(np.float32))).item())
        print(f"[attn] epoch {ep}: train_loss={tot/max(nb,1):.4f} val_loss={val_loss:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, args.out_dir / "attn_pool_probe.pt")

    grid = resolve_threshold_grid(args.threshold_grid)
    val_scores = score_split(model, val_h, args.t_max, args.batch_size, device)
    t_sel, bacc, _ = select_threshold(val_scores, val_y, grid)

    # In-domain test
    test_scores = score_split(model, test_h, args.t_max, args.batch_size, device)
    test_y = np.array([h[4] for h in test_h])
    id_metrics = {
        "method": "attn_pool", "auroc": auroc_numpy(test_y, test_scores),
        "fixed_t0.5": step_binary_metrics(test_y, test_scores, 0.5),
        "val_selected": {"threshold": t_sel, **step_binary_metrics(test_y, test_scores, t_sel)},
    }
    (args.out_dir / "in_domain_metrics.json").write_text(json.dumps(id_metrics, indent=2))
    print(f"[attn|in_domain] AUROC={id_metrics['auroc']:.4f} "
          f"val_F1={id_metrics['val_selected']['macro_f1']:.4f}", flush=True)

    # ProcessBench per subset
    summaries = []
    for sub in args.pb_subsets:
        sd = args.pb_store / sub
        if not sd.exists():
            continue
        view = ShardedRepSplit(sd)
        handles, meta = build_handles(view)
        scores = score_split(model, handles, args.t_max, args.batch_size, device)
        rows_val, m_val = evaluate_processbench(scores, meta, t_sel)
        best_ot, best_of1 = grid[0], -1.0
        for t in grid:
            _, mt = evaluate_processbench(scores, meta, t)
            if mt["F1_PB"] > best_of1:
                best_of1, best_ot = mt["F1_PB"], t
        _, m_oracle = evaluate_processbench(scores, meta, best_ot)
        with (args.out_dir / f"pb_step_scores_{sub}.jsonl").open("w") as f:
            for r in rows_val:
                f.write(json.dumps(r) + "\n")
        for tag, t, m in [("val_selected", t_sel, m_val), ("oracle", best_ot, m_oracle)]:
            summaries.append({"method": "attn_pool", "pb_name": sub, "threshold_type": tag,
                              "threshold": t, **m})
        print(f"[attn|{sub}] val(t={t_sel}) F1={m_val['F1_PB']:.4f} oracle(t={best_ot}) F1={m_oracle['F1_PB']:.4f}", flush=True)
    (args.out_dir / "eval_summary.json").write_text(json.dumps({"runs": summaries}, indent=2))
    print("[attn] done", flush=True)


if __name__ == "__main__":
    main()
