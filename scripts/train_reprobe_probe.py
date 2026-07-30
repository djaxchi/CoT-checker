#!/usr/bin/env python3
"""ReProbe: a small transformer probe over ALL of a step's last-layer tokens.

Reproduction of ReProbe (Ni et al., 2025; arXiv:2511.06209) restricted to the
representation our token store already holds: the last-layer states of every
token in a step. The learner is the paper's small (<10M-param) transformer:
project each token down, add a learned position, run a few transformer-encoder
layers, mean-pool over the step, then a linear head predicts step incorrectness.

This is the maximal-detector end of the representation x learner grid: same
`token_store` representation as attn_pool, but a full transformer instead of a
single learned query, so it isolates how much of the OOD gap is detector capacity
rather than the representation. The paper's full design also stacks all layers;
here we keep the last-layer subset we store, matching every other row's spine.

Same protocol as the rest of the harness: train on PRM800K, select the threshold
on val, report in-domain PRM800K test (AUROC + F1) and ProcessBench first-error
F1 per subset, and write pb_step_scores for offline calib-20. Data plumbing
(handles, span reading, padded collate, scoring) is reused from the attn_pool
probe so only the model differs.
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
from scripts.train_attn_pool_probe import (  # noqa: E402
    build_handles, collate, score_split,
)
from scripts.train_easy_probe_method import (  # noqa: E402
    auroc_numpy, evaluate_processbench, resolve_threshold_grid,
    select_threshold, step_binary_metrics,
)
from src.repstore.store import ShardedRepSplit  # noqa: E402


class ReProbe(nn.Module):
    """proj(d->d_model) + learned pos -> N encoder layers -> masked mean -> head.

    forward(x, mask): x is (B, T, d) padded token states, mask is (B, T) with 1
    for real tokens and 0 for padding (same contract as collate/AttnPoolProbe)."""

    def __init__(self, d: int, d_model: int = 256, nhead: int = 4, nlayers: int = 2,
                 ff: int = 1024, t_max: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d_model)
        self.pos = nn.Parameter(torch.randn(t_max, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, 1)
        self.t_max = t_max

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.proj(x) + self.pos[:T].unsqueeze(0)      # (B, T, d_model)
        pad = mask == 0                                   # True = ignore
        h = self.enc(h, src_key_padding_mask=pad)
        m = mask.unsqueeze(-1)                            # (B, T, 1)
        z = (h * m).sum(1) / m.sum(1).clamp(min=1.0)      # masked mean over step
        return self.head(z).squeeze(-1)                  # (B,)


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
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold_grid", default="0.01")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[reprobe] loading stores", flush=True)
    train_h, _ = build_handles(ShardedRepSplit(args.prm_store / args.train_stem))
    val_h, _ = build_handles(ShardedRepSplit(args.prm_store / args.val_stem))
    test_h, _ = build_handles(ShardedRepSplit(args.prm_store / args.test_stem))
    d = ShardedRepSplit(args.prm_store / args.val_stem).spec.dim

    if len(train_h) > args.train_cap:
        keep = rng.choice(len(train_h), args.train_cap, replace=False)
        train_h = [train_h[i] for i in keep]
    print(f"[reprobe] train={len(train_h)} val={len(val_h)} test={len(test_h)} d={d}", flush=True)

    model = ReProbe(d, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers,
                    ff=args.ff, t_max=args.t_max, dropout=args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[reprobe] params={n_par/1e6:.2f}M "
          f"(d_model={args.d_model} nlayers={args.nlayers} nhead={args.nhead} ff={args.ff})",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        print(f"[reprobe] epoch {ep}: train_loss={tot/max(nb,1):.4f} val_loss={val_loss:.4f} "
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
    torch.save(best_state, args.out_dir / "reprobe_probe.pt")

    grid = resolve_threshold_grid(args.threshold_grid)
    val_scores = score_split(model, val_h, args.t_max, args.batch_size, device)
    t_sel, bacc, _ = select_threshold(val_scores, val_y, grid)

    # In-domain test
    test_scores = score_split(model, test_h, args.t_max, args.batch_size, device)
    test_y = np.array([h[4] for h in test_h])
    id_metrics = {
        "method": "reprobe", "auroc": auroc_numpy(test_y, test_scores),
        "fixed_t0.5": step_binary_metrics(test_y, test_scores, 0.5),
        "val_selected": {"threshold": t_sel, **step_binary_metrics(test_y, test_scores, t_sel)},
    }
    (args.out_dir / "in_domain_metrics.json").write_text(json.dumps(id_metrics, indent=2))
    print(f"[reprobe|in_domain] AUROC={id_metrics['auroc']:.4f} "
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
            summaries.append({"method": "reprobe", "pb_name": sub, "threshold_type": tag,
                              "threshold": t, **m})
        print(f"[reprobe|{sub}] val(t={t_sel}) F1={m_val['F1_PB']:.4f} "
              f"oracle(t={best_ot}) F1={m_oracle['F1_PB']:.4f}", flush=True)
    (args.out_dir / "eval_summary.json").write_text(json.dumps({"runs": summaries}, indent=2))
    print("[reprobe] done", flush=True)


if __name__ == "__main__":
    main()
