"""Arm A of step_bert_probe_v1: supervised token-level encoder over frozen L35 token states.

Conditions (what step t's readout may depend on):

    local      tokens of step t only                         (mask, whole trace per unit)
    causal     tokens at or before each token                (mask, whole trace per unit)
    bidir -1   every token of the trace                      (mask, whole trace per unit)
    bidir W    tokens of steps t-W .. t+W                    (input cropped per target step)

Masks for local / causal / all are exact at any depth; a finite window is not
(two layers would reach 2W), so bidir W>=0 crops the input around one target per
unit instead, the same exact-window rule as v0.

Loss: BCE, each step weighted 1/n_steps(trace) (mean 1 over train). Early stopping
and the threshold use val step F1. Token data lives on the GPU in fp16.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v1_common as C  # noqa: E402
from src.probes.token_bert import TokenBert, token_mask  # noqa: E402


def load_tokens(d: Path, dev) -> tuple[dict, torch.Tensor]:
    steps = dict(np.load(d / "steps.npz"))
    X = np.load(d / "X.npy", mmap_mode="r")
    out = torch.empty(X.shape, dtype=torch.float16, device=dev)
    for i in range(0, X.shape[0], 1 << 19):
        out[i:i + (1 << 19)] = torch.from_numpy(np.array(X[i:i + (1 << 19)])).to(dev)
    return steps, out


def units(rs, condition: str, W: int) -> list[tuple[int, int, int]]:
    """(first step row, last step row + 1, target row or -1 for all steps)."""
    if condition == "bidir" and W >= 0:
        return [(s + max(0, t - W), s + min(n, t + W + 1), s + t) for s, n in rs for t in range(n)]
    return [(s, s + n, -1) for s, n in rs]


def batches(us, tok_off, budget: int, rng=None) -> list[list[int]]:
    """Group units into batches of at most `budget` padded tokens, length-sorted."""
    lens = np.array([tok_off[b] - tok_off[a] for a, b, _ in us])
    order = np.argsort(lens, kind="stable")
    out, cur, mx = [], [], 0
    for k in order:
        m = max(mx, lens[k])
        if cur and m * (len(cur) + 1) > budget:
            out.append(cur)
            cur, m = [], lens[k]
        cur.append(int(k))
        mx = m
    if cur:
        out.append(cur)
    if rng is not None:
        rng.shuffle(out)
    return out


def make_batch(X, tok_off, us, idx, condition, W, dev):
    """Readout ids are per (unit, step), so a step that appears in several crops of
    one batch is read separately in each. Returns the encoder inputs plus `pos`
    (readout ids to score) and `want` (their global step rows)."""
    sel = [us[k] for k in idx]
    lens = [int(tok_off[b] - tok_off[a]) for a, b, _ in sel]
    B, T, S = len(sel), max(lens), max(b - a for a, b, _ in sel)
    tok = torch.zeros(B, T, dtype=torch.long)
    cid = torch.full((B, T), -1, dtype=torch.long)
    bnd = torch.zeros(B, T, dtype=torch.bool)
    out_rows = np.full(B * S, -1, dtype=np.int64)
    pos = []
    for i, (a, b, t) in enumerate(sel):
        o = tok_off[a:b + 1]
        L = int(o[-1] - o[0])
        tok[i, :L] = torch.arange(int(o[0]), int(o[-1]))
        cid[i, :L] = torch.from_numpy(i * S + np.repeat(np.arange(b - a), np.diff(o)))
        bnd[i, torch.from_numpy(o[:-1] - o[0])] = True
        out_rows[i * S:i * S + b - a] = np.arange(a, b)
        pos += [i * S + t - a] if t >= 0 else list(range(i * S, i * S + b - a))
    valid = cid >= 0
    kind = "all" if sel[0][2] >= 0 else {"local": "local", "causal": "causal", "bidir": "all"}[condition]
    pos = np.array(pos)
    x = X[tok.to(X.device)].float().to(dev)
    return (x, cid.to(dev), bnd.to(dev), valid.to(dev), kind, B * S,
            torch.from_numpy(pos).to(dev), torch.from_numpy(out_rows[pos]).to(dev))


@torch.no_grad()
def predict(model, X, steps, rs, condition, W, dev, budget) -> np.ndarray:
    model.eval()
    out = np.full(len(steps["y"]), np.nan, dtype=np.float32)
    us = units(rs, condition, W)
    for idx in batches(us, steps["tok_off"], budget):
        x, cid, bnd, valid, kind, n_out, pos, want = make_batch(X, steps["tok_off"], us, idx, condition, W, dev)
        logit = model.readout(model.encode(x, bnd, token_mask(cid, valid, kind)), cid, bnd, valid, n_out)[pos]
        out[want.cpu().numpy()] = torch.sigmoid(logit).float().cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="dir with onpolicy/ and pb_<subset>/ token stores")
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--condition", choices=["local", "causal", "bidir"], required=True)
    p.add_argument("--window", type=int, default=-1)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--no_pb", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--budget", type=int, default=65536, help="padded tokens per batch")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    steps, X = load_tokens(args.data / "onpolicy", dev)
    print(f"loaded {X.shape} in {time.time() - t0:.0f}s", flush=True)
    runs, parts = C.split_runs(steps, args.split)
    tok_off = steps["tok_off"]
    y = torch.from_numpy(steps["y"].astype(np.float32)).to(dev)
    w_np = C.step_weights(parts["train"])
    w = torch.from_numpy(np.pad(w_np, (0, len(steps["y"]) - len(w_np)))).to(dev)

    model = TokenBert(in_dim=X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    us = units(parts["train"], args.condition, args.window)
    log, best, best_state, bad = [], -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        t1, tot, nb = time.time(), 0.0, 0
        for idx in batches(us, tok_off, args.budget, rng):
            x, cid, bnd, valid, kind, n_out, pos, want = make_batch(X, tok_off, us, idx, args.condition,
                                                                    args.window, dev)
            logit = model.readout(model.encode(x, bnd, token_mask(cid, valid, kind)), cid, bnd, valid, n_out)[pos]
            loss = (F.binary_cross_entropy_with_logits(logit, y[want], reduction="none") * w[want]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        pv = predict(model, X, steps, parts["val"], args.condition, args.window, dev, args.budget)
        vf1, vauc = C.val_f1(steps, pv, parts)
        log.append({"epoch": ep, "train": tot / nb, "val_f1": vf1, "val_auroc": vauc,
                    "sec": round(time.time() - t1, 1)})
        print(json.dumps(log[-1]), flush=True)
        if vf1 > best + 1e-4:
            best, bad = vf1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)

    p_all = predict(model, X, steps, list(runs.values()), args.condition, args.window, dev, args.budget)
    del X
    torch.cuda.empty_cache()
    pb = []
    if not args.no_pb:
        for s in C.PB_SUBSETS:
            zp, Xp = load_tokens(args.data / f"pb_{s}", dev)
            rp = list(C.trace_runs(zp["trace_id"]).values())
            pb.append((f"pb_{s}", zp, predict(model, Xp, zp, rp, args.condition, args.window, dev, args.budget)))
            del Xp
            torch.cuda.empty_cache()
    C.finalize(args.out_dir, args, steps, p_all, parts, log, best, pb)


if __name__ == "__main__":
    main()
