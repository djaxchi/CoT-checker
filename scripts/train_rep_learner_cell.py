#!/usr/bin/env python3
"""One cell of the representation x learner grid, under one fixed protocol.

The v1 leaderboard (`experiments/unified_harness_7b/leaderboard.md`) could not
separate the representation from the learner, because the two moved together:
the fixed-vector rows only ever got a linear head, the token-sequence rows only
ever got an attention query or a transformer, and on top of that the sequence
rows trained on a 150k subsample while the linear rows trained on all 513,810
steps, with different optimizers and epoch budgets. Every reported difference
therefore mixed at least four changes.

This script is the single entry point that makes a cell comparable to every other
cell. It fixes:

  * the data: the full PRM800K train split, no cap unless one is asked for
    explicitly, and the same val / in-domain-test / ProcessBench splits;
  * the trainer: AdamW + BCE, the same epoch budget, the same early-stopping rule
    on validation loss, for a linear head and a transformer alike;
  * the hyperparameter protocol: the same (lr x weight-decay) grid is searched
    for every cell and selected on validation AUROC, so no cell is advantaged by
    having been tuned harder than its neighbour;
  * the evaluation: the same threshold selection, the same in-domain metrics, and
    per-step ProcessBench scores written out for the same offline calib-20.

What varies is `--rep` (what the learner is shown) and `--learner` (what reads
it). The parameter count of the learner is recorded in the results so the grid
can be reported as F1_PB against capacity, one curve per representation, rather
than as a single number per row: a representation that is genuinely better
dominates at every capacity, one that only looked better because its detector was
larger converges.

Reads the compact step-span store (`scripts/build_step_span_store.py`), which
holds the pre-step boundary state followed by the step's own tokens, so every
representation here is an offline slice of one shared set of activations.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.derive_delta_from_token_store import derive_split  # noqa: E402
from scripts.train_easy_probe_method import (  # noqa: E402
    auroc_numpy, evaluate_processbench, resolve_threshold_grid,
    select_threshold, step_binary_metrics,
)
from src.harness.learners import build_learner, is_sequence, param_count  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402

# rep name -> the offline readout that derives it from the step-span store.
# `step_tokens` is the un-reduced sequence and has no vector readout.
REP_READOUT = {
    "last_token": "last",
    "step_mean": "mean",
    "step_delta": "delta",
    "step_stats": "multistat",
    "boundary_stats": "boundary_stats",
}
REPS = tuple(REP_READOUT) + ("step_tokens",)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_vectors(store_root: Path, stem: str, rep: str, cache_dir: Path | None,
                 sort: bool):
    """(X (N, D) float32, y (N,), meta) for a fixed-vector representation.

    Derivation is a pure numpy slice over the memory-mapped store, so it is cached
    on disk: the same vectors are reused by every learner on that representation,
    which is what makes the grid affordable.
    """
    readout = REP_READOUT[rep]
    cache_h = cache_dir / f"{rep}__{stem}_h.npy" if cache_dir else None
    cache_y = cache_dir / f"{rep}__{stem}_y.npy" if cache_dir else None
    cache_m = cache_dir / f"{rep}__{stem}_meta.jsonl" if cache_dir else None
    if cache_h is not None and cache_h.exists() and cache_y.exists() and (
            not sort or cache_m.exists()):
        X = np.load(cache_h, mmap_mode="r")
        y = np.load(cache_y)
        meta = ([json.loads(l) for l in cache_m.read_text().splitlines() if l.strip()]
                if sort else [])
        return X, y, meta
    X, y, meta = derive_split(store_root / stem, readout, sort=sort)
    if cache_h is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_h, X)
        np.save(cache_y, y)
        if sort:
            with cache_m.open("w") as f:
                for m in meta:
                    f.write(json.dumps(m) + "\n")
    return X, y, meta


def build_handles(view: ShardedRepSplit):
    """(shard, local_idx, step_start, n_tokens, label) per item, in view order."""
    meta = view.meta()
    y = view.y
    out = []
    for k in range(len(view)):
        rs, li = view.item_handle(k)
        m = meta[k]
        out.append((rs, li, int(m["step_start_idx"]), int(m["n_tokens"]), int(y[k])))
    return out, meta


def read_span(rs, li, step_start, t_max):
    """The step's own token states (the boundary row at index 0 is skipped)."""
    off = int(rs.offsets[li])
    end = int(rs.offsets[li + 1])
    span = np.asarray(rs.h[off + step_start:end], dtype=np.float32)
    if span.shape[0] == 0:
        span = np.asarray(rs.h[end - 1:end], dtype=np.float32)
    if span.shape[0] > t_max:
        span = span[-t_max:]
    return span


def collate_seq(handles, idx, t_max, device):
    spans = [read_span(*handles[i][:3], t_max) for i in idx]
    T = max(s.shape[0] for s in spans)
    d = spans[0].shape[1]
    x = np.zeros((len(spans), T, d), dtype=np.float32)
    mask = np.zeros((len(spans), T), dtype=np.float32)
    for j, s in enumerate(spans):
        x[j, :s.shape[0]] = s
        mask[j, :s.shape[0]] = 1.0
    yb = torch.tensor([handles[i][4] for i in idx], dtype=torch.float32, device=device)
    return (torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device), yb)


def collate_vec(X, y, idx, device):
    xb = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(device)
    yb = torch.from_numpy(np.asarray(y[idx], dtype=np.float32)).to(device)
    return xb, None, yb


# ---------------------------------------------------------------------------
# One shared trainer for every learner
# ---------------------------------------------------------------------------

def train_one(model, batches_fn, n_train, n_val, val_batches_fn, epochs, batch_size,
              lr, weight_decay, patience, device, seed):
    """AdamW + BCE with early stopping on validation loss. Identical for all cells."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(n_train)
        for i in range(0, n_train, batch_size):
            xb, mb, yb = batches_fn(order[i:i + batch_size])
            loss = F.binary_cross_entropy_with_logits(model(xb, mb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                xb, mb, yb = val_batches_fn(np.arange(i, min(i + batch_size, n_val)))
                tot += F.binary_cross_entropy_with_logits(
                    model(xb, mb), yb, reduction="sum").item()
        val_loss = tot / max(n_val, 1)
        if val_loss + 1e-8 < best:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def score_all(model, n, batches_fn, batch_size):
    model.eval()
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n))
        xb, mb, _ = batches_fn(idx)
        out[idx] = torch.sigmoid(model(xb, mb)).float().cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rep", required=True, choices=REPS)
    p.add_argument("--learner", required=True,
                   help="linear | mlp:h1024 | mlp:h1024x2 | attn_query | transformer:d256,l2")
    p.add_argument("--prm_store", required=True, type=Path,
                   help="Step-span store rep dir, e.g. <repstore>/step_spans")
    p.add_argument("--pb_store", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--vec_cache_dir", type=Path, default=None,
                   help="Where derived fixed-vector representations are cached.")
    p.add_argument("--train_stem", default="probe_train_full")
    p.add_argument("--val_stem", default="val_5k")
    p.add_argument("--test_stem", default="test_2k")
    p.add_argument("--pb_subsets", nargs="+",
                   default=["gsm8k", "math", "olympiadbench", "omnimath"])
    p.add_argument("--train_cap", type=int, default=None,
                   help="Rows to subsample from train. Default None = the FULL split; "
                        "a cap makes this cell incomparable to uncapped cells and is "
                        "recorded as such in results.json.")
    p.add_argument("--hp_search_cap", type=int, default=100000,
                   help="Rows used while searching the lr x wd grid. The winning "
                        "config is then refit on the full --train_cap rows. Same "
                        "for every cell.")
    p.add_argument("--lr_grid", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    p.add_argument("--wd_grid", type=float, nargs="+", default=[0.0, 0.01])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--t_max", type=int, default=512,
                   help="Token cap for sequence learners. 512 covers >99.9%% of "
                        "steps (p99 span is 155 tokens), so it is effectively no cap.")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold_grid", default="0.01")
    args = p.parse_args()

    seq = is_sequence(args.learner)
    if seq and args.rep != "step_tokens":
        raise SystemExit(f"learner {args.learner!r} reads sequences; rep must be step_tokens")
    if not seq and args.rep == "step_tokens":
        raise SystemExit(
            f"rep 'step_tokens' is a sequence; use a sequence learner, or the "
            f"'step_mean' rep, which is mean-pooling made explicit (the bridge cell)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = resolve_threshold_grid(args.threshold_grid)
    t0 = time.time()

    # ---- load the three PRM800K splits in the shape this learner needs -----
    if seq:
        train_h, _ = build_handles(ShardedRepSplit(args.prm_store / args.train_stem))
        val_h, _ = build_handles(ShardedRepSplit(args.prm_store / args.val_stem))
        test_h, _ = build_handles(ShardedRepSplit(args.prm_store / args.test_stem))
        d = ShardedRepSplit(args.prm_store / args.val_stem).spec.dim
        y_train = np.array([h[4] for h in train_h], dtype=np.int8)
        y_val = np.array([h[4] for h in val_h], dtype=np.int8)
        y_test = np.array([h[4] for h in test_h], dtype=np.int8)

        def mk(handles):
            return lambda idx: collate_seq(handles, idx, args.t_max, device)
        train_fn, val_fn, test_fn = mk(train_h), mk(val_h), mk(test_h)
        n_train_all = len(train_h)
    else:
        Xtr, y_train, _ = load_vectors(args.prm_store, args.train_stem, args.rep,
                                       args.vec_cache_dir, sort=False)
        Xva, y_val, _ = load_vectors(args.prm_store, args.val_stem, args.rep,
                                     args.vec_cache_dir, sort=False)
        Xte, y_test, _ = load_vectors(args.prm_store, args.test_stem, args.rep,
                                      args.vec_cache_dir, sort=False)
        d = Xtr.shape[1]
        train_fn = lambda idx: collate_vec(Xtr, y_train, idx, device)   # noqa: E731
        val_fn = lambda idx: collate_vec(Xva, y_val, idx, device)       # noqa: E731
        test_fn = lambda idx: collate_vec(Xte, y_test, idx, device)     # noqa: E731
        n_train_all = Xtr.shape[0]

    rng = np.random.default_rng(args.seed)
    n_train = n_train_all if args.train_cap is None else min(args.train_cap, n_train_all)
    train_subset = (np.arange(n_train_all) if n_train == n_train_all
                    else rng.choice(n_train_all, n_train, replace=False))
    print(f"[cell] rep={args.rep} learner={args.learner} d={d} "
          f"train={n_train}/{n_train_all} val={len(y_val)} test={len(y_test)} "
          f"device={device.type}", flush=True)

    def sub_fn(base_fn, subset):
        return lambda idx: base_fn(subset[idx])

    # ---- hyperparameter search, identical protocol for every cell ---------
    n_hp = min(args.hp_search_cap, n_train)
    hp_subset = train_subset[:n_hp]
    hp_fn = sub_fn(train_fn, hp_subset)
    n_val = len(y_val)
    trials = []
    best_cfg, best_auroc = None, -1.0
    for lr in args.lr_grid:
        for wd in args.wd_grid:
            torch.manual_seed(args.seed)
            m = build_learner(args.learner, d, t_max=args.t_max,
                              dropout=args.dropout).to(device)
            m, vloss = train_one(m, hp_fn, n_hp, n_val, val_fn, args.epochs,
                                 args.batch_size, lr, wd, args.patience, device,
                                 args.seed)
            va = auroc_numpy(y_val, score_all(m, n_val, val_fn, args.batch_size))
            trials.append({"lr": lr, "weight_decay": wd, "val_loss": vloss,
                           "val_auroc": float(va)})
            print(f"[hp] lr={lr} wd={wd} val_loss={vloss:.4f} val_auroc={va:.4f}",
                  flush=True)
            if va > best_auroc:
                best_auroc, best_cfg = va, {"lr": lr, "weight_decay": wd}
            del m

    # ---- refit the winner on the full training rows ----------------------
    torch.manual_seed(args.seed)
    model = build_learner(args.learner, d, t_max=args.t_max,
                          dropout=args.dropout).to(device)
    n_params = param_count(model)
    model, _ = train_one(model, sub_fn(train_fn, train_subset), n_train, n_val, val_fn,
                         args.epochs, args.batch_size, best_cfg["lr"],
                         best_cfg["weight_decay"], args.patience, device, args.seed)
    print(f"[cell] refit on {n_train} rows with {best_cfg}, {n_params} params",
          flush=True)

    # ---- in-domain -------------------------------------------------------
    val_scores = score_all(model, n_val, val_fn, args.batch_size)
    t_val, val_bacc, _ = select_threshold(val_scores, y_val, grid)
    test_scores = score_all(model, len(y_test), test_fn, args.batch_size)
    test_auroc = auroc_numpy(y_test, test_scores)
    t_oracle, _, _ = select_threshold(test_scores, y_test, grid)
    in_domain = {
        "auroc": float(test_auroc),
        "val_threshold": float(t_val),
        "val_bacc": float(val_bacc),
        "fixed_0.5": step_binary_metrics(y_test, test_scores, 0.5),
        "val_selected": step_binary_metrics(y_test, test_scores, t_val),
        "oracle": step_binary_metrics(y_test, test_scores, t_oracle),
    }
    print(f"[in_domain] AUROC={test_auroc:.4f} t_val={t_val:.2f}", flush=True)

    # ---- ProcessBench ----------------------------------------------------
    pb: dict[str, dict] = {}
    for sub in args.pb_subsets:
        sub_dir = args.pb_store / sub
        if not sub_dir.exists():
            print(f"[pb] skip {sub}: {sub_dir} missing", flush=True)
            continue
        if seq:
            view = ShardedRepSplit(sub_dir)
            handles, meta = build_handles(view)
            fn = lambda idx: collate_seq(handles, idx, args.t_max, device)  # noqa: E731
            scores = score_all(model, len(handles), fn, args.batch_size)
        else:
            Xpb, _, meta = load_vectors(args.pb_store, sub, args.rep,
                                        args.vec_cache_dir, sort=True)
            ypb = np.zeros(Xpb.shape[0], dtype=np.int8)
            fn = lambda idx: collate_vec(Xpb, ypb, idx, device)  # noqa: E731
            scores = score_all(model, Xpb.shape[0], fn, args.batch_size)
        _, m_val = evaluate_processbench(scores, meta, t_val)
        best_f1, best_t = -1.0, grid[0]
        for t in grid:
            _, mt = evaluate_processbench(scores, meta, t)
            if mt["F1_PB"] > best_f1:
                best_f1, best_t = mt["F1_PB"], t
        pb[sub] = {"val_selected": m_val, "oracle_F1_PB": float(best_f1),
                   "oracle_threshold": float(best_t)}
        with (args.out_dir / f"pb_step_scores_{sub}.jsonl").open("w") as f:
            for s, m in zip(scores, meta):
                f.write(json.dumps({"id": m["id"], "step_idx": int(m["step_idx"]),
                                    "label": int(m["label"]),
                                    "n_steps": int(m["n_steps"]),
                                    "score": float(s)}) + "\n")
        print(f"[pb:{sub}] F1_PB@val={m_val['F1_PB']:.4f} "
              f"F1_PB@oracle={best_f1:.4f}", flush=True)

    results = {
        "rep": args.rep,
        "learner": args.learner,
        "dim": int(d),
        "n_params": int(n_params),
        "seed": args.seed,
        "n_train": int(n_train),
        "n_train_available": int(n_train_all),
        "full_train": bool(args.train_cap is None),
        "hp": {"selected": best_cfg, "search_rows": int(n_hp), "trials": trials},
        "protocol": {"epochs": args.epochs, "patience": args.patience,
                     "batch_size": args.batch_size, "t_max": args.t_max,
                     "dropout": args.dropout,
                     "threshold_grid": args.threshold_grid},
        "in_domain": in_domain,
        "processbench": pb,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"[cell] wrote {args.out_dir}/results.json in "
          f"{results['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
