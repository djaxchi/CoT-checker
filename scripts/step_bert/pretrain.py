"""Masked-step pretraining for step_bert_probe_v0 (S2), no labels used.

Each training trace draws one target step t and one visibility mode (from
--modes); the target token is replaced by [MASK] and the model reconstructs a
fixed PCA target vector of step t from the visible steps. 15% of the
other visible steps are also masked, for robustness. Early stopping on val loss.

After training, every step of every dataset is scored in each of left / right /
both (only the modes the model was trained on): e_<mode> = mean squared
reconstruction error in the target space. These are the G1 scores.

Target space (changed 2026-10-14): plain PCA, NOT whitened, so components keep
their variance weight and low-variance (noisy) directions do not dominate the
loss. All components are divided by one shared scale sqrt(mean component
variance), so a loss of 1.0 still means "no better than predicting the mean".

Token sets:
    state         input [b_i, m_i]   target m_i
    contribution  input c_i          target c_i     (c_i = m_i - b_i)
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
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.probes.step_bert import StepBert  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inputs  # noqa: E402


def token_arrays(z, tokens: str) -> tuple[np.ndarray, np.ndarray, int]:
    b, m = z["b"], z["m"]
    if tokens == "state":
        return np.concatenate([b, m], 1), m, 2
    if tokens == "contribution":
        c = (m.astype(np.float32) - b.astype(np.float32)).astype(np.float16)
        return c, c, 1
    raise ValueError(tokens)


def trace_runs(trace_id: np.ndarray) -> dict[str, tuple[int, int]]:
    """trace -> (start row, n_steps); rows are sorted by (trace, step)."""
    runs, start = {}, 0
    ids = trace_id.tolist()
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            runs[ids[start]] = (start, i - start)
            start = i
    return runs


def batch_x(feats: torch.Tensor, runs: list[tuple[int, int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(n for _, n in runs)
    x = torch.zeros(len(runs), T, feats.shape[1], dtype=torch.float16)
    for k, (s, n) in enumerate(runs):
        x[k, :n] = feats[s:s + n]
    return x.to(device, torch.float32), torch.tensor([n for _, n in runs], device=device)


@torch.no_grad()
def score(model, feats, targets, runs, modes, device, bs=1024) -> dict[str, np.ndarray]:
    """Per-step reconstruction error for every step of every run, per mode."""
    model.eval()
    N = feats.shape[0]
    out = {m: np.full(N, np.nan, dtype=np.float32) for m in modes}
    rows_all = np.concatenate([np.arange(s, s + n) for s, n in runs])
    emb = torch.zeros(N, model.d)
    for i in range(0, len(rows_all), 4096):
        r = rows_all[i:i + 4096]
        emb[r] = model.embed(feats[r].to(device, torch.float32)[None])[0].cpu()
    items = sorted(((s, n, t) for s, n in runs for t in range(n)), key=lambda it: it[1])
    for mode in modes:
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            h, L = batch_x(emb, [(s, n) for s, n, _ in chunk], device)
            t = torch.tensor([t for _, _, t in chunk], device=device)
            pred = model.reconstruct_h(h, L, t, mode)
            rows = torch.tensor([s + t for s, _, t in chunk])
            err = ((pred - targets[rows].to(device)) ** 2).mean(1)
            out[mode][rows.numpy()] = err.cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pooled", type=Path, required=True)
    p.add_argument("--pb", type=Path, nargs="*", default=[])
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--tokens", choices=["state", "contribution"], default="state")
    p.add_argument("--modes", nargs="+", default=["both", "left", "right"])
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--targets_per_trace", type=int, default=4,
                   help="targets drawn per trace per epoch (each with its own mode)")
    p.add_argument("--extra_mask", type=float, default=0.15)
    p.add_argument("--n_target", type=int, default=256)
    p.add_argument("--n_in", type=int, default=512, help="input PCA dims per token part (scripts/step_bert/inputs.py)")
    p.add_argument("--whiten", action="store_true",
                   help="whiten the PCA target (equal weight per component); default is variance-weighted")
    p.add_argument("--seed", type=int, default=0)
    # cpu by default: MPS gave non-finite losses on mixed-mode batches (2026-10-14) and
    # this model is small enough that cpu runs an epoch in about 2 s on 1k traces
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.pooled)
    _, tgt_np, n_parts = token_arrays(z, args.tokens)
    proj = inputs.fit_or_load(z, args.tokens, args.split, args.n_in)
    x_np = inputs.reduce(z, args.tokens, proj)
    runs = trace_runs(z["trace_id"])
    split = json.loads(args.split.read_text())
    tr_runs = [runs[t] for t in split["train"]]
    va_runs = [runs[t] for t in split["val"]]

    tr_rows = np.concatenate([np.arange(s, s + n) for s, n in tr_runs])
    pca = PCA(args.n_target, whiten=args.whiten, random_state=0).fit(tgt_np[tr_rows].astype(np.float32))
    scale = 1.0 if args.whiten else float(np.sqrt(pca.explained_variance_.mean()))

    def targets_of(arr):
        return torch.from_numpy((pca.transform(arr.astype(np.float32)) / scale).astype(np.float32))

    feats, targets = torch.from_numpy(x_np), targets_of(tgt_np)
    model = StepBert(n_parts=n_parts, part_dim=args.n_in, d=args.d, n_layers=args.layers, n_target=args.n_target).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def val_loss() -> float:
        e = score(model, feats, targets, va_runs, args.modes, dev)
        rows = np.concatenate([np.arange(s, s + n) for s, n in va_runs])
        return float(np.mean([e[m][rows].mean() for m in args.modes]))

    log, best, best_state, bad = [], float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(np.repeat(np.arange(len(tr_runs)), args.targets_per_trace))
        # bucket by trace length inside chunks of 32 batches to cut padding
        chunk_n = 32 * args.batch
        order = np.concatenate([c[np.argsort([tr_runs[k][1] for k in c], kind="stable")]
                                for c in np.split(order, range(chunk_n, len(order), chunk_n))])
        batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        batches = [batches[k] for k in rng.permutation(len(batches))]
        for i, bidx in enumerate(batches):
            chunk = [tr_runs[k] for k in bidx]
            x, L = batch_x(feats, chunk, dev)
            t = torch.tensor([rng.integers(n) for _, n in chunk], device=dev)
            modes = [args.modes[k] for k in rng.integers(len(args.modes), size=len(chunk))]
            extra = torch.rand(x.shape[:2], device=dev) < args.extra_mask
            pred = model.reconstruct(x, L, t, modes, extra_mask=extra)
            rows = torch.tensor([s for s, _ in chunk]) + t.cpu()
            loss = F.mse_loss(pred, targets[rows].to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if not torch.isfinite(loss):
                bad = torch.isnan(pred).any(1).cpu().numpy()
                raise SystemExit(f"non-finite loss at epoch {ep} batch {i}: "
                                 f"{[(chunk[k], int(t[k]), modes[k]) for k in np.where(bad)[0][:5]]}")
            tot += loss.item() * len(chunk)
        vl = val_loss()
        log.append({"epoch": ep, "train": tot / len(order), "val": vl, "sec": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, args.out_dir / "model.pt")

    all_runs = list(runs.values())
    e = score(model, feats, targets, all_runs, args.modes, dev)
    np.savez(args.out_dir / "scores_onpolicy.npz", **{f"e_{m}": v for m, v in e.items()})
    for pb_path in args.pb:
        zp = np.load(pb_path)
        _, tp, _ = token_arrays(zp, args.tokens)
        xp = inputs.reduce(zp, args.tokens, proj)
        ep_ = score(model, torch.from_numpy(xp), targets_of(tp), list(trace_runs(zp["trace_id"]).values()),
                    args.modes, dev)
        np.savez(args.out_dir / f"scores_{pb_path.stem.replace('pooled_', '')}.npz",
                 **{f"e_{m}": v for m, v in ep_.items()})
    (args.out_dir / "pretrain_log.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "best_val": best,
         "pca_var_kept": float(pca.explained_variance_ratio_.sum()), "log": log}, indent=2))


if __name__ == "__main__":
    main()
