"""Supervised fine-tuning for step_bert_probe_v0 (S3).

Every labeled step is a target. Its trace is cropped to the steps the condition
may see, so the window is exact at any depth (ALiBi is relative, so cropping does
not change positions' meaning):

    local    [t, t]
    causal   [t - W, t]          (W = -1 means the whole past)
    bidir    [t - W, t + W]      (W = -1 means the whole trace)

Head: MLP over [z_t, embed(x_t), standardized pretraining scores]. The scores are
the frozen pretrained model's reconstruction errors for the modes the condition
may use (causal: e_left; bidir: e_left, e_right, e_both; local: none).

Loss: BCE, each step weighted 1 / n_steps(trace) so every trace counts equally.
Early stopping and the decision threshold both use on-policy val step F1. Writes
probabilities for every on-policy and ProcessBench step, and a metrics.json with
test step F1 (val threshold and oracle), AUROC, always-positive F1, and
ProcessBench F1_PB per subset (val threshold, calib-20 as on the leaderboard,
and per-subset oracle).
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
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inputs  # noqa: E402
from pretrain import token_arrays, trace_runs  # noqa: E402
from src.probes.step_bert import StepBert, StepBertClassifier  # noqa: E402

SCORE_MODES = {"local": [], "causal": ["left"], "bidir": ["left", "right", "both"]}


def crop(n: int, t: int, condition: str, W: int) -> tuple[int, int]:
    """Visible step range [lo, hi) of target t in a trace of n steps."""
    W = n if W < 0 else W
    if condition == "local":
        return t, t + 1
    if condition == "causal":
        return max(0, t - W), t + 1
    if condition == "bidir":
        return max(0, t - W), min(n, t + W + 1)
    raise ValueError(condition)


def make_batch(feats, extra, items, condition, W, device):
    """items: (start row, n, t). Returns x, lengths, local target index, extra, rows."""
    spans = [crop(n, t, condition, W) for _, n, t in items]
    T = max(hi - lo for lo, hi in spans)
    x = torch.zeros(len(items), T, feats.shape[1], dtype=torch.float16)
    for k, ((s, _, _), (lo, hi)) in enumerate(zip(items, spans)):
        x[k, :hi - lo] = feats[s + lo:s + hi]
    L = torch.tensor([hi - lo for lo, hi in spans])
    tl = torch.tensor([t - lo for (_, _, t), (lo, _) in zip(items, spans)])
    rows = np.array([s + t for s, _, t in items])
    ex = extra[rows].to(device) if extra is not None else None
    return x.to(device, torch.float32), L.to(device), tl.to(device), ex, rows


@torch.no_grad()
def predict(clf, feats, extra, runs, condition, W, device, bs=512) -> np.ndarray:
    clf.eval()
    out = np.full(feats.shape[0], np.nan, dtype=np.float32)
    items = [(s, n, t) for s, n in runs for t in range(n)]
    for i in range(0, len(items), bs):
        x, L, t, ex, rows = make_batch(feats, extra, items[i:i + bs], condition, W, device)
        out[rows] = torch.sigmoid(clf(x, L, t, "both", extra=ex)).cpu().numpy()
    return out


def best_threshold(y, p) -> tuple[float, float]:
    ths = np.unique(np.quantile(p, np.linspace(0.005, 0.995, 199)))
    f1s = [f1_score(y, p >= th, zero_division=0) for th in ths]
    k = int(np.argmax(f1s))
    return float(ths[k]), float(f1s[k])


def pb_correct_matrix(p, trace_id, step_idx, first_error, ths):
    """(n_traces, n_ths) bool: is each trace judged right at each threshold, plus
    a has-error flag per trace. The prediction is the first step with p >= th;
    a correct trace is right if nothing is flagged, an erroneous one if the first
    flag lands exactly on its first error."""
    order = np.lexsort((step_idx, trace_id))
    p, tid, fe = p[order], trace_id[order], first_error[order]
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    prefmax = np.empty_like(p)
    for s0, s1 in zip(starts, np.r_[starts[1:], len(p)]):
        prefmax[s0:s1] = np.maximum.accumulate(p[s0:s1])
    # index of the first flagged step = number of steps whose prefix max is below th
    pred = np.add.reduceat((prefmax[:, None] < ths[None, :]).astype(np.int32), starts, axis=0)
    n = np.diff(np.r_[starts, len(p)])[:, None]
    fe_t = fe[starts][:, None]
    has_err = fe_t[:, 0] >= 0
    right = np.where(has_err[:, None], pred == fe_t, pred == n)
    return right, has_err


def f1_from_right(right, has_err) -> np.ndarray:
    a, b = right[has_err].mean(0), right[~has_err].mean(0)
    return np.where(a + b > 0, 2 * a * b / np.maximum(a + b, 1e-12), 0.0)


def f1_pb(p, trace_id, step_idx, first_error, th) -> float:
    """ProcessBench F1: harmonic mean of exact first-error accuracy on erroneous
    traces and no-flag accuracy on correct traces."""
    return float(f1_from_right(*pb_correct_matrix(p, trace_id, step_idx, first_error, np.array([th])))[0])


def pb_thresholds(p) -> np.ndarray:
    return np.unique(np.quantile(p, np.linspace(0.005, 0.995, 199)))


def f1_pb_oracle(p, trace_id, step_idx, first_error) -> float:
    return float(f1_from_right(*pb_correct_matrix(p, trace_id, step_idx, first_error, pb_thresholds(p))).max())


def f1_pb_calib20(p, trace_id, step_idx, first_error, n_rep=20, k=20, seed=0) -> float:
    """Leaderboard protocol: pick the threshold on k held-out traces (stratified by
    has-error), apply it to the remaining traces, average over n_rep draws."""
    rng = np.random.default_rng(seed)
    right, has_err = pb_correct_matrix(p, trace_id, step_idx, first_error, pb_thresholds(p))
    err, cor = np.flatnonzero(has_err), np.flatnonzero(~has_err)
    k_err = int(round(k * len(err) / len(has_err)))
    out = []
    for _ in range(n_rep):
        cal = np.zeros(len(has_err), bool)
        cal[rng.choice(err, k_err, replace=False)] = True
        cal[rng.choice(cor, k - k_err, replace=False)] = True
        j = int(np.argmax(f1_from_right(right[cal], has_err[cal])))
        out.append(float(f1_from_right(right[~cal][:, [j]], has_err[~cal])[0]))
    return float(np.mean(out))


def load_extra(scores_dir: Path | None, name: str, modes: list[str]) -> np.ndarray | None:
    if not modes:
        return None
    e = np.load(scores_dir / f"scores_{name}.npz")
    return np.stack([e[f"e_{m}"] for m in modes], 1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pooled", type=Path, required=True)
    p.add_argument("--pb", type=Path, nargs="*", default=[])
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--tokens", choices=["state", "contribution"], default="state")
    p.add_argument("--condition", choices=["local", "causal", "bidir"], required=True)
    p.add_argument("--window", type=int, default=-1)
    p.add_argument("--pretrained_dir", type=Path, default=None,
                   help="pt_* dir: model.pt initializes the encoder, scores_*.npz give the extra features")
    p.add_argument("--no_scores", action="store_true", help="do not feed pretraining scores to the head")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_in", type=int, default=512, help="must match the pretrained model")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.pooled)
    _, _, n_parts = token_arrays(z, args.tokens)
    proj = inputs.fit_or_load(z, args.tokens, args.split, args.n_in)
    x_np = inputs.reduce(z, args.tokens, proj)
    feats = torch.from_numpy(x_np)
    y = z["y"].astype(np.float32)
    runs = trace_runs(z["trace_id"])
    split = json.loads(args.split.read_text())
    tr, va, te = ([runs[t] for t in split[k]] for k in ("train", "val", "test"))
    rows_of = lambda rs: np.concatenate([np.arange(s, s + n) for s, n in rs])  # noqa: E731

    modes = [] if (args.no_scores or args.pretrained_dir is None) else SCORE_MODES[args.condition]
    extra_np = load_extra(args.pretrained_dir, "onpolicy", modes)
    mu = sd = None
    if extra_np is not None:
        mu, sd = extra_np[rows_of(tr)].mean(0), extra_np[rows_of(tr)].std(0) + 1e-6
        extra = torch.from_numpy((extra_np - mu) / sd)
    else:
        extra = None

    enc = StepBert(n_parts=n_parts, part_dim=args.n_in)
    if args.pretrained_dir is not None:
        enc.load_state_dict(torch.load(args.pretrained_dir / "model.pt"))
    clf = StepBertClassifier(enc, n_extra=len(modes)).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.wd)

    items = [(s, n, t) for s, n in tr for t in range(n)]
    w_all = np.array([1.0 / n for _, n, _ in items])
    w_all = w_all / w_all.mean()
    yv = y[rows_of(va)]
    log, best, best_state, bad = [], -1.0, None, 0
    for ep in range(args.epochs):
        clf.train()
        t0, order, tot = time.time(), rng.permutation(len(items)), 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            x, L, t, ex, rows = make_batch(feats, extra, [items[k] for k in idx], args.condition, args.window, dev)
            logit = clf(x, L, t, "both", extra=ex)
            w = torch.from_numpy(w_all[idx]).float().to(dev)
            loss = (F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[rows]).to(dev),
                                                       reduction="none") * w).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        pv = predict(clf, feats, extra, va, args.condition, args.window, dev)[rows_of(va)]
        _, vf1 = best_threshold(yv, pv)
        log.append({"epoch": ep, "train": tot / len(items), "val_f1": vf1, "val_auroc": float(roc_auc_score(yv, pv)),
                    "sec": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
        if vf1 > best + 1e-4:
            best, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    clf.load_state_dict(best_state)

    p_all = predict(clf, feats, extra, list(runs.values()), args.condition, args.window, dev)
    th, _ = best_threshold(yv, p_all[rows_of(va)])
    rt = rows_of(te)
    yt, pt = y[rt], p_all[rt]
    prev = float(yt.mean())
    metrics = {"args": {k: str(v) for k, v in vars(args).items()}, "score_modes": modes,
               "val_threshold": th, "best_val_f1": best,
               "test": {"n": int(len(rt)), "prevalence": prev, "auroc": float(roc_auc_score(yt, pt)),
                        "f1_val_threshold": float(f1_score(yt, pt >= th)),
                        "f1_oracle": best_threshold(yt, pt)[1],
                        "always_positive_f1": 2 * prev / (1 + prev)},
               "processbench": {}, "log": log}
    np.save(args.out_dir / "probs_onpolicy.npy", p_all)
    for pb_path in args.pb:
        name = pb_path.stem.replace("pooled_", "")
        zp = np.load(pb_path)
        xp = inputs.reduce(zp, args.tokens, proj)
        ep_ = load_extra(args.pretrained_dir, name, modes)
        exp = torch.from_numpy((ep_ - mu) / sd) if ep_ is not None else None
        pp = predict(clf, torch.from_numpy(xp), exp, list(trace_runs(zp["trace_id"]).values()),
                     args.condition, args.window, dev)
        np.save(args.out_dir / f"probs_{name}.npy", pp)
        known = (zp["first_error_idx"] < 0) | (zp["step_idx"] <= zp["first_error_idx"])
        args_pb = (zp["trace_id"], zp["step_idx"], zp["first_error_idx"])
        metrics["processbench"][name] = {
            "f1_pb_val_threshold": f1_pb(pp, *args_pb, th),
            "f1_pb_oracle": f1_pb_oracle(pp, *args_pb),
            "f1_pb_calib20": f1_pb_calib20(pp, *args_pb),
            "known_step_auroc": float(roc_auc_score(zp["y"][known], pp[known])),
        }
    if metrics["processbench"]:
        for key in ("f1_pb_val_threshold", "f1_pb_calib20", "f1_pb_oracle", "known_step_auroc"):
            metrics[f"processbench_mean_{key}"] = float(np.mean(
                [v[key] for v in metrics["processbench"].values()]))
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("log", "args")}, indent=2))


if __name__ == "__main__":
    main()
