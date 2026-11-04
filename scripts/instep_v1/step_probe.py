#!/usr/bin/env python3
"""STEP baseline (Liang et al. 2026, arXiv 2601.09093) on our step labels.

STEP's step scorer, as the paper specifies it: "the last-layer hidden state of step-end token",
a two-layer MLP d -> 512 (ReLU) -> 1, weighted binary cross-entropy with positive weight
#negatives / #positives, Adam (lr 1e-4, weight decay 1e-5), batch 128, at most 20 epochs with
early stopping (patience 5). STEP trains on trace-level pseudo-labels; here it trains on the same
step labels and training split as InStep, so the two differ only in the method. Early stopping
watches the weighted BCE on the validation split (the paper does not name its criterion); the
best epoch's weights score the sets.

Inputs are read_token_inline.py encode outputs: `passive_final` (the post-norm last-layer state
at the step's last token) in <set>.shard0.npz, labels `y`, `kept`. Training states: --train_names
in --train_states, concatenated. Validation: `val` in --states, labelled from
<data>/rp/val.traces.jsonl[--val_label] (as passive_fit.py). Every other set in --states except
train* is scored per seed into <out>/seed<s>/<set>.jsonl ({"uid", "p"}, p = P(incorrect)), and
<out>/summary.json holds per-seed AP / AUROC on the ReProbe test sets and their seed mean.

    python scripts/instep_v1/step_probe.py --states $ST/phi4 --train_states $ST/phi4_fulltrain \\
        --train_names train_p0,train_p1,train_p2,train_p3 --data <R>/data --val_label qwen3 \\
        --out $OUT/scores/phi4/step
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.instep_v1.passive_fit import jl, load_set, load_x, val_labels  # noqa: E402
from scripts.reprobe_cmp.analyze import TEST, auroc, fast_ap  # noqa: E402

KEY = "passive_final"
DEFAULTS = {"hidden": 512, "lr": 1e-4, "wd": 1e-5, "batch": 128, "epochs": 20, "patience": 5}


class StepScorer(torch.nn.Module):
    def __init__(self, d: int, hidden: int = DEFAULTS["hidden"]):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, hidden)
        self.fc2 = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logits
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def n_params(d: int, hidden: int = DEFAULTS["hidden"]) -> int:
    return d * hidden + hidden + hidden + 1


def pos_weight(y: torch.Tensor) -> torch.Tensor:
    return (1 - y).sum() / y.sum()


@torch.no_grad()
def predict(m: StepScorer, X: torch.Tensor, bs: int = 8192) -> torch.Tensor:
    m.eval()
    return torch.cat([m(X[i:i + bs]) for i in range(0, len(X), bs)])


def train_seed(Xtr, ytr, Xva, yva, seed: int, a) -> tuple[StepScorer, dict]:
    torch.manual_seed(seed)
    m = StepScorer(Xtr.shape[1], a.hidden).to(Xtr.device)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr, weight_decay=a.wd)
    pw = pos_weight(ytr)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_ep, best_state, bad, hist = float("inf"), 0, None, 0, []
    for ep in range(1, a.epochs + 1):
        m.train()
        for idx in torch.randperm(len(Xtr), generator=g).split(a.batch):
            idx = idx.to(Xtr.device)
            opt.zero_grad()
            loss_fn(m(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        vl = float(loss_fn(predict(m, Xva), yva))
        hist.append(vl)
        if vl < best:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
        print(f"[step] seed {seed} epoch {ep}: val loss {vl:.4f}", flush=True)
    m.load_state_dict(best_state)
    return m, {"best_epoch": best_ep, "val_loss": best, "val_loss_by_epoch": hist, "pos_weight": float(pw)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", type=Path, required=True, help="val and test states")
    p.add_argument("--train_states", type=Path, required=True)
    p.add_argument("--train_names", required=True, help="comma list of training state files, concatenated")
    p.add_argument("--data", type=Path, required=True, help="<R>/data (rp/val.traces.jsonl, rt/all_val.items.jsonl)")
    p.add_argument("--val_label", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", default="0,1,2")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    xs, ys = [], []
    for nm in a.train_names.split(","):
        _, y, kept = load_set(a.train_states, nm)
        m = kept & (y >= 0)
        xs.append(load_x(a.train_states, nm, KEY)[m])
        ys.append(y[m])
    Xtr = torch.from_numpy(np.nan_to_num(np.concatenate(xs))).to(dev)
    ytr = torch.from_numpy(np.concatenate(ys).astype(np.float32)).to(dev)
    del xs
    rate = float(ytr.mean())
    print(f"[step] train {len(ytr)} labelled steps from {a.train_names}, positive rate {rate:.4f}", flush=True)
    if not 0.02 <= rate <= 0.6:
        raise SystemExit(f"train positive rate {rate:.4f}: wrong labels?")

    va_uids, _, va_kept = load_set(a.states, "val")
    ref = [r["uid"] for r in jl(a.data / "rt" / "all_val.items.jsonl")]
    if va_uids != ref:
        raise SystemExit(f"val meta uids ({len(va_uids)}) do not match all_val.items.jsonl ({len(ref)})")
    vl = val_labels(a.data, a.val_label)
    va_m = np.array([k and u in vl for u, k in zip(va_uids, va_kept)])
    Xva = torch.from_numpy(np.nan_to_num(load_x(a.states, "val", KEY)[va_m])).to(dev)
    yva = torch.tensor([float(vl[u]) for u, k in zip(va_uids, va_m) if k], device=dev)

    sets = sorted(z.name[:-len(".shard0.npz")] for z in a.states.glob("*.shard0.npz"))
    sets = [s for s in sets if not s.startswith("train")]
    for s in list(sets):
        with np.load(a.states / f"{s}.shard0.npz") as z:
            if KEY not in z.files:
                print(f"[step] {s}: no {KEY}, not scored", flush=True)
                sets.remove(s)
    seeds = [int(s) for s in a.seeds.split(",")]
    per_seed = {}
    for seed in seeds:
        model, info = train_seed(Xtr, ytr, Xva, yva, seed, a)
        sv = torch.sigmoid(predict(model, Xva)).cpu().numpy()
        info.update({"val_ap": fast_ap(yva.cpu().numpy(), sv), "val_auroc": auroc(yva.cpu().numpy(), sv), "test": {}})
        od = a.out / f"seed{seed}"
        od.mkdir(parents=True, exist_ok=True)
        for name in sets:
            uids, y, kept = load_set(a.states, name)
            s = torch.sigmoid(predict(model, torch.from_numpy(np.nan_to_num(load_x(a.states, name, KEY)[kept])).to(dev)))
            s = s.cpu().numpy().astype(np.float64)
            with open(od / f"{name}.jsonl", "w") as f:
                for u, v in zip([u for u, k in zip(uids, kept) if k], s):
                    f.write(json.dumps({"uid": u, "p": float(v)}) + "\n")
            if name in TEST:
                yk = y[kept]
                mk = yk >= 0
                info["test"][name] = {"ap": fast_ap(yk[mk], s[mk]), "auroc": auroc(yk[mk], s[mk]), "n": int(mk.sum())}
        info["test_mean_ap"] = float(np.mean([t["ap"] for t in info["test"].values()])) if info["test"] else None
        per_seed[str(seed)] = info
        print(f"[step] seed {seed}: best epoch {info['best_epoch']}, val AP {info['val_ap']:.4f}, "
              f"test mean AP {info['test_mean_ap']}", flush=True)
    tests = sorted({n for r in per_seed.values() for n in r["test"]})
    mean = {n: float(np.mean([per_seed[str(s)]["test"][n]["ap"] for s in seeds])) for n in tests}
    summary = {"states": str(a.states), "train": f"{a.train_states}:{a.train_names}", "val_label": a.val_label,
               "key": KEY, "hparams": {k: getattr(a, k) for k in DEFAULTS}, "params": n_params(Xtr.shape[1], a.hidden),
               "n_train": int(len(ytr)), "train_pos_rate": rate, "n_val": int(len(yva)), "seeds": seeds,
               "per_seed": per_seed, "test_seed_mean": mean,
               "overall_seed_mean": float(np.mean(list(mean.values()))) if mean else None,
               "seconds": time.perf_counter() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[step] done in {summary['seconds']:.0f}s: seed-mean test AP {summary['overall_seed_mean']} -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
