#!/usr/bin/env python3
"""InStep V1 B1 (passive linear baseline) and S1 (per-layer separability).

Reads read_token_inline.py encode outputs in --states (<set>.shard0.{npz,meta.jsonl}): the
text state at the step's last token (`passive_block<l>`, `passive_final`) and the read-token
state (`read_block<l>`). Every key: standardize on train40k, L2 logistic regression fitted on
train40k for each C in --grid (sklearn convention: C * sum logloss + ||w||^2 / 2, intercept
not penalized; full-batch LBFGS in torch, GPU when present), validation AP / AUROC per C.

B1: the (passive key, C) with the best validation AP scores every set but train40k:
<out>/<set>.jsonl, one line per kept step {"uid", "p"}, p = P(incorrect) (an `rp:` arm of
scripts/reprobe_cmp/analyze.py).
S1 (--per_layer_out): per key (passive and read) the best C, its val AP and AUROC, the
picked key, and test-set AP / AUROC of the picked model.

Validation labels come from <data>/rp/val.traces.jsonl (labels[--val_label]), as in
analyze.py; the val meta uids must equal <data>/rt/all_val.items.jsonl. Train labels are
checked against --train_items (uid -> label) and the positive rate must be in [0.02, 0.2].

    python scripts/instep_v1/passive_fit.py --states $SCRATCH/cot_mech/instep_v1/states/qwen3_8b \\
        --data <R>/data --val_label ds --train_items <R>/data/rt/ds_train.items.jsonl \\
        --out $OUT/scores/qwen3_8b/passive --per_layer_out $OUT/analysis/per_layer_qwen3_8b.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reprobe_cmp.analyze import TEST, auroc, fast_ap  # noqa: E402

GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
TRAIN = "train40k"


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def key_order(k: str) -> tuple:
    m = re.search(r"(\d+)$", k)
    return (0 if k.startswith("passive") else 1, int(m.group(1)) if m else 10**6)


def state_keys(npz: Path) -> list[str]:
    with np.load(npz) as z:
        ks = [k for k in z.files if re.fullmatch(r"passive_block\d+|passive_final|read_block\d+", k)]
    return sorted(ks, key=key_order)


def load_set(states: Path, name: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """uids, y (int, from the npz), kept (bool)."""
    with np.load(states / f"{name}.shard0.npz") as z:
        y, kept = z["y"].astype(int), z["kept"].astype(bool)
    uids = [r["uid"] for r in jl(states / f"{name}.shard0.meta.jsonl")]
    if len(uids) != len(y):
        raise SystemExit(f"{name}: {len(uids)} meta rows vs {len(y)} npz rows")
    return uids, y, kept


def load_x(states: Path, name: str, key: str) -> np.ndarray:
    with np.load(states / f"{name}.shard0.npz") as z:
        x = z[key]
    return x.reshape(len(x), -1).astype(np.float32)


def fit_path(X: torch.Tensor, y: torch.Tensor, grid: tuple[float, ...], max_iter: int = 300) -> list[tuple]:
    """L2 logistic regression for every C in grid (largest penalty first, warm started).
    X standardized (n, d) on the device. Returns [(C, w, b)]."""
    n, d = X.shape
    w = torch.zeros(d, device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    p0 = y.mean().clamp(1e-4, 1 - 1e-4)
    with torch.no_grad():
        b.fill_(float(torch.log(p0 / (1 - p0))))
    out = []
    for C in sorted(grid):
        lam = 1.0 / (C * n)
        opt = torch.optim.LBFGS([w, b], lr=1, max_iter=max_iter, history_size=20,
                                tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * lam * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)
        out.append((C, w.detach().clone(), b.detach().clone()))
    return out


class Standardizer:
    def __init__(self, X: np.ndarray):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6

    def __call__(self, X: np.ndarray, dev) -> torch.Tensor:
        return torch.from_numpy(np.nan_to_num((X - self.mu) / self.sd)).to(dev)


def prob(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(X @ w + b).cpu().numpy().astype(np.float64)


def val_labels(data: Path, label: str) -> dict:
    out = {}
    for t in jl(data / "rp" / "val.traces.jsonl"):
        for k, v in enumerate(t["labels"][label]):
            if v >= 0:
                out[f"{t['trace_id']}::{k}"] = int(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="<R>/data (rp/val.traces.jsonl, rt/all_val.items.jsonl)")
    p.add_argument("--val_label", required=True)
    p.add_argument("--train_items", type=Path, required=True, help="the items file train40k was drawn from")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per_layer_out", type=Path, default=None)
    p.add_argument("--grid", default=",".join(map(str, GRID)))
    p.add_argument("--allow_partial", action="store_true", help="smoke: val meta may be a prefix of all_val")
    p.add_argument("--keys", default=None,
                   help="comma-separated state keys to fit, e.g. passive_final (the last layer only)")
    a = p.parse_args()
    grid = tuple(float(c) for c in a.grid.split(","))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    # ---- train: labels checked against the source items file
    tr_uids, tr_y, tr_kept = load_set(a.states, TRAIN)
    src = {r["uid"]: int(r["label"]) for r in jl(a.train_items)}
    bad = [u for u, y in zip(tr_uids, tr_y) if src.get(u) != y]
    if bad:
        raise SystemExit(f"train40k: {len(bad)} labels differ from {a.train_items} (e.g. {bad[0]})")
    tr_m = tr_kept & (tr_y >= 0)
    rate = float(tr_y[tr_m].mean())
    print(f"[passive_fit] train40k {int(tr_m.sum())}/{len(tr_y)} kept labeled steps, positive rate {rate:.4f}", flush=True)
    if not 0.02 <= rate <= 0.2:
        raise SystemExit(f"train40k positive rate {rate:.4f} outside [0.02, 0.2]: wrong labels?")
    ytr = torch.from_numpy(tr_y[tr_m].astype(np.float32)).to(dev)

    # ---- val: uids must be all_val's, labels from the traces under --val_label
    va_uids, _, va_kept = load_set(a.states, "val")
    ref = [r["uid"] for r in jl(a.data / "rt" / "all_val.items.jsonl")]
    if va_uids != ref[:len(va_uids)] or (len(va_uids) != len(ref) and not a.allow_partial):
        raise SystemExit(f"val meta uids ({len(va_uids)}) do not match all_val.items.jsonl ({len(ref)})")
    vl = val_labels(a.data, a.val_label)
    va_m = np.array([k and u in vl for u, k in zip(va_uids, va_kept)])
    yva = np.array([vl[u] for u, m in zip(va_uids, va_m) if m])
    print(f"[passive_fit] val {int(va_m.sum())} labeled kept steps ({a.val_label}), positive rate {yva.mean():.4f}",
          flush=True)

    keys = state_keys(a.states / "val.shard0.npz")
    if a.per_layer_out is None:
        keys = [k for k in keys if k.startswith("passive")]
    if a.keys:
        want = a.keys.split(",")
        missing = [k for k in want if k not in keys]
        if missing:
            raise SystemExit(f"--keys {missing} not in {a.states / 'val.shard0.npz'}")
        keys = want
    if not any(k.startswith("passive") for k in keys):
        raise SystemExit(f"no passive states in {a.states}")
    rows, best = [], None
    for key in keys:
        t1 = time.perf_counter()
        Xtr = load_x(a.states, TRAIN, key)[tr_m]
        if not np.isfinite(Xtr).all():
            print(f"[passive_fit] warning: {key} train has {int((~np.isfinite(Xtr)).sum())} non-finite values",
                  flush=True)
            Xtr = np.nan_to_num(Xtr, posinf=0, neginf=0)
        std = Standardizer(Xtr)
        Xt = std(Xtr, dev)
        del Xtr
        Xv = std(load_x(a.states, "val", key)[va_m], dev)
        by_c = {}
        for C, w, b in fit_path(Xt, ytr, grid):
            s = prob(Xv, w, b)
            by_c[C] = {"val_ap": fast_ap(yva, s), "val_auroc": auroc(yva, s), "w": w, "b": b}
        Cb = max(by_c, key=lambda c: (np.nan_to_num(by_c[c]["val_ap"], nan=-1), -c))
        r = {"key": key, "kind": "passive" if key.startswith("passive") else "read",
             "block": int(key.rsplit("block", 1)[1]) if "block" in key else "final", "C": Cb,
             "val_ap": by_c[Cb]["val_ap"], "val_auroc": by_c[Cb]["val_auroc"],
             "val_ap_by_C": {str(c): v["val_ap"] for c, v in by_c.items()}}
        rows.append(r)
        print(f"[passive_fit] {key}: C {Cb:g} val AP {r['val_ap']:.4f} AUROC {r['val_auroc']:.4f} "
              f"({time.perf_counter()-t1:.0f}s)", flush=True)
        if r["kind"] == "passive" and (best is None or np.nan_to_num(r["val_ap"], nan=-1) > best[0]["val_ap"]):
            best = (r, std, by_c[Cb]["w"], by_c[Cb]["b"])
        del Xt, Xv
    r, std, w, b = best
    print(f"[passive_fit] picked {r['key']} C {r['C']:g} (val AP {r['val_ap']:.4f})", flush=True)

    # ---- score every set but train40k with the picked model
    a.out.mkdir(parents=True, exist_ok=True)
    tests = {}
    for z in sorted(a.states.glob("*.shard0.npz")):
        name = z.name[:-len(".shard0.npz")]
        if name == TRAIN:
            continue
        uids, y, kept = load_set(a.states, name)
        s = prob(std(load_x(a.states, name, r["key"])[kept], dev), w, b)
        ku = [u for u, k in zip(uids, kept) if k]
        with open(a.out / f"{name}.jsonl", "w") as f:
            for u, v in zip(ku, s):
                f.write(json.dumps({"uid": u, "p": float(v)}) + "\n")
        if name in TEST:
            yk = y[kept]
            m = yk >= 0
            tests[name] = {"ap": fast_ap(yk[m], s[m]), "auroc": auroc(yk[m], s[m]), "n": int(m.sum())}
            print(f"[passive_fit] {name}: AP {tests[name]['ap']:.4f} AUROC {tests[name]['auroc']:.4f}", flush=True)
        if not kept.all():
            print(f"[passive_fit] warning: {name} has {int((~kept).sum())} unkept steps (no score)", flush=True)
    if tests:
        print(f"[passive_fit] test mean AP over {len(tests)} sets: {np.mean([t['ap'] for t in tests.values()]):.4f}",
              flush=True)
    if a.per_layer_out is not None:
        a.per_layer_out.parent.mkdir(parents=True, exist_ok=True)
        a.per_layer_out.write_text(json.dumps({
            "states": str(a.states), "val_label": a.val_label, "grid": list(grid),
            "n_train": int(tr_m.sum()), "train_pos_rate": rate, "n_val": int(va_m.sum()),
            "val_pos_rate": float(yva.mean()), "layers": rows,
            "picked": {k: r[k] for k in ("key", "block", "C", "val_ap", "val_auroc")},
            "test": tests, "test_mean_ap": float(np.mean([t["ap"] for t in tests.values()])) if tests else None,
            "seconds": time.perf_counter() - t0}, indent=1))
    print(f"[passive_fit] done in {time.perf_counter()-t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
