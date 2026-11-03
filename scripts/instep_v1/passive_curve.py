#!/usr/bin/env python3
"""InStep V1 audit: is the passive linear baseline (B1) held back by its training budget or by
reading one layer?

B1 fits a logistic regression on 40k of the 167k DS training steps (train40k) at one text
state (picked on validation: passive_final), while InStep trains on all 167k steps and reads
every layer. Here, with passive_fit.py's estimator (standardize on the training draw, L2
logistic per C, C picked on validation AP):
  (1) learning curve: passive_final on random 5k / 10k / 20k / 40k subsets of train40k
      (nested, one fixed permutation; train40k is stored in file order, so its prefixes are not
      random);
  (2) multi-layer: the concatenation of the late text states (blocks 23..35 + final) on 40k.
For each: validation AP / AUROC at the picked C and the 8 test sets' mean PR-AUC.
Output: one JSON with every row.

    python scripts/instep_v1/passive_curve.py --states $SCRATCH/cot_mech/instep_v1/states/qwen3_8b \\
        --data <R>/data --val_label ds --out $OUT/analysis/passive_curve_qwen3_8b.json
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
from scripts.instep_v1.passive_fit import (GRID, TRAIN, Standardizer, fit_path, load_set, load_x,  # noqa: E402
                                           prob, val_labels)
from scripts.reprobe_cmp.analyze import TEST, auroc, fast_ap  # noqa: E402

CONCAT = ("passive_block23", "passive_block27", "passive_block31", "passive_block35", "passive_final")


def fit_eval(xs: dict, ys: dict, n: int | None, grid, dev) -> dict:
    """xs/ys: {"train", "val", <test sets>} arrays (rows already masked). Fits on the first n
    training rows, picks C on validation AP, scores the test sets."""
    Xtr = xs["train"] if n is None else xs["train"][:n]
    ytr = torch.from_numpy(ys["train"][:len(Xtr)].astype(np.float32)).to(dev)
    std = Standardizer(Xtr)
    Xt, Xv = std(Xtr, dev), std(xs["val"], dev)
    by_c = {}
    for C, w, b in fit_path(Xt, ytr, grid):
        s = prob(Xv, w, b)
        by_c[C] = (fast_ap(ys["val"], s), auroc(ys["val"], s), w, b)
    Cb = max(by_c, key=lambda c: (np.nan_to_num(by_c[c][0], nan=-1), -c))
    ap, au, w, b = by_c[Cb]
    test = {s: fast_ap(ys[s], prob(std(xs[s], dev), w, b)) for s in TEST}
    fitted = (std, w, b)
    return fitted, {"n_train": int(len(Xtr)), "C": Cb, "C_at_grid_edge": Cb in (min(grid), max(grid)),
            "val_ap": ap, "val_auroc": au, "val_ap_by_C": {str(c): v[0] for c, v in by_c.items()},
            "test_ap": test, "test_mean_ap": float(np.mean(list(test.values())))}


def write_scores(states: Path, keys: tuple[str, ...], fitted, out: Path, dev) -> list[str]:
    """rp-format scores {"uid", "p"} of a fitted model on every set in states but training
    draws (test sets, val, pb_*)."""
    std, w, b = fitted
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for z in sorted(states.glob("*.shard0.npz")):
        name = z.name[:-len(".shard0.npz")]
        if name.startswith(("train", "bon_")):
            continue
        uids, _, kept = load_set(states, name)
        X = np.concatenate([load_x(states, name, k)[kept] for k in keys], axis=1)
        p = prob(std(X, dev), w, b)
        with open(out / f"{name}.jsonl", "w") as f:
            for u, v in zip([u for u, k in zip(uids, kept) if k], p):
                f.write(json.dumps({"uid": u, "p": float(v)}) + "\n")
        names.append(name)
    return names


def load_xs(keys: tuple[str, ...], masks: dict, perm: np.ndarray) -> dict:
    """Masked states per split (training rows in the fixed random order perm). masks: {split:
    [(states dir, file name, row mask)]}, the parts of a split concatenated in order."""
    out = {nm: np.concatenate([np.concatenate([load_x(d, f, k)[m] for k in keys], axis=1) for d, f, m in parts])
           for nm, parts in masks.items()}
    out["train"] = out["train"][perm]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--val_label", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sizes", default="5000,10000,20000,40000", help="passive_final training sizes (0 = all)")
    p.add_argument("--concat_sizes", default="20000,0", help="concatenation training sizes (0 = all)")
    p.add_argument("--train_states", type=Path, default=None, help="dir of the training states (default --states)")
    p.add_argument("--train_names", default=TRAIN, help="comma list of training state files, concatenated")
    p.add_argument("--scores_out", type=Path, default=None,
                   help="write rp-format scores of the last passive_final and the last concatenation fit to "
                        "<dir>/final/ and <dir>/concat5/")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = GRID + (3e-5,)
    t0 = time.perf_counter()

    masks, ys = {"train": []}, {}
    tst, yl = a.train_states or a.states, []
    for nm in a.train_names.split(","):
        _, y, kept = load_set(tst, nm)
        m = kept & (y >= 0)
        masks["train"].append((tst, nm, m))
        yl.append(y[m])
    y = np.concatenate(yl)
    rate = float(y.mean())
    print(f"[curve] train {len(y)} labeled steps from {a.train_names}, positive rate {rate:.4f}", flush=True)
    if not 0.02 <= rate <= 0.2:
        raise SystemExit(f"train positive rate {rate:.4f} outside [0.02, 0.2]: wrong labels?")
    perm = np.random.default_rng(0).permutation(len(y))
    ys["train"] = y[perm]
    uids, _, kept = load_set(a.states, "val")
    vl = val_labels(a.data, a.val_label)
    m = np.array([k and u in vl for u, k in zip(uids, kept)])
    masks["val"], ys["val"] = [(a.states, "val", m)], np.array([vl[u] for u, mm in zip(uids, m) if mm])
    for s in TEST:
        _, y, kept = load_set(a.states, s)
        m = kept & (y >= 0)
        masks[s], ys[s] = [(a.states, s, m)], y[m]

    rows = []
    xs = load_xs(("passive_final",), masks, perm)
    for n in [int(v) or None for v in a.sizes.split(",")]:
        fitted, r = fit_eval(xs, ys, n, grid, dev)
        r = {"keys": ["passive_final"], **r}
        rows.append(r)
        print(f"[curve] passive_final n={r['n_train']}: C {r['C']:g} val AP {r['val_ap']:.4f} "
              f"test mean AP {r['test_mean_ap']:.4f}", flush=True)
    if a.scores_out is not None:
        print(f"[curve] scores: {write_scores(a.states, ('passive_final',), fitted, a.scores_out / 'final', dev)}",
              flush=True)
    del xs
    xs = load_xs(CONCAT, masks, perm)
    for n in [int(v) or None for v in a.concat_sizes.split(",")]:
        fitted, r = fit_eval(xs, ys, n, grid, dev)
        r = {"keys": list(CONCAT), **r}
        rows.append(r)
        print(f"[curve] concat{len(CONCAT)} n={r['n_train']}: C {r['C']:g} val AP {r['val_ap']:.4f} "
              f"test mean AP {r['test_mean_ap']:.4f}", flush=True)
    if a.scores_out is not None:
        print(f"[curve] scores: {write_scores(a.states, CONCAT, fitted, a.scores_out / 'concat5', dev)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"states": str(a.states), "train": f"{tst}:{a.train_names}", "val_label": a.val_label, "grid": list(grid),
                                 "rows": rows, "seconds": time.perf_counter() - t0}, indent=1))
    print(f"[curve] done in {time.perf_counter()-t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
