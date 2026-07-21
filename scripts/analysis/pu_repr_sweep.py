"""progress_usefulness_v0 P2 sweep: within-prefix pair accuracy over layers x pools.

Fits the progress-minus-neutral direction d_l on TRAIN pairs and evaluates on
problem-disjoint VAL:
  within-prefix pair accuracy = mean_pairs[ d . (h_progress - h_neutral) > 0 ]  (primary)
  projection AUC              = AUC of d.h separating progress vs neutral items
with a random-direction null and a within-pair label-permutation null (both ~0.5).

Runs for each (layer, pool) in the encoded tensor and for raw + L2-normalized
features. Optionally restricts to the causally-confirmed forks from P1
(--confirmed_forks pu_confirmed_forks.jsonl) so the direction is learned on
progress steps that actually raise Qwen's solve-from-here, not the annotation
prior alone.

Inputs: encoded {pu_train,pu_val}_{h.npy,y.npy,meta.jsonl} from pu_encode.py, and
the pair files pu_{train,val}_pairs.jsonl from pu_build_pairs.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import read_jsonl  # noqa: E402

# --------------------------------------------------------------------------- #
# Pure core
# --------------------------------------------------------------------------- #

def fit_direction(dh_train: np.ndarray) -> np.ndarray:
    """Unit-norm mean difference direction from train pair deltas (n, H)."""
    d = dh_train.mean(axis=0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def pair_accuracy(dh: np.ndarray, d: np.ndarray) -> float:
    """Fraction of pairs with d . (h_progress - h_neutral) > 0."""
    if len(dh) == 0:
        return float("nan")
    return float((dh @ d > 0).mean())


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC (Mann-Whitney) of scores separating labels (1=progress,0=neutral)."""
    labels = np.asarray(labels)
    n1 = int(labels.sum())
    n0 = int(len(labels) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    r1 = ranks[labels == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def random_direction_accuracy(dh_val: np.ndarray, n_dirs: int, rng: np.random.Generator) -> float:
    """Mean within-prefix accuracy over random unit directions (chance baseline)."""
    if len(dh_val) == 0:
        return float("nan")
    accs = []
    for _ in range(n_dirs):
        r = rng.standard_normal(dh_val.shape[1])
        r /= np.linalg.norm(r) + 1e-12
        accs.append(float((dh_val @ r > 0).mean()))
    return float(np.mean(accs))


def permutation_accuracy(dh_val: np.ndarray, d: np.ndarray, n_perm: int,
                         rng: np.random.Generator) -> float:
    """Null: randomly flip which member is 'progress' per pair (sign flip of dh)."""
    if len(dh_val) == 0:
        return float("nan")
    accs = []
    base = dh_val @ d
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(dh_val))
        accs.append(float((base * signs > 0).mean()))
    return float(np.mean(accs))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def _load_split(enc_dir: Path, stem: str):
    H = np.load(enc_dir / f"{stem}_h.npy")            # (n, L, P, H)
    meta = read_jsonl(enc_dir / f"{stem}_meta.jsonl")
    uid2idx = {m["item_uid"]: i for i, m in enumerate(meta)}
    return H.astype(np.float32), meta, uid2idx


def _pair_rows(pairs_path: Path, uid2idx: dict, confirmed: set | None):
    rows = []
    for pr in read_jsonl(pairs_path):
        if confirmed is not None and pr["fork_id"] not in confirmed:
            continue
        if pr["progress_uid"] in uid2idx and pr["neutral_uid"] in uid2idx:
            rows.append((uid2idx[pr["progress_uid"]], uid2idx[pr["neutral_uid"]]))
    return rows


def _pool(H: np.ndarray, lj: int, pj: int, normalize: str) -> np.ndarray:
    v = H[:, lj, pj, :]
    if normalize == "l2":
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v


def run_cell(Htr, tr_pairs, Hva, va_pairs, lj, pj, normalize, rng):
    vtr = _pool(Htr, lj, pj, normalize)
    vva = _pool(Hva, lj, pj, normalize)
    dh_tr = np.array([vtr[a] - vtr[b] for a, b in tr_pairs])
    dh_va = np.array([vva[a] - vva[b] for a, b in va_pairs])
    d = fit_direction(dh_tr)
    # val items for AUC (progress=1, neutral=0), from the val pairs
    idx, lab = [], []
    for a, b in va_pairs:
        idx += [a, b]
        lab += [1, 0]
    scores = vva[idx] @ d
    return {
        "pair_acc_val": pair_accuracy(dh_va, d),
        "auc_val": auc_from_scores(scores, np.array(lab)),
        "pair_acc_train": pair_accuracy(dh_tr, d),
        "rand_dir_acc_val": random_direction_accuracy(dh_va, 200, rng),
        "perm_acc_val": permutation_accuracy(dh_va, d, 200, rng),
        "n_train_pairs": len(tr_pairs), "n_val_pairs": len(va_pairs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enc_dir", type=Path, required=True)
    ap.add_argument("--pairs_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--confirmed_forks", type=Path, default=None,
                    help="pu_confirmed_forks.jsonl to restrict to causally-confirmed forks")
    ap.add_argument("--normalize", choices=["raw", "l2", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    Htr, _, uid2idx_tr = _load_split(args.enc_dir, "pu_train")
    Hva, _, uid2idx_va = _load_split(args.enc_dir, "pu_val")
    man = json.loads((args.enc_dir / "pu_train_manifest.json").read_text())
    layers, pools = man["layer_indices"], man["pool_order"]

    confirmed = None
    if args.confirmed_forks is not None:
        confirmed = {r["fork_id"] for r in read_jsonl(args.confirmed_forks)}
    tr_pairs = _pair_rows(args.pairs_dir / "pu_train_pairs.jsonl", uid2idx_tr, confirmed)
    va_pairs = _pair_rows(args.pairs_dir / "pu_val_pairs.jsonl", uid2idx_va, confirmed)
    print(f"[pu-sweep] train_pairs={len(tr_pairs)} val_pairs={len(va_pairs)} "
          f"confirmed_only={confirmed is not None}", flush=True)

    variants = ["raw", "l2"] if args.normalize == "both" else [args.normalize]
    results: dict = {"layers": layers, "pools": pools, "cells": []}
    for norm in variants:
        for lj, li in enumerate(layers):
            for pj, pool in enumerate(pools):
                cell = run_cell(Htr, tr_pairs, Hva, va_pairs, lj, pj, norm, rng)
                cell.update({"layer": li, "pool": pool, "normalize": norm})
                results["cells"].append(cell)
                print(f"  L{li:>2} {pool:>4} {norm:>3} | pair_acc_val={cell['pair_acc_val']:.3f} "
                      f"auc={cell['auc_val']:.3f} rand={cell['rand_dir_acc_val']:.3f}", flush=True)
    (args.out_dir / "pu_sweep_results.json").write_text(json.dumps(results, indent=2))

    # best cell by val pair accuracy
    best = max(results["cells"], key=lambda c: (c["pair_acc_val"]
                                                if c["pair_acc_val"] == c["pair_acc_val"] else -1))
    print(f"[pu-sweep] BEST pair_acc_val={best['pair_acc_val']:.3f} "
          f"@ L{best['layer']} {best['pool']} {best['normalize']} (auc={best['auc_val']:.3f})",
          flush=True)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for norm in variants:
            grid = np.full((len(pools), len(layers)), np.nan)
            for c in results["cells"]:
                if c["normalize"] == norm:
                    grid[pools.index(c["pool"]), layers.index(c["layer"])] = c["pair_acc_val"]
            fig, ax = plt.subplots(figsize=(1.4 * len(layers) + 1.5, 1.0 * len(pools) + 1.5))
            im = ax.imshow(grid, vmin=0.5, vmax=max(0.55, np.nanmax(grid)), cmap="viridis")
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([f"L{li}" for li in layers])
            ax.set_yticks(range(len(pools)))
            ax.set_yticklabels(pools)
            for r in range(len(pools)):
                for cc in range(len(layers)):
                    ax.text(cc, r, f"{grid[r, cc]:.3f}", ha="center", va="center", color="w")
            ax.set_title(f"within-prefix pair accuracy ({norm})")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            out = args.out_dir / f"pu_sweep_pairacc_{norm}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"[pu-sweep] plot -> {out}", flush=True)


if __name__ == "__main__":
    main()
