"""Assemble per-layer hidden states into one concatenated, standardized feature matrix.

This is the single new piece needed to repeat the S1 small-scale DenseLinear experiment
(probe_train_40k / val_1k -> ProcessBench PB-F1) with ALL layers combined instead of just
the last layer. The multilayer encoders write one file per layer:

    {stem}_L{idx}_h.npy   (n, hidden)   float16     one per selected layer
    {stem}_y.npy          (n,)                       labels (PRM splits)
    {stem}_meta.jsonl / pb_step_meta.jsonl           row metadata

We concatenate the selected layers column-wise into a single {stem}_h.npy of shape
(n, n_layers * hidden) and z-score every dimension. Standardization is REQUIRED here: the
DenseLinear probe (scripts/s1ms_train_dense_probe.py) trains on raw hidden states with SGD,
and concatenating layers whose norms differ by ~100x (deep layers, massive-activation dims)
would otherwise swamp the gradient. The z-score statistics are fit ONCE on the probe-train
split and reused verbatim for val and every ProcessBench subset, so train and eval live in
the same feature space (the evaluator infers in_dim from the probe weight, no other change).

After this runs, the existing stages are untouched: s1ms_train_dense_probe.py reads the
concatenated {stem}_h.npy, and evaluate_saved_probe_on_processbench.py reads the
concatenated pb_step_h.npy. Faithful to the original pipeline, only wider.

Modes:
  fit    read the train stem's per-layer files, concat, fit z-score, save scaler + {stem}_h.npy
  apply  read a split's per-layer files, concat, apply the saved scaler, save {stem}_h.npy

Usage:
  python scripts/assemble_multilayer_concat.py --mode fit \
      --in_dir CACHE --stem probe_train_40k --out_dir ASSEMBLED --scaler ASSEMBLED/scaler.npz
  python scripts/assemble_multilayer_concat.py --mode apply \
      --in_dir CACHE --stem val_1k --out_dir ASSEMBLED --scaler ASSEMBLED/scaler.npz
  python scripts/assemble_multilayer_concat.py --mode apply \
      --in_dir PB/gsm8k --stem pb_step --out_dir ASSEMBLED_PB/gsm8k --scaler ASSEMBLED/scaler.npz
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

LAYER_RE = re.compile(r"_L(\d+)_h\.npy$")


def discover_layers(in_dir: Path, stem: str) -> list[int]:
    """Return the sorted layer indices for which {stem}_L{idx}_h.npy exists."""
    layers = []
    for p in in_dir.glob(f"{stem}_L*_h.npy"):
        m = LAYER_RE.search(p.name)
        if m:
            layers.append(int(m.group(1)))
    return sorted(layers)


def load_concat(in_dir: Path, stem: str, layers: list[int]) -> np.ndarray:
    """Column-concatenate the per-layer matrices in ascending-layer order, float32."""
    mats = []
    n_ref = None
    for li in layers:
        f = in_dir / f"{stem}_L{li}_h.npy"
        if not f.exists():
            sys.exit(f"[assemble] missing layer file: {f}")
        a = np.load(f, mmap_mode="r")
        if n_ref is None:
            n_ref = a.shape[0]
        elif a.shape[0] != n_ref:
            sys.exit(f"[assemble] row mismatch in {f}: {a.shape[0]} vs {n_ref}")
        mats.append(np.asarray(a, dtype=np.float32))
    return np.concatenate(mats, axis=1)


def copy_sidecars(in_dir: Path, stem: str, out_dir: Path) -> None:
    """Copy labels and metadata next to the assembled features, unchanged."""
    for name in (f"{stem}_y.npy", f"{stem}_meta.jsonl", "pb_step_meta.jsonl"):
        src = in_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["fit", "apply"], required=True)
    p.add_argument("--in_dir", type=Path, required=True)
    p.add_argument("--stem", type=str, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--scaler", type=Path, required=True,
                   help="npz written in fit mode, read in apply mode.")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Explicit layer indices. Default: every {stem}_L*_h.npy present.")
    p.add_argument("--no_standardize", action="store_true",
                   help="Concatenate without z-scoring (faithfulness ablation).")
    p.add_argument("--save_dtype", choices=["float16", "float32"], default="float32")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    layers = args.layers if args.layers else discover_layers(args.in_dir, args.stem)
    if not layers:
        sys.exit(f"[assemble] no per-layer files {args.stem}_L*_h.npy under {args.in_dir}")
    X = load_concat(args.in_dir, args.stem, layers)
    print(f"[assemble] {args.mode} stem={args.stem} layers={layers} -> X{X.shape}", flush=True)

    if args.no_standardize:
        mean = np.zeros(X.shape[1], np.float32); std = np.ones(X.shape[1], np.float32)
    elif args.mode == "fit":
        mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        np.savez(args.scaler, mean=mean, std=std, layers=np.asarray(layers, np.int64),
                 hidden_per_layer=np.int64(X.shape[1] // len(layers)))
        print(f"[assemble] fit scaler -> {args.scaler} (dim={X.shape[1]})", flush=True)
    else:
        if not args.scaler.exists():
            sys.exit(f"[assemble] scaler not found (run fit first): {args.scaler}")
        z = np.load(args.scaler)
        if list(z["layers"]) != list(layers):
            sys.exit(f"[assemble] layer mismatch: scaler {list(z['layers'])} vs {layers}")
        mean, std = z["mean"], z["std"]
        if mean.shape[0] != X.shape[1]:
            sys.exit(f"[assemble] dim mismatch: scaler {mean.shape[0]} vs X {X.shape[1]}")

    Xs = ((X - mean) / std).astype(np.float16 if args.save_dtype == "float16" else np.float32)
    out_h = args.out_dir / f"{args.stem}_h.npy"
    np.save(out_h, Xs)
    copy_sidecars(args.in_dir, args.stem, args.out_dir)
    (args.out_dir / f"{args.stem}_assemble.json").write_text(json.dumps(
        {"stem": args.stem, "mode": args.mode, "layers": layers,
         "n": int(Xs.shape[0]), "dim": int(Xs.shape[1]),
         "standardized": not args.no_standardize, "save_dtype": args.save_dtype}, indent=2))
    print(f"[assemble] wrote {out_h}  shape={Xs.shape}  dtype={Xs.dtype}", flush=True)


if __name__ == "__main__":
    main()
