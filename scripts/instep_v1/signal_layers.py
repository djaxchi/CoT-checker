#!/usr/bin/env python3
"""InStep paper, Figure "where the signal forms": the shared linear probe of signal_geometry.py
(standardize, L2 logistic regression, C by validation AP, fitted on train40k with DeepSeek-R1
labels) on every per-layer state of several state directories, on one joint step mask. Only the
probe; none of signal_geometry.py's geometry measures.

Sources are --src name=dir[=regex]. Every directory holds <set>.shard0.npz + .meta.jsonl for
train40k, val and the 8 ReProbe test sets, with the same uids in the same order as the first
source. A step counts when it is kept in every source and labelled (val labels from
rp/val.traces.jsonl, test and train labels from the first source's npz). Keys are the npz
arrays matching the source's regex (default: every state key); arrays named p_block<l> (a
head's own P(incorrect), reprobe_layers.py) are scored directly, without a probe. For the sources
named in --mlp_src, every key is also read by STEP's own scorer (step_probe.py: MLP d -> 512 -> 1
on the raw states, its recipe and early stopping on validation loss), once per --mlp_seeds seed.

Output <out>: one JSON {"n": {set: steps}, "rows": [{"src", "key", "kind": probe|own|step_mlp, "seed", "C",
"val_ap", "test": {set: {"ap", "auroc"}}, "mean_ap", "mean_auroc"}]}.

    python scripts/instep_v1/signal_layers.py --data $D --out $OUT/signal/signal_layers.json \\
        --src s0=$ST8 prm=$SS/prm rp=$SS/reprobe rpl0=$SS/reprobe_layers/seed0='rp_block\\d+|p_block\\d+' ...
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
from scripts.instep_v1.passive_fit import GRID, Standardizer, val_labels  # noqa: E402
from scripts.instep_v1.signal_geometry import TRAIN, fit_best, jl  # noqa: E402
from scripts.instep_v1.step_probe import DEFAULTS, predict, train_seed  # noqa: E402
from scripts.reprobe_cmp.analyze import TEST, auroc, fast_ap  # noqa: E402

STATE = r"(passive|read|prm|rp)_block\d+|passive_final|sep|prm_final|reprobe_rep|p_block\d+"


def parse_src(spec: str) -> tuple[str, Path, str]:
    name, rest = spec.split("=", 1)
    d, _, rx = rest.partition("=")
    return name, Path(d), rx or STATE


def joint_mask(srcs: list[tuple[str, Path, str]], data: Path, val_label: str) -> tuple[dict, dict]:
    """({set: y}, {set: bool mask}) on the first source's uid order."""
    vl = val_labels(data, val_label)
    ys, masks = {}, {}
    for s in [TRAIN, "val", *TEST]:
        d0 = srcs[0][1]
        uids = [r["uid"] for r in jl(d0 / f"{s}.shard0.meta.jsonl")]
        with np.load(d0 / f"{s}.shard0.npz") as z:
            y, kept = z["y"].astype(int), z["kept"].astype(bool)
        if s == "val":
            y = np.array([vl.get(u, -1) for u in uids])
        for name, d, _ in srcs[1:]:
            if [r["uid"] for r in jl(d / f"{s}.shard0.meta.jsonl")] != uids:
                raise SystemExit(f"{name} {s}: uids differ from {d0}")
            with np.load(d / f"{s}.shard0.npz") as z:
                kept &= z["kept"].astype(bool)
        ys[s], masks[s] = y, kept & (y >= 0)
    return ys, masks


def load(d: Path, s: str, key: str, mask: np.ndarray) -> np.ndarray:
    with np.load(d / f"{s}.shard0.npz") as z:
        a = z[key]
    return np.nan_to_num(a.reshape(len(a), -1).astype(np.float32))[mask]


def score(src: str, d: Path, key: str, ys: dict, masks: dict, grid, dev) -> dict:
    t0 = time.perf_counter()
    if key.startswith("p_block"):  # a head's own score: no probe
        row = {"src": src, "key": key, "kind": "own", "C": None,
               "val_ap": fast_ap(ys["val"][masks["val"]], load(d, "val", key, masks["val"])[:, 0])}
        pred = {s: load(d, s, key, masks[s])[:, 0] for s in TEST}
    else:
        Xtr = load(d, TRAIN, key, masks[TRAIN])
        std = Standardizer(Xtr)
        yt = torch.from_numpy(ys[TRAIN][masks[TRAIN]].astype(np.float32)).to(dev)
        vap, C, w, b = fit_best(std(Xtr, dev), yt, std(load(d, "val", key, masks["val"]), dev),
                                ys["val"][masks["val"]], grid)
        del Xtr
        row = {"src": src, "key": key, "kind": "probe", "C": C, "val_ap": vap}
        pred = {s: (std(load(d, s, key, masks[s]), dev) @ w + b).cpu().numpy() for s in TEST}
    row["test"] = {s: {"ap": fast_ap(ys[s][masks[s]], p), "auroc": auroc(ys[s][masks[s]], p)} for s, p in pred.items()}
    row["mean_ap"] = float(np.mean([v["ap"] for v in row["test"].values()]))
    row["mean_auroc"] = float(np.mean([v["auroc"] for v in row["test"].values()]))
    print(f"[layers] {src} {key}: mean PR-AUC {row['mean_ap']:.4f} AUROC {row['mean_auroc']:.4f} "
          f"(val AP {row['val_ap']:.4f}, C {row['C']}, {time.perf_counter() - t0:.0f}s)", flush=True)
    return row


def score_mlp(src: str, d: Path, key: str, ys: dict, masks: dict, seed: int, dev) -> dict:
    t0 = time.perf_counter()
    Xtr = torch.from_numpy(load(d, TRAIN, key, masks[TRAIN])).to(dev)
    ytr = torch.from_numpy(ys[TRAIN][masks[TRAIN]].astype(np.float32)).to(dev)
    Xva = torch.from_numpy(load(d, "val", key, masks["val"])).to(dev)
    yva = torch.from_numpy(ys["val"][masks["val"]].astype(np.float32)).to(dev)
    m, info = train_seed(Xtr, ytr, Xva, yva, seed, argparse.Namespace(**DEFAULTS))
    del Xtr
    row = {"src": src, "key": key, "kind": "step_mlp", "seed": seed, "C": None, "best_epoch": info["best_epoch"],
           "val_ap": fast_ap(yva.cpu().numpy(), predict(m, Xva).cpu().numpy())}
    pred = {s: predict(m, torch.from_numpy(load(d, s, key, masks[s])).to(dev)).cpu().numpy() for s in TEST}
    row["test"] = {s: {"ap": fast_ap(ys[s][masks[s]], p), "auroc": auroc(ys[s][masks[s]], p)} for s, p in pred.items()}
    row["mean_ap"] = float(np.mean([v["ap"] for v in row["test"].values()]))
    row["mean_auroc"] = float(np.mean([v["auroc"] for v in row["test"].values()]))
    print(f"[layers] {src} {key} STEP MLP seed {seed}: mean PR-AUC {row['mean_ap']:.4f} "
          f"(best epoch {info['best_epoch']}, {time.perf_counter() - t0:.0f}s)", flush=True)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", nargs="+", required=True, help="name=dir[=key regex]; the first fixes the uid order")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--val_label", default="ds")
    p.add_argument("--grid", default=",".join(map(str, GRID)))
    p.add_argument("--mlp_src", default="", help="comma list of source names also read by STEP's MLP")
    p.add_argument("--mlp_seeds", default="0,1,2")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    srcs = [parse_src(s) for s in a.src]
    ys, masks = joint_mask(srcs, a.data, a.val_label)
    n = {s: int(m.sum()) for s, m in masks.items()}
    print(f"[layers] steps: {n}", flush=True)
    grid = tuple(float(c) for c in a.grid.split(","))
    rows = []
    for name, d, rx in srcs:
        with np.load(d / "val.shard0.npz") as z:
            keys = [k for k in z.files if re.fullmatch(rx, k) and re.fullmatch(STATE, k)]
        for k in sorted(keys, key=lambda k: (re.sub(r"\d+$", "", k), int(re.search(r"(\d*)$", k).group(1) or 99))):
            rows.append(score(name, d, k, ys, masks, grid, dev))
            if name in a.mlp_src.split(",") and not k.startswith("p_block"):
                rows += [score_mlp(name, d, k, ys, masks, int(sd), dev) for sd in a.mlp_seeds.split(",")]
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps({"src": {nm: str(d) for nm, d, _ in srcs}, "n": n, "rows": rows}, indent=1))
    print(f"[layers] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
