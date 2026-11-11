#!/usr/bin/env python3
"""InStep paper, "where the signal lives": the same linear readout and the same geometry
measures on every verifier's per-step representation, Qwen3-8B, ReProbe's DS data.

Representations (state keys; every one aligned row by row on the same steps):
  text     Qwen3-8B's state at the step's last token, every 4th block + final
           (`passive_block<l>`, `passive_final`: what ReProbe's head and the passive probe read)
  instep   the InStep step token's state, every 4th block (`read_block<l>`) + the final
           post-norm state its linear head reads (`sep`)            [--states, DS seed 0]
  prm      Qwen2.5-Math-PRM-7B at the step's separator, blocks + the state its head reads
           (`prm_block<l>`, `prm_final`)                            [--extra, signal_states.py]
  reprobe  ReProbe's released DS head, its pooled step representation (`reprobe_rep`)

For each key, fitted on train40k only:
  probe    standardize, L2 logistic regression, C picked by validation AP (passive_fit.py's
           solver and grid); test PR-AUC and AUROC per set, means over ID / OOD / all 8
  axis     the raw class-mean difference (train, unstandardized): test PR-AUC / AUROC of the
           projection on it, and the share of the total variance along it
  pcs      the top --n_pcs principal directions of the raw train states: test AUROC of each
           (sign fixed on train), the best one's index (1-based) and AUROC
For the final key of each method (--final): the held-out 2D view (x = probe logit, y = first
principal direction of the standardized states orthogonal to the probe weights), per-class
histograms of x, and the in-set ceiling (5-fold trace-grouped CV of the same probe inside each
test set) against the train-fit probe.

Outputs <out>/signal_geometry.json and <out>/points_<key>.csv (set, y, x, y2) with every
incorrect test step and --n_correct sampled correct ones per set.

    python scripts/instep_v1/signal_geometry.py --states $ST/qwen3_8b --extra $SS/prm $SS/reprobe \\
        --data $D --out $OUT/signal
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
from scripts.instep_v1.passive_fit import GRID, Standardizer, fit_path, prob, val_labels  # noqa: E402
from scripts.reprobe_cmp.analyze import TEST, auroc, fast_ap  # noqa: E402

ID = ("math", "gsm8k", "proofnet")
TRAIN = "train40k"
FINAL = ("passive_final", "sep", "prm_final", "reprobe_rep")
DEPTH = {"passive": 36, "read": 36, "prm": 28}  # blocks per backbone


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def family(key: str) -> str:
    if key.startswith("passive"):
        return "text"
    if key.startswith("read") or key == "sep":
        return "instep"
    return "prm" if key.startswith("prm") else "reprobe"


def depth(key: str) -> float:
    m = re.search(r"(passive|read|prm)_block(\d+)$", key)
    return (int(m.group(2)) + 1) / DEPTH[m.group(1)] if m else 1.0


def keys_in(d: Path) -> list[str]:
    with np.load(d / "val.shard0.npz") as z:
        return [k for k in z.files if re.fullmatch(r"(passive|read|prm)_block\d+|passive_final|sep|prm_final|reprobe_rep", k)]


def mean_groups(per: dict, field: str) -> dict:
    out = {}
    for g, ss in (("id", ID), ("ood", tuple(s for s in TEST if s not in ID)), ("all", TEST)):
        v = [per[s][field] for s in ss if s in per]
        out[g] = float(np.mean(v)) if len(v) == len(ss) else None
    return out


class Data:
    """Labels and joint masks, the same for every key: a step counts when it is kept in every
    state source and labelled (val labels from the traces, as passive_fit.py)."""

    def __init__(self, states: Path, extra: list[Path] | None, data: Path, val_label: str):
        self.dirs = [states] + list(extra or [])
        self.where = {}
        for d in self.dirs:
            for k in keys_in(d):
                if k in self.where:
                    raise SystemExit(f"state key {k} in both {self.where[k]} and {d}")
                self.where[k] = d
        self.sets = [TRAIN, "val", *TEST]
        self.uids, self.y, self.mask = {}, {}, {}
        vl = val_labels(data, val_label)
        for s in self.sets:
            uids = [r["uid"] for r in jl(states / f"{s}.shard0.meta.jsonl")]
            with np.load(states / f"{s}.shard0.npz") as z:
                y, kept = z["y"].astype(int), z["kept"].astype(bool)
            if s == "val":
                y = np.array([vl.get(u, -1) for u in uids])
            for d in self.dirs[1:]:
                if [r["uid"] for r in jl(d / f"{s}.shard0.meta.jsonl")] != uids:
                    raise SystemExit(f"{d}/{s}: uids differ from {states}/{s}")
                with np.load(d / f"{s}.shard0.npz") as z:
                    kept &= z["kept"].astype(bool)
            self.uids[s], self.y[s], self.mask[s] = uids, y, kept & (y >= 0)

    def x(self, s: str, key: str) -> np.ndarray:
        with np.load(self.where[key] / f"{s}.shard0.npz") as z:
            a = z[key]
        return np.nan_to_num(a.reshape(len(a), -1).astype(np.float32))[self.mask[s]]

    def labels(self, s: str) -> np.ndarray:
        return self.y[s][self.mask[s]]

    def traces(self, s: str) -> np.ndarray:
        return np.array([u.rsplit("::", 1)[0] for u, m in zip(self.uids[s], self.mask[s]) if m])


def fit_best(Xt: torch.Tensor, ytr: torch.Tensor, Xv: torch.Tensor, yva: np.ndarray, grid) -> tuple:
    best = None
    for C, w, b in fit_path(Xt, ytr, grid):
        ap = fast_ap(yva, prob(Xv, w, b))
        if best is None or np.nan_to_num(ap, nan=-1) > best[0]:
            best = (ap, C, w.detach().clone(), b.detach().clone())
    return best


def in_set_ceiling(X: torch.Tensor, y: np.ndarray, tr: np.ndarray, C: float, folds: int = 5) -> float:
    """5-fold CV AP inside one set, folds by trace, the same probe at the train-picked C."""
    ut = np.unique(tr)
    rng = np.random.default_rng(0)
    f_of = dict(zip(ut, rng.permutation(len(ut)) % folds))
    fold = np.array([f_of[t] for t in tr])
    s = np.zeros(len(y))
    for f in range(folds):
        te, trn = fold == f, fold != f
        if y[trn].sum() == 0:
            return float("nan")
        tt, ts = torch.from_numpy(trn).to(X.device), torch.from_numpy(te).to(X.device)
        mu, sd = X[tt].mean(0), X[tt].std(0) + 1e-6
        Xa = (X - mu) / sd
        (_, w, b), = fit_path(Xa[tt], torch.from_numpy(y[trn].astype(np.float32)).to(X.device), (C,))
        s[te] = prob(Xa[ts], w, b)
    return fast_ap(y, s)


def analyze_key(D: Data, key: str, grid, n_pcs: int, dev, final: bool, n_correct: int, out: Path) -> dict:
    t0 = time.perf_counter()
    Xtr, ytr = D.x(TRAIN, key), D.labels(TRAIN)
    std = Standardizer(Xtr)
    Xt = std(Xtr, dev)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    Xv, yva = std(D.x("val", key), dev), D.labels("val")
    vap, C, w, b = fit_best(Xt, yt, Xv, yva, grid)
    # raw geometry on the train states
    mu = Xtr.mean(0)
    Xr = torch.from_numpy(Xtr - mu).to(dev)
    dvec = Xr[yt == 1].mean(0) - Xr[yt == 0].mean(0)
    u = dvec / dvec.norm()
    total_var = float((Xr ** 2).sum() / len(Xr))
    var_axis = float(((Xr @ u) ** 2).mean())
    _, S, V = torch.pca_lowrank(Xr, q=n_pcs, center=False, niter=4)
    pc_var = (S ** 2 / len(Xr)).cpu().numpy()
    sign = torch.sign((Xr[yt == 1] @ V).mean(0) - (Xr[yt == 0] @ V).mean(0))
    # 2D view: first principal direction of the standardized train states orthogonal to w
    wu = w / w.norm()
    Xo = Xt - (Xt @ wu)[:, None] * wu[None]
    _, _, Vo = torch.pca_lowrank(Xo, q=1, center=True, niter=4)
    v2 = Vo[:, 0]
    del Xr, Xo, Xt, Xv
    per, rows = {}, []
    rng = np.random.default_rng(0)
    for s in TEST:
        X, y = D.x(s, key), D.labels(s)
        Xs = std(X, dev)
        logit = (Xs @ w + b).cpu().numpy()
        Xraw = torch.from_numpy(X - mu).to(dev)
        ax = (Xraw @ u).cpu().numpy()
        pcs = ((Xraw @ V) * sign).cpu().numpy()
        r = {"ap": fast_ap(y, logit), "auroc": auroc(y, logit), "axis_ap": fast_ap(y, ax), "axis_auroc": auroc(y, ax),
             "pc_auroc": [auroc(y, pcs[:, j]) for j in range(pcs.shape[1])], "n": int(len(y)), "pos": float(y.mean())}
        if final:
            r["ceiling_ap"] = in_set_ceiling(torch.from_numpy(X).to(dev), y, D.traces(s), C)
            y2 = (Xs @ v2).cpu().numpy()
            idx = np.flatnonzero(y == 1).tolist()
            neg = np.flatnonzero(y == 0)
            idx += rng.choice(neg, min(n_correct, len(neg)), replace=False).tolist()
            rows += [(s, int(y[i]), float(logit[i]), float(y2[i])) for i in sorted(idx)]
            r["logit_by_class"] = {"0": logit[y == 0].tolist(), "1": logit[y == 1].tolist()}
        per[s] = r
        del Xs, Xraw
    pc_mean = np.mean([per[s]["pc_auroc"] for s in TEST], 0)
    res = {"key": key, "family": family(key), "depth": depth(key), "C": C, "val_ap": vap, "dim": int(Xtr.shape[1]),
           "ap": mean_groups(per, "ap"), "auroc": mean_groups(per, "auroc"),
           "axis_ap": mean_groups(per, "axis_ap"), "axis_auroc": mean_groups(per, "axis_auroc"),
           "axis_var_share": var_axis / total_var, "pc_var_share": (pc_var / total_var).tolist(),
           "pc_auroc_mean": pc_mean.tolist(), "best_pc": int(np.argmax(pc_mean)) + 1,
           "best_pc_auroc": float(pc_mean.max()), "per_set": per}
    if final:
        res["ceiling_ap"] = mean_groups(per, "ceiling_ap")
        with open(out / f"points_{key}.csv", "w") as f:
            f.write("set,y,x,y2\n")
            for row in rows:
                f.write("%s,%d,%.4f,%.4f\n" % row)
    print(f"[signal] {key:16s} depth {res['depth']:.2f} C {C:g} val AP {vap:.3f} | test AP {res['ap']['all']:.3f} "
          f"(ID {res['ap']['id']:.3f} OOD {res['ap']['ood']:.3f}) AUROC {res['auroc']['all']:.3f} | axis AP "
          f"{res['axis_ap']['all']:.3f} var share {res['axis_var_share']:.4f} | best PC {res['best_pc']} "
          f"AUROC {res['best_pc_auroc']:.3f}" + (f" | in-set ceiling {res['ceiling_ap']['all']:.3f}" if final else "")
          + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", type=Path, required=True, help="InStep encode states (text + step token)")
    p.add_argument("--extra", type=Path, nargs="*", default=[], help="signal_states.py output dirs (prm/, reprobe/)")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--val_label", default="ds")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keys", default=None, help="comma list (default: every key found)")
    p.add_argument("--grid", default=",".join(map(str, GRID)))
    p.add_argument("--n_pcs", type=int, default=20)
    p.add_argument("--n_correct", type=int, default=400, help="sampled correct test steps per set in the points")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = Data(a.states, a.extra, a.data, a.val_label)
    keys = a.keys.split(",") if a.keys else sorted(D.where, key=lambda k: (family(k), depth(k)))
    print(f"[signal] {len(keys)} keys; steps: " + ", ".join(f"{s} {int(D.mask[s].sum())}" for s in D.sets), flush=True)
    res = [analyze_key(D, k, tuple(float(c) for c in a.grid.split(",")), a.n_pcs, dev, k in FINAL, a.n_correct, a.out)
           for k in keys]
    (a.out / "signal_geometry.json").write_text(json.dumps(
        {"states": str(a.states), "extra": [str(x) for x in a.extra], "n": {s: int(D.mask[s].sum()) for s in D.sets},
         "keys": res}, indent=1))
    print(f"[signal] -> {a.out / 'signal_geometry.json'}", flush=True)


if __name__ == "__main__":
    main()
