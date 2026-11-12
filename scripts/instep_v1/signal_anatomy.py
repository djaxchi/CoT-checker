#!/usr/bin/env python3
"""InStep paper, "where the signal lives": what each verifier's step representation is made of,
and which of its directions carry the correctness signal.

For each representation (default: the final key of each verifier):
  budget    share of the representation's variance (test steps of the 8 sets, each dimension
            standardized) explained by, in this order and each beyond the previous ones: the
            task (8 sets), the step's place in the trace (step index, log step index, relative
            position, first step), the step's length (log characters, log words), its
            correctness label, and the rest (multivariate least squares; sequential R^2)
  spectrum  PCA of the standardized train40k states; the same logistic probe (C by val AP)
            on the top k principal directions only, k = 1, 2, 4, ..., 256, and on every
            dimension: mean test AUROC over the 8 sets. Where the classifying signal sits in
            the representation's own variance ranking
  reliance  the full probe's test logit on CORRECT steps only, Spearman with step length,
            step index and relative position inside each set, mean over the 8 sets: what moves
            the score when correctness does not

    python scripts/instep_v1/signal_anatomy.py --states $ST8 --extra $SS/prm $SS/reprobe --data $D \\
        --out $OUT/signal
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
from scripts.instep_v1.passive_fit import GRID, Standardizer, prob  # noqa: E402
from scripts.instep_v1.signal_geometry import TRAIN, Data, fit_best, jl  # noqa: E402
from scripts.instep_v1.signal_pca import KEYS  # noqa: E402
from scripts.reprobe_cmp.analyze import TEST, auroc  # noqa: E402

KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FACTORS = ("task", "position", "length", "correctness")


def step_factors(D: Data, data: Path) -> dict[str, np.ndarray]:
    """Per kept test step (all 8 sets, in TEST order): set index, step index, relative
    position, characters and words of the step text."""
    out = {k: [] for k in ("task", "idx", "rel", "chars", "words")}
    for i, s in enumerate(TEST):
        uids = [u for u, m in zip(D.uids[s], D.mask[s]) if m]
        text = {r["uid"]: r["candidate_step"] for r in jl(data / "test" / f"{s}.items.jsonl")}
        last = {}
        for u in D.uids[s]:
            t, k = u.rsplit("::", 1)
            last[t] = max(last.get(t, 0), int(k))
        for u in uids:
            t, k = u.rsplit("::", 1)
            out["task"].append(i)
            out["idx"].append(int(k))
            out["rel"].append(int(k) / max(last[t], 1))
            out["chars"].append(len(text[u]))
            out["words"].append(len(text[u].split()))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def design(f: dict[str, np.ndarray], y: np.ndarray) -> dict[str, np.ndarray]:
    n = len(y)
    task = np.zeros((n, len(TEST)))
    task[np.arange(n), f["task"].astype(int)] = 1
    return {"task": task[:, 1:],
            "position": np.c_[f["idx"], np.log1p(f["idx"]), f["rel"], f["idx"] == 0],
            "length": np.c_[np.log1p(f["chars"]), np.log1p(f["words"])],
            "correctness": y[:, None].astype(float)}


def variance_budget(X: np.ndarray, blocks: dict[str, np.ndarray]) -> dict[str, float]:
    """Sequential multivariate R^2 of standardized X on the cumulative design, in FACTORS order."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    total = float((X ** 2).sum())
    out, cols, prev = {}, [np.ones((len(X), 1))], 0.0
    for name in FACTORS:
        cols.append(blocks[name])
        A = np.concatenate(cols, 1)
        coef, *_ = np.linalg.lstsq(A, X, rcond=None)
        r2 = 1 - float(((X - A @ coef) ** 2).sum()) / total
        out[name] = r2 - prev
        prev = r2
    out["rest"] = 1 - prev
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--extra", type=Path, nargs="*", default=[])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--val_label", default="ds")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keys", default=KEYS)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    D = Data(a.states, a.extra, a.data, a.val_label)
    f = step_factors(D, a.data)
    yt = np.concatenate([D.labels(s) for s in TEST])
    sets = np.concatenate([[s] * int(D.mask[s].sum()) for s in TEST])
    blocks = design(f, yt)
    res = []
    for key in a.keys.split(","):
        t0 = time.perf_counter()
        Xte = np.concatenate([D.x(s, key) for s in TEST])
        budget = variance_budget(Xte.astype(np.float64), blocks)
        Xtr, ytr = D.x(TRAIN, key), D.labels(TRAIN)
        std = Standardizer(Xtr)
        Xt = std(Xtr, "cpu")
        del Xtr
        yt_t = torch.from_numpy(ytr.astype(np.float32))
        Xv, yv = std(D.x("val", key), "cpu"), D.labels("val")
        Xs = std(Xte, "cpu")
        _, _, V = torch.pca_lowrank(Xt, q=min(max(KS), Xt.shape[1]), center=False, niter=4)
        spec = {}
        for k in [k for k in KS if k <= V.shape[1]] + ["all"]:
            P = V if k == "all" else V[:, :k]
            tr, va, te = (Xt, Xv, Xs) if k == "all" else (Xt @ P, Xv @ P, Xs @ P)
            _, C, w, b = fit_best(tr, yt_t, va, yv, GRID)
            s = prob(te, w, b)
            spec[str(k)] = float(np.mean([auroc(yt[sets == x], s[sets == x]) for x in TEST]))
            if k == "all":
                logit = (te @ w + b).numpy()
        from scipy.stats import spearmanr
        rel = {n: float(np.nanmean([spearmanr(logit[(sets == x) & (yt == 0)], v[(sets == x) & (yt == 0)]).correlation
                                    for x in TEST]))
               for n, v in (("chars", f["chars"]), ("step_idx", f["idx"]), ("rel_position", f["rel"]))}
        r = {"key": key, "budget": budget, "spectrum_auroc": spec, "reliance_correct_steps": rel}
        res.append(r)
        print(f"[anatomy] {key:14s} budget " + " ".join(f"{n} {v:.3f}" for n, v in budget.items())
              + " | spectrum " + " ".join(f"{k}:{v:.3f}" for k, v in spec.items())
              + " | reliance " + " ".join(f"{n} {v:+.2f}" for n, v in rel.items())
              + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        del Xt, Xv, Xs, Xte
    (a.out / "signal_anatomy.json").write_text(json.dumps({"n_test": int(len(yt)), "keys": res}, indent=1))


if __name__ == "__main__":
    main()
