"""Bring the numbers the probe sees down to a sane size.

Every step is stored as 4,096 numbers. On the layer we read now those numbers
swing by about +-22; on the previous backbone's layer they swung by about +-4.4,
because that layer had a normalisation step after it and this one does not. The
probe multiplies them by its weights, adds them up, and squashes the total into a
0-1 score. Five times bigger inputs give a roughly five times bigger total, so
the squash pins to the ends: half the scores land at 0.0000 or 0.9999.

That does not change which steps the probe ranks as more suspicious (AUROC held
at 0.866 on the seeds whose F1 collapsed), but it wrecks cutoff selection, and it
makes the scores meaningless for anything that needs a real confidence rather
than a ranking -- stopping a trace early, for instance.

The fix is arithmetic: for each of the positions, subtract its average across the
training split and divide by its swing, so every position sits near zero and
swings by about 1.

Sparse codes are divided but not centred. Subtracting a mean from a vector that
is 99% zeros makes it 100% non-zero and the storage argument collapses, so those
keep their zeros and only get scaled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

EPS = 1e-6


def fit(x, sample: int = 200_000, seed: int = 0, center: bool = True) -> dict:
    """Per-position average and swing, from a sample of rows.

    A sample is enough: these are summary statistics over half a million rows,
    and reading the whole 157 GiB store to compute them would cost more than the
    training it is meant to help.
    """
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= sample else np.sort(rng.choice(n, sample, replace=False))
    chunk = np.asarray(x[idx], dtype=np.float32)
    mean = chunk.mean(0) if center else np.zeros(chunk.shape[1], np.float32)
    std = chunk.std(0)
    std[std < EPS] = 1.0            # a dead position must not become inf
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
            "center": bool(center), "rows": int(len(idx))}


def apply(x: np.ndarray, stats: dict) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if stats["center"]:
        x = x - stats["mean"]
    return x / stats["std"]


def apply_sparse(values: np.ndarray, indices: np.ndarray, stats: dict) -> np.ndarray:
    """Scale only, indexed by feature, so the zeros stay zero."""
    return np.asarray(values, dtype=np.float32) / stats["std"][indices]


def save(path: str | Path, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=stats["mean"], std=stats["std"],
             center=np.array([stats["center"]]), rows=np.array([stats["rows"]]))


def load(path: str | Path) -> dict:
    z = np.load(Path(path))
    return {"mean": z["mean"], "std": z["std"],
            "center": bool(z["center"][0]), "rows": int(z["rows"][0])}


def describe(stats: dict, x_sample: np.ndarray | None = None) -> str:
    s = (f"center={stats['center']} fitted on {stats['rows']:,} rows; "
         f"swing before: median {np.median(stats['std']):.2f}, "
         f"max {stats['std'].max():.2f}")
    if x_sample is not None:
        after = apply(x_sample, stats)
        s += f"; after: std {after.std():.3f}"
    return s


def fit_sparse(indices: np.ndarray, values: np.ndarray, n_rows: int, d: int) -> dict:
    """Per-feature swing of a CSR store, counting the zeros.

    A feature that fires on 1% of steps has a small swing across the corpus, and
    that is the number to divide by. Computing it from the non-zeros alone would
    ignore the 99% of rows where the feature is absent and badly understate it.
    """
    s = np.bincount(indices, weights=values.astype(np.float64), minlength=d)
    sq = np.bincount(indices, weights=values.astype(np.float64) ** 2, minlength=d)
    mean = s / max(n_rows, 1)
    var = np.maximum(sq / max(n_rows, 1) - mean ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)
    std[std < EPS] = 1.0
    return {"mean": np.zeros(d, np.float32), "std": std, "center": False,
            "rows": int(n_rows)}


# ---------------------------------------------------------------------------
# Full whitening
# ---------------------------------------------------------------------------
# zscore divides each position by its own swing and ignores how positions move
# together. Whitening removes those correlations too, so every direction ends up
# with equal variance. That matters here because the correctness signal is a
# low-variance direction: a bottleneck that allocates capacity by variance
# discards it, and after whitening there is no variance ordering left to
# discriminate against it.
#
# It is not free. Whitening amplifies every low-variance direction, noise
# included, so it raises the signal's share of the budget without raising its
# share of the signal. Shrinkage toward the diagonal keeps that from running away
# on directions the sample barely constrains.

def fit_whiten(x, sample: int = 200_000, seed: int = 0, shrinkage: float = 0.05) -> dict:
    """Mean and a whitening matrix W with W @ cov @ W.T ~ I.

    `shrinkage` mixes the covariance toward its diagonal before inverting, which
    stops directions the sample barely pins down from being blown up.
    """
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= sample else np.sort(rng.choice(n, sample, replace=False))
    chunk = np.asarray(x[idx], dtype=np.float64)
    mean = chunk.mean(0)
    c = chunk - mean
    cov = (c.T @ c) / max(len(c) - 1, 1)
    if shrinkage > 0:
        cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, EPS)
    W = (vecs * (vals ** -0.5)) @ vecs.T          # symmetric inverse square root
    return {"mean": mean.astype(np.float32), "W": W.astype(np.float32),
            "std": np.ones(x.shape[1], np.float32), "center": True,
            "kind": "whiten", "rows": int(len(idx)), "shrinkage": float(shrinkage),
            "cond": float(vals.max() / vals.min())}


def to_torch(stats: dict, device):
    """Move a fitted transform onto the device once, for use inside collate.

    Whitening is a 4,096 x 4,096 matmul per batch, which is trivial on a GPU and
    slow in numpy, so the transform lives where the batch already is.
    """
    out = {"kind": stats.get("kind", "zscore"), "center": stats["center"],
           "mean": torch.from_numpy(stats["mean"]).to(device),
           "std": torch.from_numpy(stats["std"]).to(device)}
    if stats.get("kind") == "whiten":
        out["W"] = torch.from_numpy(stats["W"]).to(device)
    return out


def apply_torch(x, t: dict):
    """Apply a transform prepared by `to_torch` to a batch already on device."""
    if t["center"]:
        x = x - t["mean"]
    if t["kind"] == "whiten":
        return x @ t["W"].T
    return x / t["std"]
