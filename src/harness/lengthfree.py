"""Regress step length out of a representation, fitted on train, applied everywhere.

The probe trains on PRM800K steps averaging 38.8 tokens and is evaluated on
ProcessBench steps averaging 79.7 and 118.6. Step length alone scores 0.7039 on
ProcessBench, so a representation that encodes length hands the probe a boundary
fitted to the short domain and used on the long one.

Removing length is not, by itself, an improvement: `mean_residual` scores 0.7282
against plain `mean`'s 0.7470, because the length component carried real signal
along with the shortcut. The gain comes from removing it AND handing back the
scale-free part of the same information:

    mean                        0.7470
    mean with length removed    0.7282     removing it alone costs
    20 geometry features alone  0.5182     they are useless alone
    both together               0.7897     and worth 0.043 combined

The map is fitted on train only and applied unchanged to validation, test and
ProcessBench. Refitting per split would remove whatever genuine length effect
exists in each domain separately and prove nothing.
"""

from __future__ import annotations

import numpy as np

from src.harness.geom import N_GEOM


def _design(log_len: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(log_len), log_len], 1).astype(np.float64)


def fit(x: np.ndarray, n_geom: int = N_GEOM, rows: int = 200000) -> dict:
    """Least-squares map from [1, log len] to each content position, on train.

    `x` is the `mean_geom` readout: content, then n_geom geometry features, then
    log token count as the final column.
    """
    x = np.asarray(x[:rows])
    d = x.shape[1] - n_geom - 1
    if d <= 0:
        raise ValueError(f"width {x.shape[1]} leaves no content before "
                         f"{n_geom} geometry features and a length column")
    a = _design(x[:, -1].astype(np.float64))
    coef, *_ = np.linalg.lstsq(a, x[:, :d].astype(np.float64), rcond=None)
    return {"coef": coef.astype(np.float32), "d": int(d), "n_geom": int(n_geom)}


def apply(x: np.ndarray, stats: dict) -> np.ndarray:
    """Subtract the fitted length component, keep the geometry, drop the length."""
    x = np.asarray(x, dtype=np.float32)
    d, ng = stats["d"], stats["n_geom"]
    pred = (_design(x[:, -1].astype(np.float64)).astype(np.float32) @ stats["coef"])
    return np.concatenate([x[:, :d] - pred, x[:, d:d + ng]], 1)


def out_dim(stats: dict) -> int:
    return stats["d"] + stats["n_geom"]


def describe(stats: dict, sample: np.ndarray) -> str:
    d = stats["d"]
    before = np.asarray(sample[:, :d], dtype=np.float64).var(0).sum()
    after = apply(sample, stats)[:, :d].astype(np.float64).var(0).sum()
    return (f"length removed from {d} content positions, "
            f"{100 * (1 - after / max(before, 1e-12)):.2f}% of their variance; "
            f"{stats['n_geom']} geometry features kept, length column dropped "
            f"-> {out_dim(stats)}")
