"""risk_probe_v2 helpers (docs/risk_probe_v2_plan.md): the soft harmful target,
the validation split, header parsing and the weighted calib-20 metric."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from merge_rep_grid_leaderboard import (CALIB_SIZE, CALIB_SPLITS, f1_pb_from_preds,  # noqa: E402
                                        pred_matrix, quantile_grid)

_GRID = np.linspace(0.0, 1.0, 4001)
_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy < 2 has only trapz
HEADER_HEAD, HEADER_TAIL = "Problem:\n", "\n\nSolution:\n"


def soft_harmful(wrong_before, k_before, wrong_after, k_after):
    """P(theta_after > theta_before) with independent Beta(1 + wrong, 1 + right)
    posteriors on the two cut error rates, by numerical integration:
    the integral of f_after(x) * F_before(x) over [0, 1]. Scalars or arrays."""
    wb, kb, wa, ka = (np.atleast_1d(np.asarray(v, float)) for v in (wrong_before, k_before, wrong_after, k_after))
    x = _GRID[None, :]
    q = np.empty(len(wb))
    for i in range(0, len(wb), 1024):                    # bounded memory on the full label set
        s = slice(i, i + 1024)
        f_a = beta.pdf(x, 1 + wa[s, None], 1 + ka[s, None] - wa[s, None])
        F_b = beta.cdf(x, 1 + wb[s, None], 1 + kb[s, None] - wb[s, None])
        q[s] = np.clip(_trapz(f_a * F_b, _GRID, axis=1), 0.0, 1.0)
    return float(q[0]) if np.ndim(wrong_before) == 0 else q


def is_val(problem_key: str, mod: int = 10) -> bool:
    """The validation problems inside every training set (about 1 in `mod`)."""
    return int(hashlib.sha1(("val:" + problem_key).encode()).hexdigest(), 16) % mod == 0


def parse_header(header: str) -> str:
    """Problem text from a `full` view header ("Problem:\\n{p}\\n\\nSolution:\\n")."""
    if not (header.startswith(HEADER_HEAD) and header.endswith(HEADER_TAIL)):
        raise ValueError(f"unexpected header: {header[:60]!r}")
    return header[len(HEADER_HEAD):-len(HEADER_TAIL)]


def calib20_weighted(traces_v: list, traces_r: list, weights) -> float:
    """calib-20 F1_PB of score = validity + w * risk, with w and the threshold chosen
    together on each calibration draw.

    The calibration draws are exactly those of merge_rep_grid_leaderboard.
    calib20_subset (same seeds, same stratification), and at weights [0] the result
    equals calib20_subset(traces_v). Ties go to the smaller weight, then the first
    grid threshold, as in calib20_subset.
    """
    if len(traces_v) != len(traces_r):
        raise ValueError("validity and risk traces differ in number")
    labels_all = np.array([t[0] for t in traces_v])
    idx = np.arange(len(traces_v))
    combo = {w: [(lab, (np.asarray(v, float) + w * np.asarray(r, float)).tolist())
                 for (lab, v), (_, r) in zip(traces_v, traces_r)] for w in weights}
    splits = []
    for sd in range(CALIB_SPLITS):
        rng = np.random.default_rng(sd)
        err, cor = idx[labels_all != -1], idx[labels_all == -1]
        frac = CALIB_SIZE / len(traces_v)
        n_err_c = min(len(err), round(len(err) * frac))
        n_cor_c = min(len(cor), CALIB_SIZE - n_err_c)
        err_p, cor_p = rng.permutation(len(err)), rng.permutation(len(cor))
        splits.append((np.concatenate([err[err_p[:n_err_c]], cor[cor_p[:n_cor_c]]]),
                       np.concatenate([err[err_p[n_err_c:]], cor[cor_p[n_cor_c:]]])))
    evals = []
    for cal, ev in splits:
        best = (-1.0, None, None)
        for w in weights:
            tr = combo[w]
            g = quantile_grid([tr[i] for i in cal])
            preds, labels = pred_matrix([tr[i] for i in cal], g)
            f = f1_pb_from_preds(preds, labels)
            j = int(np.argmax(f))
            if f[j] > best[0]:
                best = (float(f[j]), w, g[j])
        _, w, t = best
        preds, labels = pred_matrix([combo[w][i] for i in ev], np.array([t]))
        evals.append(float(f1_pb_from_preds(preds, labels)[0]))
    return float(np.mean(evals))
