"""Feature ablation of the geometry block."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from explain_geom import score_block  # noqa: E402
from src.harness.geom import GEOM_NAMES, N_GEOM  # noqa: E402

LAMBDAS = [1e0, 1e2, 1e4]


def test_the_names_match_what_geom_feats_emits():
    """The ablation reports feature names, so a mismatch would mislabel every
    row of the result rather than fail."""
    from src.harness.geom import geom_feats

    rng = np.random.default_rng(0)
    got = geom_feats(rng.normal(size=(9, 12)).astype(np.float32),
                     rng.normal(size=12).astype(np.float32), with_len=False)
    assert len(GEOM_NAMES) == N_GEOM == len(got)
    assert len(set(GEOM_NAMES)) == N_GEOM, "duplicate feature name"


def _npz(tmp_path, d=6, n=4000, informative=0, seed=0):
    """Content plus a geometry block in which exactly one column carries signal."""
    rng = np.random.default_rng(seed)
    def blk(m):
        y = (rng.random(m) < 0.4).astype(np.float32)
        x = rng.normal(size=(m, d + N_GEOM)).astype(np.float32)
        x[:, :d] += 0.25 * y[:, None]                       # weak content signal
        x[:, d + informative] += 2.0 * y                    # one strong geometry col
        return x, y
    xt, yt = blk(n)
    xv, yv = blk(n // 4)
    xp, yp = blk(n // 4)
    p = tmp_path / "w.npz"
    np.savez(p, x_train=xt, y_train=yt, x_val=xv, y_val=yv,
             pb_x_gsm8k=xp, pb_y_gsm8k=yp)
    return p


def test_the_ablation_finds_the_one_column_that_carries_signal(tmp_path):
    z = np.load(_npz(tmp_path, informative=7))
    d = z["x_train"].shape[1] - N_GEOM
    content = list(range(d))
    geom = list(range(d, z["x_train"].shape[1]))
    base, _ = score_block(z, content, LAMBDAS, 4000)
    full, _ = score_block(z, content + geom, LAMBDAS, 4000)
    assert full > base + 0.02, "the planted signal did not register at all"

    adds = [score_block(z, content + [d + j], LAMBDAS, 4000)[0] - base
            for j in range(N_GEOM)]
    assert int(np.argmax(adds)) == 7, f"picked column {int(np.argmax(adds))}, not 7"
    assert adds[7] > 0.02
    assert max(a for j, a in enumerate(adds) if j != 7) < adds[7] / 2


def test_leave_one_out_is_near_zero_for_a_column_that_carries_nothing(tmp_path):
    """A dead feature must look dead, or the ablation cannot justify dropping it."""
    z = np.load(_npz(tmp_path, informative=0, seed=3))
    d = z["x_train"].shape[1] - N_GEOM
    content = list(range(d))
    geom = list(range(d, z["x_train"].shape[1]))
    full, _ = score_block(z, content + geom, LAMBDAS, 4000)
    loo, _ = score_block(z, content + [c for c in geom if c != d + 11],
                         LAMBDAS, 4000)
    assert abs(full - loo) < 0.01


def test_the_penalty_is_chosen_on_validation_and_never_on_processbench(tmp_path):
    """Selecting on the transfer set would turn every ablation number into an
    oracle. Corrupting ProcessBench must not change which penalty is picked."""
    p = _npz(tmp_path, informative=2)
    z = dict(np.load(p))
    d = z["x_train"].shape[1] - N_GEOM
    cols = list(range(d + N_GEOM))
    _, v1 = score_block(np.load(p), cols, LAMBDAS, 4000)
    z["pb_x_gsm8k"] = np.random.default_rng(9).normal(
        size=z["pb_x_gsm8k"].shape).astype(np.float32)
    q = tmp_path / "corrupt.npz"
    np.savez(q, **z)
    _, v2 = score_block(np.load(q), cols, LAMBDAS, 4000)
    assert v1 == v2
