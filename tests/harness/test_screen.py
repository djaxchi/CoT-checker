"""The screen has to rank representations the way the full grid would.

It is a filter, so it earns its place only if its ordering agrees with the
expensive metric. Its design was chosen by that criterion on 31 evaluated cells
(ProcessBench step AUROC 0.934 vs in-domain AUROC 0.835), and these tests check
it behaves that way on data where the right answer is known by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from screen_representation import auroc, screen  # noqa: E402

DEV = torch.device("cpu")


def make(sep, n=20000, d=24, seed=0, shift=0.0):
    """A representation whose usefulness is set by `sep`; `shift` moves the
    ProcessBench half off-distribution, so transfer can differ from fit."""
    rng = np.random.default_rng(seed)
    def block(m, sh):
        y = (rng.random(m) < 0.5).astype(np.float32)
        x = rng.normal(0, 1, (m, d)).astype(np.float32) + sh
        x[:, 0] += sep * y
        return x, y
    xtr, ytr = block(n, 0.0)
    xv, yv = block(n // 3, 0.0)
    xp, yp = block(n // 3, shift)
    return xtr, ytr, xv, yv, [(xp, yp)]


def test_auroc_matches_a_known_case():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert auroc(y, np.array([0.4, 0.3, 0.2, 0.1])) == 0.0
    assert abs(auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9


def test_a_more_separable_representation_screens_higher():
    weak = screen(*make(0.15, seed=1), DEV, epochs=8, lr=3e-3)
    strong = screen(*make(1.2, seed=1), DEV, epochs=8, lr=3e-3)
    assert strong["pb_step_auroc"] > weak["pb_step_auroc"]
    assert strong["signal_share"] > weak["signal_share"]


def test_a_representation_with_no_signal_screens_at_chance():
    r = screen(*make(0.0, seed=2), DEV, epochs=8, lr=3e-3)
    assert abs(r["pb_step_auroc"] - 0.5) < 0.08
    assert r["signal_share"] < 0.01


def test_the_transfer_number_can_disagree_with_the_in_domain_one():
    """The reason the screen leads on ProcessBench AUROC: a representation can
    fit in-domain and transfer badly. In the real grid `step_mean x mlp:h1024`
    ranked 1st of 31 in-domain and 13th on the full metric, so the screen must be
    able to see the two come apart."""
    r = screen(*make(0.8, seed=3, shift=3.0), DEV, epochs=8, lr=3e-3)
    assert r["in_domain_auroc"] > r["pb_step_auroc"]


def test_tied_scores_score_at_chance_not_perfect():
    """A saturated probe emits many identical scores. Without averaged ranks an
    all-tied vector returns AUROC 1.0, which would make the screen rank the most
    broken representations highest -- the same failure that broke calib-20."""
    y = np.array([0, 0, 1, 1])
    assert abs(auroc(y, np.array([0.9, 0.9, 0.9, 0.9])) - 0.5) < 1e-9
    half = np.array([0.0, 1.0, 0.0, 1.0])           # half tied at each end
    assert 0.0 < auroc(y, half) < 1.0


def test_signal_share_needs_no_training():
    """It is computed from the representation alone, so it is available before
    any probe is fitted and explains why a representation screens as it does."""
    xtr, ytr, xv, yv, pb = make(1.5, seed=4)
    r = screen(xtr, ytr, xv, yv, pb, DEV, epochs=1)
    assert r["signal_share"] > 0.01


def test_screen_is_fast_enough_to_be_a_filter():
    r = screen(*make(0.6, n=50000, seed=5), DEV, epochs=8, lr=3e-3)
    assert r["seconds"] < 30, "a screen slower than this stops being a screen"


def test_surface_augment_actually_concatenates(tmp_path):
    """The augment mode silently produced length-only vectors once, because a
    string patch did not match and failed quietly. Pin the widths: augment must be
    representation + 3 length terms, not 3."""
    import subprocess
    import numpy as _np
    rng = _np.random.default_rng(0)
    d = 6
    def blk(m, lo, hi):
        L = rng.integers(lo, hi, m).astype(_np.float32)
        y = (rng.random(m) < 0.5).astype(_np.float32)
        x = rng.normal(0, 1, (m, d)).astype(_np.float32)
        return x, y, L
    xt, yt, lt = blk(400, 20, 50); xv, yv, lv = blk(120, 20, 50); xp, yp, lp = blk(120, 60, 100)
    src = tmp_path / "m.npz"
    _np.savez(src, x_train=xt, y_train=yt, x_val=xv, y_val=yv, len_train=lt,
              len_val=lv, pb_x_a=xp, pb_y_a=yp, pb_len_a=lp)
    script = ROOT / "scripts" / "make_surface_baseline.py"
    for mode, expect in (("length", 1), ("length_poly", 3), ("augment", d + 3)):
        out = tmp_path / f"{mode}.npz"
        subprocess.run([sys.executable, str(script), "--npz", str(src), "--mode",
                        mode, "--out", str(out)], check=True, capture_output=True)
        z = _np.load(out)
        assert z["x_train"].shape[1] == expect, f"{mode}: {z['x_train'].shape[1]} != {expect}"
        assert z["pb_x_a"].shape[1] == expect
        assert z["x_val"].shape[1] == expect


def test_a_single_informative_feature_screens_above_chance():
    """The screen reported step length at 0.296 PB AUROC -- below chance -- while
    the feature itself separates the classes at 0.736. Cause: one fixed learning
    rate cannot serve both a 4,096-dim activation block and a 1-dim scalar, so the
    one-dimensional probe never escaped its random initialisation and kept a
    negative weight. Inputs are standardised now; pin that a lone good feature
    screens correctly."""
    rng = np.random.default_rng(0)
    def blk(m, mu0, mu1):
        y = (rng.random(m) < 0.4).astype(np.float32)
        L = np.where(y == 1, rng.normal(mu1, mu1 * 0.4, m), rng.normal(mu0, mu0 * 0.4, m))
        return np.log(np.maximum(L, 2))[:, None].astype(np.float32), y
    xt, yt = blk(20000, 32.8, 45.3)
    xv, yv = blk(4000, 32.8, 45.3)
    xp, yp = blk(4000, 79.7, 118.6)          # the real domain shift, longer steps
    raw = auroc(yp, xp[:, 0])                # what the feature is worth, no probe
    r = screen(xt, yt, xv, yv, [(xp, yp)], DEV, epochs=8, lr=1e-3)
    assert raw > 0.65
    assert r["pb_step_auroc"] > 0.6, f"1-d probe collapsed: {r['pb_step_auroc']:.3f}"
    assert abs(r["pb_step_auroc"] - raw) < 0.1
