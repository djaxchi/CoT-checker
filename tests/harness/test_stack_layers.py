"""Stacking two layers must refuse anything but identical rows in identical order."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stack_layers.py"


def _npz(path, x_off, y=None, lens=None, seed=0, n=40, d=6):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(np.float32) if y is None else y
    lens = rng.integers(3, 50, n).astype(np.float32) if lens is None else lens
    np.savez(path, x_train=rng.normal(x_off, 1, (n, d)).astype(np.float32),
             x_val=rng.normal(x_off, 1, (10, d)).astype(np.float32),
             y_train=y, y_val=np.zeros(10, np.float32),
             len_train=lens, len_val=np.ones(10, np.float32),
             pb_x_gsm8k=rng.normal(x_off, 1, (12, d)).astype(np.float32),
             pb_y_gsm8k=np.zeros(12, np.float32),
             pb_len_gsm8k=np.arange(12, dtype=np.float32))
    return y, lens


def _run(paths, out):
    return subprocess.run([sys.executable, str(SCRIPT), "--npz", *map(str, paths),
                           "--out", str(out)], capture_output=True, text=True)


def test_aligned_layers_stack_and_double_the_width(tmp_path):
    a, b, out = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "o.npz"
    y, lens = _npz(a, 0.0)
    _npz(b, 5.0, y=y, lens=lens, seed=1)
    r = _run([a, b], out)
    assert r.returncode == 0, r.stderr
    z = np.load(out)
    assert z["x_train"].shape == (40, 12)
    assert np.array_equal(z["y_train"], y)


def test_misordered_rows_are_refused_not_silently_paired(tmp_path):
    """The failure this guards against is silent: shuffled rows still stack to the
    right shape, and every step would be paired with another step's activations."""
    a, b, out = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "o.npz"
    y, lens = _npz(a, 0.0)
    perm = np.random.default_rng(3).permutation(len(y))
    _npz(b, 5.0, y=y[perm], lens=lens[perm], seed=1)
    r = _run([a, b], out)
    assert r.returncode != 0
    assert not out.exists()
    assert "not the same rows" in r.stdout + r.stderr


def test_one_layer_is_not_a_stack(tmp_path):
    a, out = tmp_path / "a.npz", tmp_path / "o.npz"
    _npz(a, 0.0)
    assert _run([a], out).returncode != 0
