"""Length-stratified AUROC: the control has to collapse, real signal has to survive."""
import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from stratified_auroc import stratified_auroc  # noqa: E402


def _pb(n=8000, seed=0):
    """ProcessBench's actual shape: first-error steps run 118.6 tokens, the rest 79.7."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(np.int64)
    ln = np.where(y == 1, rng.normal(118.6, 40, n), rng.normal(79.7, 30, n))
    return y, np.maximum(ln, 5)


def test_length_collapses_to_chance_inside_its_own_bins():
    """And only at fine bins: coarse bins leave length usable, which is why the
    script prints this control rather than assuming a bin count is enough."""
    y, ln = _pb()
    got = {k: stratified_auroc(y, ln.astype(float), ln, k) for k in (1, 5, 10, 50)}
    assert got[1] > 0.75                       # unstratified, length is worth a lot
    assert got[5] > got[10] > got[50]          # shrinks monotonically with bin count
    assert got[50] == pytest.approx(0.5, abs=0.03)


def test_a_length_only_score_also_collapses():
    """A probe that learned nothing but length must lose everything here."""
    y, ln = _pb()
    s = 3.1 * np.log(ln) - 12.0
    assert stratified_auroc(y, s, ln, 50) == pytest.approx(0.5, abs=0.03)


def test_signal_independent_of_length_survives():
    y, ln = _pb()
    rng = np.random.default_rng(1)
    s = y + rng.normal(0, 1.0, len(y))          # informative, uncorrelated with length
    plain = stratified_auroc(y, s, ln, 1)
    assert stratified_auroc(y, s, ln, 50) == pytest.approx(plain, abs=0.02)
    assert plain > 0.7


def test_bins_with_one_class_are_dropped_not_counted_as_half():
    y = np.array([1, 1, 0, 0]); ln = np.array([1.0, 2.0, 3.0, 4.0])
    # bins [1,1] and [0,0] carry no comparable pair; only a 1-bin split has any
    assert np.isnan(stratified_auroc(y, np.array([1.0, 1.0, 0.0, 0.0]), ln, 2))


def test_derived_npz_keep_their_length_arrays(tmp_path):
    """The first stratified pass skipped every residual/withlen/surface file for
    want of length arrays. Both derivers must carry them through."""
    import subprocess

    rng = np.random.default_rng(0)
    src = tmp_path / "src.npz"
    d = {"x_train": rng.normal(size=(200, 8)).astype(np.float32),
         "x_val": rng.normal(size=(50, 8)).astype(np.float32),
         "y_train": (rng.random(200) < 0.4).astype(np.float32),
         "y_val": (rng.random(50) < 0.4).astype(np.float32),
         "len_train": rng.integers(5, 90, 200).astype(np.float32),
         "len_val": rng.integers(5, 90, 50).astype(np.float32),
         "pb_x_gsm8k": rng.normal(size=(60, 8)).astype(np.float32),
         "pb_y_gsm8k": (rng.random(60) < 0.3).astype(np.float32),
         "pb_len_gsm8k": rng.integers(20, 200, 60).astype(np.float32)}
    np.savez(src, **d)
    root = Path(__file__).resolve().parents[2]
    for script, mode in [("residualize_length.py", "residual"),
                         ("residualize_length.py", "withlen"),
                         ("make_surface_baseline.py", "length"),
                         ("make_surface_baseline.py", "augment")]:
        out = tmp_path / f"{script}_{mode}.npz"
        subprocess.run([sys.executable, str(root / "scripts" / script), "--npz", str(src),
                        "--mode", mode, "--out", str(out)], check=True, capture_output=True)
        got = np.load(out)
        for k in ("len_train", "len_val", "pb_len_gsm8k"):
            assert k in got.files, f"{script} --mode {mode} dropped {k}"


def test_stratified_and_screen_fit_the_same_probe_on_the_same_rows(tmp_path):
    """These two scripts print a PB column each and are read side by side. They
    disagreed by 0.02 on the 8,192-dim stacked representations because one used
    50,000 training rows and the other 60,000. Pin that the defaults match, so a
    future divergence fails here instead of in a results table."""
    import argparse
    import importlib

    def defaults(mod_path):
        src = (Path(__file__).resolve().parents[2] / "scripts" / mod_path).read_text()
        tree = __import__("ast").parse(src)
        got = {}
        for node in __import__("ast").walk(tree):
            if (isinstance(node, __import__("ast").Call)
                    and getattr(node.func, "attr", "") == "add_argument"):
                name = node.args[0].value.lstrip("-")
                for kw in node.keywords:
                    if kw.arg == "default":
                        try:
                            got[name] = __import__("ast").literal_eval(kw.value)
                        except ValueError:
                            pass
        return got

    a, b = defaults("screen_representation.py"), defaults("stratified_auroc.py")
    for k in ("n_train", "epochs", "lr"):
        assert a[k] == b[k], (f"{k}: screen uses {a[k]}, stratified uses {b[k]}; "
                              f"their PB columns are not comparable")
