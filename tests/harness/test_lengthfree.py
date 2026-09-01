"""Length-free readout: fit on train, apply everywhere, drop the length column."""

from __future__ import annotations

import numpy as np
import pytest

from src.harness.geom import N_GEOM
from src.harness import lengthfree as lf


def _x(n=500, d=8, seed=0, slope=3.0):
    """mean_geom shaped: content that depends on length, geometry, then log len."""
    rng = np.random.default_rng(seed)
    log_len = np.log(rng.integers(5, 200, n)).astype(np.float32)
    content = rng.normal(size=(n, d)).astype(np.float32) + slope * log_len[:, None]
    geom = rng.normal(size=(n, N_GEOM)).astype(np.float32)
    return np.concatenate([content, geom, log_len[:, None]], 1), log_len


def test_it_removes_the_length_component_from_the_content():
    x, log_len = _x()
    out = lf.apply(x, lf.fit(x))
    d = x.shape[1] - N_GEOM - 1
    for j in range(d):
        r = np.corrcoef(out[:, j], log_len)[0, 1]
        assert abs(r) < 0.05, f"content column {j} still tracks length at r={r:.3f}"


def test_the_length_column_is_dropped_and_the_geometry_survives_untouched():
    x, _ = _x()
    stats = lf.fit(x)
    out = lf.apply(x, stats)
    d = x.shape[1] - N_GEOM - 1
    assert out.shape[1] == d + N_GEOM == lf.out_dim(stats)
    np.testing.assert_allclose(out[:, d:], x[:, d:d + N_GEOM])


def test_the_map_comes_from_train_and_is_not_refitted_on_the_target_split():
    """ProcessBench steps are twice as long as PRM800K's. Refitting there would
    erase the domain shift the whole correction exists to survive."""
    tr, _ = _x(seed=0)
    te, _ = _x(seed=1, slope=9.0)                 # a different length response
    stats = lf.fit(tr)
    applied = lf.apply(te, stats)
    refit = lf.apply(te, lf.fit(te))
    assert not np.allclose(applied, refit, atol=1e-3)


def test_a_width_with_no_content_left_is_refused():
    with pytest.raises(ValueError, match="no content"):
        lf.fit(np.zeros((10, N_GEOM + 1), dtype=np.float32))


def test_describe_reports_how_much_variance_length_accounted_for():
    x, _ = _x(slope=12.0)
    stats = lf.fit(x)
    msg = lf.describe(stats, x)
    assert "%" in msg and str(lf.out_dim(stats)) in msg


def test_the_cell_applies_the_train_length_map_to_processbench_too():
    """The correction is fitted on train and must reach every split the cell
    scores. If ProcessBench were left uncorrected the widths would not even
    match, but the guard belongs in the suite rather than in a crash."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "scripts"
           / "train_rep_learner_cell.py").read_text()
    assert "lstats = lfree.fit(Xtr)" in src
    for split in ("Xva", "Xte"):
        assert f"{split} = lfree.apply({split}, lstats)" in src, f"{split} uncorrected"
    assert "Xpb = lfree.apply(Xpb, lstats)" in src, "ProcessBench uncorrected"
    # and it must never be refitted anywhere but train
    assert src.count("lfree.fit(") == 1, "the length map is fitted more than once"
    ast.parse(src)


def test_the_readout_width_matches_what_the_cell_expects():
    """mean_geom is content + geometry + one length column, and the cell drops
    that column. A silent change to either side would misalign them."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.derive_delta_from_token_store import _out_dim

    d = 4096
    assert _out_dim("mean_geom", d) == d + N_GEOM + 1
    stats = lf.fit(np.random.default_rng(0).normal(
        size=(50, d + N_GEOM + 1)).astype(np.float32))
    assert lf.out_dim(stats) == d + N_GEOM


def test_the_grid_readout_reproduces_the_screen_representation_exactly(tmp_path):
    """The 0.7897 was measured by the screen, which samples the store into npz.
    The grid instead derives vectors through derive_split. If the two disagree,
    a grid result could not be attributed to the representation the search found.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    from src.repstore import STEP_SEQ, RepSpec, write_split
    from scripts.derive_delta_from_token_store import derive_split
    from src.harness.geom import geom_feats

    rng = np.random.default_rng(0)
    d, lens = 12, [4, 9, 2, 6]
    items = [rng.normal(size=(L, d)).astype(np.float32) for L in lens]
    meta = [{"step_idx": i, "label": 0, "global_index": i,
             "pre_step_boundary_idx": 0, "step_start_idx": 1}
            for i in range(len(lens))]
    sd = tmp_path / "shard_0"
    sd.mkdir(parents=True)
    write_split(sd, items, [0] * len(items), meta,
                RepSpec(name="s", kind=STEP_SEQ, dim=d, layer=-1,
                        backbone="t", readout="r"))

    got, _, _ = derive_split(tmp_path, "mean_geom", sort=False)
    for k, it in enumerate(items):
        span, bnd = it[1:], it[0]                 # what the screen pools
        want = np.concatenate([span.mean(0), geom_feats(span, bnd, with_len=False),
                               [np.log(span.shape[0])]])
        np.testing.assert_allclose(got[k], want.astype(np.float16), rtol=2e-2, atol=2e-2)
