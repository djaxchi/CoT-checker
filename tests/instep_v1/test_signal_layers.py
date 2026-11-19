import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scripts.instep_v1 import signal_layers as SL  # noqa: E402
from scripts.instep_v1.signal_geometry import TRAIN  # noqa: E402
from scripts.reprobe_cmp.analyze import TEST  # noqa: E402


def _write(d, name, uids, y, arrays, kept=None):
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{name}.shard0.npz", y=np.array(y, np.int8),
             kept=np.ones(len(y), bool) if kept is None else kept, **arrays)
    with open(d / f"{name}.shard0.meta.jsonl", "w") as f:
        for u in uids:
            f.write(json.dumps({"uid": u}) + "\n")


def _fixture(tmp_path, n=200, d=8, seed=0):
    """Source a: an informative block (read_block3) and a noise block (read_block7); source b:
    a ReProbe-like per-layer dir (rp_block3 + the head's own p_block3), one step dropped."""
    rng = np.random.default_rng(seed)
    a, b, data = tmp_path / "a", tmp_path / "b", tmp_path / "data"
    (data / "rp").mkdir(parents=True)
    vt = []
    for s in [TRAIN, "val", *TEST]:
        y = (rng.random(n) < 0.3).astype(int)
        uids = [f"{s}_t{i // 4}::{i % 4}" for i in range(n)]
        good = rng.normal(size=(n, d))
        good[:, 0] += 3.0 * y
        noise = rng.normal(size=(n, d))
        _write(a, s, uids, y, {"read_block3": good[:, None].astype(np.float16),
                               "read_block7": noise[:, None].astype(np.float16)})
        kept = np.ones(n, bool)
        kept[1] = False
        _write(b, s, uids, y, {"rp_block3": good[:, None].astype(np.float16),
                               "p_block3": (y + 0.1 * rng.normal(size=n)).astype(np.float32)}, kept)
        if s == "val":
            vt += [{"trace_id": f"val_t{i // 4}", "labels": {"ds": y[i:i + 4].tolist()}} for i in range(0, n, 4)]
    with open(data / "rp" / "val.traces.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in vt)
    return a, b, data


def test_parse_src_default_and_regex(tmp_path):
    assert SL.parse_src(f"x={tmp_path}") == ("x", tmp_path, SL.STATE)
    assert SL.parse_src(f"x={tmp_path}=rp_block\\d+")[2] == "rp_block\\d+"


def test_joint_mask_drops_steps_missing_in_any_source(tmp_path):
    a, b, data = _fixture(tmp_path)
    ys, masks = SL.joint_mask([("a", a, SL.STATE), ("b", b, SL.STATE)], data, "ds")
    assert masks["math"][1] == False and int(masks["math"].sum()) == 199  # noqa: E712


def test_joint_mask_rejects_misaligned_uids(tmp_path):
    a, b, data = _fixture(tmp_path)
    _write(b, "math", [f"other::{i}" for i in range(200)], np.zeros(200, int), {})
    with pytest.raises(SystemExit):
        SL.joint_mask([("a", a, SL.STATE), ("b", b, SL.STATE)], data, "ds")


def test_probe_reads_signal_and_own_score_skips_probe(tmp_path):
    a, b, data = _fixture(tmp_path)
    ys, masks = SL.joint_mask([("a", a, SL.STATE), ("b", b, SL.STATE)], data, "ds")
    grid = (1e-2, 1.0)
    good = SL.score("a", a, "read_block3", ys, masks, grid, "cpu")
    noise = SL.score("a", a, "read_block7", ys, masks, grid, "cpu")
    own = SL.score("b", b, "p_block3", ys, masks, grid, "cpu")
    assert good["kind"] == "probe" and good["mean_auroc"] > 0.9
    assert 0.35 < noise["mean_auroc"] < 0.65
    assert own["kind"] == "own" and own["C"] is None and own["mean_auroc"] > 0.95
    assert set(good["test"]) == set(TEST)


def test_step_mlp_row(tmp_path):
    a, b, data = _fixture(tmp_path)
    ys, masks = SL.joint_mask([("a", a, SL.STATE)], data, "ds")
    r = SL.score_mlp("a", a, "read_block3", ys, masks, 0, "cpu")
    assert r["kind"] == "step_mlp" and r["seed"] == 0 and r["mean_auroc"] > 0.85
