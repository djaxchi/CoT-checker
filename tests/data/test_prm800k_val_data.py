"""Unit tests for the PRM800K val_1k loader (synthetic fixture, no model/HF)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.prm800k_val_data import load_prm800k_val


def _make_merged(tmp_path, n=5, h=8, stem="val_1k"):
    d = tmp_path / "merged"
    d.mkdir()
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((n, h)).astype(np.float16)
    label = (rng.random(n) > 0.5).astype(np.int64)
    np.save(d / f"{stem}_h.npy", hidden)
    np.save(d / f"{stem}_y.npy", label)
    meta = [
        {
            "uid": f"u{i}", "problem_id": f"p{i % 2}", "solution_id": f"s{i}",
            "step_idx": i, "completion_idx": 0, "label": int(label[i]),
            "rating": [-1, 0, 1][i % 3], "n_tokens": 10 + i,
        }
        for i in range(n)
    ]
    (d / f"{stem}_meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
    return d, hidden, label, meta


def test_load_roundtrip_and_alignment(tmp_path):
    d, hidden, label, meta = _make_merged(tmp_path, n=6)
    data = load_prm800k_val(d)

    assert len(data) == 6
    assert data.hidden.shape == (6, 8)
    assert data.hidden.dtype == np.float32
    np.testing.assert_array_equal(data.label, label)
    np.testing.assert_array_equal(data.rating, np.array([m["rating"] for m in meta]))
    np.testing.assert_array_equal(data.step_idx, np.arange(6))
    assert list(data.uid) == [m["uid"] for m in meta]


def test_rating_defaults_to_zero_when_missing(tmp_path):
    d, *_ = _make_merged(tmp_path, n=3)
    # rewrite meta without rating to exercise the default path
    meta = [{"uid": f"u{i}", "problem_id": "p", "solution_id": f"s{i}",
             "step_idx": i, "completion_idx": 0, "label": 0} for i in range(3)]
    (d / "val_1k_meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
    data = load_prm800k_val(d)
    np.testing.assert_array_equal(data.rating, np.zeros(3, dtype=np.int64))


def test_row_mismatch_raises(tmp_path):
    d, *_ = _make_merged(tmp_path, n=4)
    # truncate meta to trigger the guard
    lines = (d / "val_1k_meta.jsonl").read_text().splitlines()
    (d / "val_1k_meta.jsonl").write_text("\n".join(lines[:3]))
    with pytest.raises(ValueError, match="row mismatch"):
        load_prm800k_val(d)
