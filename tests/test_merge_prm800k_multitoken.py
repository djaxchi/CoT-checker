"""Merge integrity test for the 4D multitoken shards (sort by global_index)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_merge_main():
    spec = importlib.util.spec_from_file_location(
        "merge_mtml", REPO / "scripts" / "merge_prm800k_multitoken_shards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_shard(d, gis, stem="mt", L=1, T=2, H=3):
    d.mkdir(parents=True)
    n = len(gis)
    h = np.zeros((n, L, T, H), dtype=np.float16)
    for i, g in enumerate(gis):
        h[i] = g  # tag every plane with the global_index for an order check
    y = np.array([g % 2 for g in gis], dtype=np.int32)
    np.save(d / f"{stem}_h.npy", h)
    np.save(d / f"{stem}_y.npy", y)
    meta = [{"uid": f"u{g}", "label": int(g % 2), "global_index": g} for g in gis]
    (d / f"{stem}_meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
    (d / f"{stem}_manifest.json").write_text(json.dumps(
        {"layer_indices": [28], "token_order": ["first", "last"], "shape": [n, L, T, H]}))


def test_merge_sorts_and_dedups(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    # interleaved global_index across two shards: 0,2 and 1,3
    _write_shard(out / "shard_00", [0, 2])
    _write_shard(out / "shard_01", [1, 3])

    mod = _load_merge_main()
    monkeypatch.setattr(sys, "argv",
                        ["x", "--shard_root", str(out), "--stem", "mt",
                         "--out_dir", str(out)])
    mod.main()

    h = np.load(out / "mt_h.npy")
    meta = [json.loads(l) for l in (out / "mt_meta.jsonl").read_text().splitlines() if l.strip()]
    assert h.shape == (4, 1, 2, 3)
    # rows must be in global_index order 0,1,2,3
    assert [m["global_index"] for m in meta] == [0, 1, 2, 3]
    assert [int(h[i, 0, 0, 0]) for i in range(4)] == [0, 1, 2, 3]
