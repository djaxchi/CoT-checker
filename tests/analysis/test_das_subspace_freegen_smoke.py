"""Smoke test for das_subspace_freegen merge (no model, synthetic solve rates)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def test_merge_recovery(tmp_path: Path):
    from scripts.das_branch.das_subspace_freegen import merge

    rng = np.random.default_rng(0)
    n = 60
    wrong = rng.uniform(0, 0.3, n)
    correct = wrong + rng.uniform(0.1, 0.3, n)          # correct solves more
    gap = correct - wrong
    df = pd.DataFrame({
        "trace_id": [f"t{i}" for i in range(n)], "layer": 12, "k_sub": 8, "seed": 0,
        "wrong": wrong, "correct": correct,
        "oracle": wrong + 0.6 * gap,
        "das": wrong + 0.3 * gap,     # partial recovery
        "random": wrong - 0.02 * gap})
    d = tmp_path / "das_train"
    d.mkdir()
    df.to_parquet(d / "fgsub_L12_k8_shard0.parquet")

    merge(argparse.Namespace(out_dir=d, layer=12, k_sub=8))
    g = json.loads((d / "gates_fgsub_L12_k8.json").read_text())
    assert g["n"] == n
    assert 0.25 < g["das"]["recovery"] < 0.35
    assert 0.55 < g["oracle"]["recovery"] < 0.65
    assert g["das"]["p_gt_random"] < 0.05          # das beats random


def test_script_imports_clean():
    r = subprocess.run(
        [sys.executable, "-c",
         "import scripts.das_branch.das_subspace_freegen as m; assert hasattr(m,'merge')"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
