"""Smoke test for das_span_oracle merge/gate math (no model, synthetic margins)."""

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


def test_merge_margin_recovery(tmp_path: Path):
    from scripts.das_branch.das_span_oracle import merge

    rng = np.random.default_rng(0)
    n = 40
    m_wrong = rng.normal(-1.0, 0.5, n)
    m_correct = m_wrong + rng.normal(1.0, 0.2, n)          # correct belief higher
    gap = m_correct - m_wrong
    rows = pd.DataFrame({
        "trace_id": [f"t{i}" for i in range(n)], "split": "test", "align": "lastk",
        "wrong_span": 8, "correct_span": 8, "inj_width": 8,
        "m_wrong": m_wrong, "m_correct": m_correct,
        "m_oracle_L20": m_wrong + 0.5 * gap,   # ~half recovery
        "m_oracle_L12": m_wrong + 0.0 * gap,   # no recovery
        "m_oracle_L26": m_wrong + 0.5 * gap,
        "m_xspan_L20": m_wrong - 0.1 * gap,    # control slightly negative
    })
    d = tmp_path / "das_span"
    d.mkdir()
    rows.to_parquet(d / "span_lastk_shard0.parquet")

    merge(argparse.Namespace(out_dir=d, align="lastk"))
    g = json.loads((d / "gates_span_lastk.json").read_text())

    assert g["n"] == n
    assert g["belief_gap_correct_minus_wrong"]["mean"] > 0.7
    assert 0.4 < g["margin_by_layer"]["L20"]["recovery_fraction"] < 0.6
    assert abs(g["margin_by_layer"]["L12"]["recovery_fraction"]) < 0.1
    assert g["margin_by_layer"]["L20"]["p"] < 0.05          # oracle > wrong
    assert "xspan_mean_minus_wrong" in g["margin_by_layer"]["L20"]


def test_script_imports_clean():
    r = subprocess.run(
        [sys.executable, "-c",
         "import scripts.das_branch.das_span_oracle as m; assert hasattr(m,'merge')"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
