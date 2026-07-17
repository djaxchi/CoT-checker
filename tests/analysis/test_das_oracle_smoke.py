"""Smoke test for das_oracle merge/gate math (no model, synthetic rollouts)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _rollouts(context: str, per_trace_rate: dict[str, float], k: int = 8):
    """One row per rollout; k rollouts per trace with the given solve rate."""
    rows = []
    for tid, rate in per_trace_rate.items():
        n_correct = round(rate * k)
        for ri in range(k):
            rows.append({"trace_id": tid, "split": "test", "context": context,
                         "rollout_idx": ri, "correct": ri < n_correct,
                         "gradeable": True, "pred": "1", "has_alt_pos": True})
    return rows


def test_merge_recovery_math(tmp_path: Path):
    from scripts.das_branch.das_oracle import merge

    traces = [f"t{i}" for i in range(30)]
    rows = []
    # wrong branch solves ~0.2, correct ~0.8, oracle_L20 recovers halfway (~0.5)
    rows += _rollouts("wrong", {t: 0.25 for t in traces})
    rows += _rollouts("correct", {t: 0.75 for t in traces})
    rows += _rollouts("oracle_L20", {t: 0.50 for t in traces})
    rows += _rollouts("oracle_L12", {t: 0.25 for t in traces})  # no recovery
    d = tmp_path / "das_branch"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "oracle_rollouts_shard0.parquet")

    merge(argparse.Namespace(out_dir=d))
    gates = json.loads((d / "gates_oracle.json").read_text())

    assert gates["n_traces"] == 30
    prem = gates["premise_wrong_vs_correct"]
    assert prem["mean_gap_correct_minus_wrong"] > 0.4  # correct beats wrong
    l20 = gates["oracle_by_layer"]["L20"]
    assert 0.4 < l20["recovery_fraction"] < 0.6         # ~halfway recovery
    l12 = gates["oracle_by_layer"]["L12"]
    assert abs(l12["recovery_fraction"]) < 0.1          # ~no recovery


def test_script_imports_clean():
    # top-level imports (transformers-free path) must load without a model
    r = subprocess.run(
        [sys.executable, "-c",
         "import scripts.das_branch.das_oracle as m; assert hasattr(m,'merge')"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
