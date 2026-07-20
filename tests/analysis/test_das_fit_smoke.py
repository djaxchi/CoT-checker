"""Smoke test for das_fit report aggregation (no model, synthetic metrics/U)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def test_report_aggregates(tmp_path: Path):
    from scripts.das_branch.das_fit import do_report

    d = tmp_path / "das_train"
    d.mkdir()
    # two seeds of one config; DAS beats random; save near-identical U's -> high overlap
    base = torch.linalg.qr(torch.randn(32, 8))[0]
    for s in (0, 1):
        (d / f"metrics_L12_k8_s{s}.json").write_text(json.dumps({
            "layer": 12, "k_sub": 8, "seed": s, "das_recovery": 0.6 + 0.02 * s,
            "random_recovery": 0.15, "oracle_recovery": 0.88,
            "p_das_gt_random": 0.01}))
        w = base + 0.01 * torch.randn(32, 8)      # small perturbation -> overlap ~1
        torch.save({"weight": w}, d / f"U_L12_k8_s{s}.pt")

    do_report(argparse.Namespace(out_dir=d))
    g = json.loads((d / "gates_das.json").read_text())
    cfg = g["configs"][0]
    assert cfg["layer"] == 12 and cfg["k_sub"] == 8 and cfg["seeds"] == 2
    assert cfg["das_minus_random_mean"] > 0.4
    assert cfg["cross_seed_overlap_mean"] > 0.9    # seeds agree on the subspace


def test_script_imports_clean():
    r = subprocess.run(
        [sys.executable, "-c",
         "import scripts.das_branch.das_fit as m; assert hasattr(m,'do_report')"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
