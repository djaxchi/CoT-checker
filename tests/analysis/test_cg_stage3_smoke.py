"""End-to-end pipeline smoke: stage1 (fake model) -> stage2 (synthetic rollouts)
-> stage3 assemble. Verifies graphs, taxonomy sites, and crosstab wiring."""

import argparse
import json
import random
import sys

import numpy as np
import pandas as pd

from scripts.causal_graph import cg_stage1_tf as s1
from scripts.causal_graph import cg_stage2_fg as s2
from scripts.causal_graph import cg_stage3_assemble as s3
from tests.analysis.test_cg_stage1_smoke import (
    FakeModel,
    FakeTok,
    _args,
    _trace,
    _write_jsonl,
)


def _probes():
    rng = np.random.default_rng(0)
    return {28: (rng.normal(size=8).astype(np.float32), 0.1),
            20: (rng.normal(size=8).astype(np.float32), -0.2)}


def _synth_rollouts(trace_ids, onpolicy):
    rng = random.Random(0)
    rows = []
    for tid in trace_ids:
        for ctx, p in (("base", 0.9), ("swap_wrong", 0.1), ("swap_xprob", 0.8),
                       ("swap_pos", 0.85)):
            for ri in range(12):
                rows.append({"arm": "forks", "trace_id": tid, "split": "test",
                             "context": ctx, "t": 1, "rollout_idx": ri,
                             "correct": rng.random() < p, "gradeable": True,
                             "pred": "4", "traj_correct": None})
    for tid, n_steps in onpolicy:
        for i in range(n_steps):
            p = 0.9 if i < n_steps // 2 else 0.4
            for ri in range(8):
                rows.append({"arm": "onpolicy", "trace_id": tid, "split": "test",
                             "context": f"prefix_{i}", "t": i, "rollout_idx": ri,
                             "correct": rng.random() < p, "gradeable": True,
                             "pred": "4", "traj_correct": True})
        for ri in range(8):
            rows.append({"arm": "onpolicy", "trace_id": tid, "split": "test",
                         "context": "swap_xtrace_1", "t": 1, "rollout_idx": ri,
                         "correct": rng.random() < 0.3, "gradeable": True,
                         "pred": "4", "traj_correct": True})
    return pd.DataFrame(rows)


def test_full_pipeline(tmp_path, monkeypatch):
    traces = [_trace(i, "val" if i % 3 == 0 else "test") for i in range(6)]
    _write_jsonl(tmp_path / "traces_forks.jsonl", traces)
    trajs = [{"traj_uid": f"g{i}", "problem": f"problem {i}", "gold": "4",
              "pred": "4", "correct": i % 2 == 0, "gradeable": True,
              "solution": "First look.\n\nCompute two plus two.\n\n"
                          "So four.\n\nAnswer \\boxed{4}."}
             for i in range(4)]
    _write_jsonl(tmp_path / "onpolicy_trajectories.jsonl", trajs)

    # stage 1
    model, tok, probes = FakeModel(), FakeTok(), _probes()
    node_rows, edge_rows, skips = [], [], []
    s1.run_forks(_args(tmp_path), model, tok, "cpu", probes,
                 node_rows, edge_rows, skips)
    s1.run_onpolicy(_args(tmp_path, arm="onpolicy"), model, tok, "cpu", probes,
                    node_rows, edge_rows, skips)
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    pd.DataFrame(node_rows).to_parquet(stage1 / "node_features_shard0.parquet")
    pd.DataFrame(edge_rows).to_parquet(stage1 / "tf_edges_shard0.parquet")
    s1.merge(_args(tmp_path, merge=True))

    # stage 2 (synthetic rollouts for the same ids)
    stage2 = tmp_path / "stage2"
    stage2.mkdir()
    _synth_rollouts([t["trace_id"] for t in traces],
                    [(f"g{i}", 4) for i in range(4)]
                    ).to_parquet(stage2 / "fg_rollouts_shard0.parquet")
    s2.merge(argparse.Namespace(run_dir=tmp_path))

    # stage 3
    monkeypatch.setattr(sys, "argv",
                        ["cg_stage3_assemble.py", "--run_dir", str(tmp_path)])
    s3.main()

    out = tmp_path / "stage3"
    graphs = list((out / "graphs").glob("*.json"))
    assert len(graphs) == 6 + 4
    ct = json.loads((out / "crosstab.json").read_text())
    assert ct["n_sites_test"] == 4  # test-split fork traces
    assert sum(ct["cells"].values()) == 4
    g = json.loads(next(p for p in graphs if p.name.startswith("tr")).read_text())
    assert g["site"] is not None and g["site"]["taxonomy"] in (
        "detected_influential", "detected_inert",
        "undetected_influential", "undetected_inert")
    fam = {e["family"] for e in g["edges"]}
    assert fam == {"tf", "fg"}
    # fg edge on the synthetic data must be a big significant drop
    fg = next(e for e in g["edges"]
              if e["family"] == "fg" and e["interv"] == "swap_wrong")
    assert fg["delta"] < -0.4 and fg["significant"]
    gon = json.loads(next(p for p in graphs if p.name.startswith("g")).read_text())
    assert "fg_curve" in gon and len(gon["fg_curve"]) == gon["n_steps"]
