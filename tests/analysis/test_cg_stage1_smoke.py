"""Smoke test for scripts/causal_graph/cg_stage1_tf.py on a deterministic fake
model: exercises run_forks / run_onpolicy / merge end-to-end (no real model)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.causal_graph import cg_stage1_tf as s1

V = 256
D = 8
N_LAYERS = 29  # hidden_states indices 0..28


class FakeOut:
    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states


class FakeModel:
    """Deterministic: logits/hidden depend only on the current token id."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn((V, V), generator=g)
        self.H = torch.randn((3, V, D), generator=g)

    def __call__(self, input_ids=None, attention_mask=None,
                 output_hidden_states=False):
        ids = input_ids.clamp(0, V - 1)
        logits = self.W[ids]
        hs = None
        if output_hidden_states:
            hs = [self.H[l % 3][ids] for l in range(N_LAYERS)]
        return FakeOut(logits, hs)


class FakeTok:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        ids = [(hash(w) % (V - 2)) + 1 for w in text.split(" ") if w]
        if text.startswith(" ") or text.startswith("\n"):
            ids = [2] + ids
        return {"input_ids": ids or [3]}


def _trace(i, split="test", alt=True):
    steps = [f"step{i} alpha beta", "compute x equals two",
             "so y equals four", "final answer is four"]
    return {"trace_id": f"tr{i}", "arm": "forks", "question": f"question {i} text",
            "steps": steps, "fork_t": 1, "wrong_step": "compute x equals three oops",
            "alt_pos_step": "compute x equals two indeed" if alt else None,
            "xprob_step": "unrelated words here now", "gt_answer": "4",
            "candidates": ["4", "5", "6", "7"], "split": split}


def _args(tmp_path, **kw):
    ns = argparse.Namespace(
        run_dir=tmp_path, model_name_or_path="fake", directions_npz=[],
        arm="forks", onpolicy_trajectories=tmp_path / "onpolicy_trajectories.jsonl",
        max_onpolicy_traces=4, shard_id=0, num_shards=1, max_seq_len=4096,
        seed=0, local_files_only=False, merge=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def probes():
    rng = np.random.default_rng(0)
    return {28: (rng.normal(size=D).astype(np.float32), 0.1),
            20: (rng.normal(size=D).astype(np.float32), -0.2)}


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_run_forks_and_merge(tmp_path, probes):
    traces = [_trace(0, "val"), _trace(1, "test", alt=False), _trace(2, "val")]
    _write_jsonl(tmp_path / "traces_forks.jsonl", traces)
    model, tok = FakeModel(), FakeTok()
    node_rows, edge_rows, skips = [], [], []
    args = _args(tmp_path)
    s1.run_forks(args, model, tok, "cpu", probes, node_rows, edge_rows, skips)

    assert len(node_rows) == 3 * 4
    # 3 swaps for traces with alt_pos, 2 otherwise
    assert len(edge_rows) == 3 + 2 + 3
    assert not skips
    e = edge_rows[0]
    assert np.isfinite(e["d_margin_final"])
    assert np.isfinite(e["probe_at_interv_l28"])
    prof = json.loads(e["d_margin_profile"])
    dlp = json.loads(e["d_logp_steps"])
    assert set(dlp.keys()) == {"2", "3"}  # downstream of fork_t=1
    assert str(len(_trace(0)["steps"])) in prof  # final boundary present
    # identical swap text at an unchanged boundary: pre-fork boundary margin delta = 0
    assert prof[str(traces[0]["fork_t"])] == pytest.approx(0.0, abs=1e-4)

    # shard files + merge
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    pd.DataFrame(node_rows).to_parquet(stage1 / "node_features_shard0.parquet")
    pd.DataFrame(edge_rows).to_parquet(stage1 / "tf_edges_shard0.parquet")
    s1.merge(_args(tmp_path, merge=True))
    gates = json.loads((stage1 / "gates_stage1.json").read_text())
    assert "G1_wrong_vs_xprob" in gates
    assert gates["G1_wrong_vs_xprob"]["n"] == 3 or gates["G1_wrong_vs_xprob"].get("skipped")


def test_run_onpolicy(tmp_path, probes):
    trajs = [{"traj_uid": f"g{i}", "problem": f"problem {i}", "gold": "4",
              "pred": "4", "correct": i % 2 == 0, "gradeable": True,
              "solution": "First we look.\n\nThen we compute two plus two.\n\n"
                          "So we get four.\n\nThe answer is \\boxed{4}."}
             for i in range(4)]
    _write_jsonl(tmp_path / "onpolicy_trajectories.jsonl", trajs)
    model, tok = FakeModel(), FakeTok()
    node_rows, edge_rows, skips = [], [], []
    args = _args(tmp_path, arm="onpolicy")
    s1.run_onpolicy(args, model, tok, "cpu", probes, node_rows, edge_rows, skips)
    assert len(node_rows) == 4 * 4
    assert len(edge_rows) == 4 * 4 * 2  # (delete, swap_xtrace) x steps x traces
    dels = [e for e in edge_rows if e["interv"] == "delete"]
    assert all(np.isnan(e["probe_at_interv_l28"]) for e in dels)
    assert all(np.isfinite(e["d_margin_final"]) for e in edge_rows)
    swap = next(e for e in edge_rows if e["interv"] == "swap_xtrace")
    assert np.isfinite(swap["probe_at_interv_l28"])
