"""Smoke tests for cg_stage2_fg pure parts (contexts, site policy, merge)."""

import argparse
import json
import random

import pandas as pd

from scripts.causal_graph import cg_stage2_fg as s2


def _trace():
    return {"trace_id": "tr0", "fork_t": 1, "question": "Q?",
            "steps": ["s0", "s1", "s2", "s3"], "wrong_step": "w1",
            "xprob_step": "x1", "alt_pos_step": "p1", "gt_answer": "4",
            "split": "test"}


def test_fork_contexts_layout():
    ctxs = dict(s2.fork_contexts(_trace()))
    assert set(ctxs) == {"base", "swap_wrong", "swap_xprob", "swap_pos"}
    assert ctxs["base"] == "Q?\ns0\ns1\n"
    assert ctxs["swap_wrong"] == "Q?\ns0\nw1\n"
    tr = _trace() | {"alt_pos_step": None}
    assert "swap_pos" not in dict(s2.fork_contexts(tr))


def test_onpolicy_contexts_no_delete_and_curve():
    tr = {"traj_uid": "g0", "problem": "P",
          "solution": "a one.\n\nb two.\n\nc three."}
    pool = [("g1", "other step words"), ("g2", "more words here")]
    ctxs = s2.onpolicy_contexts(tr, [1], pool, random.Random(0))
    names = [n for n, _ in ctxs]
    assert names == ["prefix_0", "prefix_1", "prefix_2", "swap_xtrace_1"]
    swap_prompt = dict(ctxs)["swap_xtrace_1"]
    assert "a one." in swap_prompt and "b two." not in swap_prompt


def test_pick_sites_policy():
    probe = {0: 0.1, 1: 5.0, 2: 4.0, 3: -1.0, 4: 0.0}
    sites = s2.pick_sites({}, 5, probe, random.Random(0), "probe_top2_rand2")
    assert 1 in sites and 2 in sites and len(sites) == 4
    assert s2.pick_sites({}, 3, {}, random.Random(0), "all") == [0, 1, 2]


def test_merge_gates(tmp_path):
    rows = []
    rng = random.Random(0)
    for ti in range(30):
        for ctx, p in (("base", 0.8), ("swap_wrong", 0.2), ("swap_xprob", 0.7),
                       ("swap_pos", 0.75)):
            for ri in range(8):
                rows.append({"arm": "forks", "trace_id": f"tr{ti}",
                             "split": "test", "context": ctx, "t": 1,
                             "rollout_idx": ri, "correct": rng.random() < p,
                             "gradeable": True, "pred": "4",
                             "traj_correct": None})
    d = tmp_path / "stage2"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "fg_rollouts_shard0.parquet")
    args = argparse.Namespace(run_dir=tmp_path)
    s2.merge(args)
    gates = json.loads((d / "gates_stage2.json").read_text())
    assert gates["fg_wrong_effect"]["mean_delta"] < -0.3
    assert gates["fg_wrong_effect"]["p_wilcoxon_less"] < 0.01
    assert gates["G3_fg_wrong_vs_swap_xprob"]["auc"] > 0.6
    assert gates["G2_fg_power"]["k_rollouts"] == 8
    curves = pd.read_parquet(d / "fg_curves.parquet")
    assert {"solve_rate", "ci_lo", "ci_hi"} <= set(curves.columns)
