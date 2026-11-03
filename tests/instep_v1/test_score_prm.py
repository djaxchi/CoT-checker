"""InStep V1 P1 (scripts/instep_v1/score_prm.py): trace rebuilding, PRM input, scores, gate."""

import json

import numpy as np
import pytest
import torch

from scripts.instep_v1.score_prm import (SEP, SYSTEM, batches, gate_set, load_traces, merge_set, messages,
                                         p_correct_at_seps, score_rows)


def write_jl(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def items(tid, problem, steps, sep="\n"):
    return [{"uid": f"{tid}::{k}", "id": tid, "step_idx": str(k), "problem": problem,
             "prefix": sep.join(steps[:k]), "candidate_step": s, "label": 0} for k, s in enumerate(steps)]


STEPS = ["- Step 1: 2+2=4 ", "- Step 2: 4*3=12", "<Answer>: 12"]


def test_load_traces_orders_strips_and_counts_bad_prefix(tmp_path):
    rows = items("t0", "Q0", STEPS) + items("t1", "Q1", ["- Step 1: a", "<Answer>: b"])
    rows = rows[::-1]  # shuffled order on disk
    rows[0]["step_idx"] = "1"
    p = tmp_path / "x.items.jsonl"
    write_jl(p, rows)
    trs, bad = load_traces(p)
    assert [t["id"] for t in trs] == ["t1", "t0"]
    t0 = trs[1]
    assert t0["steps"] == ["- Step 1: 2+2=4", "- Step 2: 4*3=12", "<Answer>: 12"]
    assert t0["uids"] == ["t0::0", "t0::1", "t0::2"] and t0["problem"] == "Q0"
    assert bad == 0
    # an extra blank line in a prefix is whitespace, not a mismatch; a different word is
    rows2 = items("t0", "Q0", STEPS)
    rows2[2]["prefix"] = rows2[2]["prefix"].replace("\n", "\n\n")
    rows2[1]["prefix"] = "- Step 1: 2+2=5"
    write_jl(p, rows2)
    assert load_traces(p)[1] == 1
    assert len(load_traces(p, smoke=1)[0]) == 1


def test_load_traces_rejects_uid_gap(tmp_path):
    rows = items("t0", "Q", STEPS)
    rows[1]["uid"] = "t0::5"
    p = tmp_path / "x.items.jsonl"
    write_jl(p, rows)
    with pytest.raises(ValueError):
        load_traces(p)


def test_messages_match_reprobe_prm_input():
    m = messages("What is 2+2?", ["- Step 1: 4", "<Answer>: 4"])
    assert m[0] == {"role": "system", "content": SYSTEM}
    assert SYSTEM == "Please reason step by step, and put your final answer within \\boxed{}."
    assert m[1] == {"role": "user", "content": "What is 2+2?"}
    assert m[2]["content"] == f"- Step 1: 4{SEP}<Answer>: 4{SEP}"


def test_p_correct_reads_channel_1_at_separators_only():
    sep = 9
    ids = torch.tensor([[1, 2, sep, 3, sep, 0, 0], [1, sep, 0, 0, 0, 0, 0]])
    logits = torch.zeros(2, 7, 2)
    logits[0, 2] = torch.tensor([0.0, 2.0])   # P(correct) high
    logits[0, 4] = torch.tensor([2.0, 0.0])   # P(correct) low
    logits[1, 1] = torch.tensor([0.0, 0.0])
    logits[1, 3] = torch.tensor([0.0, 5.0])   # padding: must be ignored even if it were a sep
    ids[1, 3] = sep
    pc = p_correct_at_seps(logits.to(torch.bfloat16), ids, sep, [5, 2])
    assert len(pc[0]) == 2 and len(pc[1]) == 1
    assert pc[0][0] == pytest.approx(1 / (1 + np.exp(-2)), abs=1e-2)
    assert pc[0][1] == pytest.approx(1 / (1 + np.exp(2)), abs=1e-2)
    assert pc[1][0] == pytest.approx(0.5)


def test_score_rows_orientation_and_count_check():
    t = {"id": "t0", "uids": ["t0::0", "t0::1"], "steps": ["a", "b"]}
    rows = score_rows([t], [[0.9, 0.2]])
    assert rows == [{"uid": "t0::0", "p": pytest.approx(0.1)}, {"uid": "t0::1", "p": pytest.approx(0.8)}]
    with pytest.raises(ValueError):
        score_rows([t], [[0.9]])


def test_batches_respect_token_budget_and_cover_all():
    lens = [10, 50, 30, 200, 5, 50]
    bs = batches(lens, 100)
    assert sorted(i for b in bs for i in b) == list(range(len(lens)))
    assert bs[0] == [3]  # over-long alone
    for b in bs:
        assert len(b) == 1 or len(b) * max(lens[i] for i in b) <= 100


def test_merge_set_orders_and_requires_every_uid(tmp_path):
    write_jl(tmp_path / "s.shard0.jsonl", [{"uid": "a::1", "p": 0.2}])
    write_jl(tmp_path / "s.shard1.jsonl", [{"uid": "a::0", "p": 0.1}])
    assert merge_set(tmp_path, "s", 2, ["a::0", "a::1"]) == 2
    out = [json.loads(l) for l in open(tmp_path / "s.jsonl")]
    assert out == [{"uid": "a::0", "p": 0.1}, {"uid": "a::1", "p": 0.2}]
    assert not (tmp_path / "s.shard0.jsonl").exists()
    write_jl(tmp_path / "t.shard0.jsonl", [{"uid": "a::0", "p": 0.1}])
    with pytest.raises(ValueError):
        merge_set(tmp_path, "t", 1, ["a::0", "a::1"])


def test_gate_stored_is_minus_p_correct(tmp_path):
    rng = np.random.default_rng(0)
    pc = rng.uniform(size=40)
    labels = (rng.uniform(size=40) < 1 - pc).astype(int).tolist()
    labels[0] = -1
    key = "PRM_Qwen/Qwen2.5-Math-PRM-7B"
    tr = [{"trace_id": "t", "labels": labels, "scores": {key: (-pc).tolist()}}]
    write_jl(tmp_path / "x.traces.jsonl", tr)
    ours = {f"t::{k}": 1 - pc[k] for k in range(40)}
    g = gate_set(tmp_path / "x.traces.jsonl", ours, key)
    assert g["n"] == 39 and g["pass"] and g["spearman"] == pytest.approx(1.0)
    assert g["max_abs_diff_p_vs_1_plus_stored"] < 1e-9 and g["d_ap"] == pytest.approx(0.0)
    flipped = {u: -v for u, v in ours.items()}
    assert not gate_set(tmp_path / "x.traces.jsonl", flipped, key)["pass"]
    assert gate_set(tmp_path / "x.traces.jsonl", ours, "PRM_other") is None


def test_bf16_round_matches_torch_and_gate_handles_stored_ties(tmp_path):
    from scripts.instep_v1.score_prm import bf16_round
    x = np.random.default_rng(1).uniform(size=1000).astype(np.float32)
    assert np.array_equal(bf16_round(x), torch.tensor(x).to(torch.bfloat16).float().numpy().astype(np.float64))
    # saturated PRM: stored = -bf16(P), ours = 1 - P in float32 -> raw Spearman hurt by ties, bf16 gate passes
    rng = np.random.default_rng(2)
    pc = 1 - rng.exponential(0.01, size=300).clip(0, 0.5)
    labels = (rng.uniform(size=300) < 0.2).astype(int).tolist()
    key = "PRM_Qwen/Qwen2.5-Math-PRM-7B"
    write_jl(tmp_path / "x.traces.jsonl", [{"trace_id": "t", "labels": labels,
                                             "scores": {key: (-bf16_round(pc)).tolist()}}])
    g = gate_set(tmp_path / "x.traces.jsonl", {f"t::{k}": 1 - pc[k] for k in range(300)}, key)
    assert g["spearman_bf16"] == pytest.approx(1.0) and g["d_ap_bf16"] == pytest.approx(0.0)
    assert g["pass"] and g["n_unique_stored"] < 300


def test_merge_set_rerun_rechecks_final_file(tmp_path):
    write_jl(tmp_path / "s.shard0.jsonl", [{"uid": "a::0", "p": 0.1}, {"uid": "a::1", "p": 0.2}])
    merge_set(tmp_path, "s", 1, ["a::0", "a::1"])
    assert merge_set(tmp_path, "s", 1, ["a::0", "a::1"]) == 2
    with pytest.raises(ValueError):
        merge_set(tmp_path, "s", 1, ["a::0", "a::1", "a::2"])
