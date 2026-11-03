"""Fast tests for scripts/instep_v1/score_confidence.py (no real model)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.instep_v1 import score_confidence as sc
from scripts.reprobe_cmp.reprobe_probe import load_traces


def tiny_model(vocab: int = 97):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                      tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval()


# ---------------------------------------------------------------- estimator math
def test_token_stats_matches_definition():
    torch.manual_seed(1)
    logits = torch.randn(5, 11)
    tgt = torch.tensor([0, 3, 10, 4, 4])
    ll, ent = sc.token_stats(logits, tgt)
    p = torch.softmax(logits.double(), -1)
    assert torch.allclose(ll.double(), p.log()[torch.arange(5), tgt], atol=1e-5)
    assert torch.allclose(ent.double(), -(p * p.log()).sum(-1), atol=1e-5)


def test_token_stats_ignores_masked_entries():
    logits = torch.tensor([[0.0, 0.0, -float("inf")]])
    ll, ent = sc.token_stats(logits, torch.tensor([1]))
    assert ll.item() == pytest.approx(math.log(0.5))
    assert ent.item() == pytest.approx(math.log(2))


def test_claim_scores_are_lm_polygraph_reductions():
    ll = np.log(np.array([0.5, 0.25, 1.0]))
    ent = np.array([0.1, 0.7, 0.3])
    s = sc.claim_scores(ll, ent)
    assert s["maxprob"] == pytest.approx(-np.sum(ll))
    assert s["ppl"] == pytest.approx(-np.mean(ll))
    assert s["maxent"] == pytest.approx(0.7)
    assert sc.claim_scores(np.array([]), np.array([])) == {"maxprob": 0.0, "maxent": 0.0, "ppl": 0.0}


def test_length_batches_cover_once_and_respect_budget():
    L = [5, 30, 7, 12, 40, 3, 3, 25]
    bs = sc.length_batches(L, max_tokens=60)
    flat = [i for b in bs for i in b]
    assert sorted(flat) == list(range(len(L)))
    for b in bs:
        assert len(b) == 1 or len(b) * max(L[i] for i in b) <= 60


# ---------------------------------------------------------------- U1 against a naive forward
def _trace_file(tmp_path: Path, rng) -> Path:
    rows = []
    for j in range(6):
        ctx = int(rng.integers(3, 9))
        n_reply = int(rng.integers(4, 30))
        ids = rng.integers(0, 97, ctx + n_reply).tolist()
        cuts = sorted(rng.choice(np.arange(1, n_reply), size=min(2, n_reply - 1), replace=False).tolist())
        bounds = [0] + cuts + [n_reply]
        steps = [{"aligned_token_ids": list(range(bounds[k], bounds[k + 1]))} for k in range(len(bounds) - 1)]
        rows.append({"trace_id": f"t{j}", "input_ids": ids, "context_len": ctx, "steps": steps,
                     "labels": {"ds": [0] * len(steps)}})
    p = tmp_path / "x.traces.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_u1_scores_match_naive_single_trace_forward(tmp_path):
    rng = np.random.default_rng(0)
    path = _trace_file(tmp_path, rng)
    traces = load_traces(path, None)
    model = tiny_model()
    got, empty = sc.u1_scores(model, traces, torch.device("cpu"), pad=0, max_tokens=80)
    assert empty == 0
    raw = sc.jl(path)
    for t, r in zip(traces, raw):
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([r["input_ids"]])).logits[0].double(), -1)
        ctx = r["context_len"]
        for k, s in enumerate(r["steps"]):
            pos = [ctx + i for i in s["aligned_token_ids"]]  # absolute token positions
            ll = np.array([lp[q - 1, r["input_ids"][q]].item() for q in pos])
            ent = np.array([-(lp[q - 1].exp() * lp[q - 1]).sum().item() for q in pos])
            uid = f"{t['trace_id']}::{k}"
            assert got["maxprob"][uid] == pytest.approx(-ll.sum(), rel=1e-3, abs=1e-3)
            assert got["ppl"][uid] == pytest.approx(-ll.mean(), rel=1e-3, abs=1e-3)
            assert got["maxent"][uid] == pytest.approx(ent.max(), rel=1e-3, abs=1e-3)


def test_load_traces_prompt_text_layout_matches_input_ids_layout():
    """Test-set traces (prompt_text + reply_ids) normalize to the same ids and positions as
    input_ids + context_len."""
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.jsonl"
        b = Path(d) / "b.jsonl"
        steps = [{"aligned_token_ids": [0, 1]}, {"aligned_token_ids": [2, 3, 4]}]
        a.write_text(json.dumps({"trace_id": "x", "prompt_text": "abc", "reply_ids": [5, 6, 7, 8, 9],
                                 "steps": steps, "labels": [0, 1]}) + "\n")
        b.write_text(json.dumps({"trace_id": "x", "input_ids": [97, 98, 99, 5, 6, 7, 8, 9], "context_len": 3,
                                 "steps": steps, "labels": {"ds": [0, 1]}}) + "\n")
        ta, tb = load_traces(a, Tok())[0], load_traces(b, Tok())[0]
    assert ta["ids"] == tb["ids"] and ta["ctx"] == tb["ctx"] == 3
    assert ta["steps"] == tb["steps"] == [[2, 3], [4, 5, 6]]  # logit rows predicting tokens 3..7


# ---------------------------------------------------------------- P(True)
def test_ptrue_prompt_layout():
    s = sc.ptrue_prompt("What is 2+2?", "", "- Step 1: 2+2=5")
    assert s.startswith("<|im_start|>system\nYou are a careful verifier")
    assert "Previous steps:\n(none)\n\nCurrent step:\n- Step 1: 2+2=5\n\nIs the current step correct? Answer Yes or No.<|im_end|>" in s
    assert s.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "math verifier" not in s
    s2 = sc.ptrue_prompt("Q", "- Step 1: a\n- Step 2: b", "- Step 3: c")
    assert "Previous steps:\n- Step 1: a\n- Step 2: b\n\nCurrent step:\n- Step 3: c" in s2


def _qwen_tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3-8B tokenizer not cached")


def test_ptrue_prompt_equals_chat_template():
    tok = _qwen_tok()
    s = sc.ptrue_prompt("P", "", "S")
    user = s.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    want = tok.apply_chat_template([{"role": "system", "content": sc.PTRUE_SYSTEM}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert s == want
    assert sc.yes_no_ids(tok) == (9454, 2753)


def test_ptrue_from_logits_is_renormalized_p_no():
    p = sc.ptrue_from_logits(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0]))
    want = [math.exp(0) / (math.exp(2) + math.exp(0)), math.exp(1) / (1 + math.exp(1))]
    assert p.tolist() == pytest.approx(want, rel=1e-6)


def test_ptrue_scores_match_unpadded_forward():
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 90 + 3 for c in text][-40:]}

        def encode(self, text, add_special_tokens=False):
            return {"Yes": [1], "No": [2]}[text]
    items = [{"uid": f"t::{k}", "problem": "q" * k, "prefix": "p" * (3 * k), "candidate_step": "s" + "x" * k}
             for k in range(5)]
    model = tiny_model()
    got = sc.ptrue_scores(model, Tok(), items, torch.device("cpu"), pad=0, max_tokens=100)
    for it in items:
        ids = Tok()(sc.ptrue_prompt(it["problem"], it["prefix"], it["candidate_step"]))["input_ids"]
        with torch.no_grad():
            lg = model(torch.tensor([ids])).logits[0, -1]
        assert got[it["uid"]] == pytest.approx(torch.sigmoid(lg[2] - lg[1]).item(), abs=1e-4)


# ---------------------------------------------------------------- io and gate
def test_write_and_merge_shards(tmp_path):
    root = tmp_path / "ptrue"
    sc.write_scores(root / "_shards" / "val.0of2.jsonl", {"a::0": 0.1, "a::2": 0.3})
    sc.write_scores(root / "_shards" / "val.1of2.jsonl", {"a::1": 0.2})
    sc.write_scores(root / "_shards" / "pb_math.0of2.jsonl", {"b::0": 0.5})
    with pytest.raises(SystemExit):
        sc.merge_shards(root / "_shards", root)
    sc.write_scores(root / "_shards" / "pb_math.1of2.jsonl", {"b::1": 0.6})
    assert sorted(sc.merge_shards(root / "_shards", root)) == ["pb_math", "val"]
    rows = sc.jl(root / "val.jsonl")
    assert {r["uid"]: r["p"] for r in rows} == {"a::0": 0.1, "a::1": 0.2, "a::2": 0.3}
    assert all(set(r) == {"uid", "p"} for r in rows)


def test_gate_set_passes_on_monotone_copy_and_fails_on_noise():
    rng = np.random.default_rng(0)
    traces = []
    for j in range(60):
        n = 5
        st = rng.random(n).tolist()
        traces.append({"trace_id": f"t{j}", "steps": [{}] * n, "labels": [int(v > 0.7) for v in st],
                       "scores": {"MaximumClaimProbability": st}})
    exact = {"maxprob": {f"t{j}::{k}": v for j, t in enumerate(traces)
                         for k, v in enumerate(t["scores"]["MaximumClaimProbability"])}}
    r = sc.gate_set(traces, exact)["maxprob"]
    assert r["pass"] and r["spearman"] == pytest.approx(1.0) and r["ap_ours"] == pytest.approx(r["ap_stored"])
    noise = {"maxprob": {u: float(rng.random()) for u in exact["maxprob"]}}
    assert not sc.gate_set(traces, noise)["maxprob"]["pass"]


def test_claimed_shards_split_work_between_processes(tmp_path):
    import argparse
    a = argparse.Namespace(shard="0/5", claim_dir=tmp_path / "claims")
    g1, g2 = sc.claimed_shards(a), sc.claimed_shards(a)
    got1 = [next(g1)]
    got2 = [next(g2), next(g2)]
    got1 += list(g1)
    got2 += list(g2)
    assert sorted(got1 + got2) == [(j, 5) for j in range(5)]
    assert list(sc.claimed_shards(argparse.Namespace(shard="2/5", claim_dir=None))) == [(2, 5)]
