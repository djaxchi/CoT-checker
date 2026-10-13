"""Tests for factorized_step_labels_v1 (docs/factorized_step_labels_v1_plan.md)."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.factorized_step_labels import (OffsetMeanAccumulator, extract_reprobe_answer,
                                                 gold_answer, grade_fork, marker_len,
                                                 mean_baseline, parse_reprobe_row,
                                                 position_bucket, risk_stats, select_forks,
                                                 spearman)
from src.eval.math_grade import is_equiv
from src.onpolicy.prompts import REPROBE_INSTRUCTION, reprobe_prompt

EOS = 9999


def enc(s: str) -> list[int]:
    return [ord(c) for c in s]


def dec(ids) -> str:
    return "".join("<|im_end|>" if i == EOS else chr(i) for i in ids)


def make_row(lines: list[str], answer: str, eos: bool = True, prompt: str = "P:"):
    reply = "".join(lines) + answer
    ids = enc(prompt) + enc(reply) + ([EOS] if eos else [])
    claims, c = [], 0
    for ln in lines:
        claims.append({"aligned_token_ids": list(range(c, c + len(ln))), "sentence": ln})
        c += len(ln)
    claims.append({"aligned_token_ids": list(range(c, c + len(answer) + 1)), "sentence": answer})
    return ids, enc(prompt), claims


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #

def test_reprobe_prompt_shape():
    s = reprobe_prompt("What is 1+1?")
    assert s.startswith("<|im_start|>user\nYou will be presented with a <Question>.")
    assert s.endswith("<Question>: What is 1+1?<|im_end|>\n<|im_start|>assistant\n"
                      "<think>\n\n</think>\nReasoning Steps:\n")
    assert REPROBE_INSTRUCTION.endswith("Now answer:\n<Question>: ")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def test_marker_len_char_tokens():
    line = enc("- Step 12: x = 1\n")
    assert dec(line[:marker_len(line, dec)]) == "- Step 12:"
    assert marker_len(enc("no marker here\n"), dec) == 0


def test_parse_row_spans_and_answer():
    lines = ["- Step 1: a = 1\n", "- Step 2: b = 2\n"]
    ids, pids, claims = make_row(lines, "<Answer>: 2")
    p, why = parse_reprobe_row(ids, pids, claims, dec, EOS)
    assert why == "ok"
    rep = p["reply_ids"]
    assert rep[-1] == EOS and dec(rep[:-1]) == "".join(lines) + "<Answer>: 2"
    assert [dec(rep[lo:hi]) for lo, hi in p["step_spans"]] == lines
    assert [dec(rep[c:hi]) for c, (_, hi) in zip(p["content_lo"], p["step_spans"])] == [" a = 1\n", " b = 2\n"]
    lo, hi = p["answer_span"]
    assert dec(rep[lo:hi]) == "<Answer>: 2" and p["step_claim_idx"] == [0, 1]


@pytest.mark.parametrize("mutate,reason", [
    ("prompt", "prompt_mismatch"), ("eos", "no_eos"), ("tile", "claims_not_tiling"),
    ("sentence", "sentence_decode_mismatch"),
])
def test_parse_row_rejections(mutate, reason):
    lines = ["- Step 1: a = 1\n", "- Step 2: b = 2\n"]
    ids, pids, claims = make_row(lines, "<Answer>: 2", eos=(mutate != "eos"))
    if mutate == "prompt":
        pids = enc("Q:")
    if mutate == "tile":
        claims[1]["aligned_token_ids"] = [x + 1 for x in claims[1]["aligned_token_ids"]]
    if mutate == "sentence":
        claims[0]["sentence"] = "- Step 1: a = 7\n"
    p, why = parse_reprobe_row(ids, pids, claims, dec, EOS)
    assert p is None and why == reason


def test_parse_row_answer_not_a_claim():
    lines = ["- Step 1: a = 1\n"]
    ids, pids, claims = make_row(lines, "<Answer>: 1")
    p, why = parse_reprobe_row(ids, pids, claims[:1], dec, EOS)
    assert why == "ok" and dec(p["reply_ids"][slice(*p["answer_span"])]) == "<Answer>: 1"


def test_parse_row_needs_answer_line():
    ids, pids, claims = make_row(["- Step 1: a = 1\n"], "so it is 1")
    p, why = parse_reprobe_row(ids, pids, claims[:1], dec, EOS)
    assert p is None and why == "no_answer_line"


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("reply,gold", [
    ("- Step 1: x\n<Answer>: 3/4", "\\frac{3}{4}"),
    ("<Answer>: 4√5", "4\\sqrt{5}"),
    ("<Answer>: 2√10", "2\\sqrt{10}"),
    ("<Answer>: (A) The coefficient of $x^2$", "\\text{(A)}"),
    ("<Answer>: 20/1296 = 5/324", "\\frac{5}{324}"),
    ("<Answer>: $\\boxed{7}$", "7"),
    ("<Answer>: $\\frac{1}{2}$.", "\\frac12"),
    ("<Answer>: (4, ∞)", "(4, \\infty)"),
])
def test_extract_reprobe_answer_matches_gold(reply, gold):
    assert is_equiv(extract_reprobe_answer(reply), gold)


def test_extract_without_answer_line_defers_to_project_extractor():
    assert extract_reprobe_answer("so the answer is \\boxed{12}") == "12"


def test_last_number_fallback_is_avoided():
    # math_grade's own extractor would read this as "4"
    assert extract_reprobe_answer("<Answer>: 3/4") == "3/4"


def test_gold_answer():
    assert gold_answer("so $\\boxed{\\frac{3}{16}}$.") == "\\frac{3}{16}"
    assert gold_answer("no box") is None


def test_grade_fork_harmonizes_and_fails_truncated():
    g = grade_fork([{"text": "<Answer>: 57", "truncated": False},
                    {"text": "<Answer>: 57.00", "truncated": False},
                    {"text": "<Answer>: 58", "truncated": False},
                    {"text": "<Answer>: 57", "truncated": True},
                    {"text": "nothing here", "truncated": False}], "57")
    assert [x["correct"] for x in g] == [True, True, False, False, False]
    assert g[0]["answer_key"] == g[1]["answer_key"]
    assert g[3]["gradeable"] is False and g[4]["gradeable"] is False


# --------------------------------------------------------------------------- #
# risk, selection, baselines
# --------------------------------------------------------------------------- #

def test_risk_stats_values():
    r = risk_stats([1, 1, 0, 0], [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], n_boot=200)
    assert r["err_a"] == 0.5
    assert r["err_c"] == [0.0, 0.25, 1.0]
    assert r["risk_diff"] == pytest.approx(0.5 - 1.25 / 3)
    assert r["risk_pct"] == pytest.approx(2 / 3)
    assert r["risk_z"] == pytest.approx(r["risk_diff"] / np.std([0, 0.25, 1.0], ddof=1))
    assert r["risk_diff_se"] > 0


def test_risk_stats_zero_spread():
    r = risk_stats([0] * 8, [[0] * 8, [0] * 8], n_boot=50)
    assert r["risk_diff"] == 0 and r["risk_z"] is None and r["risk_pct"] == 0.5


def test_position_bucket():
    assert [position_bucket(s, 3) for s in range(3)] == ["early", "mid", "late"]
    assert [position_bucket(s, 6) for s in range(6)] == ["early"] * 2 + ["mid"] * 2 + ["late"] * 2


def test_select_forks_one_per_problem_and_raises():
    cands = [{"trace_id": f"t{i}", "step": 0, "problem_key": f"p{i // 2}",
              "validity": "correct", "position": "early"} for i in range(6)]
    got = select_forks(cands, 3, [("correct", "early")])
    assert len({c["problem_key"] for c in got}) == 3
    with pytest.raises(ValueError):
        select_forks(cands, 4, [("correct", "early")])


def test_offset_mean_accumulator_and_baseline():
    acc = OffsetMeanAccumulator(d=2, max_offset=3, min_count=2)
    acc.add(np.array([[0, 0], [2, 2], [10, 10]]))         # body offsets 0,1; final 10
    acc.add(np.array([[2, 2], [4, 4], [6, 6], [20, 20]]))  # body offsets 0,1,2; final 20
    ref = acc.finalize()
    assert np.allclose(ref["offset_means"][0], [1, 1]) and np.allclose(ref["offset_means"][1], [3, 3])
    tail = (0 + 2 + 2 + 4 + 6) / 5
    assert np.allclose(ref["offset_means"][2], [tail, tail])  # offset 2 seen once < min_count
    assert np.allclose(ref["final_mean"], [15, 15])
    b = mean_baseline(ref, 5)
    assert b.shape == (5, 2)
    assert np.allclose(b[-1], [15, 15]) and np.allclose(b[3], [tail, tail])
    assert np.allclose(mean_baseline(ref, 1), [[15, 15]])


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [1, 1, 1]) is None


# --------------------------------------------------------------------------- #
# patching on a tiny random Qwen3
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=4, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    torch.manual_seed(0)
    m = transformers.Qwen3ForCausalLM(cfg).eval()
    ids = torch.randint(0, 64, (1, 24))
    return torch, m, ids


def test_self_patch_is_exact_noop_and_lands(tiny):
    torch, m, ids = tiny
    from src.analysis.factorized_step_labels import clean_states, patched_logprobs
    lo, hi, start = 6, 10, 10
    st, idx = clean_states(m, ids, [1, 2, 3], lo, hi)
    assert all(v == 0.0 for v in idx.values())          # block L-1 output == hidden_states[L]
    base, land0 = patched_logprobs(m, ids, start)
    assert land0 is None and base.shape[0] == ids.shape[1] - start
    for L in (1, 3):
        lp, land = patched_logprobs(m, ids, start, L, lo, hi, st[L])
        assert land == 0.0
        assert torch.allclose(lp, base, atol=1e-6)


def test_mean_patch_moves_logprobs_and_earlier_layer_moves_more(tiny):
    torch, m, ids = tiny
    from src.analysis.factorized_step_labels import clean_states, norm_matched_random, patched_logprobs
    lo, hi, start = 6, 10, 10
    st, _ = clean_states(m, ids, [1, 3], lo, hi)
    base, _ = patched_logprobs(m, ids, start)
    zero = torch.zeros_like(st[1])
    d1 = (patched_logprobs(m, ids, start, 1, lo, hi, zero)[0] - base).abs().sum()
    assert d1 > 1e-4
    r = norm_matched_random(st[3], torch.Generator().manual_seed(0))
    assert torch.allclose(r.norm(dim=-1), st[3].float().norm(dim=-1), atol=1e-4)
    lp, land = patched_logprobs(m, ids, start, 3, lo, hi, r)
    assert land == 0.0 and (lp - base).abs().sum() > 0
    # tokens before the span are untouched by any patch
    pre, _ = patched_logprobs(m, ids, 2, 1, lo, hi, zero)
    clean_pre, _ = patched_logprobs(m, ids, 2)
    assert torch.allclose(pre[:lo - 2 + 1], clean_pre[:lo - 2 + 1], atol=1e-6)


def test_necessity_from_lp_drops_boundary_token():
    from src.analysis.factorized_step_labels import necessity_from_lp
    clean = [-0.1, -0.2, -0.3, -0.4]          # tokens at absolute 10..13
    patched = [-29.0, -0.2, -1.3, -0.4]
    out = necessity_from_lp(clean, patched, 10, {"cont": (10, 14), "next": (10, 12), "ans": (12, 14)})
    assert out["boundary_drop"] == pytest.approx(28.9)
    assert out["cont"] == pytest.approx(1.0) and out["next"] == pytest.approx(0.0)
    assert out["ans"] == pytest.approx(1.0) and out["n_cont_xb"] == 3


def test_partial_spearman_removes_shared_driver():
    from src.analysis.factorized_step_labels import partial_spearman
    rng = np.random.default_rng(0)
    z = rng.normal(size=400)
    x, y = z + 0.1 * rng.normal(size=400), z + 0.1 * rng.normal(size=400)
    assert spearman(x, y) > 0.9
    assert abs(partial_spearman(x, y, [z])) < 0.2


def test_boot_ci_brackets_mean():
    from src.analysis.factorized_step_labels import boot_ci
    v = np.arange(100, dtype=float)
    lo, hi = boot_ci(lambda idx: float(v[idx].mean()), len(v), n_boot=500)
    assert lo < 49.5 < hi
