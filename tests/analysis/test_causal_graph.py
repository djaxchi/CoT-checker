"""Unit tests for src/analysis/causal_graph.py (pure + tiny-tensor parts)."""

import math
import random

import pytest
import torch

from src.analysis.causal_graph import (
    SEP_TOKEN_ID,
    assemble_ids,
    cand_token_ids,
    classify_site,
    encode_pieces,
    entropy_at,
    fg_influence,
    is_influential_fg,
    is_influential_tf,
    join_forks_to_golden,
    length_matched_step,
    localize_drops,
    null_quantile,
    per_span_mean_logprob,
    probe_logits_at,
    solve_curve,
    wilson_ci,
)


class StubTok:
    """Whitespace tokenizer: each word -> a stable int id; leading space kept as
    part of the first word (enough to test the lead-space rule)."""

    def __call__(self, text, add_special_tokens=False):
        ids = [hash(w) % 50000 for w in text.split(" ") if w != ""]
        if text.startswith(" "):
            ids = [7] + ids  # marker for the leading space
        return {"input_ids": ids}


# ---------------------------------------------------------------------------
# join_forks_to_golden
# ---------------------------------------------------------------------------

def _fork(q="Q1", t=2, prefix=("s1", "s2"), correct="s3", wrong="bad"):
    return {"fork_id": "f1", "question": q, "step_index": t + 1,
            "prefix_steps": list(prefix), "correct": correct, "wrong": wrong,
            "gt_answer": "42", "pre_generated_answer": None, "wrong_finals": [],
            "phase": 2}


def _golden(q="Q1", steps=("s1", "s2", "s3", "s4", "s5")):
    return {"trajectory_id": "traj1", "question": q, "steps": list(steps),
            "gt_answer": "42", "phase": 2}


def test_join_basic():
    counters = {}
    out = join_forks_to_golden([_fork()], [_golden()], counters=counters)
    assert len(out) == 1
    tr = out[0]
    assert tr["fork_t"] == 2
    assert tr["steps"] == ["s1", "s2", "s3", "s4", "s5"]
    assert tr["wrong_step"] == "bad"
    # fork correct == golden step at t -> no paraphrase control
    assert tr["alt_pos_step"] is None
    assert counters["joined"] == 1


def test_join_alt_pos_when_sibling_differs():
    out = join_forks_to_golden([_fork(correct="s3-alt")], [_golden()])
    assert out[0]["alt_pos_step"] == "s3-alt"


def test_join_rejects_prefix_mismatch_and_short_downstream():
    counters = {}
    out = join_forks_to_golden(
        [_fork(prefix=("s1", "sX")), _fork()],
        [_golden(steps=("s1", "s2", "s3", "s4"))],  # only 1 downstream step
        min_downstream=2, counters=counters)
    assert out == []
    assert counters["prefix_mismatch"] == 1
    assert counters["too_short_downstream"] == 1


def test_join_missing_question():
    counters = {}
    out = join_forks_to_golden([_fork(q="Q9")], [_golden()], counters=counters)
    assert out == [] and counters["no_golden_for_question"] == 1


# ---------------------------------------------------------------------------
# control pool
# ---------------------------------------------------------------------------

def test_length_matched_step_excludes_and_matches():
    pool = [("a", "one two three"), ("b", "one two three four five six seven"),
            ("self", "one two three")]
    rng = random.Random(0)
    s = length_matched_step(rng, pool, target_words=3, exclude_key="self", top_n=1)
    assert s == "one two three"
    with pytest.raises(ValueError):
        length_matched_step(rng, [("self", "x")], 1, "self")


def test_length_matched_step_deterministic():
    pool = [(f"k{i}", "w " * i) for i in range(1, 20)]
    a = length_matched_step(random.Random(3), pool, 10, "none")
    b = length_matched_step(random.Random(3), pool, 10, "none")
    assert a == b


# ---------------------------------------------------------------------------
# tokenization layout
# ---------------------------------------------------------------------------

def test_assemble_ids_spans_and_boundaries():
    pieces = [[1, 2, 3], [4, 5], [6]]
    full, spans, bounds = assemble_ids(pieces)
    assert full == [1, 2, 3, SEP_TOKEN_ID, 4, 5, SEP_TOKEN_ID, 6, SEP_TOKEN_ID]
    assert spans == [(0, 3), (4, 6), (7, 8)]
    assert bounds == [3, 6, 8]
    for b in bounds:
        assert full[b] == SEP_TOKEN_ID


def test_encode_pieces_separate():
    tok = StubTok()
    ids = encode_pieces(tok, ["a b", "c"])
    assert len(ids) == 2 and len(ids[0]) == 2 and len(ids[1]) == 1


def test_cand_token_ids_lead_space_rule():
    tok = StubTok()
    with_space = cand_token_ids(tok, "\nSo the final answer is", ["42"])
    assert with_space[0][0] == 7  # leading space injected
    no_space = cand_token_ids(tok, "answer:\n", ["42"])
    assert no_space[0][0] != 7  # suffix ends in whitespace -> no lead


# ---------------------------------------------------------------------------
# teacher-forced measurements
# ---------------------------------------------------------------------------

def test_per_span_mean_logprob_exact():
    # vocab 2, 4 positions; logits force known probabilities for next tokens
    V = 2
    logits = torch.full((4, V), 0.0)
    logits[0] = torch.tensor([math.log(0.9), math.log(0.1)])  # predicts pos 1
    logits[1] = torch.tensor([math.log(0.2), math.log(0.8)])  # predicts pos 2
    logits[2] = torch.tensor([math.log(0.5), math.log(0.5)])  # predicts pos 3
    ids = torch.tensor([0, 0, 1, 1])
    vals = per_span_mean_logprob(logits, ids, [(1, 3), (3, 4), (0, 1)])
    expected_span1 = (math.log(0.9) + math.log(0.8)) / 2
    assert vals[0] == pytest.approx(expected_span1, abs=1e-5)
    assert vals[1] == pytest.approx(math.log(0.5), abs=1e-5)
    assert math.isnan(vals[2])  # span starting at 0 with 1 token: nothing predicted


def test_entropy_at_uniform_and_peaked():
    logits = torch.zeros((2, 4))
    logits[1] = torch.tensor([100.0, 0.0, 0.0, 0.0])
    ent = entropy_at(logits, [0, 1])
    assert ent[0] == pytest.approx(math.log(4), abs=1e-4)
    assert ent[1] == pytest.approx(0.0, abs=1e-3)


def test_probe_logits_at():
    hidden = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    w = torch.tensor([3.0, 1.0])
    out = probe_logits_at(hidden, [0, 1], w, b=0.5)
    assert out == pytest.approx([3.5, 2.5])


# ---------------------------------------------------------------------------
# free-generation statistics
# ---------------------------------------------------------------------------

def test_wilson_ci_bounds_and_degenerate():
    lo, hi = wilson_ci(8, 8)
    assert 0.6 < lo < 1.0 and hi == 1.0
    lo0, hi0 = wilson_ci(0, 8)
    assert lo0 == 0.0 and 0 < hi0 < 0.4
    assert all(math.isnan(v) for v in wilson_ci(0, 0))


def test_solve_curve_and_drops():
    curve = solve_curve([[True] * 8, [True] * 6 + [False] * 2, [False] * 8])
    rates = [c["solve_rate"] for c in curve]
    assert rates == [1.0, 0.75, 0.0]
    # biggest drop first: prefix 1 -> 2 (0.75), then 0 -> 1 (0.25)
    assert localize_drops(rates, min_drop=0.25) == [1, 0]
    assert localize_drops(rates, min_drop=0.8) == []


def test_fg_influence_direction_and_ci():
    edge = fg_influence([True] * 12, [False] * 12)
    assert edge["delta"] == -1.0 and edge["recovery_rate"] == 0.0
    assert is_influential_fg(edge)
    null_edge = fg_influence([True] * 6 + [False] * 6, [True] * 6 + [False] * 6)
    assert null_edge["delta"] == 0.0 and not is_influential_fg(null_edge)


# ---------------------------------------------------------------------------
# taxonomy + calibration
# ---------------------------------------------------------------------------

def test_null_quantile_interpolates():
    assert null_quantile([-1.0, 2.0, 3.0, -4.0], q=0.5) == pytest.approx(2.5)
    assert math.isnan(null_quantile([]))


def test_classify_site_cells():
    assert classify_site(True, True) == "detected_influential"
    assert classify_site(True, False) == "detected_inert"
    assert classify_site(False, True) == "undetected_influential"
    assert classify_site(False, False) == "undetected_inert"


def test_is_influential_tf():
    assert is_influential_tf(-2.0, null_thresh=1.0)
    assert not is_influential_tf(-0.5, null_thresh=1.0)
    assert not is_influential_tf(-2.0, null_thresh=float("nan"))
