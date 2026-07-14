"""Tests for src/analysis/transition_operator.py.

The patched-forward tests run on a tiny randomly initialized Qwen2 (4 blocks,
hidden 64) so they exercise the real architecture path without any download.
The three patch invariants encode the v0.2 review fixes:
  - identity: patching a position with its own state reproduces unpatched logits;
  - propagation: a real patch changes logits at LATER (suffix) positions, i.e. the
    patched boundary's recomputed K/V is what suffix tokens attend to;
  - last-layer dead end: patching the final stream cannot reach suffix positions
    (why L28 is not a valid patch layer for Target B).
"""

from __future__ import annotations

import pytest
import torch

from src.analysis.transition_operator import (
    answer_type,
    belief_from_scores,
    build_candidates,
    candidate_mean_logprobs,
    extract_wrong_finals,
    forward_with_boundary_patch,
    gold_margin,
    integer_perturbations,
    kl_from_logits,
    normalize_answer,
    recovery_from_logits,
)


# ---------------------------------------------------------------------------
# candidate machinery
# ---------------------------------------------------------------------------

def test_answer_type_variants():
    assert answer_type("29") == "integer"
    assert answer_type("-3") == "integer"
    assert answer_type("0.25") == "decimal"
    assert answer_type("3/4") == "fraction"
    assert answer_type("\\frac{3}{4}") == "fraction"
    assert answer_type("63\\pi") == "latex_expr"
    assert answer_type("x^2") == "latex_expr"
    assert answer_type("xy") == "has_letters"
    assert answer_type("(3,4)") == "other"


def test_normalize_answer():
    assert normalize_answer("  29 ") == "29"
    assert normalize_answer("\\$0.25") == "0.25"
    assert normalize_answer("\\text{A,B,E}") == "A,B,E"
    assert normalize_answer("3/4", gold_uses_frac=True) == "\\frac{3}{4}"
    assert normalize_answer("3/4", gold_uses_frac=False) == "3/4"
    assert normalize_answer("1  +  5i") == "1 + 5i"


def test_integer_perturbations_deterministic():
    import random
    a = integer_perturbations("12", random.Random(0), 5)
    b = integer_perturbations("12", random.Random(0), 5)
    assert a == b
    assert "12" not in a
    assert len(a) == 5


def test_build_candidates_gold_first_and_typed():
    cands = build_candidates(
        "29", pre_generated="31", wrong_finals=["17", "3/4"],
        corpus_pool=["55", "\\frac{1}{2}", "60"], k=8, seed=0)
    assert cands[0] == "29"
    assert len(cands) == 8
    assert len(set(cands)) == 8
    # the fraction wrong_final and corpus fraction are type-filtered out
    assert "3/4" not in cands and "\\frac{1}{2}" not in cands
    assert "31" in cands and "17" in cands


def test_build_candidates_relaxes_type_when_pool_dry():
    cands = build_candidates("\\cot x", corpus_pool=["12", "13", "14", "15",
                                                     "16", "17", "18"], k=8, seed=0)
    assert cands[0] == "\\cot x"
    assert len(cands) == 8  # filled by relaxed-type corpus answers


def test_extract_wrong_finals():
    sample = {"label": {"steps": [
        {"completions": [
            {"text": "wrong path\n# Answer\n\n42", "rating": -1},
            {"text": "right path", "rating": 1},
            {"text": "also wrong but no final", "rating": -1},
        ]},
        {"completions": [
            {"text": "# Answer\n\n42", "rating": -1},   # duplicate
            {"text": "# Answer\n\n7", "rating": 0},     # not -1: ignored
        ]},
    ]}}
    assert extract_wrong_finals(sample) == ["42"]


def test_belief_and_margin():
    scores = [-1.0, -2.0, -3.0]
    b = belief_from_scores(scores)
    assert pytest.approx(float(b.sum()), abs=1e-6) == 1.0
    assert b[0] > b[1] > b[2]
    assert pytest.approx(gold_margin(scores)) == 1.0


# ---------------------------------------------------------------------------
# patched forward on a tiny Qwen2
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=331, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128)
    model = Qwen2ForCausalLM(config).eval()
    input_ids = torch.randint(0, 331, (1, 12))
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return model, input_ids, out


BOUNDARY = 7


def test_patch_identity(tiny):
    model, input_ids, out = tiny
    own_state = out.hidden_states[2][:, BOUNDARY, :]
    logits = forward_with_boundary_patch(model, input_ids, hs_index=2,
                                         boundary_pos=BOUNDARY,
                                         patched_state=own_state)
    assert torch.allclose(logits, out.logits, atol=1e-5)


def test_patch_propagates_to_suffix_positions(tiny):
    model, input_ids, out = tiny
    donor = out.hidden_states[2][:, BOUNDARY, :] + 0.5
    logits = forward_with_boundary_patch(model, input_ids, hs_index=2,
                                         boundary_pos=BOUNDARY,
                                         patched_state=donor)
    # positions before the boundary are untouched (causality)
    assert torch.allclose(logits[:, :BOUNDARY], out.logits[:, :BOUNDARY], atol=1e-5)
    # the boundary readout moves
    assert not torch.allclose(logits[:, BOUNDARY], out.logits[:, BOUNDARY], atol=1e-4)
    # LATER positions move too: suffix tokens attend to the patched boundary's
    # recomputed K/V (v0.2 fix 1; Target B's causal path)
    assert not torch.allclose(logits[:, BOUNDARY + 1:], out.logits[:, BOUNDARY + 1:],
                              atol=1e-4)


def test_last_layer_patch_cannot_reach_suffix(tiny):
    model, input_ids, out = tiny
    n_layers = model.config.num_hidden_layers
    donor = out.hidden_states[n_layers][:, BOUNDARY, :] + 0.5
    logits = forward_with_boundary_patch(model, input_ids, hs_index=n_layers,
                                         boundary_pos=BOUNDARY,
                                         patched_state=donor)
    # boundary logits move (final norm + LM head see the patch) ...
    assert not torch.allclose(logits[:, BOUNDARY], out.logits[:, BOUNDARY], atol=1e-4)
    # ... but no blocks remain to carry it into later positions (v0.2 fix 2:
    # why L28 is not a valid patch layer for Target B)
    assert torch.allclose(logits[:, BOUNDARY + 1:], out.logits[:, BOUNDARY + 1:],
                          atol=1e-5)


def test_recovery_bounds(tiny):
    _, _, out = tiny
    actual = out.logits[:, -1]
    pre = actual + torch.randn_like(actual)
    assert pytest.approx(recovery_from_logits(actual, actual, pre), abs=1e-6) == 1.0
    assert pytest.approx(recovery_from_logits(actual, pre, pre), abs=1e-6) == 0.0
    assert kl_from_logits(actual, actual) == pytest.approx(0.0, abs=1e-6)


def test_candidate_mean_logprobs_matches_manual(tiny):
    model, input_ids, out = tiny
    context = input_ids[0].tolist()
    cands = [[5, 9], [17, 3, 21]]
    scores = candidate_mean_logprobs(model, context, cands, pad_id=0,
                                     device="cpu")
    # manual, unbatched, no padding
    for cand, got in zip(cands, scores):
        ids = torch.tensor([context + cand])
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        lp = torch.log_softmax(logits.float(), dim=-1)
        want = float(torch.stack([
            lp[0, len(context) - 1 + i, t] for i, t in enumerate(cand)]).mean())
        assert got == pytest.approx(want, abs=1e-5)
