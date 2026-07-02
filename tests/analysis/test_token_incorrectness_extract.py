"""The GPU extraction path must match the pure numpy reference semantics.

Builds a fake forward output (CPU tensors) and checks _certainty_and_scores against
src.analysis.token_trajectory, including the critical off-by-one alignment between a
step token and the logit row that predicts it (logits[t-1]).
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.analysis.token_trajectory import (
    per_token_certainty,
    probe_scores,
    representation_stats,
)

_SCRIPT = (Path(__file__).resolve().parent.parent.parent
           / "scripts" / "analysis" / "s3_token_incorrectness_extract.py")
_spec = importlib.util.spec_from_file_location("s3_token_incorrectness_extract", _SCRIPT)
extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract)


def test_certainty_and_scores_matches_reference():
    rng = np.random.default_rng(0)
    B, S, V, H = 2, 12, 17, 8
    logits = rng.standard_normal((B, S, V)).astype(np.float32)
    hs28 = rng.standard_normal((B, S, H)).astype(np.float32)
    hs20 = rng.standard_normal((B, S, H)).astype(np.float32)
    ids = rng.integers(0, V, size=S).tolist()
    w = rng.standard_normal(H).astype(np.float32)
    bias = 0.37
    b_idx, first, last = 1, 3, 9      # step tokens ids[3..9]

    out = SimpleNamespace(logits=torch.tensor(logits))
    hs_layers = {28: torch.tensor(hs28), 20: torch.tensor(hs20)}
    device = torch.device("cpu")
    step_ids, arrs, scores, reprs = extract._certainty_and_scores(
        out, b_idx, first, last, ids, hs_layers,
        torch.tensor(w, dtype=torch.float32), bias, device, active_tau=0.5)

    # reference: predictive rows for tokens [first..last] are logits[first-1..last-1]
    pred = logits[b_idx, first - 1:last, :]
    target = np.array(ids[first:last + 1])
    ref = per_token_certainty(pred, target)

    assert step_ids == ids[first:last + 1]
    for k in ("nll", "entropy", "logit_gap", "p_top1", "p_realized"):
        np.testing.assert_allclose(arrs[k], ref[k], rtol=1e-4, atol=1e-4)

    for li, hs in ((28, hs28), (20, hs20)):
        span = hs[b_idx, first:last + 1, :]
        ref_score = probe_scores(span, w, bias)
        np.testing.assert_allclose(scores[li], ref_score, rtol=1e-4, atol=1e-4)
        ref_repr = representation_stats(span, active_tau=0.5)
        for rk in ("hidden_l2", "hidden_absmax", "hidden_nact"):
            np.testing.assert_allclose(reprs[li][rk], ref_repr[rk], rtol=1e-4, atol=1e-4)


def test_select_step_items_drops_anchors_and_caps_forks():
    rows = [
        {"role": "anchor", "candidate_step": "", "label": -1, "fork_id": "A"},
        {"role": "positive", "candidate_step": "x", "label": 0, "fork_id": "A"},
        {"role": "negative", "candidate_step": "y", "label": 1, "fork_id": "A"},
        {"role": "positive", "candidate_step": "z", "label": 0, "fork_id": "B"},
        {"role": "negative", "candidate_step": " ", "label": 1, "fork_id": "B"},  # empty step
        {"role": "negative", "candidate_step": "w", "label": 1, "fork_id": "C"},
    ]
    kept = extract.select_step_items(rows, max_forks=None)
    assert [r["fork_id"] for r in kept] == ["A", "A", "B", "C"]  # anchor + blank dropped

    capped = extract.select_step_items(rows, max_forks=2)
    assert {r["fork_id"] for r in capped} == {"A", "B"}          # first two forks only


def test_select_step_items_noop_on_heldout_rows():
    # Heldout rows carry no fork_id/role: max_forks must not drop them.
    rows = [{"uid": "u0", "candidate_step": "a", "label": 0},
            {"uid": "u1", "candidate_step": "b", "label": 1}]
    assert extract.select_step_items(rows, max_forks=1) == rows


def test_alignment_is_shifted_not_self():
    # If the code mistakenly used logits[t] (not t-1) the nll would match a different
    # target; this guards the shift explicitly with a peaked distribution.
    B, S, V, H = 1, 6, 5, 4
    logits = np.full((B, S, V), -10.0, np.float32)
    # make row r strongly predict token (r % V)
    for r in range(S):
        logits[0, r, r % V] = 10.0
    ids = [0, 1, 2, 3, 4, 0]
    first, last = 2, 4                 # step tokens ids[2..4] = [2,3,4]
    out = SimpleNamespace(logits=torch.tensor(logits))
    hs = {28: torch.zeros((B, S, H))}
    _, arrs, _, _ = extract._certainty_and_scores(
        out, 0, first, last, ids, hs, torch.zeros(H), 0.0, torch.device("cpu"))
    # predictive row for token position t is t-1: row1->tok1, row2->tok2, row3->tok3.
    # realized step tokens are [2,3,4]; row1 predicts 1 (not 2) -> high nll on first,
    # rows 2,3 predict 2,3 (not 3,4) -> also shifted -> all high nll.
    assert arrs["nll"][0] > 5.0     # row1 predicts token 1, realized 2 -> surprised
    assert arrs["nll"][1] > 5.0     # row2 predicts token 2, realized 3 -> surprised
