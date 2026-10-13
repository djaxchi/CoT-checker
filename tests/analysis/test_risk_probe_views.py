"""risk_probe_v0 view builder (docs/risk_probe_v0_plan.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "factorized_step_labels"))
from build_context_views import Tok  # noqa: E402
from build_risk_probe_views import fork_view  # noqa: E402


class CharTok:
    """Character tokenizer; special token 0 marks add_special_tokens."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": ([0] if add_special_tokens else []) + [ord(c) for c in text]}


TRACE = {"problem": "1+1?", "steps": ["Add one.", "So it is 2.", "<Answer>: 2"]}
FORK = {"fork_id": "rp_x#s1", "trace_id": "rp_x", "problem_key": "pk_x", "step_claim_idx": 1,
        "step_text": "- Step 2: So it is 2.\n", "validity": "correct", "y_ds": 0}


def test_fork_view_matches_context_ablation_full_format():
    inp, meta, enc = fork_view(Tok(CharTok()), FORK, TRACE)
    assert inp["trace_id"] == "rp_x#1" and inp["steps"] == ["Add one.", "So it is 2."]
    assert inp["header"] == "Problem:\n1+1?\n\nSolution:\n"
    assert meta["fork_id"] == "rp_x#s1" and meta["y"] == [0] and meta["split"] == "fsl_forks"
    h = len(inp["header"]) + 1                       # header chars + special token
    assert enc["fwd_step_starts"][0] == h
    # compact store: header rows, the separator before step i, step i's tokens
    assert enc["n_tokens"] == h + len("So it is 2.") and enc["step_starts"] == [h]
    assert enc["keep"] == [[0, h - 1], [enc["fwd_step_starts"][1] - 1, enc["fwd_step_ends"][1]]]


def test_fork_view_rejects_text_mismatch():
    bad = {**FORK, "step_text": "- Step 2: So it is 3.\n"}
    with pytest.raises(ValueError):
        fork_view(Tok(CharTok()), bad, TRACE)
