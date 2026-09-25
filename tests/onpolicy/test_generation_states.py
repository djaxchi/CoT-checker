"""Tests for reading step states out of one full-sequence pass."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.onpolicy.score_traces_generation_states import gather_step_states  # noqa: E402


def _h(n: int, d: int = 3) -> torch.Tensor:
    # row i carries the value i, so a slice is identified by its first entry
    return torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, d)


def test_spans_are_offset_by_the_prompt():
    h = _h(10)
    out = gather_step_states(h, n_prompt=4, spans=[(0, 2), (2, 6)], t_max=512)
    assert [x[:, 0].tolist() for x in out] == [[4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]


def test_t_max_keeps_the_first_tokens():
    out = gather_step_states(_h(10), n_prompt=0, spans=[(0, 10)], t_max=3)
    assert out[0][:, 0].tolist() == [0.0, 1.0, 2.0]


def test_empty_span_reads_its_boundary_token_instead_of_dropping_the_step():
    out = gather_step_states(_h(10), n_prompt=2, spans=[(0, 3), (3, 3)], t_max=512)
    assert len(out) == 2
    assert out[1][:, 0].tolist() == [4.0]  # generated position 2, the step's boundary
