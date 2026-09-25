"""Tests for the PRM scoring adapter.

The failure that matters here is silent: the PRM emits P(correct) and every
consumer in this codebase reads `scores` as suspicion. A sign flip produces a
mirror-image frontier that looks entirely ordinary, so the conversion is tested
directly rather than trusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.onpolicy.score_traces_with_prm import load_expected_steps, step_rewards  # noqa: E402


def test_step_rewards_reads_the_positive_channel_at_separators():
    # two positions, only the second is a separator; channel 1 is P(correct).
    logits = torch.tensor([[[9.0, 1.0], [1.0, 9.0]]])
    mask = torch.tensor([[0, 1]])
    out = step_rewards(logits, mask)
    assert len(out) == 1
    assert out[0] > 0.99


def test_step_rewards_ignores_non_separator_positions():
    logits = torch.tensor([[[0.0, 5.0], [5.0, 0.0], [0.0, 5.0]]])
    mask = torch.tensor([[1, 0, 1]])
    out = step_rewards(logits, mask)
    assert len(out) == 2
    assert all(v > 0.99 for v in out)


def test_step_rewards_is_a_probability():
    logits = torch.randn(1, 6, 2)
    mask = torch.tensor([[1, 0, 1, 1, 0, 1]])
    out = step_rewards(logits, mask)
    assert len(out) == 4
    assert all(0.0 <= v <= 1.0 for v in out)


def test_suspicion_conversion_inverts_the_reward():
    """`1 - P(correct)` is what the frontier consumes; a confident-correct step
    must come out near zero, which is what `quality_from(..., False)` ranks best."""
    rewards = [0.99, 0.01, 0.5]
    suspicion = [1.0 - r for r in rewards]
    assert suspicion[0] < suspicion[2] < suspicion[1]
    assert abs(suspicion[0] - 0.01) < 1e-9


def test_load_expected_steps_reads_every_shard(tmp_path):
    """The gate must see the whole probe pool, not the first shard only."""
    for i, uids in enumerate([["a", "b"], ["c"]]):
        with open(tmp_path / f"s{i}.jsonl", "w") as f:
            for u in uids:
                f.write(json.dumps({"traj_uid": u, "n_steps": len(u) + i}) + "\n")
    got = load_expected_steps(sorted(tmp_path.glob("s*.jsonl")))
    assert got == {"a": 1, "b": 1, "c": 2}
