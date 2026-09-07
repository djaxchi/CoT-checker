"""Tests for step-level guided decoding.

The scientific claim rests on three things being true of the loop: guided picks
the candidate the checker likes best, random does not consult the checker at all,
and plain never branches. If any of those silently broke, the arms would stop
being a controlled comparison while still producing plausible-looking numbers,
so each is pinned here.
"""

from __future__ import annotations

import random
import types

import pytest

from scripts.onpolicy import online_bon


class FakeChecker:
    """Scores by table lookup and records that it was consulted."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def score_steps(self, problem, prior_steps, candidates):
        self.calls += 1
        return [self.table[c] for c in candidates]


def _args(**kw):
    base = dict(n_candidates=3, temperature=1.5, top_p=0.95, max_new_tokens=64,
                max_steps=4, device="cpu")
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture
def patched(monkeypatch):
    """sample_candidates returns a fixed pool, so only selection is under test."""
    pools = {}

    def fake_sample(backbone, tok, problem, prior_steps, n, *a, **k):
        depth = len(prior_steps)
        pool = pools.get(depth, ["good", "mid", "bad"])
        # 7 tokens per sampled candidate, discarded branches included
        return pool[:n], 7 * n

    monkeypatch.setattr(online_bon, "sample_candidates", fake_sample)
    monkeypatch.setattr(online_bon, "grade", lambda sol, gold: "good" in sol)
    return pools


def test_guided_picks_the_candidate_with_the_lowest_suspicion(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    r = online_bon.rollout("guided", "p", "g", None, None, ch, _args(max_steps=1),
                           random.Random(0))
    assert r["steps"] == ["good"]
    assert r["chosen_scores"] == [0.1]
    assert r["pool_scores"] == [[0.1, 0.5, 0.9]]
    assert ch.calls == 1


def test_guided_is_not_fooled_by_candidate_order(patched):
    # best candidate last: a loop that quietly kept the first would pass the
    # previous test and fail this one
    ch = FakeChecker({"good": 0.9, "mid": 0.5, "bad": 0.1})
    r = online_bon.rollout("guided", "p", "g", None, None, ch, _args(max_steps=1),
                           random.Random(0))
    assert r["steps"] == ["bad"]


def test_random_arm_never_consults_the_checker(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    r = online_bon.rollout("random", "p", "g", None, None, ch, _args(max_steps=1),
                           random.Random(0))
    assert ch.calls == 0, "the control arm must not use the checker"
    assert r["steps"][0] in {"good", "mid", "bad"}
    assert r["chosen_scores"] == []


def test_random_arm_actually_varies_with_the_seed(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    picks = {
        online_bon.rollout("random", "p", "g", None, None, ch, _args(max_steps=1),
                           random.Random(s))["steps"][0]
        for s in range(30)
    }
    assert len(picks) > 1, "a control that always picks the same candidate is not random"


def test_plain_arm_draws_a_single_candidate(patched, monkeypatch):
    seen = {}

    def fake_sample(backbone, tok, problem, prior_steps, n, *a, **k):
        seen["n"] = n
        return ["good"][:n], 7 * n

    monkeypatch.setattr(online_bon, "sample_candidates", fake_sample)
    ch = FakeChecker({"good": 0.1})
    online_bon.rollout("plain", "p", "g", None, None, ch, _args(max_steps=1),
                       random.Random(0))
    assert seen["n"] == 1, "the base policy must not branch"
    assert ch.calls == 0


def test_rollout_stops_at_a_boxed_answer(patched):
    patched[0] = ["step one", "step one", "step one"]
    patched[1] = ["the answer is \\boxed{7}"] * 3
    ch = FakeChecker({"step one": 0.2, "the answer is \\boxed{7}": 0.2})
    r = online_bon.rollout("guided", "p", "7", None, None, ch, _args(max_steps=8),
                           random.Random(0))
    assert r["n_steps"] == 2, "generation must stop once the answer is written"


def test_rollout_respects_the_step_budget(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    r = online_bon.rollout("guided", "p", "g", None, None, ch, _args(max_steps=3),
                           random.Random(0))
    assert r["n_steps"] == 3


def test_is_final_only_fires_on_a_boxed_answer():
    assert online_bon.is_final("so x = \\boxed{42}")
    assert not online_bon.is_final("next we compute x = 42")
    assert not online_bon.is_final("consider the box of weight 42")


# --------------------------------------------------------------------------
# cost accounting: branching is not free and the numbers must say so
# --------------------------------------------------------------------------

def test_branching_arms_are_charged_for_discarded_candidates(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    args = _args(max_steps=2, n_candidates=3)
    guided = online_bon.rollout("guided", "p", "g", None, None, ch, args,
                                random.Random(0))
    plain = online_bon.rollout("plain", "p", "g", None, None, ch, args,
                               random.Random(0))
    assert guided["gen_tokens"] == 3 * plain["gen_tokens"], (
        "a branching arm must be billed for every sampled candidate, not just "
        "the one it kept, or guided decoding looks cheaper than it is")


def test_checker_workload_is_recorded_for_the_guided_arm_only(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    args = _args(max_steps=2, n_candidates=3)
    g = online_bon.rollout("guided", "p", "g", None, None, ch, args, random.Random(0))
    r = online_bon.rollout("random", "p", "g", None, None, ch, args, random.Random(0))
    assert g["checker_calls"] == 2 and g["scored_candidates"] == 6
    assert r["checker_calls"] == 0 and r["scored_candidates"] == 0
