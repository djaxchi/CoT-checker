"""Abandoning a trajectory mid-generation: what must never be free.

The two ways this simulation could lie to itself are charging nothing for a
trajectory that was partly written, and killing a candidate after its answer is
already on the page. Both are pinned here.
"""

from __future__ import annotations

import numpy as np

from src.analysis.onpolicy_abandon import decide, policy_outcome, run_problem, walk


def cand(scores, tokens, answer, correct, final=None):
    return {"step_scores": list(scores), "step_tokens": list(tokens),
            "answer": answer, "correct": correct,
            "final_score": max(scores) if final is None else final}


def test_an_abandoned_prefix_is_paid_for_up_to_the_step_that_killed_it():
    c = cand([0.1, 0.9, 0.1, 0.1], [10, 20, 30, 40], "2", False)
    spent, done = walk(c, tau=0.5, min_steps=1)
    assert done is False and spent == 30  # the first two steps, not all hundred


def test_a_clean_trajectory_pays_for_every_step_and_finishes():
    c = cand([0.1, 0.2], [10, 20], "2", True)
    assert walk(c, tau=0.5, min_steps=1) == (30, True)


def test_the_last_step_is_never_an_abandonment():
    """By the final step the answer is written; killing it pays the whole cost
    and throws away the thing that was bought."""
    c = cand([0.1, 0.99], [10, 20], "2", True)
    spent, done = walk(c, tau=0.5, min_steps=1)
    assert done is True and spent == 30


def test_min_steps_protects_an_early_spike():
    c = cand([0.99, 0.1, 0.1], [10, 10, 10], "2", True)
    assert walk(c, tau=0.5, min_steps=1)[1] is False
    assert walk(c, tau=0.5, min_steps=3)[1] is True


def test_the_running_maximum_is_what_condemns_a_prefix_not_the_latest_step():
    c = cand([0.9, 0.1, 0.1], [10, 10, 10], "2", True)
    spent, done = walk(c, tau=0.5, min_steps=2)
    assert done is False and spent == 20  # step 1's spike still counts at step 2


def test_a_run_that_abandons_everything_still_has_to_answer_and_pay():
    """Otherwise a threshold that kills the whole pool would score a free zero at
    a fraction of the cost, which is the cheapest way to fake a good frontier."""
    cands = [cand([0.9, 0.9], [10, 90], "2", True) for _ in range(3)]
    run = run_problem(cands, [0, 1, 2], tau=0.5, min_steps=1, n_completions=2)
    assert run["forced"] is True
    assert run["finished"] == [2]
    assert run["tokens"] == 10 + 10 + 100  # two kills, then one carried to the end


def test_drawing_stops_as_soon_as_enough_have_finished():
    cands = [cand([0.1], [10], "2", True) for _ in range(5)]
    run = run_problem(cands, [0, 1, 2, 3, 4], tau=0.5, min_steps=1, n_completions=2)
    assert run["n_drawn"] == 2 and run["tokens"] == 20


def test_only_finished_candidates_get_a_vote():
    cands = [cand([0.1], [10], "2", True), cand([0.9, 0.1], [10, 10], "3", False)]
    out = policy_outcome(cands, [1, 0], tau=0.5, min_steps=1, n_completions=1)
    assert out["finished"] == [0]
    assert out["majority"] == 1.0


def test_the_two_answer_rules_split_only_on_a_tie():
    cands = [cand([0.8], [10], "2", True), cand([0.2], [10], "3", False)]
    assert decide(cands, [0, 1]) == {"majority": 0.5, "majority+tie": 0.0}
    flipped = [cand([0.2], [10], "2", True), cand([0.8], [10], "3", False)]
    assert decide(flipped, [0, 1]) == {"majority": 0.5, "majority+tie": 1.0}


def test_a_pool_where_nothing_parsed_is_a_miss_not_a_crash():
    assert decide([cand([0.1], [10], None, False)], [0]) == {"majority": 0.0,
                                                             "majority+tie": 0.0}


def test_an_infinite_threshold_reproduces_plain_sampling():
    cands = [cand([0.9, 0.9, 0.9], [10, 10, 10], "2", True) for _ in range(3)]
    run = run_problem(cands, [0, 1, 2], tau=np.inf, min_steps=1, n_completions=3)
    assert run["tokens"] == 90 and len(run["finished"]) == 3


def test_past_the_deadline_a_candidate_is_committed_and_pays_in_full():
    """A rule with no deadline mostly kills late, once the running maximum has had
    many steps to cross; a late kill has already paid for most of the trace."""
    c = cand([0.1, 0.1, 0.99, 0.99], [10, 10, 10, 10], "2", True)
    assert walk(c, tau=0.5, min_steps=1, max_steps=2) == (40, True)
    assert walk(c, tau=0.5, min_steps=1, max_steps=None) == (30, False)


def test_the_deadline_still_allows_a_kill_inside_it():
    c = cand([0.99, 0.1, 0.1], [10, 10, 10], "2", True)
    assert walk(c, tau=0.5, min_steps=1, max_steps=2) == (10, False)
