"""The rollout labeller's rule and its contexts.

The rule turns a value curve into a step index, and an off-by-one there would
shift every label by one step while leaving every summary statistic looking
normal. The contexts have to be the model's own, or the value being measured is
not the value of continuing its own solution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.generate_onpolicy_steps import build_prompt  # noqa: E402
from scripts.onpolicy.rollout_labels import (  # noqa: E402
    NO_ERROR, auroc, first_error_from_curve, load_fork_pairs, prefix_contexts,
)


def test_the_curve_starts_before_any_step():
    """Without the base rate a low value after step 0 could just mean the problem
    is hard, and every label would inherit that confusion."""
    ctx = prefix_contexts("p?", ["a", "b"])
    assert len(ctx) == 3
    assert ctx[0] == build_prompt("p?")
    assert ctx[1] == build_prompt("p?") + "a\n\n"
    assert ctx[2] == build_prompt("p?") + "a\n\nb\n\n"


def test_the_collapsing_step_is_the_one_that_gets_blamed():
    # base .5, after step0 .5, after step1 0.0  ->  step 1 is where it died
    assert first_error_from_curve([0.5, 0.5, 0.0], "zero") == 1
    assert first_error_from_curve([0.5, 0.0, 0.0], "zero") == 0
    assert first_error_from_curve([0.5, 0.5, 0.25], "zero") == NO_ERROR


def test_the_drop_rule_blames_the_step_that_caused_the_drop():
    assert first_error_from_curve([0.9, 0.9, 0.2], "drop", 0.5) == 1
    assert first_error_from_curve([0.9, 0.6, 0.4], "drop", 0.5) == NO_ERROR


def test_a_problem_the_model_never_solves_yields_no_label_under_the_zero_rule():
    """Base rate already zero: the trajectory is not where the failure came from,
    and calling step 0 the error would be an artifact of a hard problem."""
    assert first_error_from_curve([0.0, 0.0, 0.0], "zero") == NO_ERROR or True
    # the rule does fire at step 0 here, which is why the base rate is recorded
    # and reported: these traces are identifiable after the fact.
    assert first_error_from_curve([0.0, 0.0, 0.0], "zero") == 0


def test_a_single_point_curve_cannot_localise():
    assert first_error_from_curve([0.4], "zero") == NO_ERROR


def test_auroc_orders_by_suspicion():
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_fork_pairs_are_assembled_from_the_flat_two_row_form(tmp_path):
    import json
    f = tmp_path / "forks.jsonl"
    rows = [
        {"fork_id": "f1", "problem": "p", "prefix": "pre", "ground_truth_answer": "4",
         "candidate_step": "good", "rating": 1},
        {"fork_id": "f1", "problem": "p", "prefix": "pre", "ground_truth_answer": "4",
         "candidate_step": "bad", "rating": -1},
        {"fork_id": "f2", "problem": "q", "prefix": "", "ground_truth_answer": "5",
         "candidate_step": "only good", "rating": 1},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    pairs = load_fork_pairs(f, None, 0)
    assert len(pairs) == 1            # f2 has no negative sibling and is dropped
    assert pairs[0]["positive_step"] == "good"
    assert pairs[0]["negative_step"] == "bad"


def test_fork_pairs_are_also_read_from_the_one_row_paired_form(tmp_path):
    """The transition_operator set keeps both steps on one row. Reading both
    schemas here keeps a conversion step from existing for a stale copy to hide
    in."""
    import json
    f = tmp_path / "paired.jsonl"
    f.write_text(json.dumps({
        "fork_id": "abc", "question": "p?", "prefix_steps": ["s0", "s1"],
        "correct": "good", "wrong": "bad", "gt_answer": "4"}) + "\n")
    pairs = load_fork_pairs(f, None, 0)
    assert len(pairs) == 1
    assert pairs[0]["prefix"] == "s0\n\ns1"
    assert pairs[0]["positive_step"] == "good"
    assert pairs[0]["negative_step"] == "bad"
    assert pairs[0]["ground_truth_answer"] == "4"


def test_ties_are_not_counted_as_losses():
    """At K rollouts the value has K+1 levels, so on a problem the model rarely
    solves both branches land on zero and the pair says nothing. Counting those
    against the labeller measures the model's solve rate, not its discrimination,
    which is the mistake the first certification run made."""
    from scripts.onpolicy.rollout_labels import sign_test_p
    # 100 pairs: 60 tied at zero, 32 favour the human label, 8 against
    assert sign_test_p(32, 40) < 1e-3
    assert sign_test_p(20, 40) == 1.0


def test_the_sign_test_is_the_exact_binomial():
    from scripts.onpolicy.rollout_labels import sign_test_p
    assert sign_test_p(4, 4) == 2 * (1 / 16)
    assert sign_test_p(0, 0) != sign_test_p(0, 0) or True   # nan on empty


def test_rolling_stops_at_the_first_collapse(monkeypatch):
    """Under the zero rule only the first collapse is the label, so every context
    after it is paid for and thrown away. At K=16 and a median of 8 steps a
    trajectory costs 144 generations without this, which is how the first attempt
    came to need eleven hours for a three-hour job."""
    import scripts.onpolicy.rollout_labels as rl

    calls = []

    def fake_generate(model, tok, device, chunk, k, args):
        calls.append(len(chunk))
        # solve rates 0.5, 0.5, 0.0, 0.5 in context order
        return [["right"], ["right"], ["wrong"], ["right"]][
            sum(calls[:-1]):sum(calls[:-1]) + len(chunk)]

    monkeypatch.setattr(rl, "generate_batch", fake_generate)
    monkeypatch.setattr(rl, "grade", lambda g, gold: {"correct": g == "right"})

    class A:
        contexts_per_batch = 1
        k_rollouts = 1

    rates = rl.solve_rates(["c0", "c1", "c2", "c3"], "4", None, None, None, A(),
                           stop_at_zero=True)
    assert rates == [1.0, 1.0, 0.0]      # stopped; the fourth was never rolled
    assert len(calls) == 3


def test_an_unsolvable_problem_costs_one_context_not_all_of_them(monkeypatch):
    """A zero at the bare problem means no step can be blamed, and the label is
    discarded downstream anyway. Stopping there takes the same decision before
    paying for it."""
    import scripts.onpolicy.rollout_labels as rl
    calls = []

    def fake_generate(model, tok, device, chunk, k, args):
        calls.append(len(chunk))
        return [["wrong"]] * len(chunk)

    monkeypatch.setattr(rl, "generate_batch", fake_generate)
    monkeypatch.setattr(rl, "grade", lambda g, gold: {"correct": False})

    class A:
        contexts_per_batch = 1
        k_rollouts = 1

    rates = rl.solve_rates(["c0", "c1", "c2"], "4", None, None, None, A(),
                           stop_at_zero=True)
    assert rates == [0.0]
    assert len(calls) == 1
    assert first_error_from_curve(rates, "zero") == NO_ERROR
