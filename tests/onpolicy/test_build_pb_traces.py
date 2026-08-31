"""The adapter has one job: hand the harness a ProcessBench-shaped evaluation set.

Two things it must get right. The labels have to be first-error indices, not the
outcome labels the generator writes, because F1_PB scores localisation. And where
the judge disagrees with the grader it has to resolve the conflict by a stated
policy and count how often that happened, since the count on correct
trajectories is the judge's false-alarm rate on the on-policy distribution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.encode_processbench_token_store import flatten  # noqa: E402
from scripts.onpolicy.build_pb_traces import build, read_labels, resolve_label  # noqa: E402

SOL = "Step one.\n\nStep two.\n\nSo \\boxed{4}."


def traj(uid, correct=True, gradeable=True, solution=SOL):
    return {"traj_uid": uid, "fork_id": uid.split("::")[1], "problem": "p?",
            "gold": "4", "pred": "4" if correct else "5", "correct": correct,
            "gradeable": gradeable, "solution": solution}


def run(trajs, labels, **kw):
    kw.setdefault("correct_policy", "trust_outcome")
    kw.setdefault("no_error_policy", "drop")
    kw.setdefault("min_steps", 2)
    kw.setdefault("unjudged_correct", "drop")
    return build(trajs, labels, kw["correct_policy"], kw["no_error_policy"],
                 kw["min_steps"], kw["unjudged_correct"])


def test_first_error_label_is_carried_not_the_outcome_label():
    traces, _, _ = run([traj("onpolicy::p1::g0", correct=False)],
                       {"onpolicy::p1::g0": 1})
    assert len(traces) == 1
    assert traces[0]["label"] == 1        # not "every step of a wrong solution"
    assert traces[0]["n_steps"] == 3


def test_a_correct_trajectory_gets_no_error_and_the_conflict_is_counted():
    traces, _, tally = run([traj("onpolicy::p1::g0", correct=True)],
                           {"onpolicy::p1::g0": 2})
    assert traces[0]["label"] == -1
    assert tally["conflict_correct_overridden"] == 1


def test_trust_judge_keeps_the_judges_index_instead():
    traces, _, _ = run([traj("onpolicy::p1::g0", correct=True)],
                       {"onpolicy::p1::g0": 2}, correct_policy="trust_judge")
    assert traces[0]["label"] == 2


def test_a_wrong_answer_with_no_error_found_is_dropped_by_default():
    traces, outcomes, tally = run([traj("onpolicy::p1::g0", correct=False)],
                                  {"onpolicy::p1::g0": -1})
    assert traces == []
    assert tally["conflict_error_dropped"] == 1
    assert outcomes[0]["in_pb_traces"] is False   # still available to the T2 arm


def test_last_step_policy_keeps_it_at_the_final_step():
    traces, _, _ = run([traj("onpolicy::p1::g0", correct=False)],
                       {"onpolicy::p1::g0": -1}, no_error_policy="last_step")
    assert traces[0]["label"] == 2


def test_out_of_range_and_ungradeable_and_short_traces_are_dropped():
    trajs = [traj("onpolicy::p1::g0", correct=False),
             traj("onpolicy::p2::g0", gradeable=False),
             traj("onpolicy::p3::g0", solution="One line only.")]
    traces, _, tally = run(trajs, {"onpolicy::p1::g0": 9, "onpolicy::p3::g0": 0})
    assert traces == []
    assert tally["label_out_of_range"] == 1
    assert tally["ungradeable"] == 1
    assert tally["too_few_steps"] == 1


def test_an_unjudged_trajectory_is_dropped_not_guessed():
    traces, _, tally = run([traj("onpolicy::p1::g0", correct=True)], {})
    assert traces == []
    assert tally["unjudged"] == 1


def test_output_is_what_the_processbench_encoder_consumes():
    traces, _, _ = run([traj("onpolicy::p1::g0", correct=False)],
                       {"onpolicy::p1::g0": 1})
    flat = flatten([json.loads(json.dumps(t)) for t in traces], "onpolicy")
    assert [r["step_idx"] for r in flat] == [0, 1, 2]
    assert {r["label"] for r in flat} == {1}
    assert flat[0]["prefix"] == ""
    assert flat[2]["prefix"] == "Step one.\n\nStep two."
    for r in flat:
        assert {"id", "step_idx", "label", "n_steps", "global_index"} <= set(r)


def test_resolve_label_is_total_over_the_policies():
    for judged in (None, -1, 0, 5):
        for correct in (True, False):
            for cp in ("trust_outcome", "trust_judge", "drop_conflict"):
                for np_ in ("drop", "last_step"):
                    label, tag = resolve_label(judged, correct, 3, cp, np_)
                    assert isinstance(tag, str) and tag
                    assert label is None or label == -1 or 0 <= label < 3


def test_only_incorrect_trajectories_plus_an_audit_sample_go_to_the_judge():
    """A correct trajectory takes -1 from the grader, so asking a judge about it
    buys nothing but cost. A sample goes anyway to measure false alarms."""
    from scripts.onpolicy.build_pb_traces import judge_traces
    trajs = ([traj(f"onpolicy::p{i}::g0", correct=False) for i in range(5)] +
             [traj(f"onpolicy::q{i}::g0", correct=True) for i in range(20)] +
             [traj("onpolicy::short::g0", correct=False, solution="one line")])
    out = judge_traces(trajs, min_steps=2, correct_sample=3)
    assert sum(1 for t in out if not t["traj_correct"]) == 5
    assert sum(1 for t in out if t["traj_correct"]) == 3
    assert all(len(t["steps"]) >= 2 for t in out)
    assert {"id", "problem", "steps", "traj_correct", "gold"} <= set(out[0])


def test_a_correct_trajectory_can_join_without_a_judge_and_is_marked_as_such():
    """A paid judge has a budget, and a correct trajectory already has a label
    from the grader. The assumption it carries is that no wrong step was later
    repaired, so the provenance is recorded per trace."""
    trajs = [traj("onpolicy::p1::g0", correct=True),
             traj("onpolicy::p2::g0", correct=False)]
    traces, _, tally = run(trajs, {"onpolicy::p2::g0": 1}, unjudged_correct="no_error")
    by_id = {t["id"]: t for t in traces}
    assert by_id["onpolicy::p1::g0"]["label"] == -1
    assert by_id["onpolicy::p1::g0"]["label_source"] == "grader"
    assert by_id["onpolicy::p2::g0"]["label_source"] == "judge"
    assert tally["unjudged_correct_from_grader"] == 1


def test_an_unjudged_incorrect_trajectory_is_still_dropped():
    """Its first error has no source at all, so it cannot enter the set."""
    traces, _, tally = run([traj("onpolicy::p1::g0", correct=False)], {},
                           unjudged_correct="no_error")
    assert traces == []
    assert tally["unjudged"] == 1


def test_a_label_on_a_problem_the_model_never_solves_is_dropped(tmp_path):
    """The rollout rule marks the first step after which nothing reaches the
    answer. On a problem the model never solves, that is step 0 every time, for
    reasons that have nothing to do with the step."""
    f = tmp_path / "labels.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [
        {"traj_uid": "a", "first_error": 0, "base_rate": 0.0},
        {"traj_uid": "b", "first_error": 2, "base_rate": 0.25},
    ]) + "\n")
    labels, tally = read_labels([f], min_base_rate=0.01)
    assert labels == {"b": 2}
    assert tally["dropped_unsolvable"] == 1


def test_judge_labels_have_no_base_rate_and_survive_the_filter(tmp_path):
    f = tmp_path / "labels.jsonl"
    f.write_text(json.dumps({"traj_uid": "a", "first_error": 1}) + "\n")
    labels, _ = read_labels([f], min_base_rate=0.5)
    assert labels == {"a": 1}
