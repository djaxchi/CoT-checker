import numpy as np
import pytest

from src.analysis.downstream_audit import aggregate, describe_pool, outcome, stable_fold


def test_rescue_accounting_keeps_fractional_vote_credit():
    p = describe_pool(["yes", "no"], [True, False])
    assert p["case"] == "vote_tie_with_correct"
    assert p["majority"] == .5
    assert p["oracle"] - p["majority"] == .5


def test_top_score_ties_are_order_invariant():
    p = describe_pool(["a", "b", "b"], [True, False, False])
    assert outcome(p, np.array([.4, .4, .4]), "rerank") == pytest.approx(1/3)
    assert outcome(p, np.array([.4, .4, .4]), "tiebreak") == 0


def test_zero_vote_weights_fall_back_to_majority():
    p = describe_pool(["a", "a", "b"], [True, True, False])
    assert outcome(p, np.ones(3), "wvote") == 1
    assert outcome(p, np.ones(3), "safe_wvote") == 1


def test_group_mean_must_beat_every_wrong_group():
    p = describe_pool(["a", "a", "b", "c"], [False, False, True, False])
    # Correct b beats the plurality a, but c still wins the selector.
    assert outcome(p, np.array([.9, .8, .2, .1]), "group_mean") == 0


def test_answer_identity_consistency_is_required():
    with pytest.raises(ValueError, match="Inconsistent correctness"):
        describe_pool(["a", "a"], [True, False])


def test_missing_scores_are_not_imputed_into_a_ranking():
    with pytest.raises(ValueError, match="finite"):
        aggregate([.2, np.nan], "worst")


def test_skip_first_and_mean_use_step_scores():
    assert aggregate([1, .1, .3], "worst_skip_first") == .3
    assert aggregate([.1, .3], "mean") == pytest.approx(.2)


def test_shared_question_has_shared_fold():
    assert stable_fold("same_question", 5) == stable_fold("same_question", 5)
    assert 0 <= stable_fold("same_question", 5) < 5


def test_crossfit_excludes_both_temperatures_of_the_held_out_question():
    from scripts.analysis.audit_instruct_downstream import BOUNDARY, PRM, crossfit
    a, b = f"{BOUNDARY}|worst|rerank", f"{BOUNDARY}|mean|rerank"
    pools = {t: [{"fold": i, "majority": .5} for i in range(5)] for t in ["10", "07"]}
    matrices = {t: {"majority": np.full(5, .5), a: np.array([1., .8, .8, .8, .8]),
                    b: np.array([0., .1, .1, .1, .1]),
                    f"{PRM}|worst|rerank": np.full(5, .7)} for t in pools}
    calls = {t: {k: np.zeros(5) for k in matrices[t]} for t in pools}
    first = crossfit(pools, matrices, calls)["probe_aggregation_rerank"]["choices"][0]
    for t in pools:
        matrices[t][a][0], matrices[t][b][0] = 0., 1.
    second = crossfit(pools, matrices, calls)["probe_aggregation_rerank"]["choices"][0]
    assert first == second == {"fold": 0, "policy": a}
