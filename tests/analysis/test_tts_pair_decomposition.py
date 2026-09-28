"""Check the opportunity identity and the analysis's exclusion invariants."""

import itertools

import numpy as np
import pytest

from scripts.analysis.tts_pair_decomposition import decompose_pair_pool
from src.analysis.tts_frontier import majority_expected, tie_break, vote


def test_exact_pairs_match_ordered_frontier_including_score_ties():
    answers = ["a", "b", "b", "c"]
    correct = [True, False, False, False]
    quality = [0.8, 0.8, 0.1, 0.5]
    stats = decompose_pair_pool(answers, correct, quality)
    baseline, selected = [], []
    for pair in itertools.permutations(range(4), 2):
        groups, tied, _ = vote(answers, pair)
        baseline.append(majority_expected(correct, groups, tied))
        selected.append(tie_break(correct, quality, groups, tied))
    assert stats["majority"] == pytest.approx(np.mean(baseline))
    assert stats["selected"] == pytest.approx(np.mean(selected))
    assert stats["selected"] - stats["majority"] == pytest.approx(
        stats["mixed_selected_mass"] - 0.5 * stats["mixed_pair_rate"])


def test_perfect_scorer_recovers_all_available_headroom():
    stats = decompose_pair_pool(["a", "b", "c"], [True, False, False], [1, 0, 0])
    assert stats["selected"] == stats["oracle"] == pytest.approx(2 / 3)
    assert stats["majority"] == pytest.approx(1 / 3)


def test_correct_answer_fragmentation_does_not_create_n2_correctness_lift():
    stats = decompose_pair_pool(["26", "26.00"], [True, True], [0, 1])
    assert stats["answer_disagreement_rate"] == 1
    assert stats["mixed_pair_rate"] == 0
    assert stats["selected"] == stats["majority"] == 1


@pytest.mark.parametrize("answers,correct,quality", [
    (["a", "a"], [True, False], [0, 1]),
    (["a", None], [True, False], [0, 1]),
    (["a", "b"], [True, False], [float("nan"), 1]),
    (["a"], [True], [1]),
])
def test_invalid_pool_is_rejected(answers, correct, quality):
    with pytest.raises(ValueError):
        decompose_pair_pool(answers, correct, quality)
