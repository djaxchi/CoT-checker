"""Tests for the S4 contrib-cluster core logic (representations, tags, stats)."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.contrib_cluster import (
    assign_top_tag,
    build_prefixes,
    compute_reprs,
    fit_steps_to_length,
    l2_normalize,
    surface_eta_squared,
    surface_features,
    tag_enrichment,
    tag_entropy,
    tag_step,
)
from src.data.prm800k_trajectories import audit_trajectories, reconstruct_trajectory

# ---------------------------------------------------------------------------
# Prefixes
# ---------------------------------------------------------------------------


def test_build_prefixes():
    p = build_prefixes("q?", ["s1", "s2", "s3"])
    assert p == ["q?", "q?\ns1", "q?\ns1\ns2", "q?\ns1\ns2\ns3"]


def _count_words(text: str) -> int:
    return len(text.split())


def test_fit_steps_to_length_keeps_fitting_trajectory():
    steps = ["one two", "three four"]
    assert fit_steps_to_length(_count_words, "q", steps, max_seq_len=10) == steps


def test_fit_steps_to_length_truncates_trailing_steps():
    # prefixes: q(1) -> +2 words/step; budget 6 fits q + 2 steps (5 words), not 3
    steps = ["a b", "c d", "e f"]
    assert fit_steps_to_length(_count_words, "q", steps, max_seq_len=6) == ["a b", "c d"]


def test_fit_steps_to_length_returns_none_when_min_steps_overflow():
    steps = ["a b c d e", "f g h i j", "k l"]
    assert fit_steps_to_length(_count_words, "q", steps, max_seq_len=4) is None


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------


@pytest.fixture
def H():
    rng = np.random.default_rng(0)
    return rng.normal(size=(6, 8)).astype(np.float32)  # h_0..h_5


def test_state_qres_contribution(H):
    r = compute_reprs(H)
    assert set(r) == {"state", "qres", "contribution"}
    assert all(v.shape == (5, 8) for v in r.values())
    np.testing.assert_allclose(r["state"], H[1:])
    np.testing.assert_allclose(r["qres"], H[1:] - H[0])
    np.testing.assert_allclose(r["contribution"], H[1:] - H[:-1])


def test_contribution_is_closed_form_of_recursion(H):
    """The intended recursion c_1 = h_1 - h_0, c_i = h_i - (h_0 + sum_{k<i} c_k)
    telescopes exactly to c_i = h_i - h_{i-1}; we compute the closed form."""
    r = compute_reprs(H)
    running = H[0].copy()  # h_0 + sum of previous contributions
    for i in range(1, 6):
        c = H[i] - running
        np.testing.assert_allclose(r["contribution"][i - 1], c, rtol=1e-5)
        running += c


def test_compute_reprs_rejects_short():
    with pytest.raises(ValueError):
        compute_reprs(np.zeros((1, 4), dtype=np.float32))


def test_l2_normalize():
    X = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    N = l2_normalize(X)
    np.testing.assert_allclose(N[0], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_allclose(N[1], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Tags and surface features
# ---------------------------------------------------------------------------


def test_tag_step_examples():
    t = tag_step("Let x be the number of apples, so x = 3 + 4.")
    assert t["VARIABLE_DEFINE"] and t["ARITHMETIC_COMPUTE"]
    t = tag_step("Therefore the final answer is \\boxed{7}.")
    assert t["FINAL_ANSWER"] and t["INTERMEDIATE_CONCLUSION"]
    t = tag_step("The probability of choosing a red ball is at most 1/2, so p < 0.5.")
    assert t["PROBABILITY"] and t["INEQUALITY_BOUND"]
    t = tag_step("By the Pythagorean theorem the triangle has area 6.")
    assert t["THEOREM_INVOKE"] and t["GEOMETRY_REASONING"]


def test_surface_features():
    f = surface_features("x = 1 + 2, y = \\boxed{3}")
    assert f["n_equals"] == 2
    assert f["n_digits"] == 3
    assert f["contains_boxed"] and f["contains_latex"]
    assert f["n_math_ops"] >= 1


def test_assign_top_tag_prefers_rare():
    #        tagA tagB
    M = np.array([[1, 1], [0, 1], [0, 1]], dtype=bool)
    top = assign_top_tag(M, ("A", "B"))
    assert top == ["A", "B", "B"]  # A is rarer, wins where both match


# ---------------------------------------------------------------------------
# Cluster stats
# ---------------------------------------------------------------------------


def test_tag_enrichment_math():
    labels = np.array([0, 0, 1, 1])
    M = np.array([[1], [1], [0], [0]], dtype=bool)  # tag only in cluster 0
    rows = tag_enrichment(labels, M, ("T",))
    by_c = {r["cluster"]: r for r in rows}
    assert by_c[0]["enrichment"] == pytest.approx(2.0)  # P(t|c)=1, P(t)=0.5
    assert by_c[1]["enrichment"] == pytest.approx(0.0)


def test_tag_entropy():
    labels = np.array([0, 0, 1, 1])
    ent = tag_entropy(labels, ["A", "A", "A", "B"])
    assert ent[0] == pytest.approx(0.0)
    assert ent[1] == pytest.approx(1.0)


def test_surface_eta_squared():
    labels = np.array([0, 0, 1, 1, -1])
    vals = np.array([1.0, 1.0, 5.0, 5.0, 100.0])  # noise excluded
    assert surface_eta_squared(labels, vals) == pytest.approx(1.0)
    vals2 = np.array([1.0, 5.0, 1.0, 5.0, 100.0])
    assert surface_eta_squared(labels, vals2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Trajectory reconstruction (raw PRM800K formats)
# ---------------------------------------------------------------------------


def _raw_sample(steps):
    return {
        "question": {"problem": "What is 2+2?", "ground_truth_answer": "4"},
        "label": {"steps": steps},
    }


def test_reconstruct_human_completion_path():
    steps = [
        {"human_completion": {"text": "First, add."}, "completions": []},
        {"human_completion": {"text": "2+2 = 4."}, "completions": []},
    ]
    counters: dict = {"malformed_samples": 0, "missing_problem": 0, "missing_steps": 0,
                      "truncated_paths": 0, "too_few_steps": 0}
    traj = reconstruct_trajectory(_raw_sample(steps), 0, counters)
    assert traj is not None
    assert traj["steps"] == ["First, add.", "2+2 = 4."]
    assert traj["question"] == "What is 2+2?"


def test_reconstruct_chosen_completion_path():
    steps = [
        {"chosen_completion": 1,
         "completions": [{"text": "bad"}, {"text": "Start by adding."}]},
        {"chosen_completion": 0, "completions": [{"text": "The answer is 4."}]},
    ]
    counters: dict = {"malformed_samples": 0, "missing_problem": 0, "missing_steps": 0,
                      "truncated_paths": 0, "too_few_steps": 0}
    traj = reconstruct_trajectory(_raw_sample(steps), 0, counters)
    assert traj is not None
    assert traj["steps"] == ["Start by adding.", "The answer is 4."]


def test_reconstruct_stops_at_missing_selection_and_drops_short():
    steps = [
        {"human_completion": {"text": "only step"}, "completions": []},
        {"completions": [{"text": "never selected"}]},  # no selection -> path ends
    ]
    counters: dict = {"malformed_samples": 0, "missing_problem": 0, "missing_steps": 0,
                      "truncated_paths": 0, "too_few_steps": 0}
    traj = reconstruct_trajectory(_raw_sample(steps), 0, counters)
    assert traj is None  # 1 usable step < 2
    assert counters["truncated_paths"] == 1
    assert counters["too_few_steps"] == 1


def test_audit_trajectories_counts():
    good = _raw_sample([
        {"human_completion": {"text": "a"}, "completions": []},
        {"human_completion": {"text": "b"}, "completions": []},
        {"human_completion": {"text": "c"}, "completions": []},
    ])
    bad = {"question": {"problem": "p"}, "label": {"steps": []}}
    trajs, audit = audit_trajectories([good, bad, "not-a-dict"])
    assert audit["n_raw_examples"] == 3
    assert audit["n_usable_trajectories"] == 1
    assert audit["n_usable_steps"] == 3
    assert audit["dropped"]["missing_steps"] == 1
    assert audit["dropped"]["malformed_samples"] == 1
    assert audit["steps_per_trajectory"]["max"] == 3
