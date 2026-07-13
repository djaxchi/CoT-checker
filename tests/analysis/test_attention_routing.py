"""Unit tests for the attention_routing_v0 core logic (pure numpy)."""

import numpy as np
import pytest

from src.analysis.attention_routing import (
    FEATURES,
    N_FEATURES,
    N_READS,
    READS,
    REGION_CANDIDATE,
    REGION_OLDER,
    REGION_OTHER,
    REGION_PREV1,
    REGION_QUESTION,
    assign_token_regions,
    attention_step_features,
    build_fork_segments,
    candidate_token_span,
    check_causal,
    check_row_normalized,
    count_numbers,
    count_operators,
    grounding_ratio,
    paired_regression,
    paired_stats,
    recency_ratio,
    region_token_counts,
)


def feat(name: str) -> int:
    return FEATURES.index(name)


def read(name: str) -> int:
    return READS.index(name)


# ---------------------------------------------------------------------------
# segments and token regions
# ---------------------------------------------------------------------------

class TestBuildForkSegments:
    def test_char_boundaries(self):
        text, segs = build_fork_segments("Q?", ["s1.", "s22."], "cand")
        assert text == "Q?\ns1.\ns22.\ncand"
        assert segs == [
            ("question", 0, 2),
            ("step_1", 3, 6),
            ("step_2", 7, 11),
            ("candidate", 12, 16),
        ]
        for name, a, b in segs:
            assert text[a:b] == {"question": "Q?", "step_1": "s1.",
                                 "step_2": "s22.", "candidate": "cand"}[name]

    def test_no_prefix_steps(self):
        text, segs = build_fork_segments("Q?", [], "cand")
        assert text == "Q?\ncand"
        assert [s[0] for s in segs] == ["question", "candidate"]


class TestAssignTokenRegions:
    def test_regions_and_separator_attachment(self):
        # "Q?\ns1.\ns22.\ncand" with tokens mimicking Qwen's ".\n" merges:
        # token (1,3)="?\n" starts inside the question -> question
        _, segs = build_fork_segments("Q?", ["s1.", "s22."], "cand")
        offsets = [(0, 1), (1, 3), (3, 5), (5, 7), (7, 10), (10, 12), (12, 16)]
        regions = assign_token_regions(offsets, segs)
        assert regions.tolist() == [
            REGION_QUESTION,   # "Q"
            REGION_QUESTION,   # "?\n" starts at the question's last char
            REGION_OLDER,      # "s1"
            REGION_OLDER,      # ".\n" separator attaches to step_1
            REGION_PREV1,      # "s22" (last prefix step)
            REGION_PREV1,      # ".\n"
            REGION_CANDIDATE,  # "cand"
        ]

    def test_special_token_is_other(self):
        _, segs = build_fork_segments("Q?", ["s1."], "cand")
        offsets = [(0, 0), (0, 3), (3, 7), (7, 11)]
        regions = assign_token_regions(offsets, segs)
        assert regions[0] == REGION_OTHER
        assert regions[1] == REGION_QUESTION
        assert regions[2] == REGION_PREV1  # single prefix step is prev1
        assert regions[3] == REGION_CANDIDATE

    def test_no_prefix_steps_has_no_prev_regions(self):
        _, segs = build_fork_segments("Q?", [], "cand")
        offsets = [(0, 3), (3, 7)]
        regions = assign_token_regions(offsets, segs)
        assert regions.tolist() == [REGION_QUESTION, REGION_CANDIDATE]

    def test_last_segment_must_be_candidate(self):
        with pytest.raises(ValueError):
            assign_token_regions([(0, 1)], [("question", 0, 2)])


class TestCandidateSpan:
    def test_span_and_counts(self):
        regions = np.array([REGION_QUESTION, REGION_PREV1,
                            REGION_CANDIDATE, REGION_CANDIDATE])
        assert candidate_token_span(regions) == (2, 4)
        counts = region_token_counts(regions)
        assert counts["candidate"] == 2 and counts["question"] == 1

    def test_errors(self):
        with pytest.raises(ValueError):
            candidate_token_span(np.array([REGION_QUESTION]))
        with pytest.raises(ValueError):
            candidate_token_span(np.array(
                [REGION_CANDIDATE, REGION_QUESTION, REGION_CANDIDATE]))


# ---------------------------------------------------------------------------
# attention features
# ---------------------------------------------------------------------------

def toy_attention():
    """1 head, 2 candidate tokens (abs positions 4, 5), key_len 6.

    regions: [question, question, prev1, prev1, candidate, candidate]
    t=4: uniform 0.2 over visible tokens 0..4
    t=5: all mass on j=2 (prev1)
    """
    regions = np.array([REGION_QUESTION, REGION_QUESTION, REGION_PREV1,
                        REGION_PREV1, REGION_CANDIDATE, REGION_CANDIDATE])
    attn = np.zeros((1, 2, 6), dtype=np.float32)
    attn[0, 0, :5] = 0.2
    attn[0, 1, 2] = 1.0
    return attn, regions, 4


class TestAttentionStepFeatures:
    def test_shapes(self):
        attn, regions, c0 = toy_attention()
        out = attention_step_features(attn, regions, c0)
        assert out.shape == (1, N_FEATURES, N_READS)

    def test_first_token_masses(self):
        attn, regions, c0 = toy_attention()
        out = attention_step_features(attn, regions, c0)
        first = out[0, :, read("first")]
        assert first[feat("question_mass")] == pytest.approx(0.4)
        assert first[feat("prev1_mass")] == pytest.approx(0.4)
        assert first[feat("prev_all_mass")] == pytest.approx(0.4)
        assert first[feat("older_mass")] == pytest.approx(0.0)
        assert first[feat("self_mass")] == pytest.approx(0.2)
        assert first[feat("other_mass")] == pytest.approx(0.0)
        assert first[feat("entropy")] == pytest.approx(np.log(5), rel=1e-5)
        # 0.2 * (4+3+2+1+0)
        assert first[feat("mean_distance")] == pytest.approx(2.0)
        assert first[feat("top5_mass")] == pytest.approx(1.0)
        assert first[feat("sink_mass")] == pytest.approx(0.2)

    def test_last_token_delta_attention(self):
        attn, regions, c0 = toy_attention()
        out = attention_step_features(attn, regions, c0)
        last = out[0, :, read("last")]
        assert last[feat("prev1_mass")] == pytest.approx(1.0)
        assert last[feat("question_mass")] == pytest.approx(0.0)
        assert last[feat("entropy")] == pytest.approx(0.0, abs=1e-6)
        assert last[feat("mean_distance")] == pytest.approx(3.0)  # 5 - 2
        assert last[feat("sink_mass")] == pytest.approx(0.0)

    def test_mean_read_is_token_average(self):
        attn, regions, c0 = toy_attention()
        out = attention_step_features(attn, regions, c0)
        np.testing.assert_allclose(
            out[0, :, read("mean")],
            0.5 * (out[0, :, read("first")] + out[0, :, read("last")]),
            rtol=1e-5,
        )

    def test_region_length_mismatch_raises(self):
        attn, regions, c0 = toy_attention()
        with pytest.raises(ValueError):
            attention_step_features(attn, regions[:-1], c0)


class TestChecks:
    def test_check_causal_flags_future_mass(self):
        attn, regions, c0 = toy_attention()
        assert check_causal(attn, c0) == pytest.approx(0.0)
        bad = attn.copy()
        bad[0, 0, 5] = 0.3  # query t=4 attending to j=5
        assert check_causal(bad, c0) == pytest.approx(0.3)

    def test_check_row_normalized(self):
        attn, _, _ = toy_attention()
        assert check_row_normalized(attn) == pytest.approx(0.0, abs=1e-6)
        assert check_row_normalized(attn * 0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# surface features and ratios
# ---------------------------------------------------------------------------

class TestSurface:
    def test_counts(self):
        assert count_numbers("2x^2 + 7.5x = 60") == 4  # 2, 2, 7.5, 60
        assert count_operators("2x^2 + 7.5x = 60") == 3
        assert count_numbers("no digits") == 0

    def test_ratios(self):
        g = grounding_ratio(np.array([0.3, 0.0]), np.array([0.1, 0.0]))
        assert g[0] == pytest.approx(0.75)
        assert g[1] == pytest.approx(0.0)
        r = recency_ratio(np.array([0.2]), np.array([0.4]))
        assert r[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# paired statistics
# ---------------------------------------------------------------------------

class TestPairedStats:
    def test_all_positive(self):
        rng = np.random.default_rng(0)
        delta = rng.uniform(0.5, 1.5, size=200)
        groups = np.repeat(np.arange(50), 4)
        s = paired_stats(delta, groups, n_boot=200, seed=1)
        assert s["p_gt"] == pytest.approx(1.0)
        assert s["mean"] == pytest.approx(delta.mean())
        assert s["ci_lo"] > 0
        assert s["p_sign"] < 1e-6 and s["p_wilcoxon"] < 1e-6
        assert s["n_groups"] == 50

    def test_symmetric_null(self):
        rng = np.random.default_rng(2)
        delta = rng.normal(0, 1, size=400)
        groups = np.arange(400)
        s = paired_stats(delta, groups, n_boot=200, seed=1)
        assert s["ci_lo"] < 0 < s["ci_hi"]
        assert 0.4 < s["p_gt"] < 0.6

    def test_ties_count_half(self):
        s = paired_stats(np.zeros(10), np.arange(10), n_boot=50)
        assert s["p_gt"] == pytest.approx(0.5)
        assert s["p_gt_group"] == pytest.approx(0.5)
        assert s["p_sign"] == 1.0

    def test_tests_run_on_problem_means(self):
        # 3 problems; within-problem deltas are wildly positive-heavy at the
        # fork level, but the PROBLEM means are (+1, +1, -2): a fork-level
        # sign test would see 6/9 positive; the cluster-aware one sees 2/3.
        delta = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.0, -2.0, -2.0])
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        s = paired_stats(delta, groups, n_boot=50)
        assert s["n_groups"] == 3
        assert s["p_gt"] == pytest.approx(6 / 9)
        assert s["p_gt_group"] == pytest.approx(2 / 3)
        # sign test on 3 group means, 2 positive -> p = 1.0 (binomial n=3)
        assert s["p_sign"] == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_stats(np.zeros(3), np.zeros(4))


class TestPairedRegression:
    def test_recovers_intercept(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, size=(500, 2))
        y = 2.0 + 3.0 * x[:, 0] - 1.0 * x[:, 1] + rng.normal(0, 0.1, 500)
        groups = np.repeat(np.arange(100), 5)
        r = paired_regression(y, x, groups)
        assert r["beta0"] == pytest.approx(2.0, abs=0.05)
        assert r["p0"] < 1e-10
        assert r["r2"] > 0.99
        assert r["n_groups"] == 100

    def test_null_intercept_not_significant(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, size=(300, 1))
        y = 0.5 * x[:, 0] + rng.normal(0, 1, 300)
        r = paired_regression(y, x, np.arange(300))
        assert abs(r["beta0"]) < 0.3
        assert r["p0"] > 0.01

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_regression(np.zeros(3), np.zeros((4, 2)), np.zeros(3))
