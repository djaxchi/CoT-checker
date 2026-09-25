"""The four artifact measurements of instruct_arm_v1, on synthetic inputs.

Each test builds a case where the right answer is known by construction, so a
measurement that is off by a normalisation or an index reads as a failure here
rather than as a plausible number in the Base-versus-Instruct table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis import instruct_audit as ia  # noqa: E402
from src.onpolicy.prompts import verifier_prefix  # noqa: E402


# ---- auroc -----------------------------------------------------------------

def test_auroc_perfect_inverse_and_ties():
    y = np.array([0, 0, 1, 1])
    assert ia.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert ia.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert ia.auroc(y, np.zeros(4)) == 0.5


def test_auroc_single_class_is_nan():
    assert np.isnan(ia.auroc(np.ones(3), np.arange(3.0)))


# ---- 1. outlier-dimension mass --------------------------------------------

def test_topk_share_one_hot_rows_put_everything_in_top1():
    H = np.eye(4, 16) * 7.0
    out = ia.topk_norm_share(H, ks=(1, 4))
    assert out["top1"] == pytest.approx(1.0)
    assert out["top4"] == pytest.approx(1.0)


def test_topk_share_isotropic_rows_are_k_over_d():
    H = np.ones((5, 100))
    out = ia.topk_norm_share(H, ks=(1, 10))
    assert out["top1"] == pytest.approx(0.01)
    assert out["top10"] == pytest.approx(0.10)


def test_dimension_concentration_finds_the_planted_dimension():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(500, 32))
    H[:, 5] += 40.0
    out = ia.dimension_concentration(H, k=1)
    assert out["top_dims"] == [5]
    assert out["share"] > 0.9


def test_massive_tokens_counted_by_position():
    norms = [np.array([50.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])]
    out = ia.massive_token_stats(norms, factor=5.0)
    assert out["rate"] == pytest.approx(1 / 7)
    assert out["share_at_first"] == pytest.approx(1.0)


# ---- 3. probe localisation -------------------------------------------------

def test_occlusion_share_first_against_its_uniform_baseline():
    deltas = [np.array([3.0, 1.0, 0.0, 0.0]), np.array([1.0, 1.0])]
    norms = [np.array([9.0, 1.0, 1.0, 1.0]), np.array([1.0, 2.0])]
    out = ia.occlusion_summary(deltas, norms)
    assert out["share_first"] == pytest.approx((0.75 + 0.5) / 2)
    assert out["share_first_uniform"] == pytest.approx((0.25 + 0.5) / 2)
    # the max-norm token is position 0 in the first item, position 1 in the second
    assert out["share_maxnorm"] == pytest.approx((0.75 + 0.5) / 2)


def test_occlusion_signed_deltas_use_magnitude():
    out = ia.occlusion_summary([np.array([-2.0, 2.0])], [np.array([1.0, 1.0])])
    assert out["share_first"] == pytest.approx(0.5)


def test_occlusion_all_zero_item_is_skipped_not_nan():
    out = ia.occlusion_summary([np.zeros(3), np.array([1.0, 0.0])],
                               [np.ones(3), np.ones(2)])
    assert out["n_items"] == 1
    assert out["share_first"] == pytest.approx(1.0)


# ---- 4. length and position residualisation --------------------------------

def test_residualisation_removes_a_pure_length_signal():
    rng = np.random.default_rng(1)
    n = 4000
    length = rng.uniform(1, 5, n)
    y = (length + rng.normal(0, 0.5, n) > 3).astype(int)
    score = length + rng.normal(0, 0.1, n)          # the "probe" is length plus noise
    out = ia.residual_auroc(y, score, np.c_[length])
    assert out["raw"] > 0.85
    assert out["covariates_only"] > 0.85
    assert abs(out["residual"] - 0.5) < 0.05


def test_residualisation_keeps_a_content_signal():
    rng = np.random.default_rng(2)
    n = 4000
    length = rng.uniform(1, 5, n)
    content = rng.normal(size=n)
    y = (content > 0).astype(int)
    score = content + 0.3 * length
    out = ia.residual_auroc(y, score, np.c_[length])
    assert out["residual"] > 0.95
    assert abs(out["covariates_only"] - 0.5) < 0.05


# ---- 2. attention-sink and template mass ------------------------------------

def test_char_spans_cover_the_verifier_template_exactly():
    problem, prefix = "What is 2+2?", "Step one.\n\nStep two."
    text = verifier_prefix(problem, prefix)
    spans = ia.verifier_char_spans(problem, prefix)
    rebuilt = "".join(text[a:b] for a, b, _ in spans)
    assert rebuilt == text
    cats = {c for _, _, c in spans}
    assert cats == {"template", "problem", "prior"}
    assert text[spans[1][0]:spans[1][1]] == problem


def test_char_spans_first_step_has_no_prior_category():
    spans = ia.verifier_char_spans("Q?", "")
    assert "prior" not in {c for _, _, c in spans}
    assert "".join(verifier_prefix("Q?", "")[a:b] for a, b, _ in spans) == \
        verifier_prefix("Q?", "")


def test_token_categories_from_offsets_and_step_tail():
    spans = [(0, 9, "template"), (9, 14, "problem")]
    offsets = [(0, 7), (7, 9), (9, 14)]            # 3 prefix tokens
    cats = ia.token_categories(offsets, spans, n_step=2)
    assert cats == ["sink", "template", "problem", "step", "step"]


def test_attention_mass_by_category_averages_over_step_queries():
    cats = ["sink", "template", "problem", "step", "step"]
    T = len(cats)
    A = np.zeros((T, T))
    A[3, 0] = 1.0                                  # first step query: all on sink
    A[4, 2] = 0.5; A[4, 4] = 0.5                   # second: half problem, half itself
    out = ia.attention_mass(A, cats)
    assert out["sink"] == pytest.approx(0.5)
    assert out["problem"] == pytest.approx(0.25)
    assert out["step"] == pytest.approx(0.25)
    assert sum(out.values()) == pytest.approx(1.0)


# ---- ProcessBench step labels ------------------------------------------------

def test_pb_step_labels_first_error_convention():
    meta = [{"step_idx": 0, "label": 2}, {"step_idx": 1, "label": 2},
            {"step_idx": 2, "label": 2}, {"step_idx": 3, "label": 2},
            {"step_idx": 0, "label": -1}]
    y, keep = ia.pb_step_labels(meta)
    assert keep.tolist() == [True, True, True, False, True]
    assert y[keep].tolist() == [0, 0, 1, 0]


# ---- the job script's per-split pass, end to end on a toy probe -------------

class _FakeLoader:
    """Just the SpanLoader surface run_split touches."""

    def __init__(self, spans, labels):
        import torch
        self.spans = spans; self.labels = labels; self.torch = torch
        self.handles = [(None, i, 1, len(s) + 1, int(l)) for i, (s, l) in
                        enumerate(zip(spans, labels))]

    def eval_batches(self, b):
        n = len(self.spans)
        return [np.arange(i, min(i + b, n)) for i in range(0, n, b)]

    def collate(self, idx):
        T = max(len(self.spans[i]) for i in idx); d = self.spans[0].shape[1]
        x = np.zeros((len(idx), T, d), np.float32); m = np.zeros((len(idx), T), np.float32)
        for r, i in enumerate(idx):
            x[r, :len(self.spans[i])] = self.spans[i]; m[r, :len(self.spans[i])] = 1
        t = self.torch
        return t.from_numpy(x), t.from_numpy(m), t.from_numpy(self.labels[idx].astype(np.float32))


def test_run_split_on_a_toy_transformer():
    torch = pytest.importorskip("torch")
    from scripts.analysis.instruct_artifact_audit import run_split
    from src.harness.learners import build_learner
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    spans = [rng.normal(size=(int(rng.integers(1, 9)), 16)).astype(np.float32)
             for _ in range(40)]
    y = np.array([i % 2 for i in range(40)], dtype=np.int8)
    meta = [{"step_idx": i % 5, "orig_step_start_idx": 10 + i} for i in range(40)]
    model = build_learner("transformer:d32,l1,f64,h4", 16, t_max=16).eval()
    out = run_split(model, _FakeLoader(spans, y), meta, y, np.ones(40, bool), "cpu",
                    max_occl=40, token_sample=1000)
    assert out["n_steps"] == 40
    assert 0.0 <= out["auroc"] <= 1.0
    n_multi = sum(len(s) >= 2 for s in spans)
    assert out["occlusion_auroc"]["n"] == n_multi
    assert out["occlusion"]["share_first"] >= 0
    assert out["outlier"]["n_tokens"] == sum(len(s) for s in spans)
