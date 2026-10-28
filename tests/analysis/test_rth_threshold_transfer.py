import numpy as np

from scripts.analysis import rth_threshold_transfer as tt


def test_f1_and_val_threshold():
    # error traces flag at their first error with any threshold in (0.3, 0.9); correct traces never exceed 0.3
    traces = [(1, [0.1, 0.9, 0.95]), (0, [0.95, 0.2]), (-1, [0.1, 0.3]), (-1, [0.2, 0.25])]
    assert tt.f1_at(traces, 0.5) == 1.0
    assert tt.f1_at(traces, 0.05) == 0.0          # everything flagged at step 0: error trace 0 right only
    p = np.array([0.1, 0.2, 0.7, 0.8])
    y = np.array([0, 0, 1, 1])
    t = tt.val_threshold(p, y, "bacc")
    assert 0.2 <= t < 0.7 and tt.val_threshold(p, y, "f1") < 0.7


def test_cell_report_xsub_uses_other_subsets():
    rng = np.random.default_rng(0)

    def traces(shift):
        out = []
        for i in range(60):
            fe = 1 if i % 2 else -1
            s = list(rng.uniform(0, 0.3, 3) + shift)
            if fe >= 0:
                s[fe] = 0.9 + shift / 10
            out.append((fe, s))
        return out
    cell = {s: traces(0.0) for s in tt.PB_SUBSETS}
    r = tt.cell_report(cell, None)
    assert r["mean"]["oracle"] == 1.0 and r["mean"]["xsub"] == 1.0
    assert abs(r["mean"]["true_error_rate"] - 0.5) < 1e-9


def test_em_prior_recovers_a_shifted_prior():
    rng = np.random.default_rng(0)
    # a calibrated scorer at prior 0.5 with class-conditional N(+-1, 1) logits, applied where the prior is 0.2
    y = rng.random(20000) < 0.2
    z = rng.normal(np.where(y, 1.0, -1.0), 1.0)
    p = 1 / (1 + np.exp(-2 * z))           # the Bayes posterior at prior 0.5
    assert abs(tt.em_prior(p) - 0.2) < 0.02


def test_natural_prior_rule_thresholds_at_one_minus_prior():
    traces = [(1, [0.1, 0.8]), (-1, [0.6, 0.65]), (0, [0.9])]
    cell = {s: traces for s in tt.PB_SUBSETS}
    r = tt.cell_report(cell, None, em=True, natural_prior=0.3)
    assert abs(r["thresholds"]["nat"] - 0.7) < 1e-12
    assert r["mean"]["nat"] == 1.0 and r["mean"]["p05"] < 1.0
    assert "nat" not in tt.cell_report(cell, None, em=False, natural_prior=0.3)["thresholds"]
