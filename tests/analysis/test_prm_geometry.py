"""Closed-form geometry scorers on synthetic Gaussians with a known answer."""

import math

import numpy as np
import torch

from scripts.analysis.prm_geometry import study, to_unit


def _gauss(n, d, rng, shift, scale1=1.0, nuisance=30.0):
    """Class 0 ~ N(0, diag), class 1 shifted along a LOW-variance axis (dim 1);
    dim 0 carries large label-free variance, so the raw mean axis is not hurt
    but PCA's top component is pure nuisance."""
    y = np.r_[np.zeros(n), np.ones(n)].astype(np.int64)
    sd = np.ones(d)
    sd[0] = nuisance
    X = rng.normal(size=(2 * n, d)) * sd
    X[y == 1, 1] += shift
    X[y == 1] *= np.where(np.arange(d) >= 2, scale1, 1.0)
    return X.astype(np.float32), y


def _pb(rng, d, shift):
    meta, X = [], []
    for t in range(40):
        n_steps, label = 4, (2 if t % 2 else -1)
        for j in range(n_steps):
            bad = label != -1 and j >= label
            x = rng.normal(size=d)
            x[1] += shift if bad else 0.0
            X.append(x)
            meta.append({"id": f"t{t}", "step_idx": j, "label": label, "n_steps": n_steps})
    return np.asarray(X, dtype=np.float32), meta


def _run(scale1=1.0, shift=1.5, d=16):
    rng = np.random.default_rng(0)
    tr = _gauss(4000, d, rng, shift, scale1)
    va = _gauss(1000, d, rng, shift, scale1)
    te = _gauss(1000, d, rng, shift, scale1)
    pb = {s: _pb(rng, d, shift) for s in ("gsm8k", "math", "olympiadbench", "omnimath")}
    return study(*tr, *va, *te, pb, torch.device("cpu"), knn=True)


def test_lda_matches_gaussian_prediction_and_pca_misses_the_axis():
    metrics, arrays = _run()
    lda = metrics["methods"]["lda"]["auroc_test"]
    pred = 0.5 * (1 + math.erf(1.5 / 2))          # Phi(D / sqrt 2), D = 1.5
    assert abs(metrics["mahalanobis"]["D"] - 1.5) < 0.1
    assert abs(lda - pred) < 0.03
    # the top PC is the nuisance axis: no signal there
    assert abs(arrays["pc_auroc_test"][0] - 0.5) < 0.05
    # LDA on the top-1 PC (nuisance) sees nothing; adding the 2nd PC recovers it
    k = metrics["lda_topk_pcs"]
    assert abs(k["1"]["auroc_test"] - 0.5) < 0.05
    assert abs(k["2"]["auroc_test"] - lda) < 0.02
    # with equal class covariances there is no second-order signal
    assert abs(metrics["methods"]["cov_only"]["auroc_test"] - 0.5) < 0.05
    # ProcessBench numbers exist for every subset and are in range
    m = metrics["methods"]["lda"]
    assert 0.0 <= m["pb_avg_F1_PB_oracle"] <= 1.0
    assert m["pb_avg_F1_PB_oracle"] >= m["pb_avg_F1_PB_calib20"] - 1e-9


def test_cov_only_detects_a_shape_difference():
    metrics, _ = _run(scale1=1.6, shift=0.0)
    assert metrics["methods"]["cov_only"]["auroc_test"] > 0.75
    assert abs(metrics["methods"]["lda"]["auroc_test"] - 0.5) < 0.06


def test_to_unit_is_rank_preserving():
    ref = np.array([3.0, -1.0, 2.0, 10.0])
    u = to_unit(ref)
    s = np.array([-5.0, 0.0, 2.5, 11.0])
    out = u(s)
    assert np.all(np.diff(out) >= 0) and out[0] == 0.0 and out[-1] == 1.0
