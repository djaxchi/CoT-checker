"""Unit tests for StepCorrectnessClassifier."""

import tempfile
from pathlib import Path

import torch
import pytest

from src.probes.classifier import StepCorrectnessClassifier


@pytest.fixture
def clf():
    return StepCorrectnessClassifier(input_dim=32, hidden_dim=64)


def test_output_shape(clf):
    x = torch.randn(8, 32)
    out = clf(x)
    assert out.shape == (8, 1)


def test_predict_binary(clf):
    x = torch.randn(10, 32)
    preds = clf.predict(x)
    assert preds.shape == (10,)
    assert set(preds.tolist()).issubset({0, 1})


def test_predict_threshold_zero_all_ones(clf):
    """At threshold=0 every prediction should be 1."""
    x = torch.randn(5, 32)
    preds = clf.predict(x, threshold=0.0)
    assert (preds == 1).all()


def test_predict_threshold_one_all_zeros(clf):
    """At threshold=1.0 every prediction should be 0."""
    x = torch.randn(5, 32)
    preds = clf.predict(x, threshold=1.0)
    assert (preds == 0).all()


def test_gradient_flows(clf):
    x = torch.randn(4, 32)
    y = torch.zeros(4)
    logits = clf(x).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    assert clf.net[0].weight.grad is not None


def test_save_load_roundtrip(clf, tmp_path):
    clf.eval()  # both models must be in the same mode for deterministic output
    path = tmp_path / "probe.pt"
    x = torch.randn(3, 32)
    with torch.no_grad():
        expected = clf(x)

    clf.save(path)
    loaded = StepCorrectnessClassifier.load(path)

    with torch.no_grad():
        got = loaded(x)

    assert torch.allclose(expected, got)


def test_hidden_dim_halving():
    """Second hidden layer should be hidden_dim // 2."""
    clf = StepCorrectnessClassifier(input_dim=16, hidden_dim=64)
    # net[4] is the second Linear (after LayerNorm, ReLU, Dropout)
    assert clf.net[4].out_features == 32


def test_different_batch_sizes(clf):
    for b in [1, 16, 128]:
        out = clf(torch.randn(b, 32))
        assert out.shape == (b, 1)
