"""Visibility guarantees of the step_bert_probe_v0 encoder.

The contrast the experiment rests on (bidirectional vs causal) is only valid if a
hidden step cannot move the target's output through any layer. Each test perturbs
the steps that should be invisible and requires the target output to stay fixed.
"""

import pytest
import torch

from src.probes.step_bert import StepBert, StepBertClassifier, visibility

P, PD, T = 2, 16, 9


@pytest.fixture
def model():
    torch.manual_seed(0)
    return StepBert(n_parts=P, part_dim=PD, d=32, n_layers=3, n_heads=4, ff=64,
                    dropout=0.0, n_target=8).eval()


def _x(B=2):
    torch.manual_seed(1)
    return torch.randn(B, T, P * PD)


def test_visibility_modes():
    L = torch.tensor([6])
    t = torch.tensor([2])
    assert visibility(L, t, "left", T)[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert visibility(L, t, "right", T)[0].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]
    assert visibility(L, t, "both", T)[0].tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert visibility(L, t, "local", T)[0].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert visibility(L, t, "both", T, window=1)[0].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("mode,hidden", [
    ("left", [5, 6, 7, 8]), ("right", [0, 1, 2, 3]), ("local", [0, 1, 2, 3, 5, 6, 7, 8]),
])
def test_hidden_steps_cannot_move_reconstruction(model, mode, hidden):
    x = _x()
    L, t = torch.tensor([T, T]), torch.tensor([4, 4])
    base = model.reconstruct(x, L, t, mode)
    x2 = x.clone()
    x2[:, hidden] += 5 * torch.randn_like(x2[:, hidden])
    torch.testing.assert_close(model.reconstruct(x2, L, t, mode), base)


def test_visible_steps_do_move_reconstruction(model):
    x = _x()
    L, t = torch.tensor([T, T]), torch.tensor([4, 4])
    x2 = x.clone()
    x2[:, 7] += 5 * torch.randn_like(x2[:, 7])
    assert not torch.allclose(model.reconstruct(x2, L, t, "right"), model.reconstruct(x, L, t, "right"))


def test_masked_target_content_is_hidden(model):
    x = _x()
    L, t = torch.tensor([T, T]), torch.tensor([4, 4])
    x2 = x.clone()
    x2[:, 4] += 5 * torch.randn_like(x2[:, 4])
    torch.testing.assert_close(model.reconstruct(x2, L, t, "both"), model.reconstruct(x, L, t, "both"))


def test_padding_is_invisible(model):
    x = _x()
    L, t = torch.tensor([5, 5]), torch.tensor([2, 2])
    x2 = x.clone()
    x2[:, 5:] = 100.0
    torch.testing.assert_close(model.reconstruct(x2, L, t, "both"), model.reconstruct(x, L, t, "both"))


def test_classifier_window_blocks_outside_steps(model):
    clf = StepBertClassifier(model, n_extra=3).eval()
    x = _x()
    L, t = torch.tensor([T, T]), torch.tensor([4, 4])
    extra = torch.randn(2, 3)
    base = clf(x, L, t, "both", window=1, extra=extra)
    x2 = x.clone()
    x2[:, [0, 1, 2, 6, 7, 8]] += 5
    torch.testing.assert_close(clf(x2, L, t, "both", window=1, extra=extra), base)
    x3 = x.clone()
    x3[:, 5] += 5 * torch.randn_like(x3[:, 5])  # constant shifts vanish under LayerNorm
    assert not torch.allclose(clf(x3, L, t, "both", window=1, extra=extra), base)
