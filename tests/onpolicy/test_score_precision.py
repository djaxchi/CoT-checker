"""Preserve ranking information even when probability storage saturates."""
import numpy as np
import pytest
import torch

from scripts.onpolicy.score_gen_states_multi import Cell
from scripts.onpolicy.score_traces_with_prm import step_log_odds, step_rewards


def test_prm_softmax_does_not_round_distinct_bfloat_logits_to_one():
    logits = torch.tensor([[[0., 7.], [0., 8.]]], dtype=torch.bfloat16)
    mask = torch.tensor([[True, True]])
    scores = step_rewards(logits, mask)
    assert .99 < scores[0] < scores[1] < 1
    assert step_log_odds(logits, mask) == [-7., -8.]


def test_probe_exports_logits_when_sigmoid_saturates():
    class Head(torch.nn.Module):
        def forward(self, x, mask):
            return x[:, 0]
    cell = Cell.__new__(Cell)
    cell.rep, cell.learner, cell.device = "last_token", "linear", "cpu"
    cell.seconds, cell.model = 0., Head()
    blocks = [np.array([[0.], [v]], np.float16) for v in [20., 30.]]
    result = cell.score_details(blocks)
    assert result["scores"] == [1., 1.]
    assert result["logits"] == pytest.approx([20., 30.])
    assert cell.score(blocks) == result["scores"]
