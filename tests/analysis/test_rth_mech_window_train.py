import torch

from scripts.analysis import rth_mech_window_train as m6
from scripts.read_token_probe import ReadToken, read_states
from tests.analysis.test_rth_mech_attention import tiny_model


def test_parse_layers():
    assert m6.parse_layers("0-2") == [0, 1, 2]
    assert m6.parse_layers("4-5,1,0-1") == [0, 1, 4, 5]


def test_offsets_outside_the_window_stay_zero():
    torch.manual_seed(0)
    model = tiny_model()
    rt = ReadToken(1, 32, torch.randn(1, 32) * 0.1, n_layers=3)
    opt = m6.windowed_optimizer([1])(rt, 1e-2, False)
    seqs, y = [[1, 5, 9, 3], [4, 8, 2]], torch.tensor([1.0, 0.0])
    for _ in range(5):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            rt.logit(read_states(model, rt, seqs, 0, torch.device("cpu"), grad=True)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert rt.deep[0].abs().max() == 0 and rt.deep[2].abs().max() == 0
    assert rt.deep[1].abs().max() > 0
