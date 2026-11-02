"""reprobe_inputs: the ReProbe head's inputs for the speed benchmark (scripts/analysis/inline_speed.py)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.analysis.inline_speed import reprobe_inputs  # noqa: E402
from src.probes.reprobe_head import ReProbeHead  # noqa: E402


class StubModel:
    def __init__(self, d: int):
        self.d = d
        self.model = self

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=torch.randn(*input_ids.shape, self.d))


def test_spans_and_last_step():
    seqs = [([5] * 20, [4, 9, 15]), ([5] * 12, [6, 11])]
    ri = reprobe_inputs(StubModel(16), seqs, 0, torch.device("cpu"))
    assert ri["feats"].shape == (2, 20, 16)
    assert ri["valid"][1].sum() == 12
    assert ri["claims"].shape == (5, 20)
    assert ri["owner"].tolist() == [0, 0, 0, 1, 1]
    # step 1 of row 0 spans tokens [4, 9), read at features 3..7
    assert ri["claims"][1].nonzero().flatten().tolist() == [3, 4, 5, 6, 7]
    assert ri["claims_last"].shape == (2, 20) and ri["owner_last"].tolist() == [0, 1]
    assert torch.equal(ri["claims_last"][1], ri["claims"][4])
    assert all(c.sum() > 0 for c in ri["claims"])
    head = ReProbeHead(16, 8, 1, 2, 0.0).eval()
    with torch.no_grad():
        out = head(ri["feats"], ri["valid"], ri["claims_last"], ri["owner_last"])
    assert out.shape == (2,)
