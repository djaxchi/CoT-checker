import pytest

torch = pytest.importorskip("torch")
from scripts.instep_v1 import reprobe_layers as RL  # noqa: E402


def test_restrict_labels_keeps_only_listed_steps_as_labels():
    trs = [{"trace_id": "rp1", "labels": {"ds": [0, 1, 0], "self": [1, 1, 1]}},
           {"trace_id": "rp2", "labels": {"ds": [1, 0]}}]
    out = RL.restrict_labels(trs, {"rp1::1", "rp1::2"}, "ds")
    assert len(out) == 1  # rp2 has no listed step: dropped
    assert out[0]["labels"] == {"ds": [-1, 1, 0]}
    assert trs[0]["labels"]["ds"] == [0, 1, 0]  # input untouched


def test_restrict_labels_keeps_unlabelled_steps_unlabelled():
    trs = [{"trace_id": "rp1", "labels": {"ds": [-1, 0]}}]
    out = RL.restrict_labels(trs, {"rp1::0", "rp1::1"}, "ds")
    assert out[0]["labels"]["ds"] == [-1, 0]


def test_layer_features_is_norm_of_block_output_shifted():
    """A two-block stub: features of block l = norm(block l output) without the last position."""
    class Block(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, h):
            return (h + self.k,)

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 4)
            self.layers = torch.nn.ModuleList([Block(1.0), Block(2.0)])
            self.norm = torch.nn.LayerNorm(4, elementwise_affine=False)

        def forward(self, input_ids, attention_mask):
            h = self.emb(input_ids)
            for b in self.layers:
                h = b(h)[0]
            return h

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    m = Model()
    batch = [{"ids": [1, 2, 3]}, {"ids": [4, 5]}]
    feats, valid = RL.layer_features(m, batch, "cpu", 0, [0, 1])
    assert feats[0].shape == (2, 63, 4) and valid.shape == (2, 63)
    assert valid[0, :2].all() and not valid[0, 2] and valid[1, 0] and not valid[1, 1]
    ids = torch.tensor([[1, 2]])
    want = m.model.norm(m.model.emb(ids) + 1.0)
    assert torch.allclose(feats[0][0, :2], want[0], atol=1e-5)
    assert torch.allclose(feats[1][0, :2], m.model.norm(m.model.emb(ids) + 3.0)[0], atol=1e-5)
