"""Unit tests for parametric_retrieval_component_v1: the ComponentEdit hook,
ComponentStore round-trip, and the pure donor/prompt/vector routing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.parametric_retrieval.prga_common import (
    ComponentEdit,
    ComponentStore,
)


# --------------------------------------------------------------------------- #
# a minimal model exposing model.layers[L].{self_attn, mlp} and config
# --------------------------------------------------------------------------- #

class _Sub(nn.Module):
    def __init__(self, tuple_out: bool):
        super().__init__()
        self.tuple_out = tuple_out

    def forward(self, x):
        return (x, "cache") if self.tuple_out else x


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Sub(tuple_out=True)   # attention returns a tuple
        self.mlp = _Sub(tuple_out=False)        # mlp returns a bare tensor


class _Inner(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n)])


class _FakeModel(nn.Module):
    def __init__(self, n=3, d=4):
        super().__init__()
        self.model = _Inner(n)

        class _Cfg:
            hidden_size = d
            num_hidden_layers = n
        self.config = _Cfg()


def test_component_edit_mlp_tensor_prefill():
    m = _FakeModel(d=4)
    ce = ComponentEdit(m, 1, "mlp")
    v = torch.full((4,), 7.0)
    with ce:
        ce.set([2], [v], 1.0)                    # patch position 2
        out = m.model.layers[1].mlp(torch.zeros(1, 5, 4))
    assert torch.allclose(out[0, 2], v)          # patched
    assert torch.allclose(out[0, 0], torch.zeros(4))   # untouched
    assert torch.allclose(out[0, 3], torch.zeros(4))


def test_component_edit_attn_tuple_prefill():
    m = _FakeModel(d=3)
    ce = ComponentEdit(m, 0, "attn")
    v = torch.full((3,), 5.0)
    with ce:
        ce.set([1], [v], 1.0)
        out = m.model.layers[0].self_attn(torch.zeros(1, 4, 3))
    assert isinstance(out, tuple)                # tuple structure preserved
    assert torch.allclose(out[0][0, 1], v)


def test_component_edit_noop_during_decode():
    m = _FakeModel(d=3)
    ce = ComponentEdit(m, 2, "mlp")
    v = torch.ones(3) * 9
    with ce:
        ce.set([0], [v], 1.0)
        out = m.model.layers[2].mlp(torch.full((1, 1, 3), 4.0))  # seq_len 1
    assert torch.allclose(out, torch.full((1, 1, 3), 4.0))       # unchanged


def test_component_edit_negative_idx_skips_sample():
    m = _FakeModel(d=2)
    ce = ComponentEdit(m, 0, "mlp")
    v = torch.ones(2) * 3
    with ce:
        ce.set([-1, 1], [v, v], 1.0)             # sample 0 skipped
        out = m.model.layers[0].mlp(torch.zeros(2, 3, 2))
    assert torch.allclose(out[0], torch.zeros(3, 2))     # sample 0 untouched
    assert torch.allclose(out[1, 1], v)                  # sample 1 patched


def test_component_edit_alpha_partial():
    m = _FakeModel(d=2)
    ce = ComponentEdit(m, 0, "mlp")
    v = torch.full((2,), 10.0)
    hs = torch.zeros(1, 2, 2)
    with ce:
        ce.set([1], [v], 0.5)
        out = m.model.layers[0].mlp(hs)
    assert torch.allclose(out[0, 1], torch.full((2,), 5.0))  # halfway


def test_component_edit_rejects_bad_kind():
    m = _FakeModel()
    try:
        ComponentEdit(m, 0, "resid")
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad kind")


def test_component_store_roundtrip(tmp_path):
    from safetensors.numpy import save_file

    cs_dir = tmp_path / "component_states_v1"
    cs_dir.mkdir()
    meta = pd.DataFrame({"instance_id": ["a", "b", "c"],
                         "fact_id": ["f", "f", "g"],
                         "direction": ["direct"] * 3,
                         "is_correct": [True, False, True]})
    meta.to_parquet(cs_dir / "comp_meta.parquet", index=False)
    attn = np.arange(6, dtype=np.float16).reshape(3, 2)
    mlp = (np.arange(6, dtype=np.float16).reshape(3, 2) + 100)
    save_file({"h": attn}, str(cs_dir / "attn_L00.safetensors"))
    save_file({"h": mlp}, str(cs_dir / "mlp_L00.safetensors"))

    store = ComponentStore(tmp_path)
    assert np.allclose(store.vec("b", "attn", 0), attn[1])
    assert np.allclose(store.vec("c", "mlp", 0), mlp[2])
    assert store.vec("missing", "attn", 0) is None


def test_get_vec_and_edit_routing():
    from scripts.parametric_retrieval.prgc_component import (
        get_vec,
        make_edit,
        donor_for,
        prompt_for,
    )

    class _HS:
        def vec(self, i, pos, hs_idx):
            return np.array([hs_idx], dtype=np.float32)  # encodes hs_idx

    class _CS:
        def vec(self, i, kind, layer):
            return np.array([layer], dtype=np.float32)   # encodes layer

    # full mode reads residual at hs_idx = layer + 1
    assert get_vec(_HS(), _CS(), "x", "full", 5)[0] == 6
    # component modes read the component store at layer L
    assert get_vec(_HS(), _CS(), "x", "attn", 5)[0] == 5
    assert get_vec(_HS(), _CS(), "x", "mlp", 5)[0] == 5

    m = _FakeModel(n=6)
    assert isinstance(make_edit(m, "attn", 2), ComponentEdit)
    # full maps decoder layer L -> ResidualEdit on hs_idx L+1 (layers[L])
    re = make_edit(m, "full", 2)
    assert re.block is m.model.layers[2]

    class _Row:
        donor_matched = "dm"
        donor_noop = "dn"
        donor_mismatched_type = "dt"
        donor_mismatched_rand = "dr"
        donor_reverse = "rev"
        donor_instance_id = "succ"
        recipient_instance_id = "fail"

    r = _Row()
    assert donor_for("matched", r) == "dm"
    assert donor_for("random_noise", r) == "dm"   # noise built from matched edit
    assert donor_for("reverse", r) == "rev"
    assert prompt_for("matched", r) == "fail"
    assert prompt_for("reverse", r) == "succ"      # patch failed state into success
