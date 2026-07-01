"""encode_file must read the requested hidden_states layer (default -1 = final).

Guards the backward-compatible --layer arg used to build L20-native probe features,
using stub tokenizer/model so no real weights are needed. hidden_states[k] is filled
with the constant k, so the saved vector's value reveals which layer was read.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SCRIPT = (Path(__file__).resolve().parent.parent.parent
           / "scripts" / "encode_prm800k_hidden_states.py")
_spec = importlib.util.spec_from_file_location("encode_prm800k_hidden_states", _SCRIPT)
enc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(enc)

H, N_LAYERS = 6, 4  # hidden_states has N_LAYERS+1 entries (embeddings + blocks)


class StubTok:
    """Whitespace 'tokenizer': one id per word, +1 leading id when add_special_tokens."""
    def __call__(self, text, add_special_tokens=True, truncation=False):
        ids = [7] * len(text.split())
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


class StubModel:
    def __init__(self):
        self.config = SimpleNamespace(hidden_size=H, num_hidden_layers=N_LAYERS)

    def to(self, *a, **k):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False,
                 use_cache=False):
        B, S = input_ids.shape
        hs = tuple(torch.full((B, S, H), float(k)) for k in range(N_LAYERS + 1))
        return SimpleNamespace(hidden_states=hs)


def _write_items(path):
    rows = [{
        "uid": f"u{i}", "problem_id": f"p{i}", "solution_id": f"s{i}",
        "step_idx": i, "completion_idx": 0, "label": i % 2, "rating": "-1",
        "problem": "a b c", "prefix": "d e", "candidate_step": "f g h",
    } for i in range(3)]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _encode(tmp_path, layer):
    items = tmp_path / "items.jsonl"
    _write_items(items)
    out = tmp_path / f"L{layer}"
    out.mkdir()
    enc.encode_file(
        jsonl_path=items, out_dir=out, stem="s", tokenizer=StubTok(),
        model=StubModel(), device=torch.device("cpu"), max_seq_len=64,
        batch_size=2, save_dtype=torch.float32, pad_token_id=0, layer=layer)
    return np.load(out / "s_h.npy")


def test_default_reads_final_layer(tmp_path):
    h = _encode(tmp_path, -1)
    assert h.shape == (3, H)
    np.testing.assert_allclose(h, float(N_LAYERS))  # hidden_states[-1] == layer N_LAYERS


def test_explicit_layer_read(tmp_path):
    assert np.allclose(_encode(tmp_path, 2), 2.0)
    assert np.allclose(_encode(tmp_path, 0), 0.0)


def test_layer_is_backward_compatible(tmp_path):
    # -1 and the explicit final index must give identical output.
    np.testing.assert_array_equal(_encode(tmp_path, -1), _encode(tmp_path, N_LAYERS))
