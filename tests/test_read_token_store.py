import json

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts import read_token_store as rts  # noqa: E402
from scripts.read_token_probe import ReadToken, read_all_states, tokenize  # noqa: E402
from src.repstore.store import TOKEN_SEQ, RepSpec, ShardedRepSplit, write_split  # noqa: E402
from tests.test_read_token_probe import tiny_model  # noqa: E402

D = 32


class CharTok:
    """Character ids, enough for tokenize() on a tiny model."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [1 + ord(ch) % 60 for ch in text][:40] or [1]}


def ref_store(root, n, order):
    """A reference span store whose items come in `order`, labels alternating."""
    items, metas, labels = [], [], []
    for gi, k in enumerate(order):
        items.append(np.random.default_rng(k).normal(size=(3, D)).astype(np.float32))
        metas.append({"uid": f"u{k}", "global_index": gi, "n_tokens": 3, "step_start_idx": 1,
                      "pre_step_boundary_idx": 0})
        labels.append(k % 2)
    write_split(root / "shard_00", items, labels, metas,
                RepSpec(name="step_spans", kind=TOKEN_SEQ, dim=D, layer=35, backbone="tiny",
                        readout="step_span_with_boundary"))


def source(path, n):
    with path.open("w") as f:
        for k in range(n):
            f.write(json.dumps({"uid": f"u{k}", "problem": f"p{k}", "prefix": "a\n\nb" if k % 3 else "",
                                "candidate_step": f"step {k} x", "label": k % 2}) + "\n")


def test_match_follows_reference_order_and_refuses_gaps_and_label_flips(tmp_path):
    source(tmp_path / "s.jsonl", 6)
    kind, index = rts.source_index(f"prm800k:{tmp_path / 's.jsonl'}")
    meta = [{"uid": f"u{k}"} for k in (4, 1, 2)]
    assert [it["uid"] for it in rts.match(kind, index, meta, np.array([0, 1, 0]))] == ["u4", "u1", "u2"]
    with pytest.raises(SystemExit):
        rts.match(kind, index, [{"uid": "u9"}], np.array([1]))
    with pytest.raises(SystemExit):
        rts.match(kind, index, [{"uid": "u4"}], np.array([1]))


def test_store_round_trip_matches_a_direct_read(tmp_path, monkeypatch):
    order = [5, 0, 3, 1, 4, 2]
    ref = tmp_path / "ref" / "val_5k"
    ref_store(ref, 6, order)
    source(tmp_path / "s.jsonl", 6)
    model = tiny_model()
    rt = ReadToken(2, D, torch.randn(2, D) * 0.02, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, 0.3)
    ck = tmp_path / "rt.pt"
    torch.save({"soft": rt.soft.detach(), "head": rt.head.state_dict(), "m": 2, "read": "last",
                "head_kind": "linear", "deep": rt.deep.detach()}, ck)

    monkeypatch.setattr(rts, "load_model", lambda *_a: (model, CharTok()))
    out = tmp_path / "out"
    for i in (0, 1):  # two workers, chunks of 4 items: shard_00_000 and shard_00_001
        monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--ref", str(ref),
                                         "--source", f"prm800k:{tmp_path / 's.jsonl'}",
                                         "--out_root", str(out), "--blocks", "0,1",
                                         "--append_block", "1", "--chunk", "4",
                                         "--shards", f"{i}/2"])
        rts.main()
    layers = ShardedRepSplit(out / "rt_layers" / "val_5k")
    app = ShardedRepSplit(out / "rt_append" / "val_5k")
    assert [m["uid"] for m in layers.meta()] == [f"u{k}" for k in order]
    assert list(layers.y) == [k % 2 for k in order] and app.is_vector
    _, index = rts.source_index(f"prm800k:{tmp_path / 's.jsonl'}")
    tok = CharTok()
    for g, k in enumerate(order):
        cap = {0: None, 1: None}
        H = read_all_states(model, rt, [tokenize(tok, index[f"u{k}"])], 0, torch.device("cpu"),
                            grad=False, capture=cap)
        item = layers.item(g)
        assert item.shape == (4, D)  # boundary slot, block 0, block 1, final
        assert np.allclose(item[0], H[0, -1], atol=2e-2) and np.allclose(item[-1], H[0, -1], atol=2e-2)
        assert np.allclose(item[1], cap[0][0], atol=2e-2)
        assert np.allclose(app.item(g)[0], cap[1][0], atol=2e-2)
    m0 = layers.meta()[0]
    assert m0["step_start_idx"] == 1 and m0["n_tokens"] == 4 and m0["span_n_tokens"] == 3
