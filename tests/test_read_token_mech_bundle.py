import json

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts import read_token_mech_bundle as rmb  # noqa: E402
from scripts.analysis.drv_embed import embed  # noqa: E402
from scripts.read_token_probe import ReadToken, read_all_states, tokenize  # noqa: E402
from tests.analysis.test_drv_embed import drv_cell  # noqa: E402
from tests.test_read_token_probe import tiny_model  # noqa: E402
from tests.test_read_token_store import D, CharTok, ref_store, source  # noqa: E402


def test_bundle_rows_match_direct_reads_in_reference_order(tmp_path, monkeypatch):
    order = [5, 0, 3, 1, 4, 2]
    ref = tmp_path / "ref" / "test_2k"
    ref_store(ref, 6, order)
    source(tmp_path / "s.jsonl", 6)
    model = tiny_model()
    rt = ReadToken(1, D, torch.randn(1, D) * 0.02, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, 0.3)
    ck = tmp_path / "rt.pt"
    torch.save({"soft": rt.soft.detach(), "head": rt.head.state_dict(), "m": 1, "read": "last",
                "head_kind": "linear", "deep": rt.deep.detach()}, ck)
    drv = drv_cell(tmp_path / "cell", D)
    # the same tiny model stands in for both backbones
    monkeypatch.setattr(rmb, "load_model", lambda *_a: (model, CharTok()))
    for i in (0, 1):
        monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--drv_cell", str(tmp_path / "cell"),
                                         "--ref", str(ref), "--source", f"prm800k:{tmp_path / 's.jsonl'}",
                                         "--out", str(tmp_path / "b"), "--drv_blocks", "0,2",
                                         "--shards", f"{i}/2"])
        rmb.main()
    b = rmb.load_bundle(tmp_path / "b" / "test_2k")
    assert [m["uid"] for m in b["meta"]] == [f"u{k}" for k in order]
    assert list(b["y"]) == [k % 2 for k in order] and list(b["index"]) == list(range(6))
    assert b["read"].shape == (6, 4, D) and b["last_step"].shape == (6, 4, D) and b["prm"].shape == (6, 4, D)
    # one model for both passes: the PRM rows are the last-step rows
    assert np.allclose(b["prm"], b["last_step"], atol=2e-3)
    items = {f"u{k}": {"problem": f"p{k}", "prefix": "a\n\nb" if k % 3 else "", "step": f"step {k} x"}
             for k in range(6)}
    for g, m in enumerate(b["meta"]):
        ids = tokenize(CharTok(), items[m["uid"]])
        cap = {l: None for l in range(3)}
        with torch.no_grad():
            H = read_all_states(model, rt, [ids], 0, torch.device("cpu"), grad=False, capture=cap)[:, -1]
            hs = model.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
        assert np.allclose(b["read"][g, -1], H[0], atol=2e-2)
        assert np.allclose(b["read"][g, 1], cap[1][0], atol=2e-2)
        assert np.allclose(b["last_step"][g, 0], hs[1][0, -1], atol=2e-2)
        assert np.allclose(b["last_step"][g, -1], hs[-1][0, -1], atol=2e-2)
        assert np.isclose(b["head_logit"][g], rt.logit(H).item(), atol=2e-2)
    _, L = embed(drv, np.concatenate([b["read"][:, [0, 2]], b["read"][:, 3:]], 1).astype(np.float32),
                 torch.device("cpu"))
    assert np.allclose(b["drv_logit"], L, atol=2e-2)


def test_pick_is_a_seeded_sorted_subsample():
    a, b = rmb.pick(100, 10, 0), rmb.pick(100, 10, 0)
    assert np.array_equal(a, b) and len(a) == 10 and np.all(np.diff(a) > 0)
    assert np.array_equal(rmb.pick(5, 0, 0), np.arange(5))
