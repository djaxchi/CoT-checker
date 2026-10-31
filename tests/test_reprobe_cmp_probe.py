"""reprobe_comparison_v1: the ReProbe training/scoring loop on a tiny backbone."""

import json
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts.reprobe_cmp.reprobe_probe import (  # noqa: E402
    claim_tensors, features, load_traces, micro_batches, score_traces, train)
from src.probes.reprobe_head import ReProbeHead  # noqa: E402


def tiny_model():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    m = transformers.Qwen2ForCausalLM(cfg).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def write_traces(path, n, seed):
    g = torch.Generator().manual_seed(seed)
    with open(path, "w") as f:
        for i in range(n):
            L = int(torch.randint(12, 30, (1,), generator=g))
            ids = torch.randint(1, 64, (L,), generator=g).tolist()
            ctx = 5
            cut = sorted({int(x) for x in torch.randint(1, L - ctx - 1, (2,), generator=g)})
            bounds = [0] + cut + [L - ctx - 1]
            steps = [{"aligned_token_ids": list(range(a, b))} for a, b in zip(bounds, bounds[1:]) if b > a]
            y = [int(ids[ctx + s["aligned_token_ids"][0]] % 2) for s in steps]
            f.write(json.dumps({"trace_id": f"t{i}", "input_ids": ids, "context_len": ctx, "steps": steps,
                                "labels": {"ds": y, "self": y}}) + "\n")


def test_features_shift_and_padding():
    m = tiny_model()
    tr = [{"ids": [1, 2, 3, 4], "steps": [[2]]}, {"ids": [5, 6], "steps": [[0]]}]
    f, v = features(m, tr, torch.device("cpu"), 0)
    # lengths are padded to a multiple of 64 (63 feature positions); only real ones are valid
    assert f.shape[:2] == (2, 63) and v[:, :3].tolist() == [[True, True, True], [True, False, False]]
    assert not v[:, 3:].any()
    ref = m.model(input_ids=torch.tensor([[1, 2, 3, 4]])).last_hidden_state[0, :3]
    assert torch.allclose(f[0, :3], ref, atol=1e-5)
    claims, owner, where = claim_tensors(tr, 3, torch.device("cpu"))
    assert claims.tolist() == [[0, 0, 1], [1, 0, 0]] and owner.tolist() == [0, 1] and where == [(0, 0), (1, 0)]


def test_micro_batches_cover_every_trace_once():
    tr = [{"ids": [0] * n, "steps": [[0]] * 3} for n in (5, 50, 7, 30, 9)]
    mbs = micro_batches(tr, list(range(5)), max_tokens=60, max_claim_tokens=200)
    assert sorted(i for mb in mbs for i in mb) == list(range(5))


def test_scores_do_not_depend_on_batching(tmp_path):
    m = tiny_model()
    write_traces(tmp_path / "x.jsonl", 6, 0)
    tr = load_traces(tmp_path / "x.jsonl", None)
    torch.manual_seed(1)
    h = ReProbeHead(32, head_dim=16, n_heads=2).eval()
    big = score_traces(m, {"h": (h, "masked")}, tr, torch.device("cpu"), 0, 10_000, 10_000)
    one = score_traces(m, {"h": (h, "masked")}, tr, torch.device("cpu"), 0, 1, 1)
    assert big["h"].keys() == one["h"].keys()
    for k in big["h"]:
        assert abs(big["h"][k] - one["h"][k]) < 1e-4


def test_train_writes_best_heads_and_scores(tmp_path):
    m = tiny_model()
    write_traces(tmp_path / "tr.jsonl", 10, 1)
    write_traces(tmp_path / "va.jsonl", 6, 2)
    a = SimpleNamespace(heads=["a:ds:visible", "b:self:masked"], train=tmp_path / "tr.jsonl",
                        val=tmp_path / "va.jsonl", smoke=0, seed=0, epochs=2, batch_traces=4, lr=1e-3, wd=0.1,
                        pos_weight=3.0, max_tokens=2000, max_claim_tokens=20000, out=tmp_path / "o",
                        sets=[f"va:{tmp_path / 'va.jsonl'}"], val_scores=True)
    train(a, m, None, 0, torch.device("cpu"))
    ck = torch.load(a.out / "heads.pt", weights_only=False)
    assert set(ck) == {"a", "b", "_meta"} and ck["a"]["future"] == "visible" and ck["b"]["label"] == "self"
    rows = [json.loads(l) for l in open(a.out / "scores" / "a" / "va.jsonl")]
    assert len(rows) == sum(len(json.loads(l)["steps"]) for l in open(tmp_path / "va.jsonl"))
    assert (a.out / "scores" / "b" / "val.jsonl").exists()
