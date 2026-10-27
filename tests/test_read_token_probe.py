import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts.read_token_probe import (  # noqa: E402
    ReadToken, full_forward_states, make_optimizer, read_all_states, read_states)


def tiny_model():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=128)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


SEQS = [[1, 5, 9, 3, 7, 2], [4, 8]]
DEV = torch.device("cpu")


def test_mean_equals_last_at_m1():
    model = tiny_model()
    init = torch.randn(1, 32) * 0.02
    last = read_states(model, ReadToken(1, 32, init, read="last"), SEQS, 0, DEV, grad=False)
    mean = read_states(model, ReadToken(1, 32, init, read="mean"), SEQS, 0, DEV, grad=False)
    assert torch.allclose(last, mean)


def test_mean_read_averages_all_states():
    model = tiny_model()
    rt = ReadToken(3, 32, torch.randn(3, 32) * 0.02, read="mean")
    H = read_all_states(model, rt, SEQS, 0, DEV, grad=False)
    assert H.shape == (2, 3, 32)
    assert torch.allclose(read_states(model, rt, SEQS, 0, DEV, grad=False), H.mean(1))


def test_zero_deep_matches_v0_and_nonzero_deep_matches_full_forward():
    model = tiny_model()
    init = torch.randn(2, 32) * 0.02
    v0 = read_all_states(model, ReadToken(2, 32, init), SEQS, 0, DEV, grad=False)
    rt = ReadToken(2, 32, init, n_layers=3)
    assert torch.allclose(read_all_states(model, rt, SEQS, 0, DEV, grad=False), v0)
    with torch.no_grad():
        rt.deep.normal_(0, 0.5)
    got = read_all_states(model, rt, SEQS, 0, DEV, grad=True)
    assert not torch.allclose(got.detach(), v0, atol=1e-3)
    for r, s in enumerate(SEQS):
        ref = full_forward_states(model, rt, s, DEV)
        assert torch.allclose(got[r].detach(), ref.detach(), atol=2e-4), r
    got.sum().backward()
    assert rt.deep.grad.abs().sum() > 0 and rt.soft.grad.abs().sum() > 0


def test_deep_hooks_are_removed_after_the_read():
    model = tiny_model()
    rt = ReadToken(1, 32, torch.randn(1, 32) * 0.02, n_layers=3)
    with torch.no_grad():
        rt.deep.fill_(1.0)
    read_all_states(model, rt, SEQS, 0, DEV, grad=False)
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_freeze_soft_keeps_embedding_and_trains_head():
    model = tiny_model()
    rt = ReadToken(2, 32, torch.randn(2, 32) * 0.02)
    soft0 = rt.soft.detach().clone()
    head0 = rt.head.weight.detach().clone()
    opt = make_optimizer(rt, 1e-2, freeze_soft=True)
    h = read_states(model, rt, SEQS, 0, DEV, grad=False)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(rt.logit(h), torch.tensor([1.0, 0.0]))
    opt.zero_grad(); loss.backward(); opt.step()
    assert torch.equal(rt.soft.detach(), soft0)
    assert not torch.equal(rt.head.weight.detach(), head0)


def test_mlp_head_logit_shape():
    rt = ReadToken(1, 32, torch.zeros(1, 32), head="mlp")
    assert rt.logit(torch.randn(5, 32)).shape == (5,)


@pytest.mark.slow
def test_cached_left_padded_read_state_matches_full_forward():
    name = "Qwen/Qwen2.5-0.5B"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
                                                                  local_files_only=True).eval()
    except OSError:
        pytest.skip("model not cached")
    d = model.config.hidden_size
    rt = ReadToken(3, d, torch.randn(3, d) * 0.02)
    seqs = [tok("Problem: 2+2. Step: 4.")["input_ids"], tok("Hi")["input_ids"]]
    got = read_states(model, rt, seqs, tok.eos_token_id, torch.device("cpu"), grad=True)
    got.sum().backward()
    assert rt.soft.grad is not None and rt.soft.grad.abs().sum() > 0
    for r, s in enumerate(seqs):
        ref = full_forward_states(model, rt, s, torch.device("cpu"))[-1]
        assert torch.allclose(got[r].detach(), ref.detach(), atol=2e-3), (r, (got[r] - ref).abs().max())


def test_select_prefers_smaller_m_within_tie_and_picks_m_all():
    from scripts.analysis.read_token_select import select
    assert select({"m1": 0.900, "m4": 0.901, "m16": 0.880, "m4text": 0.870}) == ("m1", "m4")
    assert select({"m1": 0.890, "m4": 0.880, "m16": 0.895, "m4text": 0.899}) == ("m4text", "m4text")
    assert select({"m1": 0.850, "m4": 0.880, "m16": 0.881, "m4text": 0.870}) == ("m4", "m4")


def _write_split(d, name, n, m, rng, pb=False):
    import json
    import numpy as np
    y = rng.integers(0, 2, n).astype(np.int8)
    allst = rng.normal(size=(n, m, 8)).astype(np.float16)
    allst[:, :, 0] += 2.0 * y[:, None]
    np.savez(d / f"{name}.shard0.npz", layers=np.array([-1]), y=y, kept=np.ones(n, bool),
             head_p_correct=np.full(n, np.nan, np.float32), sep=allst[:, -1:],
             sep_all=allst, sep_mean=allst.mean(1, keepdims=True))
    with open(d / f"{name}.shard0.meta.jsonl", "w") as f:
        for i in range(n):
            meta = {"uid": str(i), "label": int(y[i])}
            if pb:
                meta.update(id=f"t{i // 4}", step_idx=i % 4, first_error=-1)
            f.write(json.dumps(meta) + "\n")


def test_readouts_run_end_to_end(tmp_path, monkeypatch):
    import json
    import sys
    import numpy as np
    from scripts.analysis import read_token_readouts as rr
    rng = np.random.default_rng(0)
    for name, n in (("train", 200), ("val", 80), ("test", 80)):
        _write_split(tmp_path, name, n, 3, rng)
    for sub in ("gsm8k", "math", "olympiadbench", "omnimath"):
        _write_split(tmp_path, f"pb_{sub}", 80, 3, rng, pb=True)
    monkeypatch.setattr(rr, "LR_GRID", (1e-2,))
    monkeypatch.setattr(rr, "WD_GRID", (0.0,))
    monkeypatch.setattr(sys, "argv", ["x", "--configs", f"c:{tmp_path}", "--out", str(tmp_path / "o")])
    rr.main()
    rows = json.loads((tmp_path / "o" / "readouts.json").read_text())
    assert [r["method"] for r in rows] == ["linear", "mlp:h1024", "attn_query"]
    assert all(r["test_auroc"] > 0.8 for r in rows)


def test_no_cudnn_soft_pass_matches_default():
    model = tiny_model()
    rt = ReadToken(2, 32, torch.randn(2, 32) * 0.02)
    a = read_all_states(model, rt, SEQS, 0, DEV, grad=False)
    b = read_all_states(model, rt, SEQS, 0, DEV, grad=False, soft_attn="no_cudnn")
    assert torch.allclose(a, b, atol=1e-5)


def test_capture_records_offset_block_outputs_and_ends_at_the_read_state():
    from scripts.read_token_probe import capture_hooks, deep_hooks
    model = tiny_model()
    rt = ReadToken(2, 32, torch.randn(2, 32) * 0.02, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, 0.5)
    cap = {0: None, 2: None}
    H = read_all_states(model, rt, SEQS, 0, DEV, grad=False, capture=cap)
    assert set(cap) == {0, 2} and cap[0].shape == (2, 32)
    # the last block's output, normed, is the read state
    assert torch.allclose(model.model.norm(cap[2]), H[:, -1], atol=1e-5)
    # cached capture equals a full forward with the same hooks
    for r, s in enumerate(SEQS):
        e = model.get_input_embeddings()(torch.tensor([s]))
        full = torch.cat([e, rt.soft.to(model.dtype)[None]], 1)
        ref = {0: None, 2: None}
        with torch.no_grad(), deep_hooks(model, rt), capture_hooks(model, [0, 2], ref):
            model.model(inputs_embeds=full)
        for l in (0, 2):
            assert torch.allclose(cap[l][r], ref[l][0], atol=2e-4), (r, l)
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_capture_prefix_is_the_last_step_token_unchanged_by_the_read_token():
    model = tiny_model()
    rt = ReadToken(2, 32, torch.randn(2, 32) * 0.02, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, 0.5)
    pre = {0: None, 1: None, 2: None, "final": None}
    read_all_states(model, rt, SEQS, 0, DEV, grad=False, capture_prefix=pre)
    assert all(not layer._forward_hooks for layer in model.model.layers)
    for r, s in enumerate(SEQS):
        with torch.no_grad():
            hs = model.model(input_ids=torch.tensor([s]), output_hidden_states=True).hidden_states
        for l in (0, 1):  # hidden_states[l + 1] is block l's output
            assert torch.allclose(pre[l][r], hs[l + 1][0, -1], atol=2e-4), (r, l)
        assert torch.allclose(pre["final"][r], hs[-1][0, -1], atol=2e-4)
    assert torch.allclose(model.model.norm(pre[2]), pre["final"], atol=1e-5)


def test_branched_reads_equal_one_read_per_item():
    from scripts.read_token_probe import read_branched
    model = tiny_model()
    rt = ReadToken(2, 32, torch.randn(2, 32) * 0.02, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, 0.5)
    trunk = [3, 9, 4, 7, 7, 1, 5, 2, 8, 6, 11, 12]
    # shared prefixes of different lengths, a different tail, the trunk itself, a disjoint item
    items = [trunk[:4] + [20, 21], trunk[:7] + [22], trunk, trunk[:10] + [23, 24, 25], [30, 31, 32]]
    for max_rows in (2, 16):
        H = read_branched(model, rt, items, 0, DEV, max_rows=max_rows)
        for r, ids in enumerate(items):
            ref = read_all_states(model, rt, [ids], 0, DEV, grad=False)[0]
            assert torch.allclose(H[r], ref, atol=1e-4), (max_rows, r, (H[r] - ref).abs().max())


def _interv_token(kind, randomize):
    model = tiny_model()
    at = model.model.layers[0].self_attn
    rt = ReadToken(2, 32, torch.randn(2, 32) * 0.02, n_layers=3, interv=kind, rank=4, n_interv_layers=3,
                   lora_dims=(at.q_proj.out_features, at.v_proj.out_features) if kind == "lora" else None)
    with torch.no_grad():
        rt.deep.normal_(0, 0.3)
        if randomize:
            for q in rt.interv_params():
                q.normal_(0, 0.2)
    return model, rt


@pytest.mark.parametrize("kind", ["loreft", "lora"])
def test_interventions_start_as_the_identity(kind):
    from scripts.read_token_probe import ReadToken as RT
    model, rt = _interv_token(kind, randomize=False)
    plain = RT(2, 32, rt.soft.detach(), n_layers=3)
    with torch.no_grad():
        plain.deep.copy_(rt.deep)
    a = read_all_states(model, rt, SEQS, 0, DEV, grad=False)
    b = read_all_states(model, plain, SEQS, 0, DEV, grad=False)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("kind", ["loreft", "lora"])
def test_interventions_cached_branched_and_full_forward_agree(kind, tmp_path):
    from scripts.read_token_probe import read_branched, read_token_from_ckpt
    model, rt = _interv_token(kind, randomize=True)
    H = read_all_states(model, rt, SEQS, 0, DEV, grad=False)
    plain = ReadToken(2, 32, rt.soft.detach(), n_layers=3)
    with torch.no_grad():
        plain.deep.copy_(rt.deep)
    assert not torch.allclose(H, read_all_states(model, plain, SEQS, 0, DEV, grad=False), atol=1e-3)
    for r, s in enumerate(SEQS):
        assert torch.allclose(H[r], full_forward_states(model, rt, s, DEV), atol=2e-4)
    items = [[3, 9, 4, 7, 7, 1], [3, 9, 4, 20], [3, 9, 4, 7, 7, 1, 5, 2]]
    B = read_branched(model, rt, items, 0, DEV)
    for r, ids in enumerate(items):
        assert torch.allclose(B[r], read_all_states(model, rt, [ids], 0, DEV, grad=False)[0], atol=1e-4)
    ck = {"soft": rt.soft.detach(), "head": rt.head.state_dict(), "m": 2, "deep": rt.deep.detach(),
          "interv": kind, "rank": 4, "lora_dims": rt.lora_dims, "interv_state": rt.interv_state()}
    rt2 = read_token_from_ckpt(ck, model, DEV)
    assert torch.allclose(read_all_states(model, rt2, SEQS, 0, DEV, grad=False), H, atol=1e-6)
    assert all(not layer._forward_hooks for layer in model.model.layers)
    assert all(not layer.self_attn.q_proj._forward_hooks for layer in model.model.layers)


@pytest.mark.parametrize("kind", ["loreft", "lora"])
def test_one_training_step_moves_the_intervention(kind):
    model, rt = _interv_token(kind, randomize=False)
    opt = make_optimizer(rt, 1e-2, freeze_soft=False)
    before = [q.detach().clone() for q in rt.interv_params()]
    h = read_states(model, rt, SEQS, 0, DEV, grad=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(rt.logit(h), torch.tensor([1.0, 0.0]))
    opt.zero_grad(); loss.backward(); opt.step()
    assert any(not torch.equal(b, q) for b, q in zip(before, rt.interv_params()))
