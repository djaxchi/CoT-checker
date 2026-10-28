import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts.inline_fast import TEXT, FastReader, FusedReader, cache_kv, fill_inputs, layout_kinds, static_inputs  # noqa: E402
from scripts.read_token_inline import StartMarker, read_pass, text_pass  # noqa: E402
from scripts.read_token_probe import ReadToken  # noqa: E402

DEV = torch.device("cpu")
SEQS = [([1, 5, 9, 3, 7, 2, 11, 4, 6], [3, 6, 9]), ([4, 8, 13, 2], [4])]
SPOS = [[0, 3, 6], [1]]


def tiny(kind):
    torch.manual_seed(0)
    kw = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    if kind == "qwen3":
        model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(head_dim=8, **kw))
    else:
        model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**kw))
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


def token():
    rt = ReadToken(1, 32, torch.randn(1, 32) * 0.5, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, 0.5)
    return rt


def marker():
    mk = StartMarker(32, torch.randn(1, 32) * 0.5, n_layers=3)
    with torch.no_grad():
        mk.deep.normal_(0, 0.5)
    return mk


@pytest.mark.parametrize("arch", ["qwen2", "qwen3"])
@pytest.mark.parametrize("with_marker", [False, True])
@pytest.mark.parametrize("chain", [False, True])
def test_fast_reader_equals_read_pass(arch, with_marker, chain):
    model, rt = tiny(arch), token()
    mk = marker() if with_marker else None
    seqs = [s + (sp,) for s, sp in zip(SEQS, SPOS)] if with_marker else SEQS
    cache, lay = text_pass(model, seqs, 0, DEV, marker=with_marker, chain=chain)
    got = FastReader(model, rt, mk).read(cache, lay)
    T = cache.get_seq_length()
    ref = read_pass(model, rt, cache, lay, grad=False, mk=mk)
    assert cache.get_seq_length() > T  # read_pass appended; FastReader read the text part only
    assert got.shape == ref.shape == (4, 32)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_fast_reader_leaves_the_cache_untouched_and_reads_text_slots():
    model, rt = tiny("qwen3"), token()
    ids = [1, 5, 9, 3, 7]
    cache, _ = text_pass(model, [(ids[:4], [4])], 0, DEV)
    ks, vs = cache_kv(cache)
    T = ks[0].shape[2]
    # one text slot: the next token's decode step, against the plain model
    kind = torch.tensor([[TEXT]])
    mask = torch.ones(1, 1, 1, T + 1, dtype=torch.bool)
    h = FastReader(model, rt).forward(ks, vs, kind, torch.tensor([[4]]), mask, ids=torch.tensor([[ids[4]]]))
    ref = model.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1]
    assert torch.allclose(h[0, 0], ref, atol=1e-5)
    assert cache.get_seq_length() == T


def test_padded_static_inputs_give_the_same_read_states():
    model, rt, mk = tiny("qwen3"), token(), marker()
    seqs = [s + (sp,) for s, sp in zip(SEQS, SPOS)]
    cache, lay = text_pass(model, seqs, 0, DEV, marker=True, chain=True)
    fr = FastReader(model, rt, mk)
    ks, vs = cache_kv(cache)
    kind = layout_kinds(lay)
    ref = fr.forward(ks, vs, kind, lay["pos"], lay["mask"])
    st = static_inputs(fr, 2, 16, 8, False)
    st["ks"][0].fill_(3.0)  # stale values in the padded columns must not leak in
    fill_inputs(st, ks, vs, kind, lay["pos"], lay["mask"])
    got = fr.forward(**st)[:, :kind.shape[1]]
    assert torch.allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("arch", ["qwen2", "qwen3"])
@pytest.mark.parametrize("with_marker", [False, True])
def test_fused_reader_equals_read_pass_and_leaves_the_model_unchanged(arch, with_marker):
    model, rt = tiny(arch), token()
    mk = marker() if with_marker else None
    seqs = [s + (sp,) for s, sp in zip(SEQS, SPOS)] if with_marker else SEQS
    before = model.model(input_ids=torch.tensor([SEQS[0][0]])).last_hidden_state
    fr = FusedReader(model, rt, mk, compile=False)
    after = model.model(input_ids=torch.tensor([SEQS[0][0]])).last_hidden_state
    assert torch.equal(before, after)
    assert model.model.layers[0].self_attn.k_proj.weight.data_ptr() != fr.w[0][1].data_ptr()  # a view, not the start
    cache, lay = text_pass(model, seqs, 0, DEV, marker=with_marker, chain=True)
    got = fr.read(cache, lay)
    ref = read_pass(model, rt, cache, lay, grad=False, mk=mk)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_decode_inputs_put_the_read_after_its_marker():
    from scripts.analysis.inline_speed import decode_inputs
    from scripts.inline_fast import MARKER, READ
    kind, pos, mask, ids = decode_inputs(5, 1, True, DEV)
    assert kind.tolist() == [[TEXT, MARKER, READ]] and pos.tolist() == [[5, 5, 5]]
    assert mask[0, 0, :, :5].all()
    assert mask[0, 0, 2, 5 + 1] and not mask[0, 0, 0, 5 + 1] and not mask[0, 0, 2, 5]
    kind, _, mask, _ = decode_inputs(5, 0, True, DEV)
    assert kind.tolist() == [[TEXT]] and mask.shape == (1, 1, 1, 6)
