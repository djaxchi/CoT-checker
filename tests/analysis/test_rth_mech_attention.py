import torch
import transformers

from scripts.analysis import rth_mech_attention as m3
from scripts.read_token_probe import ReadToken, read_all_states, read_hooks, tokenize

DEV = torch.device("cpu")


class OffTok:
    """One token per character, with character offsets."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        out = {"input_ids": [1 + ord(ch) % 60 for ch in text] or [1]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))] or [(0, 0)]
        return out


def tiny_model():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=256)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


ITEMS = [{"problem": "2+2?", "prefix": "a\n\nb", "step": "so 4"},
         {"problem": "x", "prefix": "", "step": "ok"}]


def deep_token():
    rt = ReadToken(1, 32, torch.randn(1, 32) * 0.1, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, 0.2)
    return rt.eval()


def test_segments_match_tokenize_and_template():
    tok = OffTok()
    ids, seg = m3.segment_ids(tok, ITEMS[0])
    assert ids == tokenize(tok, ITEMS[0])
    text = "Problem:\n2+2?\n\nPrevious reasoning:\na\n\nb\n\nCurrent step:\nso 4"
    assert len(seg) == len(text)
    got = {k: "".join(c for c, s in zip(text, seg) if s == k) for k in range(5)}
    assert got[0] == "P"
    assert got[1] == "roblem:\n2+2?"
    assert got[2] == "\n\nPrevious reasoning:\na\n\nb"
    assert got[3] == "\n\nCurrent step:\n"
    assert got[4] == "so 4"


def test_mass_sums_to_one_and_matches_a_full_eager_forward():
    tok, model, rt = OffTok(), tiny_model(), deep_token()
    seg = [m3.segment_ids(tok, it) for it in ITEMS]
    att, cache, lens = m3.prefix_pass(model, [s[0] for s in seg], 0, DEV)
    h, A = m3.soft_pass(model, rt, att, cache, lens, DEV, eager=True)
    mass = m3.segment_mass(A, [s[1] for s in seg], 1)
    assert torch.allclose(mass.sum(-1), torch.ones(mass.shape[:-1]), atol=1e-5)
    # the cache was copied, so the SDPA pass afterwards sees the same prefix
    h_s, none = m3.soft_pass(model, rt, att, cache, lens, DEV, eager=False)
    assert none is None and torch.allclose(h, h_s, atol=1e-4)
    assert torch.allclose(h_s, read_all_states(model, rt, [s[0] for s in seg], 0, DEV, grad=False)[:, -1],
                          atol=1e-4)
    assert model.config._attn_implementation == "sdpa"
    # reference: one unpadded eager forward over prefix + read token, last row of attention
    model.set_attn_implementation("eager")
    try:
        for r, (ids, sg) in enumerate(seg):
            e = model.get_input_embeddings()(torch.tensor([ids]))
            with torch.no_grad(), read_hooks(model, rt):
                out = model.model(inputs_embeds=torch.cat([e, rt.soft[None]], 1), output_attentions=True)
            for l, a in enumerate(out.attentions):
                last = a[0, :, -1]  # (heads, len + 1)
                ref = torch.stack([last[:, :-1][:, torch.tensor(sg) == k].sum(-1) for k in range(5)]
                                  + [last[:, -1]], -1)
                assert torch.allclose(mass[r, l], ref, atol=1e-4)
    finally:
        model.set_attn_implementation("sdpa")


def test_summary_splits_by_label():
    import numpy as np
    mass = np.zeros((4, 2, 3, 6), np.float16)
    mass[:2, ..., 0] = 1.0   # correct items: all on the sink
    mass[2:, ..., 4] = 1.0   # incorrect items: all on the step
    y = np.array([0, 0, 1, 1])
    s = m3.summarize("t", mass, np.ones((4, 6), int), y, np.zeros(4), np.zeros(4))
    assert s["mass_correct"]["sink"] == [1.0, 1.0] and s["mass_incorrect"]["step"] == [1.0, 1.0]
    assert s["mass"]["sink"] == [0.5, 0.5] and s["gate"]["max_abs_logit_diff"] == 0.0
