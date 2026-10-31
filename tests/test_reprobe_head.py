"""reprobe_comparison_v1: the ReProbe head reimplementation and the step extractor port."""

import torch

from src.data.reprobe_steps import split_to_steps, step_item
from src.probes.reprobe_head import ReProbeHead, n_params, step_feature_index


def _head(**kw):
    torch.manual_seed(0)
    return ReProbeHead(feature_dim=16, head_dim=8, n_heads=2, dropout=0.0, **kw).eval()


def test_released_parameter_names_and_size():
    names = set(ReProbeHead().state_dict())
    for k in ("proj.0.weight", "proj.1.weight", "proj.4.weight", "proj.5.bias", "entity_embedding.weight",
              "transformer_encoder.layers.0.self_attn.in_proj_weight", "transformer_encoder.layers.0.linear1.weight",
              "transformer_encoder.layers.0.norm2.bias", "classifier.0.weight", "classifier.1.weight",
              "classifier.4.bias"):
        assert k in names
    # the released Qwen3-8B probes hold ~8.1M parameters (16.3 MB in bf16)
    assert 8.0e6 < n_params(ReProbeHead()) < 8.3e6


def test_visible_sees_future_and_masked_does_not():
    h = _head()
    feats = torch.randn(1, 6, 16)
    valid = torch.ones(1, 6, dtype=torch.bool)
    claims = torch.tensor([[0, 1, 1, 0, 0, 0]])
    t = torch.zeros(1, dtype=torch.long)
    f2 = feats.clone()
    f2[0, 4:] += 5.0  # change only tokens after the step
    with torch.no_grad():
        vis = h(feats, valid, claims, t, "visible"), h(f2, valid, claims, t, "visible")
        msk = h(feats, valid, claims, t, "masked"), h(f2, valid, claims, t, "masked")
    assert not torch.allclose(vis[0], vis[1])
    assert torch.allclose(msk[0], msk[1], atol=1e-6)


def test_padding_and_batching_do_not_change_scores():
    h = _head()
    a, b = torch.randn(4, 16), torch.randn(6, 16)
    feats = torch.zeros(2, 6, 16)
    feats[0, :4], feats[1] = a, b
    valid = torch.zeros(2, 6, dtype=torch.bool)
    valid[0, :4], valid[1] = True, True
    claims = torch.tensor([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0]])
    tr = torch.tensor([0, 0, 1])
    with torch.no_grad():
        for fut in ("visible", "masked"):
            both = h(feats, valid, claims, tr, fut)
            one = h(a[None], torch.ones(1, 4, dtype=torch.bool), claims[:2, :4], torch.zeros(2, dtype=torch.long), fut)
            assert torch.allclose(both[:2], one, atol=1e-5)


def test_reprobe_init_matches_release_rule():
    h = ReProbeHead(feature_dim=16, head_dim=8, n_heads=2).reprobe_init_()
    assert torch.all(h.proj[0].bias == 0)
    ln = h.proj[1].weight
    assert torch.all((ln >= 0) & (ln <= 1)) and ln.std() > 0  # uniform(0, 1), not ones


def test_step_feature_index_shifts_by_one():
    assert step_feature_index(10, [0, 1, 2]) == [9, 10, 11]


def _char_decode(ids):
    return "".join(chr(i) for i in ids)


def test_split_to_steps_lines_and_skips():
    text = "Reasoning Steps:\n- Step 1: a\n\n- Step 2: b\n<Answer>: 3"
    toks = [ord(c) for c in text] + [0]  # an end token, as in a sampled reply
    steps = split_to_steps(text, toks, _char_decode)
    assert [s["sentence"] for s in steps] == ["- Step 1: a\n", "- Step 2: b\n", "<Answer>: 3"]
    for s in steps:
        assert _char_decode([toks[i] for i in s["aligned_token_ids"]]).strip() == s["sentence"].strip()
    pre, body = step_item(text, steps[1])
    assert body == "- Step 2: b" and (pre + "\n" + body) == text[:steps[1]["end"] - 1]


def test_split_to_steps_rejects_mismatched_tokens():
    assert split_to_steps("abc", [ord("x")], _char_decode) == []
