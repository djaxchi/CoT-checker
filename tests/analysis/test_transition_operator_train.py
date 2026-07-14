"""Tests for src/analysis/transition_operator_train.py.

The load-bearing tests are the UpperDecoder equivalence checks: the fast
cache-based decode used in Stage 2 training must produce the same boundary logits
as the hook-based patched forward (tested ground truth from Stage 0), for the
identity patch, a perturbed patch, and a right-padded batch of unequal lengths.
"""

from __future__ import annotations

import pytest
import torch

from src.analysis.transition_operator import forward_with_boundary_patch
from src.analysis.transition_operator_train import (
    BeliefHead,
    ContrastiveProjections,
    TransitionEncoder,
    UpperDecoder,
    effect_close_mask,
    format_features,
    info_nce,
    kl_to_actual,
    percentile_of,
    rms,
)


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=331, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    return Qwen2ForCausalLM(cfg).eval()


LAYER_LO = 2  # consumes hidden_states[2], runs blocks 2..3


def _hook_boundary_logits(model, ids, h_patch):
    b = ids.shape[1] - 1
    logits = forward_with_boundary_patch(model, ids, hs_index=LAYER_LO,
                                         boundary_pos=b, patched_state=h_patch)
    return logits[:, b].float()


def test_upper_decoder_matches_hook_identity_and_perturbed(tiny):
    torch.manual_seed(1)
    ids = torch.randint(0, 331, (1, 11))
    with torch.no_grad():
        out = tiny(input_ids=ids, output_hidden_states=True)
    own = out.hidden_states[LAYER_LO][:, -1, :]
    ud = UpperDecoder(tiny, LAYER_LO)
    cache = ud.prefill(ids[:, :-1], torch.ones(1, 10, dtype=torch.long))
    base_len = ud.cache_len(cache)
    for delta in (0.0, 0.7):
        patched = own + delta
        want = _hook_boundary_logits(tiny, ids, patched)
        got = ud.decode_boundary(cache, patched, torch.tensor([10]))
        ud.crop(cache, base_len)
        assert torch.allclose(got, want, atol=1e-4), f"delta={delta}"
    # identity also matches the plain forward
    assert torch.allclose(_hook_boundary_logits(tiny, ids, own),
                          out.logits[:, -1].float(), atol=1e-4)


def test_upper_decoder_batched_right_padded(tiny):
    torch.manual_seed(2)
    ids_a = torch.randint(0, 331, (1, 12))
    ids_b = torch.randint(0, 331, (1, 8))
    # right-pad prefill inputs (without their boundary tokens: lengths 11 and 7)
    pre = torch.zeros(2, 11, dtype=torch.long)
    mask = torch.zeros(2, 11, dtype=torch.long)
    pre[0, :11], mask[0, :11] = ids_a[0, :11], 1
    pre[1, :7], mask[1, :7] = ids_b[0, :7], 1
    with torch.no_grad():
        oa = tiny(input_ids=ids_a, output_hidden_states=True)
        ob = tiny(input_ids=ids_b, output_hidden_states=True)
    patch = torch.stack([oa.hidden_states[LAYER_LO][0, -1] + 0.3,
                         ob.hidden_states[LAYER_LO][0, -1] - 0.2])
    ud = UpperDecoder(tiny, LAYER_LO)
    cache = ud.prefill(pre, mask)
    got = ud.decode_boundary(cache, patch, torch.tensor([11, 7]))
    want_a = _hook_boundary_logits(tiny, ids_a, patch[0:1])
    want_b = _hook_boundary_logits(tiny, ids_b, patch[1:2])
    assert torch.allclose(got[0:1], want_a, atol=1e-4)
    assert torch.allclose(got[1:2], want_b, atol=1e-4)


def test_upper_decoder_gradients_flow_to_patch(tiny):
    torch.manual_seed(3)
    ids = torch.randint(0, 331, (1, 9))
    with torch.no_grad():
        out = tiny(input_ids=ids, output_hidden_states=True)
    ud = UpperDecoder(tiny, LAYER_LO)
    cache = ud.prefill(ids[:, :-1], torch.ones(1, 8, dtype=torch.long))
    patched = (out.hidden_states[LAYER_LO][:, -1, :] + 0.1).requires_grad_(True)
    logits = ud.decode_boundary(cache, patched, torch.tensor([8]))
    kl_to_actual(out.logits[:, -1], logits).backward()
    assert patched.grad is not None and patched.grad.abs().sum() > 0


def test_kl_to_actual_zero_on_identical():
    x = torch.randn(3, 50)
    assert kl_to_actual(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_effect_close_mask_and_info_nce():
    e = torch.tensor([[1.0, 0.0], [0.99, 0.05], [-1.0, 0.0], [0.0, 1.0]])
    close = effect_close_mask(e, threshold=0.9)
    assert close[0, 1] and close[1, 0]          # near-duplicates masked
    assert not close[0, 2] and not close[0, 3]  # dissimilar kept
    assert not close.diagonal().any()
    za = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    loss = info_nce(za, za, close)
    assert torch.isfinite(loss)


def test_encoder_and_heads_shapes():
    enc = TransitionEncoder(hidden=32, d_model=16, n_heads=4, n_layers=1,
                            d_z=8, max_steps=6)
    z = enc(torch.randn(3, 32), torch.randn(3, 6, 32),
            torch.tensor([[1, 1, 1, 0, 0, 0]] * 3, dtype=torch.bool))
    assert z.shape == (3, 8)
    assert BeliefHead(8)(z).shape == (3, 8)
    za, ea = ContrastiveProjections(8, d_effect=10)(z, torch.randn(3, 10))
    assert za.shape == ea.shape == (3, 64)
    assert torch.allclose(za.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_format_features():
    f = format_features("So we get $x = 5$", n_tokens=7)
    assert len(f) == 9
    assert f[0] == 7.0
    assert f[4] == 1.0  # final char class dollar
    assert f[6] == 1.0  # display math
    assert f[8] == 1.0  # discourse opener "So"


def test_percentile_and_rms():
    ref = torch.arange(100.0)
    p = percentile_of(torch.tensor([0.0, 50.0, 99.5]), ref)
    assert p[0] <= 1 and 49 <= p[1] <= 52 and p[2] >= 99
    assert rms(torch.ones(2, 16)).allclose(torch.ones(2))
