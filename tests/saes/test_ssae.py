"""Unit tests for the SSAE model.

Uses stub/mock components so no real model weights or internet access needed.
The mock backbone returns fixed tensors of the correct shapes.
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from src.saes.ssae import SSAE, _sample, _append_token


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HIDDEN = 8      # tiny hidden size for speed
VOCAB  = 32
SEQ    = 6


def _make_mock_encoder(hidden=HIDDEN, vocab=VOCAB):
    """Return a mock that behaves like HuggingFace AutoModel."""
    m = MagicMock()
    m.config.hidden_size = hidden

    def forward(input_ids, attention_mask=None, **kw):
        b, s = input_ids.shape
        out = MagicMock()
        out.last_hidden_state = torch.randn(b, s, hidden)
        return out

    m.side_effect = forward
    m.__call__ = forward
    m.resize_token_embeddings = MagicMock()
    m.requires_grad_ = MagicMock(return_value=m)
    return m


def _make_mock_decoder(hidden=HIDDEN, vocab=VOCAB):
    """Return a mock that behaves like HuggingFace AutoModelForCausalLM."""
    m = MagicMock()
    m.config.hidden_size = hidden
    m.config.model_type = "qwen2"

    def forward(inputs_embeds=None, input_ids=None, attention_mask=None, **kw):
        b = inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
        s = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
        out = MagicMock()
        out.logits = torch.randn(b, s, vocab)
        return out

    m.side_effect = forward
    m.__call__ = forward
    m.resize_token_embeddings = MagicMock()
    m.requires_grad_ = MagicMock(return_value=m)

    embed = nn.Embedding(vocab, hidden)
    m.get_input_embeddings = MagicMock(return_value=embed)
    return m


def _make_tokenizer(vocab=VOCAB):
    tok = MagicMock()
    tok.__len__ = MagicMock(return_value=vocab)
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.encode = MagicMock(return_value=[10])  # "\n" → token 10
    return tok


@pytest.fixture
def ssae(monkeypatch):
    """SSAE with mocked backbones — no downloads, no GPU."""
    enc = _make_mock_encoder()
    dec = _make_mock_decoder()
    tok = _make_tokenizer()

    with patch("src.saes.ssae.AutoModel") as mock_auto_model, \
         patch("src.saes.ssae.AutoModelForCausalLM") as mock_auto_causal:
        mock_auto_model.from_pretrained.return_value = enc
        mock_auto_causal.from_pretrained.return_value = dec

        model = SSAE(
            tokenizer=tok,
            sparsity_factor=2,
            encoder_model_id="mock-model",
            decoder_model_id="mock-model",
            phase=1,
        )

    # Manually wire the real components so tests exercise real logic
    model.encoder = enc
    model.hints_encoder = _make_mock_encoder()
    model.decoder = dec

    return model


# ---------------------------------------------------------------------------
# SparseAutoencoder unit tests (via SSAE.autoencoder)
# ---------------------------------------------------------------------------

def test_autoencoder_shape(ssae):
    x = torch.randn(2, 1, HIDDEN)
    out = ssae.autoencoder(x)
    assert out.shape == (2, 1, HIDDEN * 2), "n_latents = hidden * sparsity_factor"


def test_autoencoder_nonnegative(ssae):
    x = torch.randn(4, 1, HIDDEN)
    out = ssae.autoencoder(x)
    assert (out >= 0).all()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def test_encode_output_shape(ssae):
    input_ids = torch.randint(0, VOCAB, (2, SEQ))
    attn_mask = torch.ones(2, SEQ, dtype=torch.long)

    ssae.encoder.side_effect = lambda ids, attention_mask=None, **kw: SimpleNamespace(
        last_hidden_state=torch.randn(ids.shape[0], ids.shape[1], HIDDEN)
    )

    latents = ssae.encode(input_ids, attn_mask)
    assert latents.shape == (2, 1, HIDDEN * 2)


def test_encode_l2_normalized(ssae):
    input_ids = torch.randint(0, VOCAB, (3, SEQ))
    attn_mask = torch.ones(3, SEQ, dtype=torch.long)

    ssae.encoder.side_effect = lambda ids, attention_mask=None, **kw: SimpleNamespace(
        last_hidden_state=torch.randn(ids.shape[0], ids.shape[1], HIDDEN)
    )

    latents = ssae.encode(input_ids, attn_mask)
    norms = latents.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_get_sparse_vector_shape(ssae):
    input_ids = torch.randint(0, VOCAB, (2, SEQ))
    attn_mask = torch.ones(2, SEQ, dtype=torch.long)

    ssae.encoder.side_effect = lambda ids, attention_mask=None, **kw: SimpleNamespace(
        last_hidden_state=torch.randn(ids.shape[0], ids.shape[1], HIDDEN)
    )

    vec = ssae.get_sparse_vector(input_ids, attn_mask)
    assert vec.shape == (2, HIDDEN * 2), "Should be (batch, n_latents) — no extra dim"


def test_sparse_vector_nonnegative(ssae):
    input_ids = torch.randint(0, VOCAB, (2, SEQ))
    attn_mask = torch.ones(2, SEQ, dtype=torch.long)

    ssae.encoder.side_effect = lambda ids, attention_mask=None, **kw: SimpleNamespace(
        last_hidden_state=torch.randn(ids.shape[0], ids.shape[1], HIDDEN)
    )

    vec = ssae.get_sparse_vector(input_ids, attn_mask)
    assert (vec >= 0).all()


# ---------------------------------------------------------------------------
# Last / avg token extraction helpers
# ---------------------------------------------------------------------------

def test_get_last_token_embeddings_picks_last_real():
    """Should pick the last non-padded position."""
    ssae_bare = MagicMock(spec=SSAE)
    ssae_bare._get_last_token_embeddings = SSAE._get_last_token_embeddings.__get__(ssae_bare)

    hidden = torch.zeros(2, 5, 4)
    hidden[0, 2, :] = 1.0   # last real token at index 2 for seq of len 3
    hidden[1, 4, :] = 2.0   # last real token at index 4 for seq of len 5

    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.long)

    out = ssae_bare._get_last_token_embeddings(hidden, mask)
    assert out.shape == (2, 1, 4)
    assert torch.allclose(out[0, 0, :], torch.ones(4))
    assert torch.allclose(out[1, 0, :], torch.full((4,), 2.0))


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def test_sample_greedy():
    logits = torch.zeros(10)
    logits[5] = 10.0
    token = _sample(logits, temperature=0, top_k=0, top_p=0.0)
    assert token.item() == 5


def test_sample_returns_valid_token():
    logits = torch.randn(32)
    token = _sample(logits, temperature=1.0, top_k=5, top_p=0.0)
    assert 0 <= token.item() < 32


# ---------------------------------------------------------------------------
# _append_token helper
# ---------------------------------------------------------------------------

def test_append_token_inserts_correctly():
    # insert_pos is the DESTINATION index where the new token will land.
    # src[0..ins) copies unchanged; src[ins..) shifts right by one.
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    new_tokens = torch.tensor([7, 8])
    insert_pos = torch.tensor([3, 2])  # token 7 at dst[3], token 8 at dst[2]
    unfinished = torch.ones(2, dtype=torch.long)

    new_ids, new_mask = _append_token(ids, mask, new_tokens, insert_pos, unfinished, pad_id=0)

    assert new_ids.shape == (2, 5)
    # Sequence 0: [1, 2, 3, 7, 0]  (src[0..3] unchanged, 7 at index 3, src[3]=0 shifts to 4)
    assert new_ids[0].tolist() == [1, 2, 3, 7, 0]
    # Sequence 1: [4, 5, 8, 0, 0]  (src[0..2] unchanged, 8 at index 2, src[2..] shifts right)
    assert new_ids[1].tolist() == [4, 5, 8, 0, 0]
