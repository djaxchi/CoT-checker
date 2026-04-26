"""Unit tests for Future-SSAE training utilities.

All tests run without GPU and without model downloads.
Tests cover:
  - GSM8KFutureStepDataset / _load_future_step_pairs
  - GSM8KFutureCollateFn (all four prediction_context_modes)
  - SSAE.decode_from_latents (shape + shared-latent contract)
  - masked_nll_loss
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.gsm8k_dataset import (
    GSM8KFutureCollateFn,
    GSM8KFutureStepDataset,
    _load_future_step_pairs,
)
from scripts.train_future_ssae_gsm8k import (
    masked_nll_loss,
    per_item_first_token_nll,
    per_item_masked_nll,
    PredNLLScaler,
    pred_first_token_nll_loss,
    pearson_r,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_JSONL = [
    {
        "question": "Alice has 3 apples.",
        "answer": "She buys 2 more.\nNow she has 5.\n#### 5",
    },
    {
        "question": "Bob has 10 balls.",
        "answer": "He loses 3.\nHe gives away 2.\nHe has 5 left.\n#### 5",
    },
]


@pytest.fixture
def jsonl_path(tmp_path):
    p = tmp_path / "data.jsonl"
    with open(p, "w") as f:
        for item in SAMPLE_JSONL:
            f.write(json.dumps(item) + "\n")
    return p


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.sep_token_id = 1
    tok.eos_token_id = 2
    # encode returns a small list of ints proportional to text length
    tok.encode.side_effect = lambda text, **kw: list(range(1, min(len(text) // 5 + 2, 20)))
    return tok


# ---------------------------------------------------------------------------
# _load_future_step_pairs
# ---------------------------------------------------------------------------

class TestLoadFutureStepPairs:
    def test_last_step_skipped(self, jsonl_path):
        pairs = _load_future_step_pairs(jsonl_path)
        # Sample 0 has 3 steps (splits: "She buys 2 more.", "Now she has 5.", "#### 5")
        # but the last combo merges last two lines, so effectively 2 steps → 1 pair
        # Sample 1 has 4 lines → merges last two → 3 steps → 2 pairs
        # Total: 1 + 2 = 3 pairs
        assert len(pairs) == 3

    def test_each_pair_has_required_keys(self, jsonl_path):
        pairs = _load_future_step_pairs(jsonl_path)
        for p in pairs:
            assert "question" in p
            assert "context" in p
            assert "current_step" in p
            assert "next_step" in p
            assert "prev1" in p
            assert "prev2" in p

    def test_prev1_none_for_first_step(self, jsonl_path):
        pairs = _load_future_step_pairs(jsonl_path)
        # First pair of each problem has i=0 → prev1 should be None
        first_pairs = [p for p in pairs if p["prev1"] is None]
        assert len(first_pairs) == 2  # one per problem

    def test_prev2_none_for_first_two_steps(self, jsonl_path):
        pairs = _load_future_step_pairs(jsonl_path)
        no_prev2 = [p for p in pairs if p["prev2"] is None]
        # i=0 and i=1 have no prev2 → 2 + 1 = 3 of our 3 pairs have no prev2
        # (problem 0: only pair at i=0; problem 1: pairs at i=0 and i=1)
        assert len(no_prev2) == 3

    def test_next_step_not_equal_current_step(self, jsonl_path):
        pairs = _load_future_step_pairs(jsonl_path)
        for p in pairs:
            assert p["current_step"] != p["next_step"]

    def test_question_preserved(self, jsonl_path):
        pairs = _load_future_step_pairs(jsonl_path)
        assert pairs[0]["question"] == SAMPLE_JSONL[0]["question"]


# ---------------------------------------------------------------------------
# GSM8KFutureStepDataset
# ---------------------------------------------------------------------------

class TestGSM8KFutureStepDataset:
    def test_len(self, jsonl_path, mock_tokenizer):
        ds = GSM8KFutureStepDataset(jsonl_path, mock_tokenizer)
        assert len(ds) == 3

    def test_getitem_has_required_tensors(self, jsonl_path, mock_tokenizer):
        ds = GSM8KFutureStepDataset(jsonl_path, mock_tokenizer)
        item = ds[0]
        for key in ("question_ids", "context_ids", "current_step_ids", "next_step_ids"):
            assert key in item
            assert isinstance(item[key], torch.Tensor)
            assert item[key].dtype == torch.long

    def test_prev1_none_when_first_step(self, jsonl_path, mock_tokenizer):
        ds = GSM8KFutureStepDataset(jsonl_path, mock_tokenizer)
        # idx 0 is i=0 (first step of problem 0) → prev1_ids must be None
        item = ds[0]
        assert item["prev1_ids"] is None

    def test_prev1_tensor_when_second_step(self, jsonl_path, mock_tokenizer):
        ds = GSM8KFutureStepDataset(jsonl_path, mock_tokenizer)
        # Problem 1 has i=0,1 at indices 1,2. ds[2] is i=1 of problem 1 → prev1 is not None
        item = ds[2]
        assert item["prev1_ids"] is not None
        assert isinstance(item["prev1_ids"], torch.Tensor)


# ---------------------------------------------------------------------------
# GSM8KFutureCollateFn
# ---------------------------------------------------------------------------

def _make_item(q_len=4, ctx_len=6, cur_len=5, nxt_len=5, p1_len=None, p2_len=None,
               current_step_idx=0):
    def ids(n):
        return torch.randint(3, 100, (n,), dtype=torch.long)
    return {
        "question_ids": ids(q_len),
        "context_ids": ids(ctx_len),
        "current_step_ids": ids(cur_len),
        "next_step_ids": ids(nxt_len),
        "prev1_ids": ids(p1_len) if p1_len else None,
        "prev2_ids": ids(p2_len) if p2_len else None,
        "current_step_idx": current_step_idx,
    }


class TestGSM8KFutureCollateFn:
    EOS, PAD, SEP, SPACE = 2, 2, 1, 7

    def _collate(self, mode="q_current", batch=None):
        fn = GSM8KFutureCollateFn(
            eos_token_id=self.EOS,
            pad_token_id=self.PAD,
            sep_token_id=self.SEP,
            space_token_id=self.SPACE,
            prediction_context_mode=mode,
        )
        if batch is None:
            batch = [_make_item(), _make_item(q_len=3, ctx_len=8, cur_len=4, nxt_len=6)]
        return fn(batch)

    def test_output_keys(self):
        out = self._collate()
        expected = {
            "recon_input_ids", "recon_attention_mask", "recon_loss_mask",
            "recon_hints_sep_ids", "recon_hints_sep_attn",
            "recon_sep_pos", "recon_val_len",
            "pred_input_ids", "pred_attention_mask", "pred_loss_mask",
        }
        assert expected.issubset(out.keys())

    def test_recon_batch_shape_consistent(self):
        out = self._collate()
        B = out["recon_input_ids"].shape[0]
        T = out["recon_input_ids"].shape[1]
        assert out["recon_attention_mask"].shape == (B, T)
        assert out["recon_loss_mask"].shape == (B, T)

    def test_pred_batch_shape_consistent(self):
        out = self._collate()
        B = out["pred_input_ids"].shape[0]
        T = out["pred_input_ids"].shape[1]
        assert out["pred_attention_mask"].shape == (B, T)
        assert out["pred_loss_mask"].shape == (B, T)

    def test_recon_loss_mask_zeros_context(self):
        # Context tokens must have loss_mask = 0
        batch = [_make_item(q_len=4, ctx_len=6, cur_len=5, nxt_len=5)]
        out = self._collate(batch=batch)
        sep_pos = out["recon_sep_pos"][0]
        assert out["recon_loss_mask"][0, :sep_pos].sum() == 0

    def test_recon_loss_mask_ones_on_step_region(self):
        batch = [_make_item(q_len=4, ctx_len=6, cur_len=5, nxt_len=5)]
        out = self._collate(batch=batch)
        sep_pos = out["recon_sep_pos"][0]
        val_len = out["recon_val_len"][0]
        assert out["recon_loss_mask"][0, sep_pos:val_len].sum() == val_len - sep_pos

    def test_pred_loss_mask_zeros_prefix(self):
        # Prefix tokens (question + SPACE + current_step + SEP) must have pred_loss_mask = 0
        batch = [_make_item(q_len=4, ctx_len=6, cur_len=5, nxt_len=5)]
        out = self._collate(mode="q_current", batch=batch)
        # prefix = q(4) + space(1) + cur(5) + sep(1) = 11
        prefix_len = 4 + 1 + 5 + 1
        assert out["pred_loss_mask"][0, :prefix_len].sum() == 0

    def test_pred_loss_mask_ones_on_next_step(self):
        batch = [_make_item(q_len=4, ctx_len=6, cur_len=5, nxt_len=5)]
        out = self._collate(mode="q_current", batch=batch)
        prefix_len = 4 + 1 + 5 + 1
        # next_step(5) + eos(1) = 6 tokens with loss_mask = 1
        assert out["pred_loss_mask"][0, prefix_len:prefix_len + 6].sum() == 6

    def test_all_modes_produce_valid_output(self):
        for mode in GSM8KFutureCollateFn.MODES:
            batch = [
                _make_item(p1_len=4, p2_len=3),
                _make_item(p1_len=None, p2_len=None),
            ]
            out = self._collate(mode=mode, batch=batch)
            assert out["pred_input_ids"].shape[0] == 2

    def test_q_prev1_mode_with_missing_prev1(self):
        # When prev1 is None, should fall back gracefully (no exception)
        batch = [_make_item(p1_len=None)]
        fn = GSM8KFutureCollateFn(self.EOS, self.PAD, self.SEP, self.SPACE, "q_prev1_current")
        out = fn(batch)
        assert out["pred_input_ids"].shape[0] == 1

    def test_q_prev2_mode_wider_than_q_current(self):
        # full prev context → pred sequence should be longer than q_current
        batch = [_make_item(q_len=4, ctx_len=6, cur_len=5, nxt_len=5, p1_len=5, p2_len=5)]
        out_q = self._collate(mode="q_current", batch=batch)
        out_p2 = self._collate(mode="q_prev2_current", batch=batch)
        assert out_p2["pred_input_ids"].shape[1] > out_q["pred_input_ids"].shape[1]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            GSM8KFutureCollateFn(self.EOS, self.PAD, self.SEP, self.SPACE, "bad_mode")

    def test_hints_sep_ids_shape(self):
        out = self._collate()
        B = out["recon_input_ids"].shape[0]
        assert out["recon_hints_sep_ids"].shape[0] == B
        assert out["recon_hints_sep_attn"].shape == out["recon_hints_sep_ids"].shape


# ---------------------------------------------------------------------------
# SSAE.decode_from_latents
# ---------------------------------------------------------------------------

class _TinyDecoder(nn.Module):
    """Minimal causal LM decoder stub for testing decode_from_latents."""
    def __init__(self, d: int, vocab: int):
        super().__init__()
        self._emb = nn.Embedding(vocab, d)
        self._proj = nn.Linear(d, vocab)
        self.config = MagicMock()
        self.config.hidden_size = d

    def get_input_embeddings(self):
        return self._emb

    def forward(self, inputs_embeds=None, attention_mask=None, **kw):
        out = self._proj(inputs_embeds)
        return MagicMock(logits=out)


class TestDecodeFromLatents:
    """Test SSAE.decode_from_latents using a patched model."""

    def _make_ssae(self, d=16, vocab=32, sparsity_factor=1):
        """Build a minimal SSAE with mocked backbone models."""
        from src.saes.ssae import SSAE
        from src.saes.autoencoder import SparseAutoencoder

        with patch("src.saes.ssae.AutoModel") as mock_enc_cls, \
             patch("src.saes.ssae.AutoModelForCausalLM") as mock_dec_cls:

            enc = MagicMock()
            enc.config.hidden_size = d
            enc.config._name_or_path = "mock"
            enc.resize_token_embeddings = MagicMock()
            mock_enc_cls.from_pretrained.return_value = enc

            dec = _TinyDecoder(d, vocab)
            dec.config = MagicMock()
            dec.config.hidden_size = d
            dec.config._name_or_path = "mock"
            dec.resize_token_embeddings = MagicMock()
            mock_dec_cls.from_pretrained.return_value = dec

            tok = MagicMock()
            tok.__len__ = MagicMock(return_value=vocab)
            tok.sep_token_id = 1

            model = SSAE(tokenizer=tok, sparsity_factor=sparsity_factor, phase=1)

        model.n_inputs = d
        model.n_latents = d * sparsity_factor
        model.autoencoder = SparseAutoencoder(d, d * sparsity_factor, sparsity_factor)
        model.projection_mlp = nn.Sequential(
            nn.Linear(d * sparsity_factor, d * sparsity_factor),
            nn.ReLU(),
            nn.Linear(d * sparsity_factor, d * sparsity_factor),
        )
        return model

    def test_output_shape(self):
        d, vocab, B, T = 16, 32, 2, 10
        model = self._make_ssae(d=d, vocab=vocab)
        latents = torch.randn(B, 1, d)
        input_ids = torch.randint(0, vocab, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        logits = model.decode_from_latents(latents, input_ids, attention_mask)
        # _decode removes one token and prepends sparsity_factor tokens
        # → output length = T + sparsity_factor - 1
        assert logits.shape == (B, T + model.sparsity_factor - 1, vocab)

    def test_same_latents_different_context_gives_different_logits(self):
        d, vocab, B, T = 16, 32, 1, 8
        model = self._make_ssae(d=d, vocab=vocab)
        latents = torch.randn(B, 1, d)
        ids_a = torch.randint(0, vocab, (B, T))
        ids_b = torch.randint(0, vocab, (B, T))
        attn = torch.ones(B, T, dtype=torch.long)
        logits_a = model.decode_from_latents(latents, ids_a, attn)
        logits_b = model.decode_from_latents(latents, ids_b, attn)
        # Different contexts → different logits (essentially always true)
        assert not torch.allclose(logits_a, logits_b)

    def test_decode_from_latents_does_not_rerun_encoder(self):
        d, vocab, B, T = 16, 32, 2, 8
        model = self._make_ssae(d=d, vocab=vocab)
        latents = torch.randn(B, 1, d)
        input_ids = torch.randint(0, vocab, (B, T))
        attn = torch.ones(B, T, dtype=torch.long)
        # encoder.forward should NOT be called
        model.encoder.forward = MagicMock(side_effect=RuntimeError("encoder called!"))
        # Should not raise
        model.decode_from_latents(latents, input_ids, attn)

    def test_projection_mlp_called_once(self):
        d, vocab, B, T = 16, 32, 2, 8
        model = self._make_ssae(d=d, vocab=vocab)
        latents = torch.randn(B, 1, d)
        input_ids = torch.randint(0, vocab, (B, T))
        attn = torch.ones(B, T, dtype=torch.long)
        original_mlp = model.projection_mlp
        call_count = [0]
        original_forward = original_mlp.forward
        def counting_forward(x):
            call_count[0] += 1
            return original_forward(x)
        original_mlp.forward = counting_forward
        model.decode_from_latents(latents, input_ids, attn)
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# masked_nll_loss
# ---------------------------------------------------------------------------

class TestMaskedNllLoss:
    def test_correct_prediction_gives_low_loss(self):
        B, T, V = 2, 6, 10
        logits = torch.zeros(B, T, V)
        logits[:, :, 0] = 100.0
        input_ids = torch.zeros(B, T, dtype=torch.long)
        loss_mask = torch.ones(B, T, dtype=torch.long)
        loss = masked_nll_loss(logits, input_ids, loss_mask)
        assert loss.item() < 0.01

    def test_wrong_prediction_gives_high_loss(self):
        B, T, V = 1, 4, 10
        logits = torch.zeros(B, T, V)
        logits[:, :, 9] = 100.0
        input_ids = torch.zeros(B, T, dtype=torch.long)
        loss_mask = torch.ones(B, T, dtype=torch.long)
        loss = masked_nll_loss(logits, input_ids, loss_mask)
        assert loss.item() > 5.0

    def test_ignored_positions_do_not_affect_loss(self):
        B, T, V = 2, 6, 10
        logits = torch.zeros(B, T, V)
        logits[:, :3, 9] = 100.0  # wrong, but masked out
        logits[:, 3:, 0] = 100.0  # correct, in loss region
        input_ids = torch.zeros(B, T, dtype=torch.long)
        loss_mask = torch.zeros(B, T, dtype=torch.long)
        loss_mask[:, 3:] = 1
        loss = masked_nll_loss(logits, input_ids, loss_mask)
        assert loss.item() < 0.01

    def test_returns_scalar(self):
        logits = torch.randn(3, 8, 16)
        input_ids = torch.randint(0, 16, (3, 8))
        loss_mask = torch.ones(3, 8, dtype=torch.long)
        assert masked_nll_loss(logits, input_ids, loss_mask).shape == torch.Size([])


# ---------------------------------------------------------------------------
# pred_prefix_len in collate output
# ---------------------------------------------------------------------------

class TestPredPrefixLen:
    EOS, PAD, SEP, SPACE = 2, 2, 1, 7

    def test_key_present(self):
        fn = GSM8KFutureCollateFn(self.EOS, self.PAD, self.SEP, self.SPACE)
        out = fn([_make_item()])
        assert "pred_prefix_len" in out

    def test_length_equals_batch_size(self):
        fn = GSM8KFutureCollateFn(self.EOS, self.PAD, self.SEP, self.SPACE)
        batch = [_make_item(), _make_item()]
        out = fn(batch)
        assert len(out["pred_prefix_len"]) == 2

    def test_q_current_prefix_len_value(self):
        # q_current: prefix = [q(4) | space(1) | cur(5) | SEP(1)] = 11
        fn = GSM8KFutureCollateFn(self.EOS, self.PAD, self.SEP, self.SPACE, "q_current")
        out = fn([_make_item(q_len=4, cur_len=5)])
        assert out["pred_prefix_len"][0] == 4 + 1 + 5 + 1

    def test_full_context_prefix_len_value(self):
        # full_context_current: prefix = [ctx(6) | space(1) | cur(5) | SEP(1)] = 13
        fn = GSM8KFutureCollateFn(self.EOS, self.PAD, self.SEP, self.SPACE, "full_context_current")
        out = fn([_make_item(ctx_len=6, cur_len=5)])
        assert out["pred_prefix_len"][0] == 6 + 1 + 5 + 1

    def test_prefix_len_points_to_next_step_token(self):
        # The token at pred_prefix_len should be the first next_step token
        fn = GSM8KFutureCollateFn(self.EOS, self.PAD, self.SEP, self.SPACE, "q_current")
        item = _make_item(q_len=4, cur_len=5, nxt_len=5)
        out = fn([item])
        plen = out["pred_prefix_len"][0]
        # pred_input_ids[0, plen] should equal next_step_ids[0]
        assert out["pred_input_ids"][0, plen].item() == item["next_step_ids"][0].item()

    def test_space_token_present_at_segment_boundary(self):
        # In q_current mode the prefix is [q | SPACE | cur | SEP].  Verify the
        # token at position len(q) is the space token (no "cookies.She" artifact).
        fn = GSM8KFutureCollateFn(self.EOS, self.PAD, self.SEP, self.SPACE, "q_current")
        item = _make_item(q_len=4, cur_len=5, nxt_len=5)
        out = fn([item])
        assert out["pred_input_ids"][0, 4].item() == self.SPACE
        # And current_step starts right after the space.
        assert out["pred_input_ids"][0, 5].item() == item["current_step_ids"][0].item()

    def test_space_inserted_between_all_prev_segments(self):
        # q_prev2_current: [q | SPACE | prev2 | SPACE | prev1 | SPACE | cur | SEP]
        fn = GSM8KFutureCollateFn(self.EOS, self.PAD, self.SEP, self.SPACE, "q_prev2_current")
        item = _make_item(q_len=3, cur_len=2, nxt_len=4, p1_len=2, p2_len=2)
        out = fn([item])
        # Spaces expected at positions: 3 (after q), 6 (after prev2), 9 (after prev1)
        assert out["pred_input_ids"][0, 3].item() == self.SPACE
        assert out["pred_input_ids"][0, 6].item() == self.SPACE
        assert out["pred_input_ids"][0, 9].item() == self.SPACE
        # Total prefix length: 3 + 1 + 2 + 1 + 2 + 1 + 2 + 1 = 13
        assert out["pred_prefix_len"][0] == 13


# ---------------------------------------------------------------------------
# PredNLLScaler
# ---------------------------------------------------------------------------

class TestPredNLLScaler:
    def test_initial_scale_is_first_observation(self):
        scaler = PredNLLScaler()
        scaler.step(5.0)
        assert scaler.current_scale == pytest.approx(5.0)

    def test_scale_before_first_step_defaults_to_one(self):
        scaler = PredNLLScaler()
        assert scaler.current_scale == pytest.approx(1.0)

    def test_ema_tracks_value(self):
        scaler = PredNLLScaler(decay=0.9)
        scaler.step(10.0)   # initialise
        scaler.step(1.0)    # pull down
        # After one EMA update from 10→1: 0.9*10 + 0.1*1 = 9.1
        assert scaler.current_scale == pytest.approx(9.1)

    def test_scale_does_not_go_below_min(self):
        scaler = PredNLLScaler(min_scale=1e-3)
        scaler.step(0.0)
        scaler.step(0.0)
        assert scaler.current_scale >= 1e-3

    def test_scale_decreases_toward_lower_values(self):
        scaler = PredNLLScaler(decay=0.5)
        scaler.step(100.0)
        for _ in range(50):
            scaler.step(1.0)
        assert scaler.current_scale < 10.0

    def test_read_before_write_contract(self):
        # current_scale returns last committed value, not current step's value
        scaler = PredNLLScaler()
        scaler.step(4.0)
        before = scaler.current_scale
        scaler.step(100.0)
        # before should reflect 4.0, not 100.0
        assert before == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# pred_first_token_nll_loss
# ---------------------------------------------------------------------------

class TestPredFirstTokenNllLoss:
    def test_returns_scalar(self):
        B, T, V = 3, 10, 32
        logits = torch.randn(B, T, V)
        ids = torch.randint(0, V, (B, T))
        prefix_lens = [3, 2, 4]
        loss = pred_first_token_nll_loss(logits, ids, prefix_lens)
        assert loss.shape == torch.Size([])

    def test_uses_only_first_token(self):
        B, T, V = 2, 10, 32
        # Make logits heavily predict token 7 at position prefix_len, wrong elsewhere
        logits = torch.full((B, T, V), -100.0)
        prefix_lens = [3, 4]
        targets = [5, 6]
        for i, (pl, tgt) in enumerate(zip(prefix_lens, targets)):
            logits[i, pl, tgt] = 100.0  # correct at first next-step position
        ids = torch.zeros(B, T, dtype=torch.long)
        for i, tgt in enumerate(targets):
            ids[i, prefix_lens[i]] = tgt
        loss = pred_first_token_nll_loss(logits, ids, prefix_lens)
        assert loss.item() < 0.01  # near-zero: correct prediction

    def test_wrong_prediction_gives_high_loss(self):
        B, T, V = 2, 10, 32
        logits = torch.full((B, T, V), -100.0)
        prefix_lens = [3, 4]
        for i, pl in enumerate(prefix_lens):
            logits[i, pl, 0] = 100.0  # predicts token 0
        ids = torch.full((B, T), 1, dtype=torch.long)  # targets are all token 1
        loss = pred_first_token_nll_loss(logits, ids, prefix_lens)
        assert loss.item() > 5.0


# ---------------------------------------------------------------------------
# pearson_r
# ---------------------------------------------------------------------------

class TestPearsonR:
    def test_perfect_positive_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        assert pearson_r(xs, xs) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [4.0, 3.0, 2.0, 1.0]
        assert pearson_r(xs, ys) == pytest.approx(-1.0)

    def test_uncorrelated(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0, 2.0, 2.0, 2.0]  # constant → correlation undefined / nan
        r = pearson_r(xs, ys)
        assert math.isnan(r) or abs(r) < 1e-3

    def test_returns_nan_for_single_point(self):
        r = pearson_r([1.0], [2.0])
        assert math.isnan(r)

    def test_range_is_minus_one_to_one(self):
        import random
        rng = random.Random(42)
        for _ in range(20):
            xs = [rng.gauss(0, 1) for _ in range(10)]
            ys = [rng.gauss(0, 1) for _ in range(10)]
            r = pearson_r(xs, ys)
            assert -1.0 - 1e-6 <= r <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# current_step_idx propagation
# ---------------------------------------------------------------------------

class TestCurrentStepIdxPropagation:
    EOS, PAD, SEP, SPACE = 2, 2, 1, 7

    def test_collate_emits_current_step_idx(self):
        fn = GSM8KFutureCollateFn(self.EOS, self.PAD, self.SEP, self.SPACE)
        batch = [
            _make_item(current_step_idx=0),
            _make_item(current_step_idx=2),
            _make_item(current_step_idx=5),
        ]
        out = fn(batch)
        assert out["current_step_idx"] == [0, 2, 5]

    def test_dataset_emits_current_step_idx(self, jsonl_path, mock_tokenizer):
        ds = GSM8KFutureStepDataset(jsonl_path, mock_tokenizer)
        for i in range(len(ds)):
            item = ds[i]
            assert "current_step_idx" in item
            assert isinstance(item["current_step_idx"], int)
            assert item["current_step_idx"] >= 0


# ---------------------------------------------------------------------------
# per_item_masked_nll  (used to stratify val_pred_nll by step idx)
# ---------------------------------------------------------------------------

class TestPerItemMaskedNll:
    def test_returns_per_item_tensor(self):
        B, T, V = 4, 8, 16
        logits = torch.randn(B, T, V)
        ids = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        out = per_item_masked_nll(logits, ids, mask)
        assert out.shape == (B,)

    def test_correct_prediction_per_item_low(self):
        B, T, V = 3, 4, 10
        logits = torch.full((B, T, V), -100.0)
        logits[:, :, 0] = 100.0
        ids = torch.zeros(B, T, dtype=torch.long)
        mask = torch.ones(B, T, dtype=torch.long)
        out = per_item_masked_nll(logits, ids, mask)
        assert (out < 0.01).all()

    def test_isolates_per_item_loss(self):
        # Item 0: correct on masked tokens. Item 1: wrong. Verify per-item NLLs differ.
        B, T, V = 2, 6, 10
        logits = torch.full((B, T, V), -100.0)
        ids = torch.zeros(B, T, dtype=torch.long)
        logits[0, :, 0] = 100.0          # item 0 correct
        logits[1, :, 5] = 100.0          # item 1 predicts token 5, target is 0
        mask = torch.ones(B, T, dtype=torch.long)
        out = per_item_masked_nll(logits, ids, mask)
        assert out[0].item() < 0.01
        assert out[1].item() > 5.0

    def test_zero_mask_item_safe(self):
        # Item with no masked tokens shouldn't NaN — denom is clamped at 1.0.
        B, T, V = 2, 4, 8
        logits = torch.randn(B, T, V)
        ids = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        mask[1] = 0  # item 1 has no targets
        out = per_item_masked_nll(logits, ids, mask)
        assert not torch.isnan(out).any()
        assert out[1].item() == pytest.approx(0.0)

    def test_mean_matches_global_when_uniform(self):
        # When every item has the same number of masked tokens, the mean of per-item
        # NLLs equals the global masked_nll_loss (within float tolerance).
        B, T, V = 3, 5, 12
        torch.manual_seed(0)
        logits = torch.randn(B, T, V)
        ids = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        per_item = per_item_masked_nll(logits, ids, mask).mean().item()
        global_ = masked_nll_loss(logits, ids, mask).item()
        assert per_item == pytest.approx(global_, rel=1e-5)


# ---------------------------------------------------------------------------
# per_item_first_token_nll  (used to stratify pred_first_token_nll by step idx)
# ---------------------------------------------------------------------------

class TestPerItemFirstTokenNll:
    def test_returns_per_item_tensor(self):
        B, T, V = 3, 8, 16
        logits = torch.randn(B, T, V)
        ids = torch.randint(0, V, (B, T))
        prefix_lens = [2, 3, 4]
        out = per_item_first_token_nll(logits, ids, prefix_lens)
        assert out.shape == (B,)

    def test_mean_matches_pred_first_token_nll_loss(self):
        B, T, V = 4, 10, 20
        torch.manual_seed(7)
        logits = torch.randn(B, T, V)
        ids = torch.randint(0, V, (B, T))
        prefix_lens = [2, 3, 4, 5]
        per_item = per_item_first_token_nll(logits, ids, prefix_lens).mean().item()
        scalar = pred_first_token_nll_loss(logits, ids, prefix_lens).item()
        assert per_item == pytest.approx(scalar, rel=1e-5)

    def test_per_item_isolates_correct_vs_wrong(self):
        B, T, V = 2, 6, 10
        prefix_lens = [2, 3]
        logits = torch.full((B, T, V), -100.0)
        ids = torch.zeros(B, T, dtype=torch.long)
        # item 0: correct at prefix position
        logits[0, prefix_lens[0], 0] = 100.0
        # item 1: predicts token 9, target is 0 → high loss
        logits[1, prefix_lens[1], 9] = 100.0
        out = per_item_first_token_nll(logits, ids, prefix_lens)
        assert out[0].item() < 0.01
        assert out[1].item() > 5.0
