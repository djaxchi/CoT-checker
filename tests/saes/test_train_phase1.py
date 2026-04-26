"""Unit tests for Phase 1 training utilities.

All tests run without GPU and without model downloads.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.train_ssae import (
    DWAController,
    apply_step_attention_mask,
    get_lr,
    phase1_nll_loss,
)


# ---------------------------------------------------------------------------
# DWAController
# ---------------------------------------------------------------------------

class TestDWAController:
    def test_weight_increases_when_above_target(self):
        dwa = DWAController(target=3.0, init_weight=1e-3, update_freq=5, alpha=0.01)
        # Feed sparsity > target for update_freq steps
        init_w = dwa.current_weight
        for _ in range(5):
            dwa.step(5.0)  # above target=3.0
        assert dwa.current_weight > init_w

    def test_weight_decreases_when_below_target(self):
        dwa = DWAController(target=3.0, init_weight=1e-3, update_freq=5, alpha=0.01)
        init_w = dwa.current_weight
        for _ in range(5):
            dwa.step(1.0)  # below target=3.0
        assert dwa.current_weight < init_w

    def test_weight_clamped_at_min(self):
        dwa = DWAController(
            target=3.0, init_weight=1e-6, update_freq=2, alpha=0.5, min_w=1e-6
        )
        for _ in range(10):
            dwa.step(0.0)  # always below target → would decrease
        assert dwa.current_weight >= 1e-6

    def test_weight_clamped_at_max(self):
        dwa = DWAController(
            target=3.0, init_weight=0.1, update_freq=2, alpha=0.5, max_w=0.1
        )
        for _ in range(10):
            dwa.step(100.0)  # always above target → would increase
        assert dwa.current_weight <= 0.1

    def test_no_update_before_interval(self):
        dwa = DWAController(target=3.0, init_weight=1e-3, update_freq=10)
        init_w = dwa.current_weight
        for _ in range(9):
            dwa.step(100.0)  # only 9 steps, update at 10
        assert dwa.current_weight == init_w

    def test_update_fires_exactly_at_interval(self):
        dwa = DWAController(target=3.0, init_weight=1e-3, update_freq=3, alpha=0.01)
        init_w = dwa.current_weight
        for _ in range(3):
            dwa.step(100.0)
        assert dwa.current_weight != init_w  # update fired


# ---------------------------------------------------------------------------
# get_lr
# ---------------------------------------------------------------------------

class TestGetLr:
    def test_warmup_increases_linearly(self):
        lrs = [get_lr(i, lr=1e-3, min_lr=1e-5, warmup=4, decay_iters=100)
               for i in range(4)]
        for a, b in zip(lrs, lrs[1:]):
            assert b > a

    def test_after_decay_returns_min_lr(self):
        lr = get_lr(200, lr=1e-3, min_lr=1e-5, warmup=10, decay_iters=100)
        assert lr == pytest.approx(1e-5)

    def test_cosine_stays_between_min_and_max(self):
        for step in range(0, 110, 5):
            lr = get_lr(step, lr=1e-3, min_lr=1e-5, warmup=10, decay_iters=100)
            assert 1e-5 <= lr <= 1e-3


# ---------------------------------------------------------------------------
# phase1_nll_loss
# ---------------------------------------------------------------------------

class TestPhase1NllLoss:
    def test_only_step_tokens_contribute(self):
        B, T, V = 2, 6, 10
        logits = torch.zeros(B, T, V)
        logits[:, :, 0] = 100.0  # strong prediction of token 0 everywhere
        input_ids = torch.zeros(B, T, dtype=torch.long)

        # All tokens correct → near-zero loss when mask covers all tokens
        loss_mask = torch.ones(B, T, dtype=torch.long)
        loss_all = phase1_nll_loss(logits, input_ids, loss_mask)
        assert loss_all.item() < 0.01

        # Only step tokens (last 3) included in loss
        loss_mask2 = torch.zeros(B, T, dtype=torch.long)
        loss_mask2[:, 3:] = 1
        loss_step = phase1_nll_loss(logits, input_ids, loss_mask2)
        assert loss_step.item() < 0.01

    def test_wrong_prediction_incurs_loss(self):
        B, T, V = 1, 4, 10
        logits = torch.zeros(B, T, V)
        logits[:, :, 9] = 100.0  # predicts token 9
        input_ids = torch.zeros(B, T, dtype=torch.long)  # targets are token 0
        loss_mask = torch.ones(B, T, dtype=torch.long)
        loss = phase1_nll_loss(logits, input_ids, loss_mask)
        assert loss.item() > 5.0  # large CE since prediction is wrong

    def test_ignored_positions_do_not_affect_loss(self):
        B, T, V = 2, 6, 10
        # Step tokens (last 3) are correct; context tokens (first 3) are wrong.
        logits = torch.zeros(B, T, V)
        logits[:, :3, 9] = 100.0   # wrong predictions on context (ignored)
        logits[:, 3:, 0] = 100.0   # correct predictions on step
        input_ids = torch.zeros(B, T, dtype=torch.long)  # all targets = 0
        loss_mask = torch.zeros(B, T, dtype=torch.long)
        loss_mask[:, 3:] = 1       # only step tokens contribute
        loss = phase1_nll_loss(logits, input_ids, loss_mask)
        assert loss.item() < 0.01

    def test_returns_scalar(self):
        logits = torch.randn(3, 8, 16)
        input_ids = torch.randint(0, 16, (3, 8))
        loss_mask = torch.ones(3, 8, dtype=torch.long)
        loss = phase1_nll_loss(logits, input_ids, loss_mask)
        assert loss.shape == torch.Size([])


# ---------------------------------------------------------------------------
# apply_step_attention_mask
# ---------------------------------------------------------------------------

class TestApplyStepAttentionMask:
    def test_context_tokens_never_masked(self):
        B, T = 2, 10
        attn = torch.ones(B, T, dtype=torch.long)
        sep_pos = torch.tensor([4, 3])
        val_len = torch.tensor([8, 7])

        torch.manual_seed(0)
        masked = apply_step_attention_mask(attn, sep_pos, val_len, mask_prob=0.9)

        # Context region [0..sep_pos) must remain 1 for all sequences
        for b in range(B):
            assert masked[b, : sep_pos[b].item()].sum() == sep_pos[b].item()

    def test_step_tokens_partially_masked(self):
        B, T = 4, 20
        attn = torch.ones(B, T, dtype=torch.long)
        sep_pos = torch.full((B,), 5, dtype=torch.long)
        val_len = torch.full((B,), 18, dtype=torch.long)

        torch.manual_seed(42)
        masked = apply_step_attention_mask(attn, sep_pos, val_len, mask_prob=0.5)

        step_region_sum = masked[:, 5:18].sum().item()
        total_step = B * 13
        # With mask_prob=0.5 and enough samples, roughly half should be zeroed
        assert step_region_sum < total_step  # at least some are masked

    def test_zero_prob_returns_unchanged(self):
        B, T = 2, 8
        attn = torch.ones(B, T, dtype=torch.long)
        sep_pos = torch.tensor([3, 2])
        val_len = torch.tensor([7, 6])
        masked = apply_step_attention_mask(attn, sep_pos, val_len, mask_prob=0.0)
        assert torch.equal(masked, attn)

    def test_padding_region_unaffected(self):
        B, T = 2, 10
        attn = torch.zeros(B, T, dtype=torch.long)
        attn[:, :8] = 1  # real tokens only up to position 7
        sep_pos = torch.tensor([3, 3])
        val_len = torch.tensor([8, 8])
        masked = apply_step_attention_mask(attn, sep_pos, val_len, mask_prob=1.0)
        # Positions [val_len, T) are padding and were already 0
        assert masked[:, 8:].sum() == 0
