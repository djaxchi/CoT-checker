"""Phase-1 SSAE for Qwen2.5-1.5B.

Mirror of Miaow-Lab/SSAE phase-1 (papers/SSAE/model_qwen.py + sentenceSAE.py),
adapted for Qwen2.5-1.5B and the CoT-checker repo:
- offline-only from_pretrained (local_files_only=True),
- single-step phase=1 forward only (no mean_mlp/var_mlp),
- optional auxiliary BCE head used only in ssae_contrastive representation
  training; never used for the final linear probe (per spec section 19).

The Autoencoder class is copied verbatim from sentenceSAE.Autoencoder.
The forward pass mirrors MyModel.forward(phase=1) and MyModel.decode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM


class Autoencoder(nn.Module):
    """Verbatim mirror of papers/SSAE/sentenceSAE.py::Autoencoder."""

    def __init__(self, n_latents: int, n_inputs: int, sparsity_factor: int,
                 activation: nn.Module | None = None) -> None:
        super().__init__()
        self.sparsity_factor = sparsity_factor
        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.encoder = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_inputs)  ->  (batch, 1, n_latents)
        latents_pre_act = self.encoder(x) + self.latent_bias
        return self.activation(latents_pre_act)


class QwenSSAE(nn.Module):
    """Phase-1 SSAE: encoder -> last-token latent -> projection_mlp -> decoder.

    Reproduces Miaow-Lab MyModel(phase=1). The hints_encoder is loaded and
    frozen for architectural compatibility but is not used in the phase-1
    forward (as in the official code). An auxiliary BCE head is added only
    when `contrastive=True`; it is excluded from final ProcessBench scoring.
    """

    def __init__(
        self,
        tokenizer,
        model_name_or_path: str,
        sparsity_factor: int = 1,
        phase: int = 1,
        local_files_only: bool = True,
        activation: nn.Module | None = None,
        contrastive: bool = False,
    ) -> None:
        super().__init__()
        if phase != 1:
            raise NotImplementedError("Only phase=1 is implemented; phase=2/3 are out of scope.")
        self.tokenizer = tokenizer
        self.sparsity_factor = sparsity_factor
        self.phase = phase
        self.contrastive = contrastive
        self.model_name_or_path = model_name_or_path

        # Three Qwen instances, matching official MyModel.
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        self.hints_encoder = AutoModel.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )

        new_vocab_size = len(tokenizer)
        self.encoder.resize_token_embeddings(new_vocab_size)
        self.hints_encoder.resize_token_embeddings(new_vocab_size)
        self.decoder.resize_token_embeddings(new_vocab_size)

        # n_inputs is read from the encoder config (not hardcoded).
        self.n_inputs = self.encoder.config.hidden_size
        self.n_latents = self.n_inputs * self.sparsity_factor

        self.autoencoder = Autoencoder(
            n_latents=self.n_latents,
            n_inputs=self.n_inputs,
            sparsity_factor=self.sparsity_factor,
            activation=activation if activation is not None else nn.ReLU(),
        )
        self.projection_mlp = nn.Sequential(
            nn.Linear(self.n_latents, self.n_latents),
            nn.ReLU(),
            nn.Linear(self.n_latents, self.n_latents),
        )

        # Phase 1: freeze hints_encoder (matches official phase-1 branch).
        self.hints_encoder.requires_grad_(False)

        # Caching is incompatible with gradient checkpointing and useless
        # during teacher-forced training; disable on the decoder so HF does
        # not pre-allocate KV-cache buffers.
        if hasattr(self.decoder, "config"):
            self.decoder.config.use_cache = False

        # Auxiliary BCE head, only for ssae_contrastive representation
        # training. Discarded before linear probing (spec section 19).
        self.aux_head: nn.Module | None = None
        if self.contrastive:
            self.aux_head = nn.Linear(self.n_latents, 1)

    def enable_gradient_checkpointing(self) -> None:
        """Turn on HF gradient checkpointing for encoder and decoder.

        Trades extra compute for activation memory: intermediate transformer
        block activations are not retained, they are recomputed on backward.
        hints_encoder is frozen and untouched. Caching must be off when
        checkpointing is on.
        """
        for sub in (self.encoder, self.decoder):
            if hasattr(sub, "gradient_checkpointing_enable"):
                try:
                    sub.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    sub.gradient_checkpointing_enable()
            if hasattr(sub, "config"):
                sub.config.use_cache = False

    # ------------------------------------------------------------------ utils
    def get_last_token_embeddings(self, encoder_outputs: torch.Tensor,
                                  attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, S, H), (B, S) -> (B, 1, H), picking the last non-pad token."""
        batch_size = encoder_outputs.size(0)
        last_token_indices = (attention_mask.sum(dim=1) - 1).long()
        # Vectorized form of the official per-i stack.
        last = encoder_outputs[torch.arange(batch_size, device=encoder_outputs.device),
                               last_token_indices, :]
        return last.unsqueeze(1)

    # ----------------------------------------------------------------- decode
    def decode(self, recons: torch.Tensor, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        """Mirror of MyModel.decode.

        recons: (B, sparsity_factor, n_inputs)
        Returns logits of shape (B, seq_len + sparsity_factor - 1, vocab).
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Remove the last non-pad token (teacher forcing).
        last_token_position = (attention_mask.sum(dim=1) - 1).long()
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        mask[torch.arange(batch_size, device=device), last_token_position] = False
        without_last_token_ids = input_ids[mask].reshape(batch_size, seq_len - 1)

        embedding_layer = self.decoder.get_input_embeddings()
        without_last_token_embeds = embedding_layer(without_last_token_ids)

        # Cast recons to the decoder embedding dtype (helps under autocast).
        recons = recons.to(without_last_token_embeds.dtype)

        inputs_embs = torch.cat([recons, without_last_token_embeds], dim=1)
        recons_attention_mask = torch.ones(
            (batch_size, self.sparsity_factor - 1), dtype=attention_mask.dtype, device=device
        )
        full_attention_mask = torch.cat([recons_attention_mask, attention_mask], dim=1)
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embs, attention_mask=full_attention_mask
        )
        return decoder_outputs.logits

    # ---------------------------------------------------------------- forward
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Phase-1 forward.

        Returns a dict with intermediate tensors so the trainer can run
        fail-fast finite checks at every stage and log the zero-latent rate.
        """
        batch_size = input_ids.shape[0]
        encoder_outputs = self.encoder(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state
        last_token_embeddings = self.get_last_token_embeddings(
            encoder_outputs, attention_mask
        )

        # Autoencoder: linear + bias + ReLU. We split it so the trainer can
        # see both pre-activation and post-activation tensors.
        latents_pre = (
            self.autoencoder.encoder(last_token_embeddings)
            + self.autoencoder.latent_bias
        )
        latents_relu = self.autoencoder.activation(latents_pre)

        # Stable L2 normalization with eps=1e-8. Equivalent to the official
        # F.normalize(p=2) on non-zero vectors; on (near-)zero rows it
        # divides by eps instead of by 0, which prevents NaN propagation.
        latent_norm = latents_relu.norm(p=2, dim=-1, keepdim=True)
        # zero_norm_frac: fraction of (batch, sparsity_factor) latent vectors
        # whose pre-normalize L2 norm is below 1e-8 -- collapsed latents.
        zero_norm_frac = (latent_norm.detach() < 1e-8).float().mean()
        safe_norm = latent_norm.clamp_min(1e-8)
        latents_normed = latents_relu / safe_norm

        sparsity_loss = latents_normed.abs().sum(dim=(1, 2))  # (B,)
        latents_clean = latents_normed

        aux_logit: torch.Tensor | None = None
        if self.contrastive and self.aux_head is not None:
            # Aux head sees post-normalize, pre-noise latents.
            aux_logit = self.aux_head(latents_clean.squeeze(1).float())  # (B, 1)

        if self.training:
            noise = torch.randn_like(latents_normed) * 0.01
            latents = latents_normed + noise
        else:
            latents = latents_normed

        proj_out = self.projection_mlp(latents)  # (B, 1, n_latents)
        recons = proj_out.view(batch_size, self.sparsity_factor, self.n_inputs)
        logits = self.decode(recons, input_ids, attention_mask)

        return {
            "encoder_last_hidden": encoder_outputs,
            "last_token_embeddings": last_token_embeddings,
            "latents_pre": latents_pre,
            "latents_relu": latents_relu,
            "latent_norm": latent_norm,
            "latents_normed": latents_normed,
            "latents": latents,
            "latents_clean": latents_clean,
            "proj_out": proj_out,
            "logits": logits,
            "sparsity_loss_sum": sparsity_loss.sum(),
            "aux_logit": aux_logit,
            "zero_norm_frac": zero_norm_frac,
        }

    # --------------------------------------------------- inference for latents
    @torch.no_grad()
    def encode_latents(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """Return normalized SSAE latents (B, n_latents) for downstream probing."""
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        last_token_embeddings = self.get_last_token_embeddings(encoder_outputs, attention_mask)
        latents = self.autoencoder(last_token_embeddings)
        # Use the same eps-stabilized normalize as the training forward.
        norm = latents.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        latents = latents / norm
        return latents.squeeze(1)
