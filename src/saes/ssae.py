"""Step-level Sparse AutoEncoder (SSAE).

Architecture (paper: arXiv:2603.03031, Miaow-Lab):
  Phase 1 — Reconstruction:
    Encoder  : LLM backbone encodes [context | SEP | step] → last-token h_k ∈ ℝ^d
    Projector: SparseAutoencoder h_k → h_c ∈ ℝ^(d * F)  (sparse via ReLU + L1 reg)
    Proj-MLP : h_c → recons ∈ ℝ^(F, d)                  (F "reconstruction tokens")
    Decoder  : LLM backbone autoregressively reconstructs step tokens
               given [recons | context_without_last_token]

  Phase 2 — Distribution:
    HintsEncoder + mean_mlp + var_mlp learn P(h_c | context)

  Phase 3 — Fine-tune var_mlp only.

Ported and documented from the SSAE reference implementation (Miaow-Lab/SSAE).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .autoencoder import SparseAutoencoder, TopKActivation

__all__ = ["SSAE", "TopKActivation"]

# Map local training paths → public HuggingFace model IDs
_LOCAL_TO_HF: dict[str, str] = {
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
}


def _resolve_model_id(name: str) -> str:
    """Convert a possibly-local training path to a HuggingFace model ID.

    Handles patterns like:
      "./Qwen2.5-0.5B"  → "Qwen/Qwen2.5-0.5B"   (relative path from training)
      "/abs/path/model" → "/abs/path/model"        (absolute local path, keep)
      "Qwen/Qwen2.5-0.5B" → "Qwen/Qwen2.5-0.5B" (already an HF id, keep)
    """
    # Absolute path that actually exists on disk → use as-is
    if Path(name).is_absolute() and Path(name).exists():
        return name
    # Strip leading "./" or "../" to get the bare model directory name
    stem = name.lstrip("./")
    # If still contains "/" it looks like a proper HF owner/model id
    if "/" in stem:
        return stem
    return _LOCAL_TO_HF.get(stem, stem)


class SSAE(nn.Module):
    """Step-Level Sparse AutoEncoder wrapping a causal LM backbone.

    Args:
        tokenizer: Tokenizer for the backbone model(s).
        sparsity_factor: Expansion ratio for the sparse projector.
            n_latents = hidden_size * sparsity_factor.
        activation: Non-linearity for the sparse projector (default: ReLU).
        encoder_model_id: HuggingFace model ID for the encoder backbone.
        decoder_model_id: HuggingFace model ID for the decoder backbone.
            Defaults to encoder_model_id.
        phase: Training phase (1, 2, or 3). Controls which parameters are frozen.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sparsity_factor: int = 1,
        activation: nn.Module | None = None,
        encoder_model_id: str = "Qwen/Qwen2.5-0.5B",
        decoder_model_id: str | None = None,
        phase: int = 1,
        dtype: torch.dtype | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if decoder_model_id is None:
            decoder_model_id = encoder_model_id

        self.tokenizer = tokenizer
        self.sparsity_factor = sparsity_factor
        self.freeze_encoder = freeze_encoder

        # --- Backbone models ---
        kwargs = {} if dtype is None else {"dtype": dtype}
        self.encoder = AutoModel.from_pretrained(encoder_model_id, **kwargs)
        self.hints_encoder = AutoModel.from_pretrained(encoder_model_id, **kwargs)
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model_id, **kwargs)

        # Resize embeddings if tokenizer was extended (e.g. added SEP token)
        vocab_size = len(tokenizer)
        self.encoder.resize_token_embeddings(vocab_size)
        self.hints_encoder.resize_token_embeddings(vocab_size)
        self.decoder.resize_token_embeddings(vocab_size)

        d = self.encoder.config.hidden_size
        n_latents = d * sparsity_factor
        self.n_inputs = d
        self.n_latents = n_latents

        # --- Sparse projector ---
        self.autoencoder = SparseAutoencoder(
            n_inputs=d,
            n_latents=n_latents,
            sparsity_factor=sparsity_factor,
            activation=activation,
        )

        # --- Projection MLP: latents → reconstruction tokens ---
        self.projection_mlp = nn.Sequential(
            nn.Linear(n_latents, n_latents),
            nn.ReLU(),
            nn.Linear(n_latents, n_latents),
        )

        # --- Distribution MLPs (phase 2 / 3) ---
        self.mean_mlp = nn.Sequential(
            nn.Linear(d, n_latents // 8),
            nn.LeakyReLU(),
            nn.Linear(n_latents // 8, n_latents // 2),
            nn.LeakyReLU(),
            nn.Linear(n_latents // 2, n_latents),
        )
        self.var_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.LeakyReLU(),
            nn.Linear(d, n_latents),
            nn.Linear(n_latents, n_latents),
        )

        self.phase = phase
        self._freeze_for_phase(phase)

    # ------------------------------------------------------------------
    # Phase-specific parameter freezing
    # ------------------------------------------------------------------

    def _freeze_for_phase(self, phase: int) -> None:
        if phase == 1:
            self.var_mlp.requires_grad_(False)
            self.mean_mlp.requires_grad_(False)
            self.hints_encoder.requires_grad_(False)
            if self.freeze_encoder:
                self.encoder.requires_grad_(False)
        elif phase == 2:
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.autoencoder.requires_grad_(False)
            self.projection_mlp.requires_grad_(False)
        elif phase == 3:
            self.hints_encoder.requires_grad_(False)
            self.mean_mlp.requires_grad_(False)
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.autoencoder.requires_grad_(False)
            self.projection_mlp.requires_grad_(False)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | os.PathLike,
        *,
        hf_repo: str = "Miaow-Lab/SSAE-Checkpoints",
        device: str = "cpu",
        phase: int = 1,
    ) -> "SSAE":
        """Load a pretrained SSAE from a .pt checkpoint.

        Args:
            checkpoint: Either a local path to a .pt file, OR just the filename
                (e.g. ``"gsm8k-385k_Qwen2.5-0.5b_spar-10.pt"``) which will be
                downloaded from ``hf_repo`` automatically.
            hf_repo: HuggingFace repo containing the checkpoints.
            device: torch device string.
            phase: Training phase to set on the loaded model.

        Returns:
            SSAE with pretrained weights loaded.

        Example::

            model = SSAE.from_checkpoint("gsm8k-385k_Qwen2.5-0.5b_spar-10.pt")
        """
        path = Path(checkpoint)
        if not path.exists():
            # Try downloading from HuggingFace Hub
            from huggingface_hub import hf_hub_download

            print(f"Downloading {path.name} from {hf_repo} …")
            local = hf_hub_download(repo_id=hf_repo, filename=path.name)
            path = Path(local)

        print(f"Loading checkpoint from {path} …")
        ckpt = torch.load(path, map_location=device, weights_only=False)

        cfg = ckpt["config"]
        sparsity_factor = int(cfg["sparsity_factor"])
        encoder_name = _resolve_model_id(ckpt["encoder_name"])
        decoder_name = _resolve_model_id(ckpt["decoder_name"])

        print(f"  encoder : {encoder_name}")
        print(f"  decoder : {decoder_name}")
        print(f"  sparsity_factor: {sparsity_factor}")

        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        # The reference training adds a <sep> token separating context from step
        if tokenizer.sep_token is None:
            tokenizer.add_special_tokens({"sep_token": "<sep>"})
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

        # Detect checkpoint dtype so backbones load in the same dtype — avoids
        # lossy bfloat16 round-trips when the checkpoint was saved in float32.
        sample_tensor = next(iter(ckpt["model"].values()))
        ckpt_dtype = sample_tensor.dtype

        model = cls(
            tokenizer=tokenizer,
            sparsity_factor=sparsity_factor,
            encoder_model_id=encoder_name,
            decoder_model_id=decoder_name,
            phase=phase,
            dtype=ckpt_dtype,
        ).to(device)

        model.load_state_dict(ckpt["model"], strict=True)
        print(
            f"  loaded  : step {ckpt.get('global_step', '?')}, "
            f"best_val_loss {ckpt.get('best_val_loss', '?'):.4f}"
        )
        return model

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _get_last_token_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the last *real* token embedding for each sequence.

        Args:
            hidden_states: (batch, seq_len, d)
            attention_mask: (batch, seq_len)  1 = real, 0 = pad

        Returns:
            (batch, 1, d)
        """
        last_idx = (attention_mask.sum(dim=1) - 1).long()  # (batch,)
        batch_size = hidden_states.shape[0]
        embeddings = torch.stack(
            [hidden_states[i, last_idx[i], :] for i in range(batch_size)]
        ).unsqueeze(1)
        return embeddings  # (batch, 1, d)

    def _get_avg_token_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average-pool over real (non-padded) token embeddings.

        Returns:
            (batch, 1, d)
        """
        mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        avg = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return avg.unsqueeze(1)  # (batch, 1, d)

    # ------------------------------------------------------------------
    # Core encode: step + context → sparse latent vector
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a step (with its context prepended) to a sparse latent vector.

        The caller should format input_ids as [context | SEP | step] before
        passing them here.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            latents: (batch, 1, n_latents)  — sparse, L2-normalised
        """
        hidden = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        h_k = self._get_last_token_embeddings(hidden, attention_mask)
        # Cast to autoencoder dtype in case backbone uses bfloat16/float16
        h_k = h_k.to(self.autoencoder.encoder.weight.dtype)
        latents = self.autoencoder(h_k)
        latents = F.normalize(latents, p=2, dim=-1)
        return latents

    @torch.no_grad()
    def encode_dense(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a step to the raw backbone embedding, skipping the sparse bottleneck.

        Returns h_k (last-token hidden state) L2-normalised, matching the
        normalisation applied in encode(). Used as the dense ablation baseline.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            h_k: (batch, 1, n_inputs)  — dense, L2-normalised
        """
        hidden = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        h_k = self._get_last_token_embeddings(hidden, attention_mask)
        h_k = h_k.float()
        h_k = F.normalize(h_k, p=2, dim=-1)
        return h_k

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hints_sep_ids: torch.Tensor,
        hints_sep_attention_masks: torch.Tensor,
    ) -> tuple:
        """Training forward pass. Behaviour depends on self.phase.

        Phase 1 returns: (latents, loss_sparsity, logits)
        Phase 2/3 return: (nll_loss, mean_error)
        """
        batch_size = input_ids.shape[0]

        if self.phase == 1:
            hidden = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            h_k = self._get_last_token_embeddings(hidden, attention_mask)
            h_k = h_k.to(self.autoencoder.encoder.weight.dtype)
            latents = self.autoencoder(h_k)
            latents = F.normalize(latents, p=2, dim=-1)
            loss_sparsity = latents.abs().sum(dim=(1, 2))  # (batch,)

            # Add small Gaussian noise for robustness (σ=0.01 per paper)
            latents = latents + torch.randn_like(latents) * 0.01

            recons = self.projection_mlp(latents)  # (batch, 1, n_latents)
            recons = recons.view(batch_size, self.sparsity_factor, self.n_inputs)  # (batch, F, d)
            logits = self._decode(recons, input_ids, attention_mask)

            return latents, loss_sparsity.sum(), logits

        elif self.phase in (2, 3):
            # Encode step to get ground-truth latents
            hidden = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            h_k = self._get_last_token_embeddings(hidden, attention_mask)
            h_k = h_k.to(self.autoencoder.encoder.weight.dtype)
            latents = self.autoencoder(h_k)
            latents = F.normalize(latents, p=2, dim=-1)

            # Encode hints (context only) to learn the latent distribution
            hints_hidden = self.hints_encoder(
                hints_sep_ids, attention_mask=hints_sep_attention_masks
            ).last_hidden_state

            if self.phase == 2:
                hints_emb = self._get_last_token_embeddings(hints_hidden, hints_sep_attention_masks)
            else:  # phase 3
                hints_emb = self._get_avg_token_embeddings(hints_hidden, hints_sep_attention_masks)

            mean = self.mean_mlp(hints_emb)  # (batch, 1, n_latents)
            log_var = self.var_mlp(hints_emb)  # (batch, 1, n_latents)
            var = torch.exp(log_var)

            log_likelihood = -((latents - mean) ** 2) / (2 * var) - 0.5 * torch.log(
                2 * torch.pi * var
            )
            nll_loss = -log_likelihood.mean()
            mean_error = torch.sqrt(((latents - mean) ** 2).mean())
            return nll_loss, mean_error

    # ------------------------------------------------------------------
    # Decoder (teacher-forcing reconstruction)
    # ------------------------------------------------------------------

    def _decode(
        self,
        recons: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct step tokens via teacher forcing.

        Prepends F reconstruction tokens to the context embeddings and runs
        the decoder LM in one forward pass.

        Args:
            recons:         (batch, F, d) — projection of sparse latents
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, seq_len + F - 1, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Remove last token (teacher-forcing shift)
        last_pos = (attention_mask.sum(dim=1) - 1).long()
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        mask[torch.arange(batch_size), last_pos] = False
        without_last = input_ids[mask].reshape(batch_size, seq_len - 1)

        embed = self.decoder.get_input_embeddings()
        embed_dtype = embed.weight.dtype
        inputs_embs = torch.cat([recons.to(embed_dtype), embed(without_last)], dim=1)

        recons_mask = torch.ones(
            (batch_size, self.sparsity_factor - 1), dtype=torch.long, device=device
        )
        full_mask = torch.cat([recons_mask, attention_mask], dim=1)

        return self.decoder(inputs_embeds=inputs_embs, attention_mask=full_mask).logits

    # ------------------------------------------------------------------
    # Prediction branch decoder (Future-SSAE)
    # ------------------------------------------------------------------

    def decode_from_latents(
        self,
        latents: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forcing decode using pre-computed latents without re-running the encoder.

        Used by Future-SSAE to share h_hat_k between the reconstruction and
        prediction branches, avoiding a second encoder forward pass.

        Args:
            latents:        (batch, 1, n_latents) — from the reconstruction forward pass
            input_ids:      (batch, T) — full teacher-forcing sequence (e.g. [q | s_k | SEP | s_{k+1} | EOS])
            attention_mask: (batch, T)

        Returns:
            logits: (batch, T + sparsity_factor - 1, vocab_size)
        """
        batch_size = latents.shape[0]
        recons = self.projection_mlp(latents)
        recons = recons.view(batch_size, self.sparsity_factor, self.n_inputs)
        return self._decode(recons, input_ids, attention_mask)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_sparse_vector(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the sparse latent vector for an encoded step.

        Convenience wrapper around encode() that returns h_c squeezed to
        (batch, n_latents) for easy inspection.

        Args:
            input_ids:      (batch, seq_len)  — [context | SEP | step]
            attention_mask: (batch, seq_len)

        Returns:
            (batch, n_latents)
        """
        latents = self.encode(input_ids, attention_mask)
        return latents.squeeze(1)  # (batch, n_latents)

    @torch.no_grad()
    def generate_from_latents(
        self,
        latents: torch.Tensor,
        hints_sep_ids: torch.Tensor,
        hints_sep_attention_masks: torch.Tensor,
        temperature: float = 0.6,
        top_k: int = 0,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> torch.Tensor:
        """Decode a sparse latent vector into text autoregressively.

        Args:
            latents:                  (batch, 1, n_latents)
            hints_sep_ids:            (batch, hints_len)
            hints_sep_attention_masks:(batch, hints_len)

        Returns:
            decode_text_ids: (batch, generated_len)
        """
        batch_size = latents.shape[0]
        device = latents.device

        recons = self.projection_mlp(latents)
        recons = recons.view(batch_size, self.sparsity_factor, self.n_inputs)

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        newline_id = self.tokenizer.encode("\n")[0]
        decode_ids = torch.full((batch_size, 1), newline_id, dtype=torch.long, device=device)
        unfinished = torch.ones(batch_size, dtype=torch.long, device=device)

        embed = self.decoder.get_input_embeddings()

        for _ in range(max_new_tokens):
            hints_embs = embed(hints_sep_ids).to(recons.dtype)
            inputs_embs = torch.cat([recons, hints_embs], dim=1)
            recons_mask = torch.ones(
                (batch_size, self.sparsity_factor), dtype=torch.long, device=device
            )
            full_mask = torch.cat([recons_mask, hints_sep_attention_masks], dim=1)

            out = self.decoder(inputs_embeds=inputs_embs, attention_mask=full_mask)

            # Pick the logit at the last *real* token position (offset by recons prefix).
            # Tokens are appended at the end so the last real token is always at
            # (number of real tokens - 1) + sparsity_factor in the combined sequence.
            valid_len = (hints_sep_attention_masks == 1).sum(dim=1) - 1  # (batch,)
            last_logit_pos = valid_len + self.sparsity_factor
            last_logits = out.logits[torch.arange(batch_size), last_logit_pos, :]

            next_tokens = torch.stack(
                [_sample(last_logits[i], temperature, top_k, top_p) for i in range(batch_size)]
            ).unsqueeze(1)

            newly_done = (next_tokens.squeeze(1) == self.tokenizer.eos_token_id) & (unfinished == 1)
            unfinished[newly_done] = 0

            to_add = next_tokens.clone().squeeze(1)
            to_add[unfinished == 0] = pad_id
            decode_ids = torch.cat([decode_ids, to_add.unsqueeze(1)], dim=1)

            if unfinished.max() == 0:
                break

            # Insert the new token AFTER the last real token (valid_len points to sep /
            # last generated token).  This keeps [context | sep | t1 | t2 | ...] order,
            # matching the training format, while keeping padding contiguous at the end.
            hints_sep_ids, hints_sep_attention_masks = _append_token(
                hints_sep_ids,
                hints_sep_attention_masks,
                to_add,
                valid_len + 1,
                unfinished,
                pad_id,
            )

        return decode_ids


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Sample or greedily decode a single-step logit vector."""
    logits = logits.float()
    if temperature == 0:
        return torch.argmax(logits)

    logits = logits / temperature

    if top_k > 0:
        values, _ = torch.topk(logits, k=top_k)
        logits[logits < values[-1]] = -float("inf")
    elif top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits[sorted_idx[remove]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _append_token(
    ids: torch.Tensor,
    mask: torch.Tensor,
    new_tokens: torch.Tensor,
    insert_positions: torch.Tensor,
    unfinished: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert a generated token into the hints sequence.

    The new token is placed at insert_positions[i] in the output.
    Elements originally at positions [0, insert_pos) are copied unchanged;
    elements at [insert_pos, end) shift right by one.

    Args:
        ids:              (batch, seq_len)
        mask:             (batch, seq_len)
        new_tokens:       (batch,) — token ids to insert
        insert_positions: (batch,) — destination index for each new token
        unfinished:       (batch,) — 1 if the sequence is still being generated
        pad_id:           padding token id
    """
    batch_size = ids.shape[0]
    old_len = ids.shape[1]
    new_len = old_len + 1
    device = ids.device

    new_ids = torch.full((batch_size, new_len), pad_id, dtype=ids.dtype, device=device)
    new_mask = torch.zeros((batch_size, new_len), dtype=mask.dtype, device=device)

    src_idx = torch.arange(old_len, device=device).unsqueeze(0)  # (1, old_len)
    dst_idx = torch.arange(new_len, device=device).unsqueeze(0)  # (1, new_len)
    ins = insert_positions.unsqueeze(1)  # (batch, 1)

    # Elements before the insertion point: src[0..ins) → dst[0..ins)
    before_src = src_idx < ins  # (batch, old_len)
    before_dst = dst_idx < ins  # (batch, new_len) — same count per row
    new_ids[before_dst] = ids[before_src]
    new_mask[before_dst] = mask[before_src]

    # Insert new token at the insertion position (for unfinished sequences)
    active = unfinished == 1
    if active.any():
        new_ids[active, insert_positions[active]] = new_tokens[active]
        new_mask[active, insert_positions[active]] = 1

    # Elements at and after the insertion point: src[ins..) → dst[ins+1..)
    after_src = src_idx >= ins  # (batch, old_len)
    after_dst = dst_idx >= (ins + 1)  # (batch, new_len) — same count per row
    new_ids[after_dst] = ids[after_src]
    new_mask[after_dst] = mask[after_src]

    return new_ids, new_mask
