"""transition_operator_v0 Stage 2: trainable components + frozen-decoder decode path.

Pieces (plan sections 4-7, v0.3):
  TransitionEncoder  E(S_{t-1}, H_t[:-1]) -> z          (trained)
  Linear D           z -> residual edit at the boundary  (trained, A arms only)
  BeliefHead h_B     z -> d_belief (8)                   (trained, B arms only)
  UpperDecoder       frozen blocks (layer_lo..end) + norm + LM head, decoding the
                     patched boundary state against the pre-context K/V cache with
                     gradients ONLY through the patched position (incl. its
                     recomputed K/V). Equivalence with the hook implementation in
                     transition_operator.forward_with_boundary_patch is unit-tested.
  info_nce           symmetric CLIP-style with the effect-distance mask (cos > 0.9
                     measured-effect negatives are dropped, sibling included).
"""

from __future__ import annotations

import re

import torch
from torch import nn


# ---------------------------------------------------------------------------
# trained modules
# ---------------------------------------------------------------------------

class TransitionEncoder(nn.Module):
    """[s-slot(S_{t-1}); x_1..x_{T}] -> 2-layer Transformer -> z at the s-slot."""

    def __init__(self, hidden: int = 3584, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 2, d_z: int = 64, max_steps: int = 192):
        super().__init__()
        self.proj_state = nn.Sequential(nn.Linear(hidden, d_model),
                                        nn.LayerNorm(d_model))
        self.proj_step = nn.Sequential(nn.Linear(hidden, d_model),
                                       nn.LayerNorm(d_model))
        self.pos = nn.Embedding(max_steps + 1, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0)
        self.blocks = nn.TransformerEncoder(block, n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_z), nn.LayerNorm(d_z))

    def forward(self, s_prev: torch.Tensor, h_steps: torch.Tensor,
                step_mask: torch.Tensor) -> torch.Tensor:
        """s_prev (B,hidden); h_steps (B,T,hidden); step_mask (B,T) True=valid."""
        B, T, _ = h_steps.shape
        seq = torch.cat([self.proj_state(s_prev)[:, None, :],
                         self.proj_step(h_steps)], dim=1)
        seq = seq + self.pos(torch.arange(T + 1, device=seq.device))[None]
        pad = torch.cat([torch.ones(B, 1, dtype=torch.bool,
                                    device=seq.device), step_mask], dim=1)
        out = self.blocks(seq, src_key_padding_mask=~pad)
        return self.head(out[:, 0])


class BeliefHead(nn.Module):
    def __init__(self, d_z: int = 64, d_hidden: int = 64, d_out: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_z, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_out))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ContrastiveProjections(nn.Module):
    """z -> R^64 and measured effect [dL_64 ; d_belief_8] -> R^64, both L2-normed."""

    def __init__(self, d_z: int = 64, d_effect: int = 72, d_out: int = 64):
        super().__init__()
        self.z_proj = nn.Linear(d_z, d_out)
        self.g = nn.Sequential(nn.Linear(d_effect, d_out), nn.GELU(),
                               nn.Linear(d_out, d_out))

    def forward(self, z: torch.Tensor, effect: torch.Tensor):
        za = nn.functional.normalize(self.z_proj(z), dim=-1)
        ea = nn.functional.normalize(self.g(effect), dim=-1)
        return za, ea


# ---------------------------------------------------------------------------
# frozen-decoder boundary decode with gradients
# ---------------------------------------------------------------------------

class UpperDecoder:
    """Blocks [layer_lo..n) + final norm + LM head of a frozen CausalLM.

    prefill() runs the pre context WITHOUT its final boundary token (no grad) and
    returns the model's own DynamicCache; decode_boundary() runs the patched
    layer-`layer_lo` boundary state through the upper blocks against that cache.
    The boundary K/V is computed fresh from the patched state (cache-surgery
    semantics; the cache never holds an unpatched boundary in the upper blocks).
    """

    def __init__(self, model, layer_lo: int, mask_value: float | None = None):
        self.model = model
        self.layer_lo = layer_lo
        self.layers = model.model.layers[layer_lo:]
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary = model.model.rotary_emb
        # additive-mask fill for padded cache slots. finfo.min saturates to -inf
        # in half precision, which is a known NaN source in sdpa BACKWARD; a
        # large-but-finite value (-1e4) kills the softmax weight equally well
        # and keeps gradients finite.
        self.mask_value = mask_value

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        out = self.model.model(input_ids=input_ids, attention_mask=attn_mask,
                               use_cache=True)
        return out.past_key_values

    def cache_len(self, cache) -> int:
        return int(cache.layers[self.layer_lo].keys.shape[2])

    def crop(self, cache, length: int) -> None:
        cache.crop(length)

    def decode_boundary(self, cache, h_patch: torch.Tensor,
                        ctx_lens: torch.Tensor) -> torch.Tensor:
        """h_patch (B, hidden): patched layer-`layer_lo` state of the boundary
        token; ctx_lens (B,): true (unpadded) prefill length per row, which is
        also the boundary's position index. Returns boundary logits (B, vocab).
        NOTE: mutates the cache (appends the boundary K/V); crop() back before
        decoding a second branch against the same prefill."""
        B = h_patch.shape[0]
        device = h_patch.device
        L = self.cache_len(cache)
        pos = ctx_lens.to(device).view(B, 1)
        h = h_patch[:, None, :].to(self.norm.weight.dtype)
        pos_emb = self.rotary(h, pos)
        # additive mask over [cached slots 0..L-1, new slot L]
        fill = self.mask_value if self.mask_value is not None \
            else torch.finfo(h.dtype).min
        slots = torch.arange(L + 1, device=device)[None, :]
        valid = (slots < ctx_lens.to(device)[:, None]) | (slots == L)
        mask = torch.where(valid, 0.0, fill).to(h.dtype)[:, None, None, :]
        cache_position = torch.tensor([L], device=device)
        for layer in self.layers:
            h = layer(h, attention_mask=mask, past_key_values=cache,
                      use_cache=True, cache_position=cache_position,
                      position_embeddings=pos_emb)
            if isinstance(h, tuple):
                h = h[0]
        return self.lm_head(self.norm(h))[:, 0].float()


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def kl_to_actual(actual_logits: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """L_A = KL(p_actual || p_pred), full vocab, float32, mean over batch."""
    lp = torch.log_softmax(actual_logits.float(), dim=-1)
    lq = torch.log_softmax(pred_logits.float(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1).mean()


def effect_close_mask(effects: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    """(N,N) bool, True where the measured effects of i and j are too similar to
    serve as negatives (cos > threshold). Diagonal False (the positive)."""
    e = nn.functional.normalize(effects, dim=-1)
    close = (e @ e.T) > threshold
    close.fill_diagonal_(False)
    return close


def info_nce(za: torch.Tensor, ea: torch.Tensor, close: torch.Tensor,
             tau: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE with masked near-duplicate negatives."""
    logits = za @ ea.T / tau
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(close, neg_inf)
    target = torch.arange(len(za), device=za.device)
    return 0.5 * (nn.functional.cross_entropy(masked, target)
                  + nn.functional.cross_entropy(masked.T, target))


# ---------------------------------------------------------------------------
# format features (frozen list from the plan) for the dL-space residualization
# ---------------------------------------------------------------------------

_DISCOURSE = re.compile(r"^(So|Now|Then|Next|Therefore|First|Finally)\b", re.IGNORECASE)
_ENDS_EQ = re.compile(r"=[^=]*$")

_CHAR_CLASSES = ("digit", "letter", "period", "dollar", "other")


def format_features(text: str, n_tokens: int) -> list[float]:
    """[n_tokens, one-hot final char class (5), display math, ends-with-eq,
    discourse opener] -> 9 dims."""
    t = text.rstrip()
    last = t[-1] if t else " "
    if last.isdigit():
        cls = "digit"
    elif last.isalpha():
        cls = "letter"
    elif last == ".":
        cls = "period"
    elif last == "$":
        cls = "dollar"
    else:
        cls = "other"
    onehot = [1.0 if cls == c else 0.0 for c in _CHAR_CLASSES]
    lines = t.splitlines() or [""]
    return [float(n_tokens), *onehot,
            1.0 if "$" in t else 0.0,
            1.0 if _ENDS_EQ.search(lines[-1]) else 0.0,
            1.0 if _DISCOURSE.search(t.lstrip()) else 0.0]


# ---------------------------------------------------------------------------
# D(z) naturalness audits (A1-A4 reference stats; full audit in the train script)
# ---------------------------------------------------------------------------

def rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean(-1).sqrt()


def percentile_of(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """For each value, its percentile within the reference distribution (CPU)."""
    ref = reference.detach().float().cpu().sort().values
    idx = torch.searchsorted(ref, values.detach().float().cpu().contiguous())
    return 100.0 * idx / len(ref)
