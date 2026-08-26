"""Load a Qwen-Scope TopK SAE and encode stored residual states through it.

Qwen-Scope (arXiv:2605.11887) publishes SAEs for Qwen3-8B-Base on the residual
stream at every one of the 36 layers, width 65,536 (16x expansion), TopK, in two
sparsity settings (k=50 and k=100). That is the substrate the SAE arm needs and
the reason for the backbone choice: the previous backbone had no base-model SAE
at all, which forced an Instruct-matched compromise.

One file per layer, `layer{N}.sae.pt`, a plain state_dict:
    W_enc (d_sae, d_model)   b_enc (d_sae,)
    W_dec (d_model, d_sae)   b_dec (d_model,)

TopK inference, matching the reference `app.py` shipped in the SAE repo:
    pre  = h @ W_enc.T + b_enc        (no b_dec subtraction)
    z    = topk(relu(pre), k)         (ReLU first, then TopK)
    hhat = z @ W_dec.T + b_dec

Both details matter and neither is guessable. Subtracting b_dec before encoding
(the convention several other SAE suites use, including the BatchTopK SAE already
in this repo) and taking TopK of the raw pre-activation instead of its ReLU
together give FVU ~5 on correctly-paired states, versus well under 1 when done
their way.

Their hook is `model.model.layers[N].register_forward_hook` capturing `out[0]`,
i.e. the OUTPUT of block N. So `layer{N}.sae.pt` pairs with `hidden_states[N+1]`,
and `layer34.sae.pt` pairs with a store encoded at `--layer 35`.

**Layer pairing is checkable, and worth checking.** HF's `output_hidden_states`
gives len(layers)+1 entries where index i is the *input* to block i, so block N's
output is `hidden_states[N+1]`. The final entry is special: HF returns it after
the model's final RMSNorm, so `hidden_states[-1]` is not a raw resid_post at all
and no SAE reconstructs it (measured FVU 224.65). `fvu` is the gate: a correct
pairing with a correct convention sits well below 1, anything else does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def find_snapshot(hf_cache: str | Path, repo_id: str) -> Path:
    """Resolve a local HF snapshot dir. Compute nodes have no internet, so the
    weights must already be there; this fails loudly rather than trying to fetch."""
    name = "models--" + repo_id.replace("/", "--")
    root = Path(hf_cache) / "hub" / name / "snapshots"
    snaps = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if not snaps:
        raise FileNotFoundError(
            f"no local snapshot for {repo_id} under {root}; download it on the "
            f"login node first (compute nodes have no internet)")
    return snaps[-1]


class TopKSAE:
    """Qwen-Scope TopK sparse autoencoder over one layer's residual stream."""

    def __init__(self, W_enc, b_enc, W_dec, b_dec, k: int, device=None,
                 dtype=torch.float32):
        dev = device or torch.device("cpu")
        self.W_enc = W_enc.to(dev, dtype)
        self.b_enc = b_enc.to(dev, dtype)
        self.W_dec = W_dec.to(dev, dtype)
        self.b_dec = b_dec.to(dev, dtype)
        self.k = int(k)
        self.device = dev
        self.dtype = dtype

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[1]

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[0]

    @classmethod
    def from_snapshot(cls, snapshot: str | Path, layer: int, device=None,
                      dtype=torch.float32) -> "TopKSAE":
        snapshot = Path(snapshot)
        cfg = json.loads((snapshot / "config.json").read_text())
        path = snapshot / f"layer{layer}.sae.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not present; Qwen-Scope ships one file per layer and only "
                f"the layers you downloaded are on disk")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        return cls(sd["W_enc"], sd["b_enc"], sd["W_dec"], sd["b_dec"],
                   k=cfg["k"], device=device, dtype=dtype)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """(n, d_model) -> (n, d_sae) with exactly k non-zeros per row."""
        pre = torch.relu(h.to(self.device, self.dtype) @ self.W_enc.T + self.b_enc)
        vals, idx = torch.topk(pre, self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z

    def encode_sparse(self, h: torch.Tensor):
        """(values (n,k), indices (n,k)) without materializing the dense code.

        A dense 65,536-wide code is 128 KB per token at float16; the sparse form
        is what makes pooling over a corpus of steps tractable.
        """
        pre = torch.relu(h.to(self.device, self.dtype) @ self.W_enc.T + self.b_enc)
        vals, idx = torch.topk(pre, self.k, dim=-1)
        return vals, idx

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.to(self.device, self.dtype) @ self.W_dec.T + self.b_dec

    def fvu(self, h: torch.Tensor) -> float:
        """Fraction of variance unexplained. The layer-pairing gate: a correct
        pairing sits well below 1, a wrong one at or above it."""
        h = h.to(self.device, self.dtype)
        hhat = self.decode(self.encode(h))
        return float(((h - hhat) ** 2).sum() / ((h - h.mean(0)) ** 2).sum())
