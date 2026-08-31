"""Learned bottlenecks that try to keep the correctness signal, not the variance.

The sparse-dictionary result said something specific: a bottleneck trained to
reconstruct spends its budget on the directions that explain the most variance,
and step correctness is not one of them. It is a distributed, low-variance margin
(~0.01% of variance, section 15), so an SAE at 50-of-65,536 features discards
it -- costing up to 0.195 F1, with the damage shrinking the more tokens are
pooled and more features survive.

These bottlenecks vary one thing: what the compression is asked to preserve.

    recon        reconstruct the input                      unsupervised, the baseline
    recon_white  reconstruct in a whitened metric           unsupervised
    mixed        reconstruct AND predict correctness        supervised, one knob
    ib           predict correctness through a noisy code   supervised

`recon_white` is the interesting unsupervised one. Reconstruction error in the
raw metric is dominated by the high-variance directions; measuring it after
whitening makes every direction cost the same to lose, so the encoder has no
reason to sacrifice a low-variance one. It never sees a label.

`mixed` has a single knob, beta, running from pure reconstruction to pure
detection. The useful output is not one number but the curve: how much
reconstruction fidelity is given up per point of F1 recovered.

`ib` is the information-bottleneck form of the same question -- compress as hard
as possible while keeping what predicts y -- implemented as a variational encoder
whose KL term is the compression pressure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

OBJECTIVES = ("recon", "recon_white", "mixed", "ib")


class Bottleneck(nn.Module):
    """encoder -> code -> (decoder, correctness head).

    The code is the representation under test. Everything else exists only to
    train it, and is discarded when the code is handed to the grid's learners.
    """

    def __init__(self, d_in: int, d_code: int, objective: str = "recon",
                 hidden: int = 0, variational: bool = False):
        super().__init__()
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        self.objective = objective
        self.variational = variational or objective == "ib"
        out_dim = d_code * 2 if self.variational else d_code
        if hidden:
            self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                     nn.Linear(hidden, out_dim))
        else:
            self.enc = nn.Linear(d_in, out_dim)
        self.dec = nn.Linear(d_code, d_in)
        self.head = nn.Linear(d_code, 1)
        self.d_code = d_code

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The representation. Deterministic even for the variational form, so
        what the grid evaluates is a fixed function of the input."""
        h = self.enc(x)
        return h[:, :self.d_code] if self.variational else h

    def forward(self, x: torch.Tensor):
        h = self.enc(x)
        if self.variational:
            mu, logvar = h[:, :self.d_code], h[:, self.d_code:].clamp(-8, 8)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        else:
            mu, logvar, z = h, None, h
        return z, mu, logvar

    def loss(self, x, y, beta: float = 1.0, whiten: torch.Tensor | None = None,
             kl_weight: float = 1e-3):
        """Returns (total, parts) so the trade-off can be read, not just the sum."""
        z, mu, logvar = self.forward(x)
        parts = {}
        total = x.new_zeros(())

        if self.objective in ("recon", "recon_white", "mixed"):
            err = self.dec(z) - x
            if self.objective == "recon_white" and whiten is not None:
                # Measure the error after whitening: losing a low-variance
                # direction then costs as much as losing a high-variance one.
                err = err @ whiten.T
            recon = (err ** 2).mean()
            parts["recon"] = float(recon.detach())
            total = total + recon

        if self.objective in ("mixed", "ib"):
            bce = F.binary_cross_entropy_with_logits(self.head(z).squeeze(-1), y)
            parts["bce"] = float(bce.detach())
            total = total + beta * bce

        if self.objective == "ib":
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            parts["kl"] = float(kl.detach())
            total = total + kl_weight * kl

        parts["total"] = float(total.detach())
        return total, parts


@torch.no_grad()
def signal_share(codes: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of the code's variance explained by the class means.

    The number this whole exercise is about. Raw states put ~0.01% of their
    variance on correctness; a bottleneck that helps should raise that share, and
    one that allocates by variance (a reconstruction SAE) lowers it. Reported
    beside F1 so a gain can be attributed rather than merely observed.

    This is the one-way between-group variance ratio, not the variance along the
    class-difference direction. The latter has a floor of about 1/d_code for any
    code at all -- a random 4-dimensional code scores 0.25 on it and looks
    informative when it is not.
    """
    z = codes.float()
    pos, neg = labels == 1, labels == 0
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    n = n_p + n_n
    mu = z.mean(0)
    between = (n_p * (z[pos].mean(0) - mu).pow(2).sum()
               + n_n * (z[neg].mean(0) - mu).pow(2).sum()) / n
    total = z.var(0, unbiased=False).sum()
    return float(between / total.clamp(min=1e-12))
