"""How much did this step move the model's own next-token belief?

Every representation in this study reads the residual stream directly. None has
ever been pushed through the unembedding, which is the one place the model's
internal state becomes a statement about what it thinks comes next. Two shifts
are measurable from what is already stored, and they ask different questions:

  along the step   the belief at the pre-step boundary against the belief at the
                   step's last token. A step that goes wrong may be one that
                   moves the model somewhere it was not heading.
  across layers    the same position read at layer 26 and at layer 35. The logit
                   lens reading of how much the late blocks revised the
                   prediction. Layer 26 already beats layer 35 as a probe input
                   here, so the two layers demonstrably disagree about something.

The features are divergences and ranks, never raw logits, so nothing here can
encode an activation scale or a token count.

A caution recorded in advance rather than after the fact. This project's
token-trajectory work found step incorrectness is diffuse and correlates with
per-token entropy at only -0.20, and latent_memory_v0 found the answer-belief
readout behaves as an answer shortcut with intermediate recall at the floor. So
the entropy entries below are expected to be weak. The belief SHIFT is the actual
hypothesis: not how uncertain the model is, but how far the step moved it.
"""

from __future__ import annotations

import numpy as np

N_BELIEF = 11


def _logsoftmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def _entropy(logp: np.ndarray) -> float:
    return float(-(np.exp(logp) * logp).sum())


def _kl(logp: np.ndarray, logq: np.ndarray) -> float:
    return float((np.exp(logp) * (logp - logq)).sum())


def _js(logp: np.ndarray, logq: np.ndarray) -> float:
    m = np.log(0.5 * (np.exp(logp) + np.exp(logq)) + 1e-30)
    return 0.5 * _kl(logp, m) + 0.5 * _kl(logq, m)


def shift(logits_a: np.ndarray, logits_b: np.ndarray) -> np.ndarray:
    """Five numbers comparing two next-token beliefs, a before and b after."""
    la, lb = _logsoftmax(logits_a), _logsoftmax(logits_b)
    ta, tb = int(la.argmax()), int(lb.argmax())
    # where a's favourite token ended up in b's ranking, in log rank so a drop
    # from first to hundredth reads as a bigger move than hundredth to two
    # hundredth
    rank = int((lb > lb[ta]).sum())
    return np.array([
        _kl(lb, la),
        _js(la, lb),
        _entropy(lb) - _entropy(la),
        float(ta == tb),
        np.log1p(rank),
    ], dtype=np.float32)


def belief_feats(boundary_logits: np.ndarray, end_logits: np.ndarray,
                 end_logits_mid_layer: np.ndarray | None = None) -> np.ndarray:
    """Eleven numbers: the shift along the step, the shift across layers, and the
    two absolute uncertainties the shifts are measured between."""
    la, lb = _logsoftmax(boundary_logits), _logsoftmax(end_logits)
    out = [shift(boundary_logits, end_logits)]
    out.append(shift(end_logits_mid_layer, end_logits)
               if end_logits_mid_layer is not None else np.zeros(5, np.float32))
    out.append(np.array([_entropy(la), _entropy(lb)][:1] + [_entropy(lb)],
                        dtype=np.float32))
    return np.concatenate(out)[:N_BELIEF]
