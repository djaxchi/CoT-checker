"""das_branch_subspace_v0 Phase 3: DAS shared-U subspace training core.

The whole-span full-state oracle transfers correct-vs-wrong answer behaviour (L12
recovers 88% of the belief margin, 35% of the free-gen solve gap; the boundary vector
recovered ~0). DAS now asks the actual hypothesis: is that transferable branch
variable a LOW-DIMENSIONAL distributed subspace? We learn one orthonormal projector
U in R^{d x k}, SHARED across all candidate-step positions, and interchange only that
subspace between siblings:

    H_patched[p] = H_base[p] + U U^T (H_donor[p] - H_base[p])   for every span p.

Qwen is frozen; only U trains, against a correctness-typed margin objective (raise the
gold-answer margin when the correct span is injected into the wrong base). Success =
held-out margin/solve recovery that beats a same-k random subspace, is correct-sibling
specific, and gives a cross-seed-consistent subspace (small principal angles).

The donor/base span states H_donor, H_base are precomputed and detached (constants);
grad flows to U only through the injected states and the frozen upper blocks.
"""

from __future__ import annotations

import torch


class SubspaceU(torch.nn.Module):
    """Learnable d x k projector with orthonormal columns (thin-QR each forward, so
    the span is well-defined and the interchange is a genuine subspace swap)."""

    def __init__(self, d: int, k: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(d, k, generator=g) / (d ** 0.5))

    def forward(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.weight)      # (d, k), orthonormal columns
        return q


def interchange_states(base_span: torch.Tensor, donor_span: torch.Tensor,
                       Q: torch.Tensor) -> torch.Tensor:
    """H_base + (H_donor - H_base) projected onto span(Q). Shapes: spans (n, d),
    Q (d, k). Differentiable in Q."""
    diff = donor_span - base_span                # (n, d)
    proj = (diff @ Q) @ Q.transpose(-1, -2)      # (n, d)
    return base_span + proj


def span_candidate_logprobs_grad(model, context_ids: list[int],
                                 cand_ids_list: list[list[int]], pad_id: int, device,
                                 layer: int, lo: int, hi: int,
                                 states: torch.Tensor) -> torch.Tensor:
    """Grad-enabled analogue of das_span.span_candidate_logprobs: mean per-token
    log-prob of each candidate, with the span [lo, hi) at ``layer`` replaced by
    ``states`` (which carry grad to U). Returns a (num_cand,) tensor."""
    from src.analysis.das_span import make_span_patch_hook

    rows, spans = [], []
    for cand in cand_ids_list:
        rows.append(context_ids + cand)
        spans.append((len(context_ids), len(context_ids) + len(cand)))
    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, :len(r)] = 1
    handle = model.model.layers[layer - 1].register_forward_hook(
        make_span_patch_hook(lo, hi, states))
    try:
        logits = model(input_ids=input_ids.to(device),
                       attention_mask=mask.to(device)).logits
    finally:
        handle.remove()
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    outs = []
    for i, (clo, chi) in enumerate(spans):
        tok_ids = input_ids[i, clo:chi].to(device)
        lp = logprobs[i, clo - 1:chi - 1, :].gather(-1, tok_ids[:, None]).squeeze(-1)
        outs.append(lp.mean())
    return torch.stack(outs)


def margin_ce_loss(cand_logprobs: torch.Tensor, gold_idx: int = 0) -> torch.Tensor:
    """Cross-entropy pushing the candidate distribution onto the gold answer, i.e.
    maximise the gold margin. Softmax over the candidate mean-logprobs."""
    return torch.nn.functional.cross_entropy(
        cand_logprobs[None, :], torch.tensor([gold_idx], device=cand_logprobs.device))


def subspace_overlap(Q1: torch.Tensor, Q2: torch.Tensor) -> float:
    """Mean squared principal-angle cosine between two orthonormal d x k bases: the
    squared singular values of Q1^T Q2 averaged over k. 1 = identical span, 0 =
    orthogonal. Cross-seed identifiability metric."""
    k = Q1.shape[1]
    s = torch.linalg.svdvals(Q1.transpose(-1, -2) @ Q2)
    return float((s.clamp(max=1.0) ** 2).sum() / k)
