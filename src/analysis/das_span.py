"""das_branch_subspace_v0 whole-step-span interchange core.

The boundary (single-vector) oracle was null on the final answer while verified live
on the next token (Phase 1b + sanity). This module tests the successor hypothesis:
correctness is distributed across the WHOLE candidate-step token span, not its
boundary. It patches every step-token position at a layer so the upper blocks
regenerate their K/V from the patched span during prefill (the required propagation,
not post-hoc cache editing).

Tokenization is the S6 SEP-id scheme via causal_graph.encode_pieces/assemble_ids, so
the candidate-step content span is spans[-1] and, because the prefix pieces are
identical across siblings, an equal-length correct/wrong pair occupies the IDENTICAL
absolute positions (RoPE-clean interchange). The last-k alignment (extension) injects
the correct step's final k states into the wrong step's final k positions even when
lengths differ; that is not position-clean and is reported separately.
"""

from __future__ import annotations

import torch

from src.analysis.causal_graph import ELICITATION_SUFFIX, assemble_ids, encode_pieces
from src.analysis.transition_operator import SEP_TOKEN_ID, gold_margin  # noqa: F401


def fork_span_ids(tok, tr: dict, branch: str) -> tuple[list[int], int, int]:
    """(full_ids, lo, hi) for one branch of a fork under the SEP-id scheme.

    branch: 'correct' -> golden step tr['steps'][t]; 'wrong' -> tr['wrong_step'].
    (lo, hi) is the half-open content span of the candidate step (spans[-1])."""
    t = tr["fork_t"]
    step = tr["steps"][t] if branch == "correct" else tr["wrong_step"]
    pieces = [tr["question"]] + tr["steps"][:t] + [step]
    full, spans, _ = assemble_ids(encode_pieces(tok, pieces))
    lo, hi = spans[-1]
    return full, lo, hi


def aligned_positions(wrong_span: tuple[int, int], correct_span: tuple[int, int],
                      mode: str, k: int = 8) -> tuple[int, int, int, int]:
    """Injection window in the wrong ctx and donor window in the correct ctx.

    'equal'  : spans must be the same length; whole span aligned 1-1 (identical
               absolute positions given the shared prefix).
    'lastk'  : the final min(k, n_wrong, n_correct) positions of each span.
    Returns (inject_lo, inject_hi, donor_lo, donor_hi) with equal widths."""
    wlo, whi = wrong_span
    clo, chi = correct_span
    if mode == "equal":
        if (whi - wlo) != (chi - clo):
            raise ValueError("equal mode requires same-length spans")
        return wlo, whi, clo, chi
    if mode == "lastk":
        kk = min(k, whi - wlo, chi - clo)
        return whi - kk, whi, chi - kk, chi
    raise ValueError(f"unknown mode {mode}")


def make_span_patch_hook(lo: int, hi: int, states: torch.Tensor):
    """Forward hook overwriting residual positions [lo, hi) with ``states`` (shape
    (hi-lo, d)) during any pass whose sequence covers the span (prefill). During
    incremental decode the sequence length is 1 < hi, so the hook is a no-op and the
    patched K/V cached at prefill drives generation."""
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if hs.shape[1] >= hi:
            hs = hs.clone()
            hs[:, lo:hi, :] = states.to(hs.dtype).to(hs.device)
            return (hs,) + tuple(output[1:]) if is_tuple else hs
        return output
    return hook


def extract_span_states(model, input_ids: torch.Tensor, layer: int,
                        lo: int, hi: int) -> torch.Tensor:
    """Residual states hidden_states[layer] at positions [lo, hi) for a single
    sequence. layer is 1-based (embedding output = index 0). Returns (hi-lo, d)."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[layer][0, lo:hi, :].detach()


def span_candidate_logprobs(model, context_ids: list[int],
                            cand_ids_list: list[list[int]], pad_id: int, device,
                            layer: int | None = None, lo: int | None = None,
                            hi: int | None = None,
                            states: torch.Tensor | None = None) -> list[float]:
    """Mean per-token log-prob of each candidate after context_ids, one batched
    right-padded forward, optionally with a span patch at layer/[lo,hi). The span is
    inside the shared context, so a single hook patches all candidate rows."""
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
    handle = None
    if states is not None:
        handle = model.model.layers[layer - 1].register_forward_hook(
            make_span_patch_hook(lo, hi, states))
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids.to(device),
                           attention_mask=mask.to(device)).logits
    finally:
        if handle is not None:
            handle.remove()
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for i, (clo, chi) in enumerate(spans):
        tok_ids = input_ids[i, clo:chi].to(device)
        lp = logprobs[i, clo - 1:chi - 1, :].gather(-1, tok_ids[:, None]).squeeze(-1)
        out.append(float(lp.mean()))
    return out


def generate_with_span_patch(model, tok, device, context_ids: list[int], k: int,
                             temperature: float, top_p: float, max_new_tokens: int,
                             layer: int | None = None, lo: int | None = None,
                             hi: int | None = None,
                             states: torch.Tensor | None = None) -> list[str]:
    """K sampled continuations of context_ids, optionally with a span patch applied
    during prefill (so generation attends to the patched K/V)."""
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    handle = None
    if states is not None:
        handle = model.model.layers[layer - 1].register_forward_hook(
            make_span_patch_hook(lo, hi, states))
    try:
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids, do_sample=True, temperature=temperature,
                top_p=top_p, max_new_tokens=max_new_tokens, num_return_sequences=k,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
    finally:
        if handle is not None:
            handle.remove()
    return tok.batch_decode(gen[:, input_ids.shape[1]:], skip_special_tokens=True)


def suffix_ids(tok) -> list[int]:
    """Frozen elicitation suffix token ids (S6 winner), appended after a context to
    read the gold-answer margin."""
    return tok(ELICITATION_SUFFIX, add_special_tokens=False)["input_ids"]
