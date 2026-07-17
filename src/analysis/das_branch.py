"""das_branch_subspace_v0 core: activation interchange at a reasoning-step boundary.

Hypothesis: is there a distributed internal variable at a fork's candidate-step
boundary whose interchange causally transfers correct-vs-wrong branch behaviour?

This module holds the primitives shared by the oracle gate (Phase 1b) and the DAS
subspace search (Phase 2):

  - a generation-safe forward hook that replaces the residual state at ONE boundary
    position during prefill only (a no-op during incremental decode, so the patched
    K/V is cached once and every generated token attends to it);
  - boundary-state extraction at a chosen layer;
  - generate-with-patch for free-generation rollouts.

Frozen constants and the fork/candidate machinery are reused verbatim from
src/analysis/transition_operator (SEP-id tokenization) and the cg free-gen prompt
format (scripts/causal_graph/cg_stage2_fg.fork_contexts), so the boundary token is
the trailing newline of the candidate step in both branches.

Design constraint (S6 Stage-0): a full-state boundary patch recovers the next-token
distribution (0.90-1.0) but ~0 of the immediately-elicited answer belief. The oracle
asks the untested question: does that next-token steering compound over FREE
generation into recovery of the correct final answer? Only a positive oracle makes
the Phase-2 subspace search worth the compute.
"""

from __future__ import annotations

import torch


def make_boundary_patch_hook(boundary_pos: int, patched_state: torch.Tensor):
    """Forward hook for a decoder block that overwrites the residual state at
    ``boundary_pos`` with ``patched_state`` during PREFILL only.

    Generation safety: incremental decode steps feed a single new token, so the
    layer output has sequence length 1 and never covers ``boundary_pos``; the hook
    is then a no-op and the already-patched K/V from prefill stays in the cache.
    """
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if hs.shape[1] > boundary_pos:  # prefill (or any pass covering the boundary)
            hs = hs.clone()
            hs[:, boundary_pos, :] = patched_state.to(hs.dtype).to(hs.device)
            return (hs,) + tuple(output[1:]) if is_tuple else hs
        return output
    return hook


def extract_boundary_state(model, input_ids: torch.Tensor, layer: int,
                           position: int) -> torch.Tensor:
    """Residual state ``hidden_states[layer]`` at ``position`` for a single
    sequence. ``layer`` is 1-based like ``output_hidden_states`` (index 0 is the
    embedding output, index i the output of block i), matching the hs_index
    convention in transition_operator.forward_with_boundary_patch. Returns a
    detached (d,) tensor."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[layer][0, position, :].detach()


def generate_with_patch(model, tok, device, prompt: str, k: int,
                        temperature: float, top_p: float, max_new_tokens: int,
                        layer: int | None = None,
                        patched_state: torch.Tensor | None = None) -> list[str]:
    """K sampled continuations of ``prompt``; if ``patched_state`` is given, the
    residual at ``layer`` and the prompt's last token is replaced during prefill.

    One prompt at a time (num_return_sequences=k), so the boundary position is
    unambiguously the final prompt token (no left-padding). ``layer`` is 1-based;
    the hook lands on model.model.layers[layer - 1] whose output is
    hidden_states[layer]."""
    enc = tok(prompt, return_tensors="pt").to(device)
    boundary_pos = enc["input_ids"].shape[1] - 1
    handle = None
    if patched_state is not None:
        if layer is None:
            raise ValueError("layer required with patched_state")
        block = model.model.layers[layer - 1]
        handle = block.register_forward_hook(
            make_boundary_patch_hook(boundary_pos, patched_state))
    try:
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens, num_return_sequences=k,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
    finally:
        if handle is not None:
            handle.remove()
    width = enc["input_ids"].shape[1]
    return tok.batch_decode(gen[:, width:], skip_special_tokens=True)


def fork_branch_prompts(tr: dict) -> dict[str, str]:
    """Correct- and wrong-branch free-gen prompts for one fork, in the exact
    cg_stage2_fg.fork_contexts string format (question + prefix steps + candidate
    step, newline-joined, trailing newline = the boundary token).

    correct = golden step tr['steps'][t] (the donor / source branch)
    wrong   = tr['wrong_step']            (the base branch to be rescued)"""
    t = tr["fork_t"]
    prefix = [tr["question"]] + tr["steps"][:t]
    return {
        "correct": "\n".join(prefix + [tr["steps"][t]]) + "\n",
        "wrong": "\n".join(prefix + [tr["wrong_step"]]) + "\n",
    }
