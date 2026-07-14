"""transition_operator_v0 core: answer candidates + patched frozen-decoder passes.

Spec: docs/transition_operator_v0_plan.md. Two halves:

1. Pure candidate machinery for Target B: answer typing, the frozen normalization
   function, distractor construction (priority: pre_generated > wrong-branch finals >
   corpus same-type > integer perturbations), wrong-final extraction from raw sessions.

2. Torch machinery for the frozen-decoder patch: a forward hook on the block whose
   output is hidden_states[L] replaces the boundary position's state over a FULL
   forward of the sequence. This realizes the plan's cache-surgery semantics exactly:
   blocks 1..L keep the original boundary, blocks L+1..n and every later (suffix)
   token see the patched boundary through its recomputed K/V. Patching a position
   with its own state must reproduce the unpatched logits (identity test).

The separator token is appended at the ID level (SEP_TOKEN_ID) rather than via string
concatenation, so the readout token is identical by construction even where BPE would
merge a trailing newline into the previous token.
"""

from __future__ import annotations

import random
import re

import torch

SEP_TOKEN_ID = 198  # "\n" in the Qwen2.5 tokenizer

ANSWER_TYPES = ("integer", "decimal", "fraction", "latex_expr", "has_letters", "other")


def answer_type(ans: str) -> str:
    a = ans.strip()
    if re.fullmatch(r"-?\d+", a):
        return "integer"
    if re.fullmatch(r"-?\d*\.\d+", a):
        return "decimal"
    if re.fullmatch(r"-?\d+\s*/\s*\d+", a) or "\\frac" in a:
        return "fraction"
    if "\\" in a or "^" in a:
        return "latex_expr"
    if re.search(r"[a-zA-Z]", a):
        return "has_letters"
    return "other"


def normalize_answer(ans: str, gold_uses_frac: bool = False) -> str:
    """Frozen normalization from the plan: whitespace, \\$ and \\text{} wrappers,
    a/b -> \\frac{a}{b} only when the gold uses \\frac."""
    a = ans.strip()
    a = re.sub(r"^\\\$", "", a)
    a = re.sub(r"\\\$$", "", a)
    m = re.fullmatch(r"\\text\{(.+)\}", a)
    if m:
        a = m.group(1).strip()
    if gold_uses_frac:
        m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", a)
        if m:
            a = f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    a = re.sub(r"\s+", " ", a)
    return a


def integer_perturbations(gold: str, rng: random.Random, n: int) -> list[str]:
    """Deterministic distractor fillers for integer golds: sign flip, +/-1, +/-2,
    digit swap, x10, order shuffled by rng."""
    g = int(gold)
    cands = {str(-g), str(g + 1), str(g - 1), str(g + 2), str(g - 2), str(g * 10)}
    digits = str(abs(g))
    if len(digits) >= 2:
        swapped = digits[1] + digits[0] + digits[2:]
        cands.add(("-" if g < 0 else "") + str(int(swapped)))
    cands.discard(str(g))
    out = sorted(cands)
    rng.shuffle(out)
    return out[:n]


def build_candidates(gold: str,
                     pre_generated: str | None = None,
                     wrong_finals: tuple[str, ...] | list[str] = (),
                     corpus_pool: tuple[str, ...] | list[str] = (),
                     k: int = 8,
                     seed: int = 0) -> list[str]:
    """Gold-first candidate list of length <= k, deduped after normalization.

    Distractor priority (plan section 6): pre_generated_answer, wrong-branch finals,
    same-type corpus answers, integer perturbations. Type match is enforced for
    sources 1-3; if the pool runs dry for non-integer golds the type constraint is
    relaxed on the corpus pool (last resort, so the set still has k entries).
    """
    rng = random.Random(seed)
    gold_frac = "\\frac" in gold
    gtype = answer_type(gold)
    gold_n = normalize_answer(gold, gold_frac)
    seen = {gold_n}
    out = [gold_n]

    def push(raw: str) -> None:
        if len(out) >= k:
            return
        c = normalize_answer(raw, gold_frac)
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    typed = [c for c in ([pre_generated] if pre_generated else [])
             if answer_type(c) == gtype]
    for c in typed:
        push(c)
    for c in wrong_finals:
        if answer_type(c) == gtype:
            push(c)
    pool_same = [c for c in corpus_pool if answer_type(c) == gtype]
    rng.shuffle(pool_same)
    for c in pool_same:
        push(c)
    if len(out) < k and gtype == "integer":
        for c in integer_perturbations(gold_n, rng, k):
            push(c)
    if len(out) < k:  # last resort: relax type on the corpus pool
        pool_any = list(corpus_pool)
        rng.shuffle(pool_any)
        for c in pool_any:
            push(c)
    return out


_ANSWER_RE = re.compile(r"#\s*Answer\s*\n+\s*(.+)")


def extract_wrong_finals(sample: dict) -> list[str]:
    """Final answers stated inside rating -1 completions of one raw PRM800K session
    (distractor source 2). Cheap regex on '# Answer' blocks; deduped, order stable."""
    label = sample.get("label")
    steps = label.get("steps") if isinstance(label, dict) else sample.get("steps")
    if not isinstance(steps, list):
        return []
    finals: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        for comp in step.get("completions") or []:
            if not isinstance(comp, dict) or comp.get("rating") != -1:
                continue
            text = comp.get("text")
            if not isinstance(text, str):
                continue
            for m in _ANSWER_RE.finditer(text):
                a = m.group(1).strip()
                if a and a not in seen:
                    seen.add(a)
                    finals.append(a)
    return finals


# ---------------------------------------------------------------------------
# Frozen-decoder patched forward
# ---------------------------------------------------------------------------

def _boundary_patch_hook(boundary_pos: int, patched_state: torch.Tensor):
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        hs = hs.clone()
        hs[:, boundary_pos, :] = patched_state.to(hs.dtype).to(hs.device)
        return (hs,) + tuple(output[1:]) if is_tuple else hs
    return hook


def forward_with_boundary_patch(model,
                                input_ids: torch.Tensor,
                                hs_index: int | None = None,
                                boundary_pos: int | None = None,
                                patched_state: torch.Tensor | None = None,
                                attention_mask: torch.Tensor | None = None,
                                ) -> torch.Tensor:
    """Full-sequence logits with hidden_states[hs_index] replaced by patched_state at
    boundary_pos (all batch rows). hs_index is 1-based like output_hidden_states, so
    the hook lands on model.model.layers[hs_index - 1]. patched_state=None -> plain
    forward. No grad; Stage 2 training will re-expose this with grad enabled."""
    handle = None
    if patched_state is not None:
        if hs_index is None or boundary_pos is None:
            raise ValueError("hs_index and boundary_pos required with patched_state")
        layer = model.model.layers[hs_index - 1]
        handle = layer.register_forward_hook(
            _boundary_patch_hook(boundary_pos, patched_state))
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        if handle is not None:
            handle.remove()
    return out.logits


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) over the last dim, computed in float32."""
    lp = torch.log_softmax(p_logits.float(), dim=-1)
    lq = torch.log_softmax(q_logits.float(), dim=-1)
    return float((lp.exp() * (lp - lq)).sum(dim=-1).mean())


def recovery_from_logits(actual: torch.Tensor, oracle: torch.Tensor,
                         pre: torch.Tensor) -> float:
    """Gate-5 recovery = 1 - KL(actual||oracle) / KL(actual||pre). 1 = full recovery,
    0 = no better than the unpatched pre distribution, < 0 = worse."""
    denom = kl_from_logits(actual, pre)
    if denom <= 0:
        return float("nan")
    return 1.0 - kl_from_logits(actual, oracle) / denom


def candidate_mean_logprobs(model,
                            context_ids: list[int],
                            cand_ids_list: list[list[int]],
                            pad_id: int,
                            device,
                            hs_index: int | None = None,
                            boundary_pos: int | None = None,
                            patched_state: torch.Tensor | None = None,
                            ) -> list[float]:
    """Mean per-token log-prob of each candidate continuation of context_ids, one
    batched (right-padded) forward for all candidates; optionally with the boundary
    patch active. Right padding is safe: causal attention never lets real positions
    see the pads."""
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
    logits = forward_with_boundary_patch(
        model, input_ids.to(device), hs_index=hs_index, boundary_pos=boundary_pos,
        patched_state=patched_state, attention_mask=mask.to(device))
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for i, (lo, hi) in enumerate(spans):
        tok_ids = input_ids[i, lo:hi].to(device)
        lp = logprobs[i, lo - 1:hi - 1, :].gather(-1, tok_ids[:, None]).squeeze(-1)
        out.append(float(lp.mean()))
    return out


def belief_from_scores(scores: list[float]) -> torch.Tensor:
    """Softmax belief over the candidate set from mean-token log-probs."""
    return torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=-1)


def gold_margin(scores: list[float]) -> float:
    """logp_gold - max distractor logp; gold is index 0 by construction."""
    return scores[0] - max(scores[1:])
