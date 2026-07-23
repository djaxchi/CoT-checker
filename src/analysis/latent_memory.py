"""latent_memory_v0: per-trace capacity oracle for CoT -> latent-memory compression.

Question: how few FREE residual vectors, injected at layer L in place of a CoT's step
tokens, let the frozen base model reproduce the full-CoT answer belief? This is the
oracle-first gate before any amortised compressor (see docs/latent_memory_v0_plan.md).

Design rule (as in transition_operator_v0): every component that CAN be the frozen model
IS the frozen model. The only trained parameters are the m latent vectors z, optimised
per trace. Injection reuses das_span.make_span_patch_hook (overwrites a residual span at
layer L; upper blocks recompute K/V during prefill; gradient flows to z). The behavioural
readout reuses the S6 elicitation-suffix candidate machinery.

The latent span sits inside the shared context (before the elicitation suffix), so a
single forward hook patches every candidate row at once, exactly as span_candidate_logprobs
does for whole-step-span patching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.analysis.das_span import make_span_patch_hook
from src.analysis.transition_operator import SEP_TOKEN_ID, belief_from_scores


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def latent_context_ids(question_ids: list[int], m: int, placeholder_id: int,
                       sep_id: int = SEP_TOKEN_ID) -> tuple[list[int], int, int]:
    """Skeleton `[question] SEP [P]*m SEP`; the m placeholder positions are the latent
    span. Returns (full_ids, lo, hi) with [lo, hi) the half-open latent span. The
    question stays as real tokens; only the CoT is replaced by the m latent slots."""
    if m < 1:
        raise ValueError("m must be >= 1")
    full = list(question_ids) + [sep_id]
    lo = len(full)
    full += [placeholder_id] * m
    hi = len(full)
    full += [sep_id]
    return full, lo, hi


def full_cot_context_ids(question_ids: list[int], step_ids: list[list[int]],
                         sep_id: int = SEP_TOKEN_ID) -> list[int]:
    """Teacher context `[question] SEP [step_1] SEP ... [step_n] SEP` (SEP-id scheme)."""
    full = list(question_ids) + [sep_id]
    for s in step_ids:
        full += list(s) + [sep_id]
    return full


def no_cot_context_ids(question_ids: list[int], sep_id: int = SEP_TOKEN_ID) -> list[int]:
    """Floor context `[question] SEP` (no reasoning)."""
    return list(question_ids) + [sep_id]


# ---------------------------------------------------------------------------
# Pooling baselines / initialisation
# ---------------------------------------------------------------------------

def chunk_pool_states(step_states: torch.Tensor, m: int,
                      mode: str = "mean") -> torch.Tensor:
    """Pool a (T, d) stack of step-token residuals into (m, d) by splitting the T
    positions into m contiguous chunks and pooling each. mode in {"mean","max"}.

    Contiguous chunking preserves coarse token order (unlike global max pooling), and
    the result is both the fixed pooling baseline at width m and a sensible init for the
    optimiser. When T < m the last rows repeat the final chunk."""
    if step_states.dim() != 2:
        raise ValueError("step_states must be (T, d)")
    t = step_states.shape[0]
    if t == 0:
        raise ValueError("no step states to pool")
    bounds = [round(i * t / m) for i in range(m + 1)]
    out = []
    for i in range(m):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:  # empty chunk (T < m): reuse the last non-empty position
            a, b = max(0, min(a, t - 1)), max(1, min(a + 1, t))
        chunk = step_states[a:b]
        out.append(chunk.mean(0) if mode == "mean" else chunk.amax(0))
    return torch.stack(out, 0)


# ---------------------------------------------------------------------------
# Grad-enabled candidate scoring under a latent patch
# ---------------------------------------------------------------------------

def _patch_module(model, layer: int):
    """Module whose forward output carries the latent span. layer==0 -> the token
    embedding (soft-input-token / gist-style injection, full-stack control); layer>=1
    -> output of block `layer` (residual injection, blocks 1..layer-1 uncontrolled)."""
    if layer == 0:
        return model.model.embed_tokens
    return model.model.layers[layer - 1]


def candidate_scores_grad(model, context_ids: list[int], cand_ids_list: list[list[int]],
                          pad_id: int, device, layer: int, lo: int, hi: int,
                          states: torch.Tensor) -> torch.Tensor:
    """Mean per-token log-prob of each candidate continuation of context_ids, one
    batched right-padded forward, with the latent span [lo, hi) at `layer` overwritten
    by `states` (m, d). Returns a (num_cand,) tensor WITH grad to `states`.

    Grad-enabled twin of das_span.span_candidate_logprobs (which runs under no_grad).
    layer==0 injects at the embedding (soft input tokens); layer>=1 hooks the output of
    block `layer` (1-based), the das_span residual-patch convention."""
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
    handle = _patch_module(model, layer).register_forward_hook(
        make_span_patch_hook(lo, hi, states))
    try:
        logits = model(input_ids=input_ids.to(device),
                       attention_mask=mask.to(device)).logits
    finally:
        handle.remove()
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for i, (clo, chi) in enumerate(spans):
        tok_ids = input_ids[i, clo:chi].to(device)
        lp = logprobs[i, clo - 1:chi - 1, :].gather(-1, tok_ids[:, None]).squeeze(-1)
        out.append(lp.mean())
    return torch.stack(out)


def gold_margin_t(scores: torch.Tensor) -> torch.Tensor:
    """logp_gold - max distractor logp (gold index 0), differentiable."""
    return scores[0] - scores[1:].max()


# ---------------------------------------------------------------------------
# Per-trace optimiser
# ---------------------------------------------------------------------------

@dataclass
class OracleResult:
    z: torch.Tensor                      # optimised latent (m, d), detached
    scores: list[float]                  # final candidate mean-logprobs
    margin: float                        # gold margin under the optimised latent
    loss_history: list[float] = field(default_factory=list)


def optimize_latent(model, context_ids: list[int], lo: int, hi: int, layer: int,
                    cand_ids_list: list[list[int]], suffix_ids: list[int],
                    teacher_belief: torch.Tensor, init_states: torch.Tensor,
                    pad_id: int, device, epochs: int = 60, lr: float = 5e-2
                    ) -> OracleResult:
    """Optimise the m latent vectors (frozen model) to match the teacher belief over the
    candidate set (KL(teacher || student)). context_ids already contains the m placeholder
    slots at [lo, hi); the elicitation suffix is spliced in right after the context so the
    latent span stays inside the shared prefix that the single patch hook covers.

    teacher_belief is the full-CoT belief over the SAME candidate order (softmax of the
    teacher's mean-logprobs). init_states (m, d) seeds z (e.g. chunk-mean-pooled steps)."""
    scoring_ctx = list(context_ids) + list(suffix_ids)
    z = torch.nn.Parameter(init_states.detach().to(device).float().clone())
    teacher_belief = teacher_belief.detach().to(device).float()
    opt = torch.optim.Adam([z], lr=lr)
    history: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        scores = candidate_scores_grad(model, scoring_ctx, cand_ids_list, pad_id,
                                       device, layer, lo, hi, z)
        student_logbelief = torch.log_softmax(scores, dim=-1)
        loss = torch.sum(teacher_belief * (teacher_belief.clamp_min(1e-9).log()
                                           - student_logbelief))
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        final = candidate_scores_grad(model, scoring_ctx, cand_ids_list, pad_id,
                                      device, layer, lo, hi, z)
    scores_l = [float(s) for s in final]
    return OracleResult(z=z.detach(), scores=scores_l,
                        margin=scores_l[0] - max(scores_l[1:]),
                        loss_history=history)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def recovery(margin_latent: float, margin_no_cot: float, margin_full_cot: float) -> float:
    """Fraction of the full-CoT answer-belief benefit recovered by the latent memory:
    (margin_latent - margin_no_cot) / (margin_full_cot - margin_no_cot). NaN when the
    full-CoT benefit is degenerate (denominator ~0)."""
    denom = margin_full_cot - margin_no_cot
    if abs(denom) < 1e-6:
        return float("nan")
    return (margin_latent - margin_no_cot) / denom


# ---------------------------------------------------------------------------
# Memory-swap (test B) + trace-probe (test A) helpers
# ---------------------------------------------------------------------------

def joint_candidate_texts(b_gold: str, a_gold: str, b_distractors: list[str],
                          a_distractors: list[str], k: int = 8) -> tuple[list[str], int, int]:
    """Candidate list for a swap pair: recipient gold, donor gold, then distractors from
    both traces, deduped. Returns (texts, idx_b_gold, idx_a_gold). If the two golds
    normalise to the same string the pair is unusable and idx_a == idx_b."""
    from src.analysis.transition_operator import normalize_answer
    gold_frac = "\\frac" in b_gold
    b_n = normalize_answer(b_gold, gold_frac)
    a_n = normalize_answer(a_gold, gold_frac)
    out, seen = [b_n], {b_n}
    idx_a = 0 if a_n == b_n else -1
    if idx_a == -1:
        out.append(a_n); seen.add(a_n); idx_a = 1
    for raw in list(b_distractors) + list(a_distractors):
        if len(out) >= k:
            break
        c = normalize_answer(raw, gold_frac)
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out, 0, idx_a


def belief_masses(scores: list[float], idx_b: int, idx_a: int) -> tuple[float, float]:
    """Softmax belief over the candidate set; return (mass on recipient gold b,
    mass on donor gold a)."""
    b = belief_from_scores(scores)
    return float(b[idx_b]), float(b[idx_a])


def random_like(state: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Gaussian control matched to `state`'s per-vector norm: an injection of the same
    magnitude but no learned content, to show any donor pull is content-specific."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(state.shape, generator=g).to(state.dtype).to(state.device)
    scale = state.norm(dim=-1, keepdim=True) / r.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return r * scale


def donor_win(mass_a: float, mass_b: float) -> bool:
    """The injected donor memory made the donor's answer outrank the recipient's own."""
    return mass_a > mass_b


# Trace-probe (test A): recall of an intermediate value, a question DIFFERENT from the
# final answer, with a clean gold extracted straight from the reasoning text.
PROBE_SUFFIX = "\nAn intermediate value computed in the solution is"

import re as _re

_INT_RE = _re.compile(r"-?\d+")


def extract_ints(text: str) -> list[str]:
    """Integer literals in `text`, in order (used for probe gold/distractor mining)."""
    return _INT_RE.findall(text)


def pick_probe_target(question: str, steps: list[str], gt_answer: str,
                      min_abs: int = 3) -> tuple[str, int] | None:
    """Pick an intermediate integer to probe: it must appear in a middle step, not in the
    question, differ from the final answer, and have |value| >= min_abs (skip 0/1/2, too
    guessable). Returns (gold_str, step_idx) or None if no clean target exists.

    Middle steps only (exclude the first step and the last two, which usually state the
    answer), so the target is genuinely intermediate reasoning, not the conclusion. Among
    the qualifying integers the most distinctive (largest |value|) is chosen, so the gold
    is less guessable from the question and the floor stays low."""
    if len(steps) < 4:
        return None
    q_ints = set(extract_ints(question))
    ans_ints = set(extract_ints(gt_answer))
    best = None  # (abs_value, step_idx, tok_s)
    for idx in range(1, len(steps) - 2):
        for tok_s in extract_ints(steps[idx]):
            if tok_s in q_ints or tok_s in ans_ints:
                continue
            try:
                v = abs(int(tok_s))
            except ValueError:
                continue
            if v < min_abs:
                continue
            if best is None or v > best[0]:
                best = (v, idx, tok_s)
    return (best[2], best[1]) if best else None


def pick_probe_targets(question: str, steps: list[str], gt_answer: str,
                       n_targets: int = 3, min_abs: int = 3,
                       cue_words: int = 8) -> list[dict]:
    """Up to n_targets distinguishable intermediate-value probes, one per distinct middle
    step, spread across the trace. Each probe is anchored to the phrase immediately before
    the number in its step (its "cue"), so different probes are distinct queries: the model
    must complete "<cue>" with the value that followed it in the reasoning. Returns dicts
    {gold, step_idx, cue}. Skips numbers in the question/answer, |value| < min_abs, and any
    number with no textual cue before it (start of step)."""
    if len(steps) < 4:
        return []
    q_ints = set(extract_ints(question))
    ans_ints = set(extract_ints(gt_answer))
    out: list[dict] = []
    used_steps: set[int] = set()
    used_golds: set[str] = set()
    for idx in range(1, len(steps) - 2):
        if idx in used_steps:
            continue
        step = steps[idx]
        for m in _INT_RE.finditer(step):
            tok_s = m.group()
            if tok_s in q_ints or tok_s in ans_ints or tok_s in used_golds:
                continue
            try:
                if abs(int(tok_s)) < min_abs:
                    continue
            except ValueError:
                continue
            cue = " ".join(step[: m.start()].split()[-cue_words:]).strip()
            if len(cue) < 3 or not any(c.isalpha() for c in cue):
                continue
            out.append({"gold": tok_s, "step_idx": idx, "cue": cue})
            used_steps.add(idx); used_golds.add(tok_s)
            break
        if len(out) >= n_targets:
            break
    return out


def multi_target_scores(model, prepped: list[tuple[list[int], list[list[int]]]],
                        pad_id: int, device, layer: int, lo: int, hi: int,
                        states: torch.Tensor) -> list[torch.Tensor]:
    """Mean per-token log-probs for several queries in ONE batched forward. `prepped` is a
    list of (context_ids, cand_ids_list); every context shares the SAME latent span [lo,hi)
    (same absolute positions, since the query suffix comes after it), so a single patch hook
    covers all rows. Returns one (num_cand,) grad tensor per query. This is the batched twin
    of candidate_scores_grad across queries, the hot path for the joint optimiser."""
    rows, meta = [], []  # meta[i] = (query_idx, clo, chi)
    for qj, (ctx, cand_ids) in enumerate(prepped):
        for cand in cand_ids:
            rows.append(ctx + cand)
            meta.append((qj, len(ctx), len(ctx) + len(cand)))
    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, :len(r)] = 1
    handle = _patch_module(model, layer).register_forward_hook(
        make_span_patch_hook(lo, hi, states))
    try:
        logits = model(input_ids=input_ids.to(device),
                       attention_mask=mask.to(device)).logits
    finally:
        handle.remove()
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    per_query: list[list[torch.Tensor]] = [[] for _ in prepped]
    for i, (qj, clo, chi) in enumerate(meta):
        tok_ids = input_ids[i, clo:chi].to(device)
        lp = logprobs[i, clo - 1:chi - 1, :].gather(-1, tok_ids[:, None]).squeeze(-1)
        per_query[qj].append(lp.mean())
    return [torch.stack(x) for x in per_query]


def optimize_latent_multi(model, context_ids: list[int], lo: int, hi: int, layer: int,
                          targets: list[dict], init_states: torch.Tensor, pad_id: int,
                          device, epochs: int = 60, lr: float = 5e-2) -> dict:
    """Optimise ONE latent to satisfy several queries at once. Each target is a dict
    {cand_ids, suffix_ids, teacher_belief}; the loss sums KL(teacher || student) across
    targets, so the single latent must answer every query. Returns {z, margins, scores,
    loss_history} with per-target final margins (gold index 0). One batched forward per
    epoch (multi_target_scores), so cost is ~independent of the number of queries."""
    z = torch.nn.Parameter(init_states.detach().to(device).float().clone())
    prepped = [(list(context_ids) + list(t["suffix_ids"]), t["cand_ids"]) for t in targets]
    beliefs = [t["teacher_belief"].detach().to(device).float() for t in targets]
    opt = torch.optim.Adam([z], lr=lr)
    history: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        scores = multi_target_scores(model, prepped, pad_id, device, layer, lo, hi, z)
        loss = z.new_zeros(())
        for s, tb in zip(scores, beliefs):
            logbelief = torch.log_softmax(s, dim=-1)
            loss = loss + torch.sum(tb * (tb.clamp_min(1e-9).log() - logbelief))
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        final = multi_target_scores(model, prepped, pad_id, device, layer, lo, hi, z)
    all_scores = [[float(x) for x in s] for s in final]
    margins = [s[0] - max(s[1:]) for s in all_scores]
    return {"z": z.detach(), "margins": margins, "scores": all_scores,
            "loss_history": history}
