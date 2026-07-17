"""cot_causal_graph_v0 core: trace construction, edge math, taxonomy.

Spec: docs/cot_causal_graph_v0_plan.md. Three separable per-step quantities on real
CoT traces: Detection (probe), Influence (intervention effect on downstream text
likelihood / answer margin / rollout accuracy), Repair (causal recovery by a later
step). Two edge families: teacher-forced (fixed continuation) and free-generation
(step -> answer only).

Reuses the S6 frozen machinery verbatim (separator-id tokenization, candidate set,
frozen elicitation suffix, batched candidate scoring): src/analysis/transition_operator.
Boundary-state patching is deliberately absent (pre-falsified for answer-level
effects, S6 Stage 0), and downstream probe-score deltas are diagnostics, never
causal evidence (S3 Stage 5).
"""

from __future__ import annotations

import math
import random

import torch

from src.analysis.transition_operator import (
    SEP_TOKEN_ID,
    candidate_mean_logprobs,
    gold_margin,
)

ELICITATION_SUFFIX = "\nSo the final answer is"  # S6 Stage-0 winner, frozen

TAXONOMY = ("detected_influential", "detected_inert",
            "undetected_influential", "undetected_inert")


# ---------------------------------------------------------------------------
# Trace construction (pure)
# ---------------------------------------------------------------------------

def join_forks_to_golden(forks: list[dict], goldens: list[dict],
                         min_downstream: int = 2,
                         counters: dict | None = None) -> list[dict]:
    """Join fork rows to a complete golden trajectory of the same session.

    Key: exact question text; the golden trajectory must extend the fork's golden
    prefix through the fork position with >= min_downstream further steps. Fork
    ``step_index`` is 1-based, so the 0-based fork position is t = step_index - 1;
    the intervention replaces golden steps[t] and steps[t+1:] stay teacher-forced.
    ``alt_pos_step`` is the fork's +1 side when it differs from the golden step
    (paraphrase-level control), else None. One golden per fork (first match).
    """
    if counters is None:
        counters = {}
    counters.setdefault("no_golden_for_question", 0)
    counters.setdefault("prefix_mismatch", 0)
    counters.setdefault("too_short_downstream", 0)
    counters.setdefault("joined", 0)
    by_q: dict[str, list[dict]] = {}
    for g in goldens:
        by_q.setdefault(g["question"], []).append(g)

    out: list[dict] = []
    for fk in forks:
        t = int(fk["step_index"]) - 1
        cands = by_q.get(fk["question"])
        if not cands:
            counters["no_golden_for_question"] += 1
            continue
        matched = None
        saw_short = False
        for g in cands:
            steps = g["steps"]
            if steps[:t] != fk["prefix_steps"]:
                continue
            if len(steps) < t + 1 + min_downstream:
                saw_short = True
                continue
            matched = g
            break
        if matched is None:
            counters["too_short_downstream" if saw_short
                     else "prefix_mismatch"] += 1
            continue
        golden_step = matched["steps"][t]
        alt = fk["correct"] if fk["correct"] != golden_step else None
        counters["joined"] += 1
        out.append({
            "trace_id": f'{fk["fork_id"]}::{matched["trajectory_id"]}',
            "arm": "forks",
            "question": fk["question"],
            "steps": list(matched["steps"]),
            "fork_t": t,
            "wrong_step": fk["wrong"],
            "alt_pos_step": alt,
            "gt_answer": matched.get("gt_answer") or fk.get("gt_answer"),
            "pre_generated_answer": fk.get("pre_generated_answer"),
            "wrong_finals": fk.get("wrong_finals") or [],
            "phase": fk.get("phase"),
        })
    return out


def length_matched_step(rng: random.Random, pool: list[tuple[str, str]],
                        target_words: int, exclude_key: str,
                        top_n: int = 5) -> str:
    """Off-topic control step: from a (key, text) pool, drop entries whose key
    matches ``exclude_key``, rank by |word-count - target_words|, and pick
    uniformly among the top_n closest. Deterministic given rng."""
    eligible = [(k, s) for k, s in pool if k != exclude_key]
    if not eligible:
        raise ValueError("empty control pool after exclusion")
    ranked = sorted(eligible, key=lambda ks: (abs(len(ks[1].split()) - target_words),
                                              ks[0], ks[1]))
    return rng.choice(ranked[:top_n])[1]


# ---------------------------------------------------------------------------
# Tokenization layout (pure given a tokenizer)
# ---------------------------------------------------------------------------

def encode_pieces(tok, pieces: list[str]) -> list[list[int]]:
    """Per-piece token ids, each piece tokenized separately (the S6 rule: string
    joining would let BPE merge newlines and break span identities)."""
    return [tok(p, add_special_tokens=False)["input_ids"] for p in pieces]


def assemble_ids(piece_ids: list[list[int]]) -> tuple[list[int],
                                                      list[tuple[int, int]],
                                                      list[int]]:
    """Flat ids with a SEP after every piece, plus per-piece content spans and
    boundary (SEP) positions.

    Returns (full_ids, spans, boundaries): spans[i] = (lo, hi) half-open span of
    piece i's content tokens in full_ids; boundaries[i] = index of the SEP token
    that closes piece i (the boundary readout position of S6)."""
    full: list[int] = []
    spans: list[tuple[int, int]] = []
    boundaries: list[int] = []
    for ids in piece_ids:
        lo = len(full)
        full.extend(ids)
        spans.append((lo, len(full)))
        boundaries.append(len(full))
        full.append(SEP_TOKEN_ID)
    return full, spans, boundaries


def cand_token_ids(tok, suffix: str, answers: list[str]) -> list[list[int]]:
    """Candidate continuations tokenized after the suffix (S6 to_stage0 rule:
    a single leading space unless the suffix already ends in whitespace)."""
    lead = "" if suffix[-1].isspace() else " "
    return [tok(lead + a, add_special_tokens=False)["input_ids"] for a in answers]


# ---------------------------------------------------------------------------
# Teacher-forced measurements (torch, model-agnostic)
# ---------------------------------------------------------------------------

def per_span_mean_logprob(logits: torch.Tensor, input_ids: torch.Tensor,
                          spans: list[tuple[int, int]]) -> list[float]:
    """Teacher-forced mean token log-prob per span. logits/input_ids are a single
    sequence (L, V) and (L,). Position p's token is predicted by logits at p-1,
    so a span starting at 0 contributes from its second token on."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for lo, hi in spans:
        lo_eff = max(lo, 1)
        if hi <= lo_eff:
            out.append(float("nan"))
            continue
        tgt = input_ids[lo_eff:hi]
        lp = logprobs[lo_eff - 1:hi - 1, :].gather(-1, tgt[:, None]).squeeze(-1)
        out.append(float(lp.mean()))
    return out


def entropy_at(logits: torch.Tensor, positions: list[int]) -> list[float]:
    """Next-token entropy (nats) of the distribution AT each position."""
    out = []
    for p in positions:
        lp = torch.log_softmax(logits[p].float(), dim=-1)
        out.append(float(-(lp.exp() * lp).sum()))
    return out


def probe_logits_at(hidden: torch.Tensor, positions: list[int],
                    w: torch.Tensor, b: float) -> list[float]:
    """Linear probe readout w.h + b at given positions of a (L, d) hidden-state
    tensor. Positive = incorrect, matching the deployed probe convention."""
    w = w.to(hidden.dtype).to(hidden.device)
    return [float(hidden[p] @ w + b) for p in positions]


def margin_at_boundary(model, ctx_ids: list[int], suffix_ids: list[int],
                       cand_ids: list[list[int]], pad_id: int, device) -> float:
    """Gold answer margin read at one boundary: gold mean-logprob minus best
    distractor, candidates scored after ctx + frozen suffix in one batched
    forward (S6 machinery)."""
    scores = candidate_mean_logprobs(model, ctx_ids + suffix_ids, cand_ids,
                                     pad_id, device)
    return gold_margin(scores)


# ---------------------------------------------------------------------------
# Free-generation statistics (pure)
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def solve_curve(grades_per_prefix: list[list[bool]]) -> list[dict]:
    """Per-prefix solve rate s_i with Wilson CI from K graded rollouts each."""
    out = []
    for i, g in enumerate(grades_per_prefix):
        k, n = sum(bool(x) for x in g), len(g)
        lo, hi = wilson_ci(k, n)
        out.append({"prefix": i, "n": n, "solve_rate": (k / n) if n else float("nan"),
                    "ci_lo": lo, "ci_hi": hi})
    return out


def localize_drops(s_curve: list[float], min_drop: float = 0.25) -> list[int]:
    """Indices i where the solve rate falls by >= min_drop from prefix i to i+1,
    i.e. step i+1 is where the trajectory becomes doomed (largest first)."""
    drops = [(s_curve[i] - s_curve[i + 1], i) for i in range(len(s_curve) - 1)]
    hits = [(d, i) for d, i in drops if d >= min_drop]
    return [i for _, i in sorted(hits, key=lambda x: (-x[0], x[1]))]


def fg_influence(base_grades: list[bool], interv_grades: list[bool]) -> dict:
    """Free-generation step->answer edge: solve-rate delta with a CI from the
    difference of independent Wilson intervals (conservative)."""
    kb, nb = sum(base_grades), len(base_grades)
    ki, ni = sum(interv_grades), len(interv_grades)
    sb = kb / nb if nb else float("nan")
    si = ki / ni if ni else float("nan")
    b_lo, b_hi = wilson_ci(kb, nb)
    i_lo, i_hi = wilson_ci(ki, ni)
    return {"base_rate": sb, "interv_rate": si, "delta": si - sb,
            "delta_ci_lo": i_lo - b_hi, "delta_ci_hi": i_hi - b_lo,
            "recovery_rate": si}


# ---------------------------------------------------------------------------
# Taxonomy + calibration (pure)
# ---------------------------------------------------------------------------

def null_quantile(control_deltas: list[float], q: float = 0.95) -> float:
    """Magnitude threshold from matched-control edges: the q-quantile of |delta|
    (linear interpolation). Real edges must exceed this to count as influential."""
    if not control_deltas:
        return float("nan")
    vals = sorted(abs(v) for v in control_deltas)
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (pos - lo) * (vals[hi] - vals[lo])


def classify_site(detected: bool, influential: bool) -> str:
    """The 2x2 cell of one error site (repair is a separate overlay)."""
    return (("detected_" if detected else "undetected_")
            + ("influential" if influential else "inert"))


def is_influential_fg(edge: dict) -> bool:
    """Influence call from a free-generation edge: solve rate dropped and the
    conservative CI of the delta excludes zero."""
    return edge["delta"] < 0 and edge["delta_ci_hi"] < 0


def is_influential_tf(delta_margin: float, null_thresh: float) -> bool:
    """Influence call from a teacher-forced answer edge vs the control null."""
    if math.isnan(null_thresh):
        return False
    return abs(delta_margin) > null_thresh
