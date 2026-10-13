"""factorized_step_labels_v1 core: downstream risk and necessity of one step.

Plan: docs/factorized_step_labels_v1_plan.md.

Everything here works on ReProbe's released token ids, never on re-tokenized
text: the trajectories were sampled under `reprobe_prompt` and the reply's ids
are in each row, so prefixes, step spans and continuations are exact. The
pieces:

  - token-exact parsing of a released row into prompt, step-line spans and the
    answer line (`parse_reprobe_row`);
  - answer extraction for the "<Answer>: X" format, handed to the project
    grader (`src/eval/math_grade.py`) with the per-fork answer-identity
    harmonization of regrade_trajectories.py;
  - downstream-risk statistics from per-continuation errors;
  - the position-matched mean baseline and the norm-matched random control;
  - teacher-forced continuation log-probs under a span patch, reusing
    `make_span_patch_hook` from das_span, with a landing check.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np

from src.eval.math_grade import (answer_key, extract_final_answer, is_equiv,
                                 last_boxed_only_string, remove_boxed)

STEP_LINE = re.compile(r"^\s*-\s*Step\s+\d+\s*:")
ANSWER_LINE = re.compile(r"^\s*<\s*Answer\s*>", re.I)
ANSWER_CAPTURE = re.compile(r"^\s*<\s*Answer\s*>\s*:?\s*(.*?)\s*$", re.I | re.M)
POSITIONS = ("early", "mid", "late")


# --------------------------------------------------------------------------- #
# Token-exact parsing of a released row
# --------------------------------------------------------------------------- #

def marker_len(line_ids: Sequence[int], decode: Callable[[list[int]], str]) -> int:
    """Number of leading tokens of a step line that spell its "- Step k:" marker.

    The content starts right after the token that completes the colon. Returns 0
    when the line has no marker or nothing follows it.
    """
    for j in range(1, len(line_ids)):
        if STEP_LINE.match(decode(list(line_ids[:j]))):
            return j
    return 0


def parse_reprobe_row(input_ids: Sequence[int], prompt_ids: Sequence[int],
                      claims: Sequence[dict], decode: Callable[[list[int]], str],
                      eos_id: int) -> tuple[dict | None, str]:
    """Split one released row into prompt, step lines and answer line, by token.

    `claims` are the row's claims (`aligned_token_ids` reply-relative, `sentence`).
    Kept only if: the prompt ids match `prompt_ids`; the reply contains an EOS
    (the model stopped on its own); the step-line claims tile the reply from 0
    contiguously and each decodes byte-identically to its sentence; the tokens
    between the last step line and the EOS decode to an "<Answer>" line. Returns
    (parsed, "ok") or (None, reason).

    Spans are half-open and reply-relative. `step_claim_idx[s]` is the claim (and
    manifest label) index of step line s; `content_lo[s]` skips the marker.
    """
    ids = [int(x) for x in input_ids]
    P = len(prompt_ids)
    if ids[:P] != [int(x) for x in prompt_ids]:
        return None, "prompt_mismatch"
    rep = ids[P:]
    if eos_id not in rep:
        return None, "no_eos"
    eos = rep.index(eos_id)
    spans, cidx, cursor = [], [], 0
    for k, c in enumerate(claims):
        al = c.get("aligned_token_ids")
        al = [] if al is None else [int(x) for x in al]
        sent = c.get("sentence") or ""
        if not STEP_LINE.match(sent):
            if ANSWER_LINE.match(sent):
                break                           # the answer claim ends the step lines
            return None, "non_step_claim"
        if not al or al != list(range(cursor, cursor + len(al))):
            return None, "claims_not_tiling"
        if decode(rep[al[0]:al[-1] + 1]) != sent:
            return None, "sentence_decode_mismatch"
        spans.append((al[0], al[-1] + 1)); cidx.append(k); cursor = al[-1] + 1
    if not spans:
        return None, "no_step_lines"
    if any(STEP_LINE.match(c.get("sentence") or "") for c in claims[len(spans):]):
        return None, "step_after_answer"
    ans_text = decode(rep[cursor:eos])
    if cursor >= eos or not ANSWER_LINE.match(ans_text):
        return None, "no_answer_line"
    content_lo = []
    for lo, hi in spans:
        m = marker_len(rep[lo:hi], decode)
        if m == 0 or lo + m >= hi:
            return None, "no_step_marker"
        content_lo.append(lo + m)
    return {"prompt_len": P, "reply_ids": rep[:eos + 1], "step_spans": spans,
            "content_lo": content_lo, "step_claim_idx": cidx,
            "answer_span": (cursor, eos), "answer_text": ans_text}, "ok"


def position_bucket(s: int, n_steps: int) -> str:
    """Thirds of the trace's step lines: early, mid, late."""
    return POSITIONS[min(2, (3 * s) // n_steps)]


def fork_order_key(trace_id: str, step: int) -> str:
    return hashlib.sha1(f"{trace_id}:{step}".encode()).hexdigest()


def select_forks(candidates: list[dict], per_cell: int,
                 cells: Sequence[tuple[str, str]]) -> list[dict]:
    """Fill each (validity, position) cell with `per_cell` forks, one per problem.

    Candidates are visited in sha1(trace_id, step) order, cell by cell in the
    given order, and a problem used by an earlier pick is skipped. Raises if a
    cell cannot be filled, so a short stratum is visible rather than silently
    rebalanced.
    """
    used: set[str] = set()
    out = []
    order = sorted(candidates, key=lambda c: fork_order_key(c["trace_id"], c["step"]))
    for cell in cells:
        got = 0
        for c in order:
            if (c["validity"], c["position"]) != cell or c["problem_key"] in used:
                continue
            out.append(c); used.add(c["problem_key"]); got += 1
            if got == per_cell:
                break
        if got < per_cell:
            raise ValueError(f"cell {cell}: only {got} of {per_cell} forks available")
    return out


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def gold_answer(solution: str | None) -> str | None:
    """The boxed answer of a ground-truth solution, or None."""
    if not isinstance(solution, str) or not solution:
        return None
    b = last_boxed_only_string(solution)
    return remove_boxed(b) if b is not None else None


def extract_reprobe_answer(reply: str) -> str | None:
    """Final answer of a reply in ReProbe's format.

    Reads the last "<Answer>: X" line; unwraps a \\boxed{} or one $...$ pair and
    a trailing period. Without such a line, defers to the project extractor. The
    project extractor alone would read "<Answer>: 3/4" as "4" (it falls back to
    the last number), which is why this exists.
    """
    hits = ANSWER_CAPTURE.findall(reply)
    if not hits:
        return extract_final_answer(reply)
    s = hits[-1].strip()
    b = last_boxed_only_string(s)
    if b is not None:
        return remove_boxed(b)
    s = s.rstrip(".").strip()
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s.strip("$").strip()
    return plain_to_latex(s) or None


# ReProbe's replies write answers as plain text ("4√5", "(4, ∞)", "20/1296 = 5/324",
# "(A) The coefficient of x^2"), where MATH gold is LaTeX. Measured on the v2 dev
# traces both annotators label error-free, these spellings were most of the
# grader's misses. Spelling only: equivalence stays math_grade's.
_UNICODE = (("√", "\\sqrt"), ("∞", "\\infty"), ("π", "\\pi"), ("×", "\\times"),
            ("≤", "\\le"), ("≥", "\\ge"), ("°", ""), ("−", "-"))
_CHOICE = re.compile(r"^\(?([A-E])\)?(?:[\s.:]|$)")


def plain_to_latex(s: str) -> str:
    m = _CHOICE.match(s)
    if m:
        return f"({m.group(1)})"
    if "=" in s and not re.search(r"[<>]", s):
        rhs = s.rsplit("=", 1)[1].strip()
        s = rhs or s
    s = _SQRT_ARG.sub(lambda m: "\\sqrt{" + (m.group(1) or m.group(2)) + "}", s)
    for u, t in _UNICODE:
        s = s.replace(u, t)
    return s.strip()


# "√10" and "√(x+1)" need braces: math_grade's _fix_sqrt braces one character only.
_SQRT_ARG = re.compile(r"√\(([^()]*)\)|√(\d+|[A-Za-z])")


def grade_fork(replies: list[dict], gold: str) -> list[dict]:
    """Grade every continuation of one fork, sharing one grade per answer identity.

    Each item needs `text` (the whole reply) and `truncated`. A truncated or
    unparseable reply is an error. Within the fork, replies with the same
    `answer_key` are correct if any of them matches the gold (§21.13), so two
    spellings of one answer never get different grades.
    """
    out = []
    for r in replies:
        pred = None if r["truncated"] else extract_reprobe_answer(r["text"])
        ok = pred is not None and bool(is_equiv(pred, gold))
        out.append({"pred": pred, "answer_key": answer_key(pred) if pred is not None else None,
                    "correct_raw": ok, "gradeable": pred is not None})
    group: dict[str, bool] = defaultdict(bool)
    for g in out:
        if g["answer_key"] is not None:
            group[g["answer_key"]] |= g["correct_raw"]
    for g in out:
        g["correct"] = bool(group[g["answer_key"]]) if g["answer_key"] is not None else False
    return out


# --------------------------------------------------------------------------- #
# Downstream risk
# --------------------------------------------------------------------------- #

def risk_stats(err_a: Sequence[bool], err_c: Sequence[Sequence[bool]],
               n_boot: int = 2000, seed: int = 0) -> dict:
    """Where the real step's continuation error rate sits among the alternatives'.

    err_a: per-continuation errors after the real step (k values).
    err_c: one list of per-continuation errors per alternative (m lists).
    risk_diff = err(A) - mean_j err(C_j); risk_pct = share of alternatives with a
    lower error rate than A, ties counted half; risk_z = risk_diff / sd_j err(C_j)
    when that sd is above 0. The SE resamples continuations within every
    condition and alternatives among themselves.
    """
    a = np.asarray(err_a, dtype=float)
    cs = [np.asarray(c, dtype=float) for c in err_c]
    ea, ec = float(a.mean()), np.array([c.mean() for c in cs])
    diff = ea - float(ec.mean())
    pct = float(((ec < ea).sum() + 0.5 * (ec == ea).sum()) / len(ec))
    sd = float(ec.std(ddof=1)) if len(ec) > 1 else 0.0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ab = rng.choice(a, len(a)).mean()
        pick = rng.integers(0, len(cs), len(cs))
        boots[b] = ab - np.mean([rng.choice(cs[j], len(cs[j])).mean() for j in pick])
    return {"err_a": ea, "err_c": ec.tolist(), "err_c_mean": float(ec.mean()),
            "err_c_sd": sd, "risk_diff": diff, "risk_pct": pct,
            "risk_z": diff / sd if sd > 0 else None, "risk_diff_se": float(boots.std(ddof=1))}


# --------------------------------------------------------------------------- #
# Ablation baselines
# --------------------------------------------------------------------------- #

class OffsetMeanAccumulator:
    """Position-matched mean of step-content states at one layer.

    Every content token but the last is binned by its offset from the content
    start; the last (the newline that ends the line) goes to its own bin, since
    the boundary token is a different kind of position. Offsets seen fewer than
    `min_count` times, or beyond `max_offset`, fall back to the mean over all
    non-final offsets (the tail mean).
    """

    def __init__(self, d: int, max_offset: int = 128, min_count: int = 20):
        self.d, self.max_offset, self.min_count = d, max_offset, min_count
        self.sums = np.zeros((max_offset, d), dtype=np.float64)
        self.counts = np.zeros(max_offset, dtype=np.int64)
        self.final_sum = np.zeros(d, dtype=np.float64)
        self.final_count = 0
        self.all_sum = np.zeros(d, dtype=np.float64)
        self.all_count = 0

    def add(self, span_states: np.ndarray) -> None:
        x = np.asarray(span_states, dtype=np.float64)
        if x.shape[0] == 0:
            return
        self.final_sum += x[-1]; self.final_count += 1
        body = x[:-1]
        n = min(len(body), self.max_offset)
        self.sums[:n] += body[:n]; self.counts[:n] += 1
        self.all_sum += body.sum(0); self.all_count += len(body)

    def finalize(self) -> dict:
        if self.final_count == 0 or self.all_count == 0:
            raise ValueError("no reference states accumulated")
        tail = self.all_sum / self.all_count
        means = np.where((self.counts >= self.min_count)[:, None],
                         self.sums / np.maximum(self.counts, 1)[:, None], tail[None, :])
        return {"offset_means": means.astype(np.float32), "counts": self.counts.copy(),
                "final_mean": (self.final_sum / self.final_count).astype(np.float32),
                "tail_mean": tail.astype(np.float32), "n_final": self.final_count,
                "n_body": self.all_count}


def mean_baseline(ref: dict, n: int) -> np.ndarray:
    """(n, d) position-matched baseline for a content span of n tokens."""
    if n < 1:
        raise ValueError("empty span")
    om, tail = ref["offset_means"], ref["tail_mean"]
    body = [om[j] if j < len(om) else tail for j in range(n - 1)]
    return np.stack(body + [ref["final_mean"]]).astype(np.float32)


def norm_matched_random(states, generator):
    """Random directions with each row's norm (torch tensors, float32 out)."""
    import torch
    x = states.float()
    r = torch.randn(x.shape, generator=generator, device="cpu").to(x.device)
    r = r / r.norm(dim=-1, keepdim=True)
    return r * x.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------- #
# Teacher-forced continuation log-probs under a span patch
# --------------------------------------------------------------------------- #

def token_logprobs(logits, input_ids, start: int):
    """log P(input_ids[t] | ids[:t]) for t in [start, T), float32, chunked."""
    import torch
    out = []
    T = input_ids.shape[-1]
    for c0 in range(start, T, 512):
        c1 = min(T, c0 + 512)
        lp = torch.log_softmax(logits[0, c0 - 1:c1 - 1, :].float(), dim=-1)
        out.append(lp.gather(-1, input_ids[0, c0:c1, None]).squeeze(-1))
    return torch.cat(out)


def patched_logprobs(model, input_ids, start: int, layer: int | None = None,
                     lo: int | None = None, hi: int | None = None, states=None):
    """Per-token log-probs of input_ids[start:] with hidden_states[layer][lo:hi]
    replaced by `states` (das_span.make_span_patch_hook on block layer-1), or
    unpatched when `states` is None.

    Landing check: a pre-hook on block `layer` reads that block's input over the
    span; it must equal the patch exactly. Returns (logprobs, landing_max_abs),
    landing_max_abs None when unpatched.
    """
    import torch
    from src.analysis.das_span import make_span_patch_hook
    handles, seen = [], {}
    if states is not None:
        st = states.to(next(model.parameters()).dtype)
        handles.append(model.model.layers[layer - 1].register_forward_hook(
            make_span_patch_hook(lo, hi, st)))

        def pre(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            seen["x"] = h[0, lo:hi].detach()
        handles.append(model.model.layers[layer].register_forward_pre_hook(pre, with_kwargs=True))
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
    finally:
        for h in handles:
            h.remove()
    lp = token_logprobs(logits, input_ids, start)
    land = None
    if states is not None:
        land = float((seen["x"].float() - st.float().to(seen["x"].device)).abs().max())
    return lp, land


def clean_states(model, input_ids, layers: Sequence[int], lo: int, hi: int):
    """Clean hidden_states[L][lo:hi] for every L, plus the index check: block L-1's
    output must equal hidden_states[L] (returns its max abs difference per L)."""
    import torch
    cap, handles = {}, []
    for L in layers:
        def hook(module, args, output, L=L):
            hs = output[0] if isinstance(output, tuple) else output
            cap[L] = hs[0, lo:hi].detach()
        handles.append(model.model.layers[L - 1].register_forward_hook(hook))
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    st = {L: out.hidden_states[L][0, lo:hi].detach() for L in layers}
    idx = {L: float((st[L].float() - cap[L].float()).abs().max()) for L in layers}
    return st, idx


def segment_sums(lp: np.ndarray, start: int, segments: dict[str, tuple[int, int]]) -> dict:
    """Sum of per-token log-probs over absolute [lo, hi) segments, lp indexed from start."""
    return {k: float(lp[lo - start:hi - start].sum()) for k, (lo, hi) in segments.items()}


def necessity_from_lp(lp_clean: Sequence[float], lp_patched: Sequence[float], start: int,
                      segments: dict[str, tuple[int, int]]) -> dict:
    """Necessity with the boundary token split out.

    The first continuation token (absolute position `start`) is predicted at the
    span's last position, which is the patched state itself, so its drop is a
    direct readout of the overwritten state, not dependence carried by attention.
    In the pilot it carried the whole L35 effect on last-step forks (about 29 nats
    on "<" of "<Answer>"). Every segment is therefore summed from start + 1, and
    the boundary token's drop is reported on its own.
    """
    d = np.asarray(lp_clean, float) - np.asarray(lp_patched, float)
    out = {"boundary_drop": float(d[0])}
    for k, (lo, hi) in segments.items():
        lo = max(lo, start + 1)
        out[k] = float(d[lo - start:hi - start].sum()) if hi > lo else 0.0
    out["n_cont_xb"] = segments["cont"][1] - (start + 1)
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rho with average ranks; None when undefined."""
    a, b = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return None

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v)); r[order] = np.arange(len(v), dtype=float)
        for u in np.unique(v):
            m = v == u
            r[m] = r[m].mean()
        return r
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return None if math.isnan(rho) else rho


def _ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v)); r[order] = np.arange(len(v), dtype=float)
    for u in np.unique(v):
        m = v == u
        r[m] = r[m].mean()
    return r


def partial_spearman(x: Sequence[float], y: Sequence[float],
                     covs: Sequence[Sequence[float]]) -> float | None:
    """Spearman rho of x and y after linearly removing covariates from all ranks."""
    a, b = np.asarray(x, float), np.asarray(y, float)
    C = [np.asarray(c, float) for c in covs]
    ok = np.isfinite(a) & np.isfinite(b)
    for c in C:
        ok &= np.isfinite(c)
    if ok.sum() < len(C) + 3:
        return None
    Z = np.column_stack([np.ones(ok.sum())] + [_ranks(c[ok]) for c in C])
    ra, rb = _ranks(a[ok]), _ranks(b[ok])
    ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]
    eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
    if ea.std() < 1e-12 or eb.std() < 1e-12:
        return None
    return float(np.corrcoef(ea, eb)[0, 1])


def boot_ci(stat: Callable[[np.ndarray], float | None], n: int, n_boot: int = 2000,
            seed: int = 0) -> list[float] | None:
    """Percentile 95% interval of stat(idx) over n_boot resamples of range(n).
    Forks are one per problem, so resampling forks is resampling problems."""
    rng = np.random.default_rng(seed)
    vals = [v for v in (stat(rng.integers(0, n, n)) for _ in range(n_boot)) if v is not None]
    if len(vals) < n_boot // 2:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
