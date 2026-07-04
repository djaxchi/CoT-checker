"""Core logic for the parametric_retrieval_geometry_v0 experiment.

WikiProfile retrieval-regime geometry on Qwen2.5-7B-Instruct hidden states.
Pure functions only (no torch / no model calls) so everything here is unit
testable: answer normalization + deterministic grading, behavioral
retrieval-class assignment, stratified fact sampling, prompt templates, and
token-position finding for hidden-state extraction.

Layer convention: hs_idx is the index into the HF hidden_states tuple
(hidden_states[0] = embeddings, hidden_states[k] = post block k-1), so
hs_idx=20 <=> block_idx=19. Both labels are stored everywhere.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HS_INDICES = [4, 8, 12, 16, 20, 24, 28]

QA_FAMILIES = ["direct", "direct_natural", "reverse", "reverse_natural"]
ALL_FAMILIES = QA_FAMILIES + ["completion"]

RETRIEVAL_CLASSES = ["direct_retrieval", "reasoning_unlocked",
                     "unstable_retrieval", "non_retrieved"]
CONTROL_CLASSES = ["ctrl_retrieved", "ctrl_unstable", "ctrl_non_retrieved"]

GBC_BIN_LABELS = ["low", "mid", "high", "very_high"]

DIRECT_TEMPLATE = ("Answer the question with a short answer only.\n"
                   "Question: {question}\n"
                   "Answer:")
COT_TEMPLATE = ("Think step by step, then give the final answer in the format:\n"
                "Final answer: <short answer>\n\n"
                "Question: {question}")
COMPLETION_TEMPLATE = ("Complete the following text with the missing entity. "
                       "Reply with only the completion.\n"
                       "Text: {completion}")

FINAL_ANSWER_RE = re.compile(r"final answer\s*[:\-]\s*", re.IGNORECASE)

# direct-mode positions (fixed) and cot-mode positions (fixed + ragged
# sentence_end ranks); order below is trajectory order.
DIRECT_POSITIONS = ["final_prompt_token", "first_generated_token",
                    "final_answer_token"]
COT_POSITIONS = ["final_prompt_token", "first_generated_token", "sentence_end",
                 "token_before_final_answer", "first_final_answer_token",
                 "final_answer_token"]
MAX_SENTENCE_ENDS = 20


def block_idx(hs_idx: int) -> int:
    """hidden_states tuple index -> transformer block index (resid_post)."""
    if hs_idx < 1:
        raise ValueError(f"hs_idx {hs_idx} has no owning block (0 = embeddings)")
    return hs_idx - 1


# ---------------------------------------------------------------------------
# Answer normalization + grading
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"^(a|an|the)\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """lowercase, NFKC, accent folding, punctuation->space, collapse
    whitespace, strip a single leading article."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = "".join(c for c in unicodedata.normalize("NFKD", t)
                if not unicodedata.combining(c))  # dōjinshi -> dojinshi
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 1,914 -> 1914 before de-punct
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    t = _ARTICLE_RE.sub("", t)
    return t


def digit_groups(text: str) -> list[str]:
    """Canonical digit runs: thousands separators removed (1,914 -> 1914),
    leading zeros stripped (01 -> 1) so date formats compare equal."""
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return [str(int(g)) for g in re.findall(r"\d+", t)]


def _digit_dominant(norm_gold: str) -> bool:
    compact = norm_gold.replace(" ", "")
    if not compact:
        return False
    n_digits = sum(c.isdigit() for c in compact)
    return n_digits >= max(1, len(compact) / 2)


def grade_answer(pred: str, gold: str) -> tuple[bool, str]:
    """Deterministic v0 grading.

    Returns (correct, grading_status) with grading_status in
    exact | containment | normalized_number | ambiguous | failed.
    ambiguous = matched only under loose criteria (very short spans);
    counted as incorrect for labels but stored for inspection.
    """
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if not ng or not np_:
        return False, "failed"
    if np_ == ng:
        return True, "exact"
    contained = ng in np_ or np_ in ng
    if contained and min(len(np_), len(ng)) >= 4:
        return True, "containment"
    if _digit_dominant(ng):
        gg, gp = digit_groups(ng), digit_groups(np_)
        if gg and all(g in gp for g in gg):
            return True, "normalized_number"
    if contained:
        return False, "ambiguous"
    return False, "failed"


def extract_cot_final_answer(text: str) -> tuple[str, bool]:
    """Answer span after the LAST 'Final answer:' marker; falls back to the
    last non-empty line when the marker is missing (marker_found=False)."""
    matches = list(FINAL_ANSWER_RE.finditer(text))
    if matches:
        ans = text[matches[-1].end():].strip().splitlines()
        return (ans[0].strip() if ans else ""), True
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), False


# ---------------------------------------------------------------------------
# Behavioral labels
# ---------------------------------------------------------------------------

def assign_retrieval_class(direct_greedy_correct: bool,
                           direct_pass_at_4: bool,
                           cot_greedy_correct: bool,
                           cot_pass_at_4: bool) -> str:
    """Hard mutually-exclusive v0 label (priority chain as specified).

    pass@4 flags are computed over the 4 sampled generations only (greedy
    excluded), so direct_greedy_correct and direct_pass_at_4 are independent.
    """
    if direct_greedy_correct:
        return "direct_retrieval"
    if (not direct_pass_at_4) and cot_greedy_correct:
        return "reasoning_unlocked"
    if direct_pass_at_4 or cot_pass_at_4:
        return "unstable_retrieval"
    return "non_retrieved"


def soft_flags(direct_greedy_correct: bool,
               direct_pass_at_4: bool,
               cot_greedy_correct: bool,
               cot_pass_at_4: bool) -> dict[str, bool]:
    return {
        "reasoning_unlocked_soft": (not direct_pass_at_4) and cot_pass_at_4,
        "direct_unstable": (not direct_greedy_correct) and direct_pass_at_4,
        "cot_unstable": (not cot_greedy_correct) and cot_pass_at_4,
    }


def completion_control_class(direct_greedy_correct: bool,
                             direct_pass_at_4: bool) -> str:
    """Auxiliary label for the completion control (direct-only, no CoT arm)."""
    if direct_greedy_correct:
        return "ctrl_retrieved"
    if direct_pass_at_4:
        return "ctrl_unstable"
    return "ctrl_non_retrieved"


# ---------------------------------------------------------------------------
# Fact sampling
# ---------------------------------------------------------------------------

def gbc_bins(gbc: pd.Series) -> pd.Series:
    """Quartile popularity bins over the FULL fact table (rank-based so ties
    cannot collapse bins): low / mid / high / very_high."""
    ranks = gbc.rank(method="first")
    return pd.qcut(ranks, 4, labels=GBC_BIN_LABELS).astype(str)


def stratified_fact_sample(facts: pd.DataFrame, n_facts: int,
                           seed: int = 42) -> pd.DataFrame:
    """Sample n_facts rows stratified by gbc_bin x category with proportional
    (largest-remainder) allocation. Expects a gbc_bin column already assigned
    on the full table. Deterministic under a fixed seed."""
    if n_facts > len(facts):
        raise ValueError(f"n_facts={n_facts} > available facts {len(facts)}")
    rng = np.random.default_rng(seed)
    cells = facts.groupby(["gbc_bin", "category"], sort=True, observed=True)
    keys, sizes = zip(*[(k, len(g)) for k, g in cells])
    sizes = np.asarray(sizes, dtype=float)
    exact = n_facts * sizes / sizes.sum()
    quota = np.floor(exact).astype(int)
    remainder = n_facts - quota.sum()
    order = np.argsort(-(exact - quota), kind="stable")
    for i in order[:remainder]:
        quota[i] += 1
    quota = np.minimum(quota, sizes.astype(int))
    short = n_facts - quota.sum()  # redistribute if any cell was capped
    while short > 0:
        for i in np.argsort(-(sizes - quota), kind="stable"):
            if short == 0:
                break
            if quota[i] < sizes[i]:
                quota[i] += 1
                short -= 1
    picked = []
    for (key, group), q in zip(cells, quota):
        if q == 0:
            continue
        idx = rng.choice(len(group), size=int(q), replace=False)
        picked.append(group.iloc[np.sort(idx)])
    out = pd.concat(picked, ignore_index=True)
    assert len(out) == n_facts
    return out


def build_prompt_instances(facts: pd.DataFrame) -> pd.DataFrame:
    """Explode sampled facts into prompt instances: 4 QA forms per fact
    (core geometry) + 1 completion form (direct-only auxiliary control).

    Gold mapping: direct/direct_natural -> direct_answer (object),
    reverse/reverse_natural -> reverse_answer (subject),
    completion -> object.
    """
    meta_cols = ["fact_id", "page_title", "item_id", "gbc", "gbc_bin",
                 "category", "subject", "object", "subject_type", "object_type"]
    rows = []
    for _, f in facts.iterrows():
        base = {c: f[c] for c in meta_cols}
        for family in QA_FAMILIES:
            gold = f["direct_answer"] if family.startswith("direct") \
                else f["reverse_answer"]
            rows.append({**base, "question_id": f"{f['fact_id']}::{family}",
                         "family": family, "question": f[family],
                         "gold_answer": gold, "is_control": False})
        rows.append({**base, "question_id": f"{f['fact_id']}::completion",
                     "family": "completion", "question": f["completion"],
                     "gold_answer": f["object"], "is_control": True})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_user_message(question: str, family: str, prompt_mode: str) -> str:
    """User-turn text (chat template applied by the caller)."""
    if family == "completion":
        if prompt_mode != "direct":
            raise ValueError("completion is a direct-only control")
        return COMPLETION_TEMPLATE.format(completion=question)
    if prompt_mode == "direct":
        return DIRECT_TEMPLATE.format(question=question)
    if prompt_mode == "cot":
        return COT_TEMPLATE.format(question=question)
    raise ValueError(f"unknown prompt_mode {prompt_mode!r}")


# ---------------------------------------------------------------------------
# Token positions
# ---------------------------------------------------------------------------

def sentence_end_char_indices(text: str, before_char: int | None = None) -> list[int]:
    """Char indices (inclusive) of sentence-final . ! ? followed by whitespace
    or end of text; optionally only boundaries strictly before before_char."""
    out = [m.start() for m in re.finditer(r"[.!?](?=\s|$)", text)]
    if before_char is not None:
        out = [c for c in out if c < before_char]
    return out


def char_to_token(offsets: list[int], char_idx: int) -> int:
    """Map a char index to the token containing it. offsets[i] = cumulative
    decoded length AFTER token i (strictly increasing except empty tokens)."""
    for i, end in enumerate(offsets):
        if char_idx < end:
            return i
    return len(offsets) - 1


def compute_positions(prompt_len: int, n_gen: int, gen_text: str,
                      gen_offsets: list[int], prompt_mode: str) -> list[dict]:
    """Token positions (indices into the full prompt+generation sequence) at
    which hidden states are extracted.

    Returns [{position_name, position_rank, token_index}] in trajectory order.
    final_prompt_token = last token of the rendered chat prompt (index
    prompt_len-1). Generated token g sits at sequence index prompt_len+g.
    """
    pos = [{"position_name": "final_prompt_token", "position_rank": 0,
            "token_index": prompt_len - 1}]
    if n_gen <= 0:
        return pos
    pos.append({"position_name": "first_generated_token", "position_rank": 0,
                "token_index": prompt_len})
    last_gen = prompt_len + n_gen - 1
    if prompt_mode == "direct":
        pos.append({"position_name": "final_answer_token", "position_rank": 0,
                    "token_index": last_gen})
        return pos

    matches = list(FINAL_ANSWER_RE.finditer(gen_text))
    marker_end = matches[-1].end() if matches else None
    for rank, c in enumerate(
            sentence_end_char_indices(gen_text, before_char=marker_end)
            [:MAX_SENTENCE_ENDS]):
        pos.append({"position_name": "sentence_end", "position_rank": rank,
                    "token_index": prompt_len + char_to_token(gen_offsets, c)})
    if marker_end is not None and marker_end < len(gen_text):
        t_ans = char_to_token(gen_offsets, marker_end)
        if t_ans > 0:
            pos.append({"position_name": "token_before_final_answer",
                        "position_rank": 0,
                        "token_index": prompt_len + t_ans - 1})
        pos.append({"position_name": "first_final_answer_token",
                    "position_rank": 0, "token_index": prompt_len + t_ans})
    pos.append({"position_name": "final_answer_token", "position_rank": 0,
                "token_index": last_gen})
    return pos
