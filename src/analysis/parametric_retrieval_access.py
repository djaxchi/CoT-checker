"""Core logic for the parametric_retrieval_access_v1 experiment.

Latent factual knowledge vs retrieval access on Qwen2.5-7B-Instruct: same-fact
mixed-outcome paraphrase pairs, fact-disjoint splits, matched candidate sets
for answer-content decoding, and token-span helpers for multi-position
hidden-state extraction. Pure functions only (no torch / no model calls).

Reuses answer normalization and grading from
src.analysis.parametric_retrieval (v0), which stays the single source of
truth for grade_answer / extract_cot_final_answer.

Directions are separate retrieval tasks: direct asks subject+relation->object
(gold = direct_answer), reverse asks object+inverse->subject (gold =
reverse_answer). The queried entity mentioned in the question is the fact
subject for direct and the fact object for reverse.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from src.analysis.parametric_retrieval import normalize_answer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTIONS = ["direct", "reverse"]

# seed question columns in WikiProfile, per direction
SEED_VARIANTS = {
    "direct": ["direct", "direct_natural"],
    "reverse": ["reverse", "reverse_natural"],
}

# Instruction wrappers. Every wrapper ends with "Question: {q}\nAnswer:" so
# the answer-prefix position (the ':' of "Answer:") is standardized across
# paraphrases; only the instruction framing varies (recorded as template_id).
PARAPHRASE_WRAPPERS: dict[str, str] = {
    "w0": "Answer the question with a short answer only.\n"
          "Question: {q}\nAnswer:",
    "w1": "Question: {q}\nAnswer:",
    "w2": "Give just the answer, with no explanation.\n"
          "Question: {q}\nAnswer:",
    "w3": "This is a factual quiz. Reply with only the answer.\n"
          "Question: {q}\nAnswer:",
    "w4": "Please answer the following question as briefly as possible.\n"
          "Question: {q}\nAnswer:",
    "w5": "Answer concisely.\nQuestion: {q}\nAnswer:",
}

COT_TEMPLATE = ("Think step by step, then give the final answer in the "
                "format:\nFinal answer: <short answer>\n\n"
                "Question: {question}")

SPLIT_NAMES = ["train", "val", "test"]
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)

# pre-generation positions; first_generated_token is auxiliary/post-decision
POSITION_NAMES = ["entity_first", "entity_last", "entity_mean",
                  "question_last", "answer_prefix", "final_prompt_token",
                  "first_generated_token"]

FACT_META_COLS = ["fact_id", "page_title", "item_id", "gbc", "gbc_bin",
                  "category", "subject", "object", "subject_type",
                  "object_type"]


def gold_column(direction: str) -> str:
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}")
    return "direct_answer" if direction == "direct" else "reverse_answer"


def entity_of(fact_row, direction: str) -> str:
    """Queried entity mentioned in the question."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}")
    return str(fact_row["subject"] if direction == "direct"
               else fact_row["object"])


def answer_type_column(direction: str) -> str:
    """Type of the ANSWER entity (for matched negatives)."""
    return "object_type" if direction == "direct" else "subject_type"


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------

def leaks_answer(question: str, gold: str) -> bool:
    """True when the normalized gold answer appears inside the normalized
    question text (such prompts reveal the answer and are dropped)."""
    ng, nq = normalize_answer(gold), normalize_answer(question)
    return bool(ng) and ng in nq


def build_access_instances(facts: pd.DataFrame) -> pd.DataFrame:
    """Explode facts into direct-mode paraphrase instances plus one canonical
    CoT instance per fact x direction.

    Per fact x direction: len(SEED_VARIANTS) x len(PARAPHRASE_WRAPPERS)
    direct paraphrases (prompt_mode='direct', user_message = wrapper applied
    to the seed question) and 1 CoT instance (prompt_mode='cot', canonical
    seed = the template question, COT_TEMPLATE). Instances whose question
    leaks the gold answer are dropped; a whole fact x direction is dropped
    when its canonical seed leaks (the CoT arm would be meaningless).
    """
    rows = []
    for _, f in facts.iterrows():
        base = {c: f[c] for c in FACT_META_COLS}
        for direction in DIRECTIONS:
            gold = str(f[gold_column(direction)])
            entity = entity_of(f, direction)
            for seed in SEED_VARIANTS[direction]:
                question = str(f[seed])
                if leaks_answer(question, gold):
                    continue
                for wid, wrapper in PARAPHRASE_WRAPPERS.items():
                    rows.append({
                        **base,
                        "instance_id": f"{f['fact_id']}::{direction}::"
                                       f"{seed}::{wid}",
                        "direction": direction,
                        "prompt_mode": "direct",
                        "seed_variant": seed,
                        "template_id": wid,
                        "paraphrase_id": f"{seed}::{wid}",
                        "question": question,
                        "user_message": wrapper.format(q=question),
                        "gold_answer": gold,
                        "entity": entity,
                    })
            canonical = str(f[SEED_VARIANTS[direction][0]])
            if not leaks_answer(canonical, gold):
                rows.append({
                    **base,
                    "instance_id": f"{f['fact_id']}::{direction}::cot",
                    "direction": direction,
                    "prompt_mode": "cot",
                    "seed_variant": SEED_VARIANTS[direction][0],
                    "template_id": "cot",
                    "paraphrase_id": "cot",
                    "question": canonical,
                    "user_message": COT_TEMPLATE.format(question=canonical),
                    "gold_answer": gold,
                    "entity": entity,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fact-disjoint splits
# ---------------------------------------------------------------------------

def assign_fact_splits(facts: pd.DataFrame,
                       fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
                       seed: int = 42) -> dict[str, str]:
    """fact_id -> train/val/test, fact-disjoint, stratified by
    gbc_bin x category with largest-remainder allocation. Both directions of
    a fact share its split by construction. Deterministic under a fixed seed.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    for _, group in facts.groupby(["gbc_bin", "category"], sort=True,
                                  observed=True):
        ids = sorted(group.fact_id.astype(str))
        perm = rng.permutation(len(ids))
        exact = np.asarray(fractions) * len(ids)
        quota = np.floor(exact).astype(int)
        order = np.argsort(-(exact - quota), kind="stable")
        for i in order[: len(ids) - quota.sum()]:
            quota[i] += 1
        start = 0
        for name, q in zip(SPLIT_NAMES, quota):
            for j in perm[start:start + q]:
                out[ids[j]] = name
            start += q
    return out


# ---------------------------------------------------------------------------
# Mixed-outcome selection and pairing
# ---------------------------------------------------------------------------

def group_outcomes(graded: pd.DataFrame, min_success: int = 2,
                   min_fail: int = 2) -> pd.DataFrame:
    """Aggregate direct-mode grading per fact x direction.

    graded needs columns: instance_id, fact_id, direction, prompt_mode,
    is_correct. Returns one row per fact x direction with n_paraphrases,
    n_success, n_fail, p_direct and is_mixed (>= min_success successes AND
    >= min_fail failures among greedy paraphrase outcomes).
    """
    d = graded[graded.prompt_mode == "direct"]
    agg = (d.groupby(["fact_id", "direction"], sort=True)
           .agg(n_paraphrases=("is_correct", "size"),
                n_success=("is_correct", "sum"))
           .reset_index())
    agg["n_success"] = agg.n_success.astype(int)
    agg["n_fail"] = agg.n_paraphrases - agg.n_success
    agg["p_direct"] = agg.n_success / agg.n_paraphrases
    agg["is_mixed"] = ((agg.n_success >= min_success)
                       & (agg.n_fail >= min_fail))
    return agg


def build_pairs(graded: pd.DataFrame, mixed: pd.DataFrame,
                max_pairs_per_group: int = 8, seed: int = 42) -> pd.DataFrame:
    """Matched (success, fail) instance pairs within each mixed
    fact x direction group.

    All success x fail combinations are enumerated in sorted instance order;
    when a group exceeds max_pairs_per_group, a deterministic subsample is
    drawn. Returns pair_id, fact_id, direction, donor_instance_id (success),
    recipient_instance_id (fail).
    """
    rng = np.random.default_rng(seed)
    d = graded[graded.prompt_mode == "direct"]
    keys = mixed.loc[mixed.is_mixed, ["fact_id", "direction"]]
    rows = []
    for fact_id, direction in keys.itertuples(index=False):
        g = d[(d.fact_id == fact_id) & (d.direction == direction)]
        succ = sorted(g.loc[g.is_correct, "instance_id"])
        fail = sorted(g.loc[~g.is_correct, "instance_id"])
        combos = [(s, f) for s in succ for f in fail]
        if len(combos) > max_pairs_per_group:
            idx = rng.choice(len(combos), size=max_pairs_per_group,
                             replace=False)
            combos = [combos[i] for i in np.sort(idx)]
        for r, (s, f) in enumerate(combos):
            rows.append({"pair_id": f"{fact_id}::{direction}::p{r}",
                         "fact_id": fact_id, "direction": direction,
                         "donor_instance_id": s,
                         "recipient_instance_id": f})
    return pd.DataFrame(
        rows, columns=["pair_id", "fact_id", "direction",
                       "donor_instance_id", "recipient_instance_id"])


# ---------------------------------------------------------------------------
# Candidate sets (Experiment A)
# ---------------------------------------------------------------------------

def parse_choices(raw) -> list[str]:
    """WikiProfile MC choices are stringified python lists."""
    if isinstance(raw, list):
        return [str(c) for c in raw]
    try:
        val = ast.literal_eval(str(raw))
        return [str(c) for c in val] if isinstance(val, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def build_candidate_set(facts: pd.DataFrame, fact_id: str, direction: str,
                        k: int = 32, seed: int = 42) -> dict:
    """Gold answer + up to k-1 hard negatives for one fact x direction.

    Negatives, deduplicated under normalize_answer and never equal to the
    gold: (1) the fact's own MC distractors (type-matched by construction),
    then (2) answers of OTHER facts with the same answer-entity type, drawn
    same-category+same-gbc_bin first, then same-category, then any.
    Deterministic under a fixed seed (per-fact rng so sets are independent
    of iteration order).
    """
    row = facts.loc[facts.fact_id == fact_id]
    if row.empty:
        raise KeyError(f"fact_id {fact_id!r} not in facts table")
    row = row.iloc[0]
    gold = str(row[gold_column(direction)])
    seen = {normalize_answer(gold)}
    negatives: list[str] = []

    def add(cand: str) -> None:
        n = normalize_answer(str(cand))
        if n and n not in seen and len(negatives) < k - 1:
            seen.add(n)
            negatives.append(str(cand))

    choice_col = ("direct_choices" if direction == "direct"
                  else "reverse_choices")
    if choice_col in row.index:
        for c in parse_choices(row[choice_col]):
            add(c)

    type_col = answer_type_column(direction)
    ans_col = gold_column(direction)
    pool = facts[(facts.fact_id != fact_id)
                 & (facts[type_col] == row[type_col])]
    tiers = [
        pool[(pool.category == row["category"])
             & (pool.gbc_bin == row["gbc_bin"])],
        pool[pool.category == row["category"]],
        pool,
    ]
    rng = np.random.default_rng(
        seed + (hash(f"{fact_id}::{direction}") % (2 ** 31)))
    for tier in tiers:
        if len(negatives) >= k - 1:
            break
        cands = sorted(tier[ans_col].astype(str).unique())
        for i in rng.permutation(len(cands)):
            add(cands[i])
            if len(negatives) >= k - 1:
                break
    return {"fact_id": fact_id, "direction": direction, "gold": gold,
            "negatives": negatives}


# ---------------------------------------------------------------------------
# Token spans for extraction
# ---------------------------------------------------------------------------

def find_ci(haystack: str, needle: str) -> tuple[int, int] | None:
    """First case-insensitive occurrence as (start, end) char span."""
    if not needle:
        return None
    i = haystack.lower().find(needle.lower())
    return None if i < 0 else (i, i + len(needle))


def span_to_token_range(offsets: list[tuple[int, int]], start: int,
                        end: int) -> tuple[int, int] | None:
    """Map a [start, end) char span to (first_token, last_token) inclusive,
    using HF fast-tokenizer offset mapping. Tokens with empty offsets
    (special tokens) never match. None when no token overlaps the span."""
    toks = [i for i, (s, e) in enumerate(offsets)
            if e > s and s < end and e > start]
    return (toks[0], toks[-1]) if toks else None


def compute_access_positions(rendered: str, offsets: list[tuple[int, int]],
                             question: str, entity: str,
                             user_message: str) -> list[dict]:
    """Extraction positions for one rendered chat prompt.

    offsets = fast-tokenizer offset mapping of the FULL rendered chat text
    (len(offsets) == prompt token count). Emits, when locatable:
      entity_first/entity_last/entity_mean  queried-entity mention inside the
                                            question text
      question_last                         last non-space question char
      answer_prefix                         last non-space user-message char
                                            (the ':' of 'Answer:')
      final_prompt_token                    always
      first_generated_token                 always (index prompt_len; caller
                                            must teacher-force >= 1 token)
    Rows: {position_name, token_start, token_end} inclusive; single-token
    positions have token_start == token_end. entity_mean spans the mention.
    """
    prompt_len = len(offsets)
    pos: list[dict] = []

    def single(name: str, tok_idx: int) -> None:
        pos.append({"position_name": name, "token_start": tok_idx,
                    "token_end": tok_idx})

    q0 = rendered.rfind(question)
    if q0 >= 0:
        span = find_ci(rendered[q0:q0 + len(question)], entity)
        if span is not None:
            tr = span_to_token_range(offsets, q0 + span[0], q0 + span[1])
            if tr is not None:
                single("entity_first", tr[0])
                single("entity_last", tr[1])
                pos.append({"position_name": "entity_mean",
                            "token_start": tr[0], "token_end": tr[1]})
        q_strip = question.rstrip()
        if q_strip:
            tr = span_to_token_range(offsets, q0 + len(q_strip) - 1,
                                     q0 + len(q_strip))
            if tr is not None:
                single("question_last", tr[1])
    m0 = rendered.rfind(user_message)
    if m0 >= 0:
        m_strip = user_message.rstrip()
        tr = span_to_token_range(offsets, m0 + len(m_strip) - 1,
                                 m0 + len(m_strip))
        if tr is not None:
            single("answer_prefix", tr[1])
    single("final_prompt_token", prompt_len - 1)
    single("first_generated_token", prompt_len)
    return pos
