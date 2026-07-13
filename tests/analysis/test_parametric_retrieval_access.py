"""Unit tests for the parametric_retrieval_access_v1 core logic."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.parametric_retrieval_access import (
    DIRECTIONS,
    PARAPHRASE_WRAPPERS,
    SEED_VARIANTS,
    SPLIT_NAMES,
    assign_fact_splits,
    build_access_instances,
    compute_access_positions,
    build_candidate_set,
    build_pairs,
    entity_of,
    find_ci,
    gold_column,
    group_outcomes,
    leaks_answer,
    parse_choices,
    span_to_token_range,
)


def make_facts(n: int = 12) -> pd.DataFrame:
    rows = []
    cats = ["Arts", "Science"]
    bins = ["low", "mid", "high", "very_high"]
    for i in range(n):
        rows.append({
            "fact_id": f"f{i:03d}",
            "page_title": f"Page {i}",
            "item_id": f"Q{i}",
            "gbc": 100 + i,
            "gbc_bin": bins[i % 4],
            "category": cats[i % 2],
            "subject": f"Subject{i}",
            "object": f"Object{i}",
            "subject_type": "PERSON",
            "object_type": "ORG",
            "direct": f"Which organization did Subject{i} found?",
            "direct_answer": f"Object{i}",
            "direct_natural": f"What org was founded by Subject{i}?",
            "direct_choices": str([f"Object{i}", "Acme Corp",
                                   f"Object{(i + 1) % n}", "Globex"]),
            "reverse": f"Who founded Object{i}?",
            "reverse_answer": f"Subject{i}",
            "reverse_natural": f"Object{i} was founded by whom?",
            "reverse_choices": str([f"Subject{i}", "John Doe",
                                    f"Subject{(i + 1) % n}", "Jane Roe"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# wrappers and instances
# ---------------------------------------------------------------------------

def test_wrappers_standardize_answer_prefix():
    for wid, w in PARAPHRASE_WRAPPERS.items():
        assert w.endswith("Question: {q}\nAnswer:"), wid


def test_build_access_instances_counts_and_ids():
    facts = make_facts(3)
    inst = build_access_instances(facts)
    n_para = len(SEED_VARIANTS["direct"]) * len(PARAPHRASE_WRAPPERS)
    # per fact: n_para direct-mode paraphrases + 1 cot, per direction
    assert len(inst) == 3 * len(DIRECTIONS) * (n_para + 1)
    assert inst.instance_id.is_unique
    d = inst[(inst.fact_id == "f001") & (inst.direction == "direct")
             & (inst.prompt_mode == "direct")]
    assert len(d) == n_para
    assert set(d.template_id) == set(PARAPHRASE_WRAPPERS)
    assert (d.gold_answer == "Object1").all()
    assert (d.entity == "Subject1").all()
    r = inst[(inst.fact_id == "f001") & (inst.direction == "reverse")]
    assert (r.gold_answer == "Subject1").all()
    assert (r.entity == "Object1").all()


def test_build_access_instances_no_leakage():
    facts = make_facts(2)
    # make one direct seed leak the gold answer
    facts.loc[0, "direct"] = "Which organization, Object0, did Subject0 found?"
    inst = build_access_instances(facts)
    kept = inst[(inst.fact_id == "f000") & (inst.direction == "direct")]
    # canonical seed leaks -> cot dropped; only direct_natural paraphrases stay
    assert set(kept.seed_variant) == {"direct_natural"}
    assert "cot" not in set(kept.template_id)
    for _, row in inst.iterrows():
        assert not leaks_answer(row.question, row.gold_answer)


def test_gold_and_entity_helpers():
    facts = make_facts(1)
    row = facts.iloc[0]
    assert gold_column("direct") == "direct_answer"
    assert gold_column("reverse") == "reverse_answer"
    assert entity_of(row, "direct") == "Subject0"
    assert entity_of(row, "reverse") == "Object0"
    with pytest.raises(ValueError):
        gold_column("sideways")


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def test_assign_fact_splits_disjoint_and_deterministic():
    facts = make_facts(100)
    s1 = assign_fact_splits(facts, seed=7)
    s2 = assign_fact_splits(facts, seed=7)
    assert s1 == s2
    assert set(s1) == set(facts.fact_id)
    assert set(s1.values()) <= set(SPLIT_NAMES)
    counts = pd.Series(list(s1.values())).value_counts()
    assert counts["train"] > counts["val"]
    assert abs(counts["train"] - 60) <= 8  # stratified, near 60/20/20
    assert assign_fact_splits(facts, seed=8) != s1


def test_assign_fact_splits_bad_fractions():
    with pytest.raises(ValueError):
        assign_fact_splits(make_facts(4), fractions=(0.5, 0.2, 0.2))


# ---------------------------------------------------------------------------
# outcomes and pairs
# ---------------------------------------------------------------------------

def make_graded(outcomes: dict[str, list[bool]]) -> pd.DataFrame:
    """outcomes: fact_id -> per-paraphrase correctness (direction 'direct')."""
    rows = []
    for fid, oks in outcomes.items():
        for j, ok in enumerate(oks):
            rows.append({"instance_id": f"{fid}::direct::seed::w{j}",
                         "fact_id": fid, "direction": "direct",
                         "prompt_mode": "direct", "is_correct": ok})
    return pd.DataFrame(rows)


def test_group_outcomes_mixed_flag():
    graded = make_graded({
        "f0": [True, True, False, False],   # mixed
        "f1": [True, False, False, False],  # only 1 success -> not mixed
        "f2": [True] * 4,                   # pure success
        "f3": [False] * 4,                  # pure fail
    })
    agg = group_outcomes(graded, min_success=2, min_fail=2)
    m = agg.set_index("fact_id")
    assert bool(m.loc["f0", "is_mixed"])
    assert not bool(m.loc["f1", "is_mixed"])
    assert not bool(m.loc["f2", "is_mixed"])
    assert not bool(m.loc["f3", "is_mixed"])
    assert m.loc["f0", "p_direct"] == 0.5
    assert m.loc["f1", "n_fail"] == 3


def test_build_pairs_cap_and_determinism():
    graded = make_graded({"f0": [True] * 4 + [False] * 4})
    mixed = group_outcomes(graded)
    p1 = build_pairs(graded, mixed, max_pairs_per_group=6, seed=3)
    p2 = build_pairs(graded, mixed, max_pairs_per_group=6, seed=3)
    assert len(p1) == 6  # capped below the 16 combos
    pd.testing.assert_frame_equal(p1, p2)
    g = make_graded({"f0": [True] * 4 + [False] * 4})
    full = build_pairs(g, mixed, max_pairs_per_group=100, seed=3)
    assert len(full) == 16
    donors = set(full.donor_instance_id)
    recips = set(full.recipient_instance_id)
    assert donors.isdisjoint(recips)
    correct = set(g.loc[g.is_correct, "instance_id"])
    assert donors <= correct and recips.isdisjoint(correct)


def test_build_pairs_empty_when_nothing_mixed():
    graded = make_graded({"f0": [True] * 4})
    mixed = group_outcomes(graded)
    pairs = build_pairs(graded, mixed)
    assert pairs.empty
    assert list(pairs.columns) == ["pair_id", "fact_id", "direction",
                                   "donor_instance_id",
                                   "recipient_instance_id"]


# ---------------------------------------------------------------------------
# candidate sets
# ---------------------------------------------------------------------------

def test_parse_choices():
    assert parse_choices("['a', 'b']") == ["a", "b"]
    assert parse_choices(["x"]) == ["x"]
    assert parse_choices("not a list") == []


def test_build_candidate_set_gold_excluded_and_matched():
    facts = make_facts(12)
    cs = build_candidate_set(facts, "f003", "direct", k=8, seed=1)
    assert cs["gold"] == "Object3"
    assert len(cs["negatives"]) == 7
    norms = [n.lower() for n in cs["negatives"]]
    assert "object3" not in norms
    assert len(set(norms)) == len(norms)  # deduplicated
    # MC distractors come first
    assert "Acme Corp" in cs["negatives"][:3]
    # deterministic
    assert build_candidate_set(facts, "f003", "direct", k=8, seed=1) == cs
    rev = build_candidate_set(facts, "f003", "reverse", k=8, seed=1)
    assert rev["gold"] == "Subject3"
    assert all("Object" not in n for n in rev["negatives"])


def test_build_candidate_set_unknown_fact():
    with pytest.raises(KeyError):
        build_candidate_set(make_facts(2), "nope", "direct")


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------

def test_find_ci():
    assert find_ci("Who founded OpenAI?", "openai") == (12, 18)
    assert find_ci("abc", "zzz") is None
    assert find_ci("abc", "") is None


def test_span_to_token_range():
    # tokens: [special(0,0)] "Who"(0,3) " founded"(3,11) " Open"(11,16)
    # "AI"(16,18) "?"(18,19)
    offsets = [(0, 0), (0, 3), (3, 11), (11, 16), (16, 18), (18, 19)]
    assert span_to_token_range(offsets, 12, 18) == (3, 4)
    assert span_to_token_range(offsets, 0, 3) == (1, 1)
    assert span_to_token_range(offsets, 100, 110) is None
    # special token (0,0) never matches even for span starting at 0
    assert span_to_token_range(offsets, 0, 1) == (1, 1)


def test_compute_access_positions():
    question = "Who founded OpenAI?"
    user_message = f"Question: {question}\nAnswer:"
    rendered = f"<|im_start|>user\n{user_message}<|im_end|>\n"
    # character-level offsets: one token per char, plus a trailing special
    offsets = [(i, i + 1) for i in range(len(rendered) - 1)] + [(0, 0)]
    pos = compute_access_positions(rendered, offsets, question, "openai",
                                   user_message)
    by = {p["position_name"]: p for p in pos}
    assert set(by) == {"entity_first", "entity_last", "entity_mean",
                       "question_last", "answer_prefix",
                       "final_prompt_token", "first_generated_token"}
    ent = rendered.find("OpenAI")
    assert by["entity_first"]["token_start"] == ent
    assert by["entity_last"]["token_start"] == ent + len("OpenAI") - 1
    assert by["entity_mean"]["token_start"] == ent
    assert by["entity_mean"]["token_end"] == ent + len("OpenAI") - 1
    assert by["question_last"]["token_start"] == rendered.find("?")
    assert by["answer_prefix"]["token_start"] == rendered.find(
        "Answer:") + len("Answer:") - 1
    assert by["final_prompt_token"]["token_start"] == len(offsets) - 1
    assert by["first_generated_token"]["token_start"] == len(offsets)


def test_compute_access_positions_entity_missing():
    question = "Who founded the lab?"
    user_message = f"Question: {question}\nAnswer:"
    rendered = f"<|im_start|>user\n{user_message}<|im_end|>\n"
    offsets = [(i, i + 1) for i in range(len(rendered))]
    pos = compute_access_positions(rendered, offsets, question, "OpenAI",
                                   user_message)
    names = {p["position_name"] for p in pos}
    assert "entity_first" not in names
    assert {"question_last", "answer_prefix", "final_prompt_token",
            "first_generated_token"} <= names
