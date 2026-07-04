"""Unit tests for the parametric_retrieval_geometry_v0 core logic."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.parametric_retrieval import (
    QA_FAMILIES,
    assign_retrieval_class,
    block_idx,
    build_prompt_instances,
    build_user_message,
    char_to_token,
    completion_control_class,
    compute_positions,
    digit_groups,
    extract_cot_final_answer,
    gbc_bins,
    grade_answer,
    normalize_answer,
    sentence_end_char_indices,
    soft_flags,
    stratified_fact_sample,
)

# ---------------------------------------------------------------------------
# normalization + grading
# ---------------------------------------------------------------------------


def test_normalize_answer_basics():
    assert normalize_answer("  The Boardwalk. ") == "boardwalk"
    assert normalize_answer("Jean-Luc Picard") == "jean luc picard"
    assert normalize_answer("An  apple") == "apple"
    assert normalize_answer("THE") == "the"  # bare article is not stripped


def test_grade_exact_and_containment():
    assert grade_answer("The Boardwalk", "Boardwalk") == (True, "exact")
    assert grade_answer("It was the Boardwalk club in Manchester",
                        "Boardwalk") == (True, "containment")
    assert grade_answer("Paris", "Paris, France") == (True, "containment")
    assert grade_answer("London", "Boardwalk") == (False, "failed")


def test_grade_numbers_and_dates():
    assert grade_answer("It happened in 1,914.", "1914") == (True, "containment")
    assert grade_answer("the year was nineteen... i mean 1914",
                        "1914") == (True, "containment")
    # digit-dominant gold with different formatting
    ok, status = grade_answer("01/07/1914", "1 July 1914")
    assert ok and status == "normalized_number"
    assert grade_answer("1915", "1914")[0] is False


def test_grade_short_span_is_ambiguous_not_correct():
    ok, status = grade_answer("axe", "ax")  # containment but too short
    assert ok is False and status == "ambiguous"


def test_grade_empty():
    assert grade_answer("", "Boardwalk") == (False, "failed")


def test_grade_accent_folding():
    assert grade_answer("dojinshi", "dōjinshi") == (True, "exact")
    assert grade_answer("Beyonce Knowles", "Beyoncé")[0] is True


def test_digit_groups_strip_thousands_separators():
    assert digit_groups("12,345 people in 1914") == ["12345", "1914"]


def test_extract_cot_final_answer():
    text = "Let me think. Oasis formed in Manchester.\nFinal answer: the Boardwalk"
    ans, found = extract_cot_final_answer(text)
    assert found and ans == "the Boardwalk"
    # last marker wins
    text2 = "Final answer: X\nWait.\nFinal answer: Y"
    assert extract_cot_final_answer(text2) == ("Y", True)
    # fallback: last non-empty line
    ans, found = extract_cot_final_answer("Some reasoning\nBoardwalk\n\n")
    assert not found and ans == "Boardwalk"


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


def test_retrieval_class_priority_chain():
    # direct greedy correct dominates everything
    assert assign_retrieval_class(True, True, True, True) == "direct_retrieval"
    # reasoning_unlocked needs direct pass@4 to FAIL and cot greedy to succeed
    assert assign_retrieval_class(False, False, True, True) == "reasoning_unlocked"
    # cot greedy correct but direct pass@4 also correct -> unstable
    assert assign_retrieval_class(False, True, True, True) == "unstable_retrieval"
    # only sampled cot succeeds -> unstable, not reasoning_unlocked (hard label)
    assert assign_retrieval_class(False, False, False, True) == "unstable_retrieval"
    assert assign_retrieval_class(False, True, False, False) == "unstable_retrieval"
    assert assign_retrieval_class(False, False, False, False) == "non_retrieved"


def test_retrieval_class_mutually_exclusive_and_total():
    from itertools import product
    for flags in product([False, True], repeat=4):
        cls = assign_retrieval_class(*flags)
        assert cls in {"direct_retrieval", "reasoning_unlocked",
                       "unstable_retrieval", "non_retrieved"}


def test_soft_flags():
    f = soft_flags(False, False, False, True)
    assert f == {"reasoning_unlocked_soft": True, "direct_unstable": False,
                 "cot_unstable": True}
    f = soft_flags(False, True, True, True)
    assert f == {"reasoning_unlocked_soft": False, "direct_unstable": True,
                 "cot_unstable": False}


def test_completion_control_class():
    assert completion_control_class(True, False) == "ctrl_retrieved"
    assert completion_control_class(False, True) == "ctrl_unstable"
    assert completion_control_class(False, False) == "ctrl_non_retrieved"


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def make_fact_table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    cats = ["Arts", "Science", "People"]
    df = pd.DataFrame({
        "fact_id": [f"p{i}__0" for i in range(n)],
        "page_title": [f"Page {i}" for i in range(n)],
        "item_id": [f"Q{i}" for i in range(n)],
        "gbc": rng.integers(10, 10_000_000, n),
        "category": rng.choice(cats, n),
        "subject": "subj", "object": "obj",
        "subject_type": "PERSON", "object_type": "LOCATION",
        "direct": "dq?", "direct_natural": "dnq?",
        "reverse": "rq?", "reverse_natural": "rnq?",
        "completion": "The thing was", "direct_answer": "obj",
        "reverse_answer": "subj",
    })
    df["gbc_bin"] = gbc_bins(df["gbc"])
    return df


def test_gbc_bins_are_balanced_quartiles():
    df = make_fact_table(400)
    counts = df["gbc_bin"].value_counts()
    assert set(counts.index) == {"low", "mid", "high", "very_high"}
    assert counts.max() - counts.min() <= 1
    # ordering: low bin has smaller gbc than very_high bin
    assert (df[df.gbc_bin == "low"].gbc.median()
            < df[df.gbc_bin == "very_high"].gbc.median())


def test_stratified_sample_size_determinism_and_proportions():
    df = make_fact_table(200)
    s1 = stratified_fact_sample(df, 80, seed=7)
    s2 = stratified_fact_sample(df, 80, seed=7)
    assert len(s1) == 80
    assert list(s1.fact_id) == list(s2.fact_id)
    assert s1.fact_id.is_unique
    # proportional: each gbc_bin gets ~20 of 80
    bin_counts = s1.gbc_bin.value_counts()
    assert bin_counts.max() - bin_counts.min() <= 2


def test_stratified_sample_rejects_oversized_request():
    df = make_fact_table(50)
    with pytest.raises(ValueError):
        stratified_fact_sample(df, 51)


def test_build_prompt_instances_families_and_gold():
    df = make_fact_table(3)
    inst = build_prompt_instances(df)
    assert len(inst) == 3 * 5
    fam_counts = inst.family.value_counts()
    for fam in QA_FAMILIES + ["completion"]:
        assert fam_counts[fam] == 3
    assert (inst[inst.family == "direct"].gold_answer == "obj").all()
    assert (inst[inst.family == "reverse_natural"].gold_answer == "subj").all()
    assert (inst[inst.family == "completion"].is_control).all()
    assert (~inst[inst.family != "completion"].is_control).all()
    assert inst.question_id.is_unique


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_build_user_message():
    d = build_user_message("Where?", "direct", "direct")
    assert d.startswith("Answer the question with a short answer only.")
    assert "Where?" in d
    c = build_user_message("Where?", "direct", "cot")
    assert "Final answer: <short answer>" in c
    comp = build_user_message("Oasis played at", "completion", "direct")
    assert "Oasis played at" in comp
    with pytest.raises(ValueError):
        build_user_message("x", "completion", "cot")


def test_block_idx_convention():
    assert block_idx(20) == 19
    assert block_idx(28) == 27
    with pytest.raises(ValueError):
        block_idx(0)


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------


def offsets_from_pieces(pieces):
    out, total = [], 0
    for p in pieces:
        total += len(p)
        out.append(total)
    return out


def test_char_to_token():
    offsets = offsets_from_pieces(["He", "llo", " world"])  # [2, 5, 11]
    assert char_to_token(offsets, 0) == 0
    assert char_to_token(offsets, 2) == 1
    assert char_to_token(offsets, 10) == 2
    assert char_to_token(offsets, 99) == 2  # clamped


def test_sentence_end_char_indices():
    text = "First. Second! Third? Final answer: X."
    idxs = sentence_end_char_indices(text)
    assert idxs[:3] == [5, 13, 20]
    idxs_before = sentence_end_char_indices(text, before_char=22)
    assert idxs_before == [5, 13, 20]


def test_compute_positions_direct():
    pieces = ["The", " Board", "walk"]
    pos = compute_positions(prompt_len=10, n_gen=3, gen_text="".join(pieces),
                            gen_offsets=offsets_from_pieces(pieces),
                            prompt_mode="direct")
    by_name = {p["position_name"]: p["token_index"] for p in pos}
    assert by_name == {"final_prompt_token": 9, "first_generated_token": 10,
                       "final_answer_token": 12}


def test_compute_positions_direct_zero_gen():
    pos = compute_positions(10, 0, "", [], "direct")
    assert [p["position_name"] for p in pos] == ["final_prompt_token"]


def test_compute_positions_cot_with_marker():
    pieces = ["Think", ".", " More", " thought", ".", " Final",
              " answer", ":", " Board", "walk"]
    text = "".join(pieces)  # "Think. More thought. Final answer: Boardwalk"
    offsets = offsets_from_pieces(pieces)
    pos = compute_positions(prompt_len=5, n_gen=len(pieces), gen_text=text,
                            gen_offsets=offsets, prompt_mode="cot")
    names = [(p["position_name"], p["position_rank"]) for p in pos]
    assert ("final_prompt_token", 0) in names
    assert ("first_generated_token", 0) in names
    # two reasoning sentence ends, none inside the final-answer span
    ends = [p for p in pos if p["position_name"] == "sentence_end"]
    assert [e["position_rank"] for e in ends] == [0, 1]
    assert [e["token_index"] for e in ends] == [5 + 1, 5 + 4]
    by_name = {p["position_name"]: p["token_index"] for p in pos}
    # answer text " Boardwalk" starts at char 34 -> token 8
    assert by_name["first_final_answer_token"] == 5 + 8
    assert by_name["token_before_final_answer"] == 5 + 7
    assert by_name["final_answer_token"] == 5 + len(pieces) - 1


def test_compute_positions_cot_without_marker():
    pieces = ["No", " marker", " here", "."]
    pos = compute_positions(3, 4, "".join(pieces),
                            offsets_from_pieces(pieces), "cot")
    names = {p["position_name"] for p in pos}
    assert "first_final_answer_token" not in names
    assert "token_before_final_answer" not in names
    assert "final_answer_token" in names
