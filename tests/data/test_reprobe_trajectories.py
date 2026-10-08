"""ReProbe release parsing, label mapping, truncation and slices (v2 plan section 9)."""

import math
from collections import Counter

from src.data.prm_trajectories import Y_UNKNOWN
from src.data.reprobe_trajectories import merge_duplicates, parse_row, post_error_slices, strip_step

NAN = float("nan")


def claim(sentence, n_tok=3):
    return {"sentence": sentence, "claim_text": sentence, "aligned_token_ids": list(range(n_tok))}


def row(steps, verified, reply=None, q="What is 2+2?"):
    reply = reply if reply is not None else "".join(s + "\n" for s in steps)
    return {"question": q, "answer": "4", "reply": reply, "claims": [claim(s) for s in steps], "verified": verified}


S = ["- Step 1: Add 2 and 2.", "- Step 2: That is 5.", "- Step 3: So the sum is 5.", "<Answer>: 5"]


def test_strip_prefix_and_label_mapping():
    c = Counter()
    t = parse_row(row(S, [0.0, 1.0, 0.0, NAN]), "r", c)
    assert t["steps"] == ["Add 2 and 2.", "That is 5.", "So the sum is 5.", "<Answer>: 5"]
    assert t["y"] == [0, 1, 0, Y_UNKNOWN]
    assert t["label_mask"] == [True, True, True, False]
    assert t["first_error"] == 1 and t["finished"]
    assert strip_step("  -  Step 12 :  x") == "x"


def test_post_error_steps_keep_their_own_labels():
    t = parse_row(row(S, [0.0, 1.0, 0.0, 1.0]), "r", Counter())
    pre, post = post_error_slices(t["y"], t["label_mask"])
    assert pre == [True, True, False, False]
    assert post == [False, False, True, True]
    assert t["y"][2] == 0  # a correct step after an error is not propagated


def test_truncated_final_step_masked_only_when_cut_mid_line():
    steps = S[:3]
    cut = parse_row(row(steps, [0.0, 0.0, 1.0], reply="".join(s + "\n" for s in steps)[:-1]), "r", Counter())
    assert cut["label_mask"] == [True, True, False] and not cut["finished"]
    whole = parse_row(row(steps, [0.0, 0.0, 1.0]), "r", Counter())  # ends with newline: complete step
    assert whole["label_mask"] == [True, True, True]


def test_length_mismatch_and_empty_alignment_masked():
    c = Counter()
    t = parse_row(row(S, [0.0, 1.0]), "r", c)
    assert t is None and c["traces_label_length_mismatch"] == 1 and c["drop_no_label"] == 1
    r = row(S, [0.0, 1.0, 0.0, 0.0])
    r["claims"][1]["aligned_token_ids"] = []
    t = parse_row(r, "r", Counter())
    assert t["label_mask"] == [True, False, True, True]


def test_unaligned_claim_rejected():
    r = row(S, [0.0, 0.0, 0.0, 0.0])
    r["claims"][2]["sentence"] = "- Step 3: something never written"
    c = Counter()
    assert parse_row(r, "r", c) is None and c["drop_unaligned"] == 1


def test_dedup_conflict_masked_and_ids_stable():
    c = Counter()
    a = parse_row(row(S, [0.0, 1.0, 0.0, 0.0]), "a", c)
    b = parse_row(row(S, [0.0, 0.0, 0.0, 0.0]), "b", c)
    [m] = merge_duplicates([a, b], c)
    assert m["label_mask"][1] is False and m["y"][1] == Y_UNKNOWN
    assert m["trace_id"].startswith("rp_") and m["n_annotations"] == 2
    [m2] = merge_duplicates([b, a], Counter())
    assert m2["trace_id"] == m["trace_id"]
