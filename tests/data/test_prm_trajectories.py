"""Raw PRM800K phase-2 alignment, label masks, dedup and problem splits."""

from collections import Counter

from src.data.prm_trajectories import (
    Y_UNKNOWN, align_record, assign_splits, merge_duplicates, problem_key, processbench_trace,
)

PRE = ["Step A.", "Step B.", "Step C.", "Step D."]


def comp(text, rating, flagged=None):
    return {"text": text, "rating": rating, "flagged": flagged}


def record(steps, pre=PRE, problem="What is 1+1?", finish="found_error"):
    return {"question": {"problem": problem, "pre_generated_steps": list(pre)},
            "label": {"steps": steps, "finish_reason": finish}}


def st(comps, chosen=0, human=None):
    return {"completions": comps, "chosen_completion": chosen, "human_completion": human}


def test_original_continuation_kept_and_labels_aligned():
    rec = record([st([comp("Step A.", 1)]), st([comp("Step B.", 0)]),
                  st([comp("Step C.", -1), comp("Alt C.", 1)], chosen=None)])
    a = align_record(rec, "r0", Counter())
    assert a["steps"] == PRE  # unlabeled step D kept as context
    assert a["y"] == [0, 0, 1, Y_UNKNOWN]
    assert a["label_mask"] == [True, True, True, False]
    assert a["ratings"][:3] == [1, 0, -1]
    assert a["first_error"] == 2


def test_borrowed_suffix_after_human_repair_is_masked():
    # Step B is repaired by a human: later labels refer to the human branch,
    # even though a completion text matches the original step C.
    rec = record([st([comp("Step A.", 1)]),
                  st([comp("Step B.", -1)], chosen=None, human={"text": "Fixed B."}),
                  st([comp("Step C.", 1)]), st([comp("Step D.", 1)])])
    c = Counter()
    a = align_record(rec, "r1", c)
    assert a["steps"] == PRE  # the input is the ORIGINAL trajectory, never the repair
    assert a["y"] == [0, 1, Y_UNKNOWN, Y_UNKNOWN]
    assert a["label_mask"] == [True, True, False, False]
    assert a["step_status"][2:] == ["off_branch", "off_branch"]
    assert c["branch_diverged"] == 1


def test_alternative_chosen_earlier_masks_later():
    rec = record([st([comp("Step A.", 0), comp("Alt A.", 1)], chosen=1),
                  st([comp("Step B.", 1)])])
    a = align_record(rec, "r2", Counter())
    assert a["label_mask"] == [True, False, False, False]


def test_flagged_unrated_and_contradictory_are_masked():
    rec = record([st([comp("Step A.", 1)]), st([comp("Step B.", 1, flagged=True)]),
                  st([comp("Step C.", None)]), st([comp("Step D.", 1), comp("Step D.", -1)], chosen=None)],
                 finish="give_up")
    a = align_record(rec, "r3", Counter())
    assert a["label_mask"] == [True, False, False, False]
    assert a["step_status"][1:] == ["flagged", "unrated", "contradictory"]


def test_rating_map():
    rec = record([st([comp("Step A.", 1)]), st([comp("Step B.", 0)]), st([comp("Step C.", -1)], chosen=None)])
    a = align_record(rec, "r4", Counter())
    assert a["y"][:3] == [0, 0, 1]


def test_bad_problem_and_no_label_dropped():
    c = Counter()
    assert align_record(record([st([comp("Step A.", 1)])], finish="bad_problem"), "x", c) is None
    assert align_record(record([st([comp("Step A.", None)])], finish="give_up"), "y", c) is None
    assert c["drop_bad_problem"] == 1 and c["drop_no_trustworthy_label"] == 1


def test_dedup_merges_agreeing_and_masks_conflicts():
    c = Counter()
    r1 = align_record(record([st([comp("Step A.", 1)]), st([comp("Step B.", 1)])]), "a", c)
    r2 = align_record(record([st([comp("Step A.", 0)]), st([comp("Step B.", -1)], chosen=None)]), "b", c)
    [m] = merge_duplicates([r1, r2], c)
    assert m["y"][0] == 0 and m["label_mask"][0]  # +1 vs 0: both correct
    assert not m["label_mask"][1]  # -1 vs +1: conflicting, no adjudication
    assert m["source_record_ids"] == ["a", "b"] and m["n_annotations"] == 2


def _traces(problems):
    out = []
    for i, p in enumerate(problems):
        out.append({"problem_key": problem_key(p), "first_error": 1 if i % 3 else None})
    return out


def test_duplicate_problem_text_never_crosses_splits():
    base = [f"Problem number {i}" for i in range(400)]
    # same problem, different spacing/case and different record ids
    dups = [f"  problem   NUMBER {i} " for i in range(0, 400, 7)]
    tr = _traces(base + dups)
    a = assign_splits(tr, seed=1729)
    for i in range(0, 400, 7):
        assert problem_key(base[i]) == problem_key(dups[i // 7])
    assert set(a.values()) == {"train", "dev", "calib", "test"}
    n = Counter(a.values())
    assert abs(n["dev"] - n["calib"]) <= 2
    assert 0.75 < n["train"] / len(a) < 0.85


def test_pb_overlap_exclusion_and_determinism():
    tr = _traces([f"P{i}" for i in range(100)])
    ex = {problem_key("P3")}
    a1 = assign_splits(tr, seed=1729, exclude_problem_keys=ex)
    a2 = assign_splits(tr, seed=1729, exclude_problem_keys=ex)
    assert a1 == a2 and a1[problem_key("P3")] == "excluded_pb_overlap"


def test_processbench_known_label_convention():
    t = processbench_trace({"id": "x", "problem": "q", "steps": ["a", "b", "c", "d"], "label": 1}, "math")
    assert t["y"] == [0, 1, Y_UNKNOWN, Y_UNKNOWN] and t["label_mask"] == [True, True, False, False]
    t = processbench_trace({"id": "y", "problem": "q", "steps": ["a", "b"], "label": -1}, "math")
    assert t["y"] == [0, 0] and all(t["label_mask"])
