"""Unit tests for progress_usefulness_v0 P1 rollout aggregation (no model/GPU).

Covers context assembly (shared prefix + candidate append, empty-prefix case) and
the pure utility/gate aggregation over synthetic rollout rows.
"""

from scripts.progress_usefulness.pu_rollouts import (
    gate_summary,
    pair_contexts,
    pair_utilities,
    solve_rates,
)


def _item(prefix, step, problem="P?", gt="42"):
    return {"problem": problem, "prefix": prefix, "candidate_step": step,
            "ground_truth_answer": gt}


def test_pair_contexts_three_named_contexts_with_prefix():
    prog = _item("step A", "PROG step")
    neu = _item("step A", "NEU step")
    gt, ctxs = pair_contexts(prog, neu)
    assert gt == "42"
    names = [n for n, _ in ctxs]
    assert names == ["base", "progress", "neutral"]
    texts = dict(ctxs)
    # base = head + prefix (no candidate); progress/neutral append their step
    assert texts["base"].endswith("step A\n\n")
    assert "PROG step\n\n" in texts["progress"] and texts["progress"].startswith(texts["base"])
    assert "NEU step\n\n" in texts["neutral"] and texts["neutral"].startswith(texts["base"])
    # progress and neutral share the identical base prefix, differ only in the step
    assert texts["progress"].replace("PROG step\n\n", "") == \
           texts["neutral"].replace("NEU step\n\n", "")


def test_pair_contexts_empty_prefix():
    prog = _item("", "PROG step")
    neu = _item("", "NEU step")
    _, ctxs = pair_contexts(prog, neu)
    texts = dict(ctxs)
    # with no prefix, base is just the problem head and ends at "Solution:\n"
    assert texts["base"].endswith("Solution:\n")
    assert texts["progress"] == texts["base"] + "PROG step\n\n"


def _rows(fork, base_c, prog_c, neu_c, k=4):
    """k rollouts per context with the given number correct."""
    out = []
    for ctx, ncorrect in (("base", base_c), ("progress", prog_c), ("neutral", neu_c)):
        for i in range(k):
            out.append({"fork_id": fork, "split": "train", "context": ctx,
                        "correct": i < ncorrect})
    return out


def test_solve_rates_counts():
    rows = _rows("f1", base_c=1, prog_c=3, neu_c=2, k=4)
    sr = solve_rates(rows)
    assert sr["f1"]["base"] == (1, 4)
    assert sr["f1"]["progress"] == (3, 4)
    assert sr["f1"]["neutral"] == (2, 4)


def test_pair_utilities_computes_deltas():
    rows = _rows("f1", 1, 3, 2, k=4)
    (u,) = pair_utilities(rows)
    assert u["sr_base"] == 0.25 and u["sr_progress"] == 0.75 and u["sr_neutral"] == 0.5
    assert u["U_progress"] == 0.5 and u["U_neutral"] == 0.25


def test_pair_utilities_skips_incomplete_forks():
    rows = [{"fork_id": "f2", "context": "base", "correct": True},
            {"fork_id": "f2", "context": "progress", "correct": True}]  # no neutral
    assert pair_utilities(rows) == []


def test_gate_summary_means_and_confirmed():
    rows = _rows("f1", 1, 4, 2, k=4) + _rows("f2", 2, 2, 3, k=4)
    utils = pair_utilities(rows)
    g = gate_summary(utils)
    assert g["n_forks"] == 2
    # f1: U_prog=0.75>U_neu=0.25 (confirmed); f2: U_prog=0.0<U_neu=0.25 (not)
    assert g["n_confirmed_Uprog_gt_Uneu"] == 1
    assert g["frac_forks_Uprog_gt_Uneu"] == 0.5
    assert abs(g["mean_U_progress"] - ((0.75 + 0.0) / 2)) < 1e-9
    assert abs(g["mean_U_neutral"] - ((0.25 + 0.25) / 2)) < 1e-9


def test_gate_summary_empty():
    assert gate_summary([])["n_forks"] == 0
