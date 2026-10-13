"""step_value_labels_v1: cut offsets, per-step risk, chunked sampler, set building, labeling."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "step_value_labels"))

from src.analysis.step_value_labels import cut_offsets, sample_rounds, step_risks  # noqa: E402

EOS = 0


def test_cut_offsets_tiling():
    assert cut_offsets([[0, 4], [4, 9], [9, 12]], 12) == [0, 4, 9, 12]


@pytest.mark.parametrize("spans,ans", [([[1, 4], [4, 9]], 9), ([[0, 4], [5, 9]], 9), ([[0, 4], [4, 9]], 10)])
def test_cut_offsets_rejects_gaps(spans, ans):
    with pytest.raises(ValueError):
        cut_offsets(spans, ans)


def test_step_risks_difference_and_se():
    err = {0: [0, 0, 1, 1], 1: [1, 1, 1, 1], 2: [1, 1, 1, 1]}
    rs = step_risks(err, 2)
    assert [r["step"] for r in rs] == [0, 1]
    assert rs[0]["risk"] == pytest.approx(0.5)
    assert rs[0]["se"] == pytest.approx(math.sqrt(0.25 / 4))
    assert rs[0]["harmful"] is False              # 0.5 is exactly 2 SE, not above
    assert rs[1]["risk"] == 0 and rs[1]["se"] == 0 and rs[1]["harmful"] is False


def test_step_risks_skips_steps_without_both_cuts():
    rs = step_risks({3: [0, 1], 4: [1, 1]}, 6)
    assert [r["step"] for r in rs] == [3]


def scripted_gen(seqs: dict[int, list[int]], calls: list):
    """gen() continuing row r's scripted sequence; the context's first id is r."""
    def gen(ctxs, max_new):
        calls.append((len(ctxs), max_new))
        out = []
        for c in ctxs:
            seq, done = seqs[c[0]], len(c) - 1
            g = seq[done:done + max_new]
            out.append(g + [EOS] * (max_new - len(g)))   # padding after the end, as HF pads
        return out
    return gen


def test_sample_rounds_continues_across_chunks():
    seqs = {0: [5, 6, 7, 8, 9, EOS], 1: [5, EOS], 2: [5] * 20}
    calls = []
    res = sample_rounds(scripted_gen(seqs, calls), [[0], [1], [2]], [100, 100, 10], {EOS}, chunk=2, batch=8)
    assert res[0] == {"ids": [5, 6, 7, 8, 9, EOS], "eos": True, "truncated": False}
    assert res[1] == {"ids": [5, EOS], "eos": True, "truncated": False}
    assert res[2] == {"ids": [5] * 10, "eos": False, "truncated": True}   # budget, no EOS
    assert all(m <= 2 for _, m in calls)
    assert calls[0][0] == 3 and calls[1][0] == 2                           # finished rows drop out


def test_sample_rounds_eos_beyond_budget_is_truncated_and_zero_budget_skipped():
    seqs = {0: [5, 5, 5, EOS], 1: [5, EOS]}
    calls = []
    res = sample_rounds(scripted_gen(seqs, calls), [[0], [1]], [3, 0], {EOS}, chunk=8, batch=8)
    assert res[0] == {"ids": [5, 5, 5], "eos": False, "truncated": True}
    assert res[1] == {"ids": [], "eos": False, "truncated": True}
    assert calls == [(1, 3)]


def test_sample_rounds_batches():
    seqs = {r: [5, EOS] for r in range(5)}
    calls = []
    sample_rounds(scripted_gen(seqs, calls), [[r] for r in range(5)], [10] * 5, {EOS}, chunk=4, batch=2)
    assert [n for n, _ in calls] == [2, 2, 1]


def fork(step=1):
    return {"fork_id": "t1#s1", "trace_id": "t1", "problem_key": "pk1", "validity": "incorrect", "gold": "5",
            "prompt_ids": [1, 2], "reply_ids": list(range(10, 22)), "n_steps": 3, "step": step,
            "step_spans": [[0, 4], [4, 8], [8, 10]], "answer_span": [10, 11]}


def test_validation_records_cut_pair():
    from build_trace_set import validation_records
    r = validation_records([fork(1), fork(2)])
    assert r[0]["cuts"] == [1, 2] and r[0]["cut_offsets"] == [4, 8]
    assert r[1]["cuts"] == [2, 3] and r[1]["cut_offsets"] == [8, 10]   # last step: the answer line's start
    assert r[0]["group"] == "fork_incorrect" and r[0]["trace_id"] == "t1#s1"


def test_select_quota_order_and_problem_cap():
    from build_trace_set import select, sha1
    recs = [{"trace_id": f"e{i}", "problem_key": "pA" if i < 3 else f"p{i}", "group": "error"} for i in range(5)]
    recs += [{"trace_id": f"h{i}", "problem_key": "pA", "group": "hard_clean"} for i in range(2)]
    recs += [{"trace_id": f"z{i}", "problem_key": f"q{i}", "group": "easy"} for i in range(4)]
    out = select(recs, {"error": None, "hard_clean": 5, "easy": 2}, max_per_problem=2)
    errs = [r for r in out if r["group"] == "error"]
    assert len(errs) == 4                                        # pA capped at 2 of its 3
    assert not [r for r in out if r["group"] == "hard_clean"]    # pA already full
    easy = [r["trace_id"] for r in out if r["group"] == "easy"]
    assert easy == sorted([f"z{i}" for i in range(4)], key=sha1)[:2]


class CharTok:
    """ids >= 10 are characters (newline included); 0 is EOS."""
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if i >= 10)


def ids(s):
    return [ord(ch) for ch in s]


def test_label_group_end_to_end():
    from label_values import label_group
    reply = ids("- Step 1: a\n") + ids("- Step 2: b\n") + ids("<Answer>: 5") + [EOS]
    t = {"trace_id": "t", "problem_key": "pk", "group": "error", "gold": "5", "n_steps": 2,
         "prompt_ids": [1], "reply_ids": reply, "cuts": [0, 1, 2], "cut_offsets": [0, 12, 24]}
    calls = []
    # the model's reply is a function of its text so far: from cut 0 it takes a wrong
    # first step and answers 4; once the real step 1 is in the context it answers 5
    targets = [ids("- Step 1: x\n<Answer>: 4") + [EOS], reply]

    def gen(ctxs, max_new):
        calls.append(len(ctxs))
        out = []
        for c in ctxs:
            so_far = c[1:]
            tgt = next(w for w in targets if w[:len(so_far)] == so_far)
            seq = tgt[len(so_far):]
            out.append((seq + [EOS] * max_new)[:max_new])
        return out
    args = SimpleNamespace(k=3, max_reply_tokens=4096, chunk=8, gen_batch=4)
    (row,) = label_group(gen, CharTok(), [(0, t)], args, {EOS})
    assert [c["k"] for c in row["cuts"]] == [3, 3, 3]
    assert [c["err"] for c in row["cuts"]] == [1.0, 0.0, 0.0]
    assert [s["risk"] for s in row["steps"]] == [-1.0, 0.0]
    assert all(n <= 4 for n in calls)


def test_step_table_view_ids_and_phase():
    from summarize_values import step_table
    r = {"trace_id": "rp_x", "problem_key": "pk", "group": "error", "n_steps": 3, "step_claim_idx": [0, 1, 3],
         "y_ds": [0, 1, 1], "mask_ds": [True, True, True], "y_self": [0, 1, 0], "mask_self": [True] * 3,
         "first_error_ds": 1, "cuts": [{"cut": 0, "err": 0.25}],
         "steps": [{"step": i, "risk": 0.0, "harmful": False, "err_before": 0.0} for i in range(3)]}
    T = step_table([r])
    assert [t["view_id"] for t in T] == ["rp_x#0", "rp_x#1", "rp_x#3"]
    assert [t["phase"] for t in T] == ["pre_error", "first_error", "post_error"]
    assert T[0]["err0"] == 0.25 and T[2]["relpos"] == 1.0


def test_validate_against_track1(tmp_path):
    import json
    from summarize_values import validate
    t1 = [{"fork_id": f"f{i}", "risk": {"err_c_mean": 0.5, "err_a": 0.5 + 0.1 * (i % 2), "risk_diff": 0.1 * (i % 2)}}
          for i in range(40)]
    (tmp_path / "track1.shard00.jsonl").write_text("".join(json.dumps(r) + "\n" for r in t1))
    rows = [{"trace_id": f"f{i}", "fork_step": 2, "cuts": [{"cut": 2, "err": 0.5}, {"cut": 3, "err": 0.5 + 0.1 * (i % 2)}]}
            for i in range(40)]
    v = validate(rows, tmp_path, 0.03)
    assert v["n_forks"] == 40 and v["pass"]
    assert v["err_after_minus_err_a"]["mean"] == pytest.approx(0.0)


def claim(sent, lo):
    return {"sentence": sent, "aligned_token_ids": list(range(lo, lo + len(sent)))}


def test_parse_unfinished_row_keeps_complete_lines():
    from src.analysis.step_value_labels import parse_unfinished_row
    l1, l2, frag = "- Step 1: a\n", "- Step 2: bb\n", "- Step 3: cut"
    prompt = [1, 2]
    reply = ids(l1 + l2 + frag)
    cl = [claim(l1, 0), claim(l2, len(l1)), claim(frag, len(l1) + len(l2))]
    p, why = parse_unfinished_row(prompt + reply, prompt, cl, CharTok().decode, EOS)
    assert why == "ok"
    assert p["step_spans"] == [(0, 12), (12, 25)] and p["end"] == 25 and p["n_dropped_tokens"] == len(frag)
    assert p["reply_ids"] == reply[:25] and p["step_claim_idx"] == [0, 1]
    assert parse_unfinished_row(prompt + reply + [EOS], prompt, cl, CharTok().decode, EOS)[1] == "has_eos"
    assert parse_unfinished_row([9] + reply, prompt, cl, CharTok().decode, EOS)[1] == "prompt_mismatch"


def test_batches_respect_row_and_token_caps():
    from src.analysis.step_value_labels import _batches
    assert _batches([0, 1, 2, 3, 4], [10] * 5, 2, None, 8) == [[0, 1], [2, 3], [4]]
    # (rows) x (longest + chunk) <= 60: 3 rows of length 10 with chunk 10 fit, a 4th does not
    assert _batches([0, 1, 2, 3], [10, 10, 10, 10], 99, 60, 10) == [[0, 1, 2], [3]]
    # a long row starts its own batch once it would push the padded size over the cap
    assert _batches([0, 1, 2], [5, 5, 50], 99, 70, 5) == [[0, 1], [2]]


def test_sample_rounds_pool_rebatches_running_rows():
    seqs = {r: [5] * (3 if r < 4 else 9) + [EOS] for r in range(6)}
    calls = []
    res = sample_rounds(scripted_gen(seqs, calls), [[r] for r in range(6)], [50] * 6, {EOS}, chunk=4, batch=4)
    assert all(x["eos"] for x in res) and [len(x["ids"]) for x in res] == [4] * 4 + [10] * 2
    assert calls[:2] == [(4, 4), (2, 4)]       # round 1: two batches of the 6 rows
    assert calls[2][0] == 2                     # round 2: only the 2 long rows, one batch
