"""tts_tables: best-of-N rows (pass@1, pass@N, majority, stored, arms), beam rows on common questions."""

import json

import numpy as np

from scripts.instep_v1 import tts_tables as T

UH = T.UHEAD


def _w(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_bon_table(tmp_path):
    bon = tmp_path / "bon"
    (bon / "inputs").mkdir(parents=True)
    # 2 questions x 3 candidates; q0: k1 correct, q1: k0, k2 correct
    ok = {(0, 0): False, (0, 1): True, (0, 2): False, (1, 0): True, (1, 1): False, (1, 2): True}
    key = {(0, 0): "a", (0, 1): "b", (0, 2): "a", (1, 0): "c", (1, 1): "d", (1, 2): "c"}
    _w(bon / "math.graded.jsonl", [{"qi": q, "k": k, "ok": v, "key": key[(q, k)]} for (q, k), v in ok.items()])
    stored = {(0, 0): 0.9, (0, 1): 0.1, (0, 2): 0.5, (1, 0): 0.9, (1, 1): 0.1, (1, 2): 0.5}
    _w(bon / "inputs" / "math.meta.jsonl", [{"trace_id": f"t{q}{k}", "qi": q, "k": k, "stored": {UH: stored[(q, k)]}}
                                           for (q, k) in ok])
    arm = tmp_path / "arm"
    arm.mkdir()
    # rp arm: per-step P(incorrect), max step = trace score; picks k1 for q0 (right), k2 for q1 (right)
    p = {"t00": [0.2, 0.8], "t01": [0.1, 0.3], "t02": [0.5], "t10": [0.6], "t11": [0.9], "t12": [0.05, 0.4]}
    _w(arm / "bon_math.jsonl", [{"uid": f"{t}::{i}", "p": v} for t, vs in p.items() for i, v in enumerate(vs)])
    out = T.bon_table(bon, {"InStep-DS": ("rp", [str(arm)])}, ("math", "trips"), 50, [("InStep-DS", "pass@1")])
    rows = {r["row"]: r for r in out["rows"]}
    assert np.isclose(rows["Qwen3-8B pass@1 (lower bound)"]["math"], 100 * (1 / 3 + 2 / 3) / 2)
    assert rows["Qwen3-8B pass@N (upper bound)"]["math"] == 100
    assert rows["Majority voting"]["math"] == 50  # q0 majority "a" wrong, q1 "c" right
    assert rows["ReProbe, Attn+Logit, DeepSeek-anno"]["math"] == 50  # lowest stored: q0 k1 right, q1 k1 wrong
    assert rows["InStep, DeepSeek-anno"]["math"] == 100
    assert rows["InStep, DeepSeek-anno"]["trips"] is None and rows["InStep, DeepSeek-anno"]["Overall"] is None
    assert rows["Skywork-PRM-1.5B"]["note"] and "math" not in rows["Skywork-PRM-1.5B"]
    assert out["notes"] == {"trips": "not graded yet"}
    assert out["pairs"][0]["sets"] == ["math"]


def test_beam_table_common_questions_and_judge(tmp_path):
    beam = tmp_path / "beam"
    beam.mkdir()
    rows = [{"set": "math", "qi": 0, "arm": "random", "correct": False, "seconds": 2.0},
            {"set": "math", "qi": 0, "arm": "instep_ds", "correct": True, "seconds": 3.0},
            {"set": "math", "qi": 1, "arm": "instep_ds", "correct": True},  # random never ran q1: dropped
            {"set": "proofnet", "qi": 5, "arm": "instep_ds", "correct": None},
            {"set": "proofnet", "qi": 5, "arm": "random", "correct": None}]
    _w(beam / "x.jsonl", rows)
    judged = {"proofnet": {"5|instep_ds": True, "5|random": False}}
    out = T.beam_table(beam, judged, ("math", "proofnet"), 50, [])
    r = {x["row"]: x for x in out["rows"]}
    assert r["InStep, DeepSeek-anno"]["math"] == 100 and r["Random scores (same search)"]["math"] == 0
    assert r["InStep, DeepSeek-anno"]["proofnet"] == 100 and r["Random scores (same search)"]["proofnet"] == 0
    assert out["info"]["math"]["questions"] == 1
    # seconds per question, on the same common questions as the accuracy
    assert out["seconds"]["instep_ds"]["math"] == 3.0 and out["seconds"]["random"]["math"] == 2.0
    assert out["seconds"]["random"]["proofnet"] is None  # not logged


def test_averages_follow_paper():
    row = {"math": 90, "gsm8k": 90, "proofnet": 90, "trips": 10, "calendar": 30}
    a = T.averages(row, ("math", "gsm8k", "proofnet", "trips", "calendar"))
    assert a == {"ID": 90.0, "OOD": 20.0, "Overall": 62.0}


def test_bon_curve_restricts_to_the_first_n_candidates(tmp_path):
    bon = tmp_path / "bon"
    (bon / "inputs").mkdir(parents=True)
    # 1 question x 4 candidates; only k2 is correct; the arm scores k2 lowest
    ok = {0: False, 1: False, 2: True, 3: False}
    _w(bon / "math.graded.jsonl", [{"qi": 0, "k": k, "ok": v, "key": "abca"[k]} for k, v in ok.items()])
    _w(bon / "inputs" / "math.meta.jsonl", [{"trace_id": f"t{k}", "qi": 0, "k": k, "stored": {UH: 0.5}}
                                           for k in ok])
    # a set whose pool is too small for the largest N is left out of the curve
    _w(bon / "calendar.graded.jsonl", [{"qi": 0, "k": 0, "ok": True, "key": "x"}])
    _w(bon / "inputs" / "calendar.meta.jsonl", [{"trace_id": "c0", "qi": 0, "k": 0, "stored": {UH: 0.5}}])
    arm = tmp_path / "arm"
    arm.mkdir()
    _w(arm / "bon_math.jsonl", [{"uid": f"t{k}::0", "p": 0.1 if k == 2 else 0.9} for k in ok])
    _w(arm / "bon_calendar.jsonl", [{"uid": "c0::0", "p": 0.1}])
    out = T.bon_curve(bon, {"InStep-self": ("rp", [str(arm)])}, ("math", "calendar"), (1, 2, 3, 4))
    assert out["sets"] == ["math"] and out["n"] == [1, 2, 3, 4]
    assert out["curves"]["InStep-self"] == [0.0, 0.0, 100.0, 100.0]  # k2 enters at N = 3
    assert out["curves"]["majority"][:2] == [0.0, 0.0]  # "a" wins (first on ties) at N = 1, 2
