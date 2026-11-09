"""tts_judge: verdict parsing and the merge into best-of-N grades, beam grades and agreement."""

import argparse
import json

from scripts.instep_v1 import tts_judge as J


def test_parse_grade():
    assert J.parse_grade("Solution comments: fine\n<Grade>: Correct\n<end of response>") is True
    assert J.parse_grade("<Grade>: (Incorrect)") is False
    assert J.parse_grade("<think>maybe <Grade>: Correct</think>\n<Grade>: Incorrect") is False
    assert J.parse_grade("no verdict here") is None


def test_merge(tmp_path):
    items = [{"jid": "bon|trips|0|0", "set": "trips"}, {"jid": "bon|trips|0|1", "set": "trips"},
             {"jid": "beam|proofnet|3|rt_self", "set": "proofnet"},
             {"jid": "val|math|1|2", "set": "math", "prog": True}, {"jid": "val|math|1|3", "set": "math", "prog": False}]
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    v = [{"jid": "bon|trips|0|0", "correct": True}, {"jid": "bon|trips|0|1", "correct": None},
         {"jid": "beam|proofnet|3|rt_self", "correct": False},
         {"jid": "val|math|1|2", "correct": True}, {"jid": "val|math|1|3", "correct": True}]
    (tmp_path / "verdicts.shard0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in v))
    bon = tmp_path / "bon"
    bon.mkdir()
    J.cmd_merge(argparse.Namespace(out=tmp_path, bon=bon))
    g = [json.loads(l) for l in open(bon / "trips.graded.jsonl")]
    assert [(r["qi"], r["k"], r["ok"], r["judged"]) for r in g] == [(0, 0, True, True), (0, 1, False, False)]
    assert json.loads((tmp_path / "beam_judged.json").read_text()) == {"proofnet": {"3|rt_self": False}}
    ag = json.loads((tmp_path / "judge_agreement.json").read_text())["math"]
    assert ag["n"] == 2 and ag["agreement"] == 0.5 and ag["prog_wrong_judge_correct"] == 1


def test_prompts_pass_a_real_bool_for_thinking():
    """Qwen3's template checks `enable_thinking is false`; the CLI's integer 0 must become False."""
    seen = []

    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
            seen.append(enable_thinking)
            return "x"

    it = {"problem": "p", "reference": "r", "solution": "s"}
    J._prompts(Tok(), [it], 0)
    J._prompts(Tok(), [it], 1)
    assert seen == [False, True] and all(type(v) is bool for v in seen)


def test_needed_candidates_keeps_candidate_zero_and_every_pick():
    meta = [{"qi": q, "k": k} for q in (0, 1) for k in range(3)]
    a = [0.5, 0.1, 0.9, 0.2, 0.8, 0.3]  # picks q0 k1, q1 k0
    b = [0.5, 0.9, 0.1, 0.9, 0.8, 0.3]  # picks q0 k2, q1 k2
    assert J.needed_candidates(meta, [a, b]) == {(0, 0), (1, 0), (0, 1), (0, 2), (1, 2)}


def test_build_only_limits_bon_items(tmp_path, monkeypatch):
    import pandas as pd
    from scripts.reprobe_cmp import bon_d0
    df = pd.DataFrame([{"q": "Q", "qi": 0, "k": k, "answer": "A", "reply": f"r{k}"} for k in range(5)])
    monkeypatch.setattr(bon_d0, "load_pool", lambda root, s: df)
    (tmp_path / "picks.json").write_text(json.dumps({"trips": [[0, 0], [0, 3]]}))
    J.cmd_build(argparse.Namespace(pools=tmp_path, beam=None, sets="trips", val_per_set=0, what="bon",
                                   out=tmp_path / "j", only=tmp_path / "picks.json"))
    items = [json.loads(l) for l in open(tmp_path / "j" / "items.jsonl")]
    assert [it["jid"] for it in items] == ["bon|trips|0|0", "bon|trips|0|3"]


def test_build_beam_arms_filter(tmp_path, monkeypatch):
    import pandas as pd
    from scripts.reprobe_cmp import bon_d0
    df = pd.DataFrame([{"q": "Q", "qi": 0, "k": 0, "answer": "A", "reply": "r"}])
    monkeypatch.setattr(bon_d0, "load_pool", lambda root, s: df)
    beam = tmp_path / "beam"
    beam.mkdir()
    rows = [{"set": "trips", "qi": 0, "arm": a, "steps": ["s"]} for a in ("prm7b", "prm800k", "instep_ds")]
    (beam / "trips.v1.part0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    J.cmd_build(argparse.Namespace(pools=tmp_path, beam=beam, sets="trips", val_per_set=0, what="beam",
                                   out=tmp_path / "j", only=None, arms="prm7b,prm800k"))
    items = [json.loads(l) for l in open(tmp_path / "j" / "items.jsonl")]
    assert [it["jid"] for it in items] == ["beam|trips|0|prm7b", "beam|trips|0|prm800k"]
