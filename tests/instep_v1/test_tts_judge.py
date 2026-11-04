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
