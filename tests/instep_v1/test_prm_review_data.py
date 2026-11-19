import json

import numpy as np

from scripts.encode_pooled_multilayer import prm800k_items
from scripts.instep_v1.prm_review_data import build_traces, draw_indices, items_of
from tests.test_read_token_inline import CharTok


def test_draw_indices_are_the_training_draw(tmp_path):
    rows = [{"uid": f"u{i}", "problem": f"q{i % 7}", "prefix": "", "candidate_step": "s", "label": i % 2}
            for i in range(50)]
    path = tmp_path / "f.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for seed in (0, 1, 2):
        got = [rows[i]["uid"] for i in draw_indices(len(rows), 20, seed)]
        assert got == [it["uid"] for it in prm800k_items(path, 20, seed)]
    assert list(draw_indices(5, 0, 0)) == list(range(5))


def test_build_traces_steps_are_the_member_items(monkeypatch):
    import src.onpolicy.prompts as pr
    monkeypatch.setattr(pr, "chat_prompt", lambda problem: f"<{problem}>")
    rows = [{"uid": "a", "problem": "p", "prefix": "", "candidate_step": "ab", "label": 0},
            {"uid": "b", "problem": "p", "prefix": "ab", "candidate_step": "cd", "label": 1},
            {"uid": "c", "problem": "p", "prefix": "ab\n\ncd", "candidate_step": "e", "label": 0},
            {"uid": "d", "problem": "p", "prefix": "ab", "candidate_step": "xy", "label": 1},  # other branch
            {"uid": "e", "problem": "q", "prefix": "zz", "candidate_step": "w", "label": 1}]   # prefix not drawn
    traces, empty = build_traces(CharTok(), items_of(rows))
    assert empty == 0
    by = {tuple(t["uids"]): t for t in traces}
    t = by[("a", "b", "c")]
    text = "<p>ab\n\ncd\n\ne"
    assert t["context_len"] == 3 and len(t["input_ids"]) == len(text)
    reply = text[3:]
    assert ["".join(reply[i] for i in s["aligned_token_ids"]) for s in t["steps"]] == ["ab", "cd", "e"]
    assert t["labels"] == [0, 1, 0]
    t = by[("e",)]
    assert "".join("<q>zz\n\nw"[3 + i] for i in t["steps"][0]["aligned_token_ids"]) == "w"
    assert sorted(u for t in traces for u in t["uids"]) == ["a", "b", "c", "d", "e"]
    assert np.all([len(t["steps"]) == len(t["labels"]) for t in traces])
