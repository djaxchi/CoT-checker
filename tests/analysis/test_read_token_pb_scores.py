import json

import numpy as np
import pytest

from scripts.analysis import read_token_pb_scores as rps


def _enc(tmp_path):
    enc = tmp_path / "enc"
    enc.mkdir()
    meta = [{"id": "t1", "step_idx": 0, "first_error": 1, "label": 0},
            {"id": "t1", "step_idx": 1, "first_error": 1, "label": 1},
            {"id": "t0", "step_idx": 0, "first_error": -1, "label": 0}]
    for sub in rps.PB_SUBSETS:
        np.savez(enc / f"pb_{sub}.shard0.npz", kept=np.ones(3, bool),
                 head_p_correct=np.array([0.9, 0.2, 0.7], np.float32))
        (enc / f"pb_{sub}.shard0.meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    return enc


def test_traces_follow_the_reference_order(tmp_path, monkeypatch):
    enc = _enc(tmp_path)
    like = tmp_path / "like"
    like.mkdir()
    for sub in rps.PB_SUBSETS:
        (like / f"pb_step_scores_{sub}.jsonl").write_text(
            json.dumps({"id": "t0"}) + "\n" + json.dumps({"id": "t1"}) + "\n")
    monkeypatch.setattr("sys.argv", ["x", "--enc", str(enc), "--out", str(tmp_path / "o"), "--like", str(like)])
    rps.main()
    rows = [json.loads(l) for l in (tmp_path / "o" / "pb_step_scores_math.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["t0", "t1"]
    assert rows[1] == {"id": "t1", "label": 1, "n_steps": 2, "scores": pytest.approx([0.1, 0.8])}


def test_a_missing_trace_stops_the_run(tmp_path, monkeypatch):
    enc = _enc(tmp_path)
    like = tmp_path / "like"
    like.mkdir()
    for sub in rps.PB_SUBSETS:
        (like / f"pb_step_scores_{sub}.jsonl").write_text(json.dumps({"id": "t0"}) + "\n")
    monkeypatch.setattr("sys.argv", ["x", "--enc", str(enc), "--out", str(tmp_path / "o"), "--like", str(like)])
    with pytest.raises(SystemExit):
        rps.main()
