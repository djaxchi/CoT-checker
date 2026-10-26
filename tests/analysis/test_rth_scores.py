import json

import numpy as np

from scripts.analysis import rth_scores


def test_traces_in_leaderboard_format_and_order(tmp_path, monkeypatch):
    meta = [{"id": "t0", "step_idx": 0, "label": 1, "n_steps": 2}, {"id": "t0", "step_idx": 1, "label": 1, "n_steps": 2},
            {"id": "t1", "step_idx": 0, "label": -1, "n_steps": 1}]
    for sub in rth_scores.PB_SUBSETS:
        part = tmp_path / "b" / sub / "part_0"
        part.mkdir(parents=True)
        np.save(part / "head_logit.npy", np.array([-2.0, 3.0, -1.0], np.float32))
        np.save(part / "index.npy", np.arange(3))
        (part / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    monkeypatch.setattr("sys.argv", ["x", "--bundle", str(tmp_path / "b"), "--out", str(tmp_path / "o")])
    rth_scores.main()
    rows = [json.loads(l) for l in (tmp_path / "o" / "pb_step_scores_math.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["t0", "t1"] and [r["label"] for r in rows] == [1, -1]
    assert np.allclose(rows[0]["scores"], 1 / (1 + np.exp(-np.array([-2.0, 3.0]))))
