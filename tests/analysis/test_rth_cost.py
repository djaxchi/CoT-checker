import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts.analysis import rth_cost  # noqa: E402
from scripts.read_token_probe import ReadToken  # noqa: E402
from tests.test_read_token_probe import tiny_model  # noqa: E402
from tests.test_read_token_store import D, CharTok  # noqa: E402


class ChatTok(CharTok):
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        return "".join(m["content"] for m in msgs)


def test_reports_all_three_scorers(tmp_path, monkeypatch):
    model = tiny_model()
    rt = ReadToken(1, D, torch.randn(1, D) * 0.02, n_layers=3)
    ck = tmp_path / "rt.pt"
    torch.save({"soft": rt.soft.detach(), "head": rt.head.state_dict(), "m": 1, "read": "last",
                "head_kind": "linear", "deep": rt.deep.detach()}, ck)
    monkeypatch.setattr(rth_cost, "load_model", lambda *_a: (model, ChatTok()))
    trajs = tmp_path / "t.jsonl"
    trajs.write_text("".join(json.dumps({"traj_uid": f"q{k}", "problem": "p", "gradeable": True,
                                         "solution": "a\n\nb\n\nc"}) + "\n" for k in range(4)))
    monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--trajectories", str(trajs), "--n", "3",
                                     "--warmup", "1", "--out", str(tmp_path / "e1.json")])
    rth_cost.main()
    res = json.loads((tmp_path / "e1.json").read_text())
    for k in ("rth", "passive", "prm"):
        assert res[k]["traces"] == 3 and res[k]["sec_per_step"] > 0
