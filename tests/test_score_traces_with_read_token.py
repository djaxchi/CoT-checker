import json

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts.onpolicy import score_traces_with_read_token as sr  # noqa: E402
from scripts.read_token_probe import ReadToken, read_all_states, tokenize  # noqa: E402
from tests.test_read_token_probe import tiny_model  # noqa: E402
from tests.test_read_token_store import D, CharTok  # noqa: E402

DEV = torch.device("cpu")


def _setup(tmp_path, monkeypatch):
    model = tiny_model()
    rt = ReadToken(1, D, torch.randn(1, D) * 0.02, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, 0.3)
    ck = tmp_path / "tok" / "read_token.pt"
    ck.parent.mkdir()
    torch.save({"soft": rt.soft.detach(), "head": rt.head.state_dict(), "m": 1, "read": "last",
                "head_kind": "linear", "deep": rt.deep.detach()}, ck)
    monkeypatch.setattr(sr, "load_model", lambda *_a: (model, CharTok()))
    trajs = tmp_path / "t.jsonl"
    rows = [{"traj_uid": f"q{k}::g0", "fork_id": f"q{k}", "problem": f"p{k}", "correct": k % 2 == 0,
             "gradeable": True, "solution": "a b\n\nc d\n\ne" if k else "only"} for k in range(3)]
    trajs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return model, rt, ck, trajs, rows


def test_scores_are_the_head_on_each_step_read_under_its_prefix(tmp_path, monkeypatch):
    model, rt, ck, trajs, rows = _setup(tmp_path, monkeypatch)
    out = tmp_path / "s.jsonl"
    monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--trajectories", str(trajs),
                                     "--out", str(out), "--chunk", "2"])
    sr.main()
    got = {r["traj_uid"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert got["q1::g0"]["n_steps"] == 3 and got["q0::g0"]["n_steps"] == 1
    assert got["q1::g0"]["cell"] == "rth_tok"
    for r in rows:
        for k, it in enumerate(sr.step_items(r)):
            with torch.no_grad():
                h = read_all_states(model, rt, [tokenize(CharTok(), it)], 0, torch.device("cpu"),
                                    grad=False)[:, -1]
                want = torch.sigmoid(rt.logit(h)).item()
            assert np.isclose(got[r["traj_uid"]]["scores"][k], want, atol=2e-3)
    assert sr.step_items(rows[1])[2] == {"problem": "p1", "prefix": "a b\n\nc d", "step": "e"}


def test_segmentation_gate_refuses_a_mismatch(tmp_path, monkeypatch):
    _, _, ck, trajs, _ = _setup(tmp_path, monkeypatch)
    ref = tmp_path / "prm.jsonl"
    ref.write_text(json.dumps({"traj_uid": "q1::g0", "n_steps": 2}) + "\n")
    monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--trajectories", str(trajs),
                                     "--out", str(tmp_path / "s.jsonl"), "--verify_against", str(ref)])
    with pytest.raises(SystemExit):
        sr.main()


def test_branched_scores_match_and_compare_reports_agreement(tmp_path, monkeypatch, capsys):
    _, _, ck, trajs, _ = _setup(tmp_path, monkeypatch)
    full, br = tmp_path / "full.jsonl", tmp_path / "br.jsonl"
    monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--trajectories", str(trajs), "--out", str(full)])
    sr.main()
    monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--trajectories", str(trajs), "--out", str(br),
                                     "--branched", "--compare_to", str(full)])
    sr.main()
    a = {r["traj_uid"]: r["scores"] for r in map(json.loads, full.read_text().splitlines())}
    b = {r["traj_uid"]: r["scores"] for r in map(json.loads, br.read_text().splitlines())}
    assert a.keys() == b.keys()
    for u in a:
        assert np.allclose(a[u], b[u], atol=1e-4)
    assert "max_abs_diff" in capsys.readouterr().out


def test_fork_ids_keeps_only_the_listed_questions(tmp_path, monkeypatch):
    _, _, ck, trajs, _ = _setup(tmp_path, monkeypatch)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps(["q1", "q2"]))
    out = tmp_path / "s.jsonl"
    monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--trajectories", str(trajs),
                                     "--out", str(out), "--fork_ids", str(keep), "--branched"])
    sr.main()
    got = [json.loads(l)["problem_id"] for l in out.read_text().splitlines()]
    assert got == ["q1", "q2"]


def test_inline_scores_are_a_read_token_after_each_step_of_the_chat_text(tmp_path, monkeypatch):
    from src.onpolicy.prompts import chat_prompt
    from tests.test_read_token_inline import CliTok
    model, rt, ck, trajs, rows = _setup(tmp_path, monkeypatch)
    tok = CliTok()
    monkeypatch.setattr(sr, "load_model", lambda *_a: (model, tok))
    rows = [dict(r, prompt_style="chat") for r in rows]
    trajs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "s.jsonl"
    monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--trajectories", str(trajs),
                                     "--out", str(out), "--inline", "--max_tokens", "2000"])
    sr.main()
    got = {r["traj_uid"]: r["scores"] for r in map(json.loads, out.read_text().splitlines())}
    steps = ["a b", "c d", "e"]
    for k in range(3):
        ids = tok(chat_prompt("p1") + "\n\n".join(steps[:k + 1]) + "\n\n")["input_ids"]
        with torch.no_grad():
            ref = torch.sigmoid(rt.logit(read_all_states(model, rt, [ids], 0, DEV, grad=False)[:, -1]))
        assert abs(got["q1::g0"][k] - float(ref)) < 1e-4, k
    manifest = json.loads((tmp_path / "s_manifest.json").read_text())
    assert manifest["template"] == "inline_chat" and manifest["tokens"] > 0


def test_inline_refuses_a_pool_sampled_under_another_prompt(tmp_path, monkeypatch):
    from tests.test_read_token_inline import CliTok
    model, rt, ck, trajs, rows = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(sr, "load_model", lambda *_a: (model, CliTok()))
    monkeypatch.setattr("sys.argv", ["x", "--ckpt", str(ck), "--trajectories", str(trajs),
                                     "--out", str(tmp_path / "s.jsonl"), "--inline"])
    with pytest.raises(SystemExit):
        sr.main()
