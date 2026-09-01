"""The API judge must be the same judge, and must not be able to overspend.

Two risks. If the prompt or the parser drifted from the local judges', the
bake-off table would be comparing prompts rather than models, and the number that
decides which labeller we trust would be meaningless. And a run that costs money
must stop where it was told to and resume without paying twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from scripts.onpolicy import judge_openrouter as api  # noqa: E402
from scripts.onpolicy import judge_steps as local  # noqa: E402


class Args:
    retry_empty = False
    tell_outcome = True
    show_gold = False
    cot = False
    n_votes = 1
    max_tokens = 64
    temperature = 0.0
    reasoning_effort = None
    retries = 0
    retry_delay = 0.0
    timeout = 5.0


TRACE = {"id": "t0", "problem": "p?", "steps": ["a", "b", "c"],
         "final_answer_correct": False, "label": 1}


def test_the_api_judge_sends_the_prompt_the_local_judges_were_scored_on(monkeypatch):
    sent = {}

    def fake_call(prompt, model, key, args):
        sent["prompt"] = prompt
        return {"choices": [{"message": {"content": "Answer: 2"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.001}}

    monkeypatch.setattr(api, "call", fake_call)
    row = api.judge_one(TRACE, "m", "k", Args())
    expected = local.build_prompt(TRACE["problem"], TRACE["steps"],
                                  local.trace_outcome(TRACE), None, None, False, False)
    assert sent["prompt"] == expected
    assert "INCORRECT final answer" in sent["prompt"]
    assert row["first_error"] == 1          # one-based 2 becomes zero-based 1
    assert row["parse_ok"] is True


def test_a_reasoning_model_that_never_answers_is_a_parse_failure(monkeypatch):
    """An empty content field with a full reasoning trace means the model spent
    its budget thinking. Reading the verdict out of the reasoning would score a
    different task than the local judges were scored on."""
    monkeypatch.setattr(api, "call", lambda *a, **k: {
        "choices": [{"message": {"content": "", "reasoning": "Step 2 is wrong..."}}],
        "usage": {"cost": 0.002}})

    class A(Args):
        retry_empty = False
    row = api.judge_one(TRACE, "m", "k", A())
    assert row["parse_ok"] is False
    assert row["first_error"] == api.NO_ERROR


def test_votes_are_pooled_the_same_way_as_the_local_judge(monkeypatch):
    replies = iter(["Answer: 2", "Answer: 3", "Answer: 2"])
    monkeypatch.setattr(api, "call", lambda *a, **k: {
        "choices": [{"message": {"content": next(replies)}}], "usage": {"cost": 0.0}})

    class A(Args):
        n_votes = 3
    row = api.judge_one(TRACE, "m", "k", A())
    assert row["votes"] == [1, 2, 1]
    assert row["first_error"] == 1
    assert row["usage"]["cost"] == 0.0


def test_the_key_is_read_from_the_environment_or_refused(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(api.Path, "home", staticmethod(lambda: Path("/nonexistent")))
    with pytest.raises(SystemExit):
        api.api_key(None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "abc")
    assert api.api_key(None) == "abc"
    assert api.api_key("explicit") == "explicit"


def test_a_retryable_error_is_retried_and_a_fatal_one_is_not(monkeypatch):
    import urllib.error
    calls = {"n": 0}

    def flaky(req, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 429, "rate", None, None)

    monkeypatch.setattr(api.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(api.time, "sleep", lambda s: None)

    class A(Args):
        retries = 2
    with pytest.raises(RuntimeError):
        api.call("p", "m", "k", A())
    assert calls["n"] == 3            # the original plus two retries

    calls["n"] = 0

    def fatal(req, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 401, "unauthorized", None, None)

    monkeypatch.setattr(api.urllib.request, "urlopen", fatal)
    with pytest.raises(RuntimeError):
        api.call("p", "m", "k", A())
    assert calls["n"] == 1            # a bad key is not worth four more tries


def write_traces(path: Path, n: int) -> None:
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"t{i}", "problem": "p?", "steps": ["a", "b"],
                                "final_answer_correct": False, "label": 1}) + "\n")


def run_main(monkeypatch, argv, cost_per_call=0.01, reply="Answer: 2"):
    calls = {"n": 0}

    def fake_call(prompt, model, key, args):
        calls["n"] += 1
        return {"choices": [{"message": {"content": reply}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                          "cost": cost_per_call}}

    monkeypatch.setattr(api, "call", fake_call)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(sys, "argv", ["judge_openrouter.py", *argv])
    api.main()
    return calls


def test_the_spend_cap_stops_the_run(tmp_path, monkeypatch, capsys):
    traces, out = tmp_path / "tr.jsonl", tmp_path / "out.jsonl"
    write_traces(traces, 40)
    calls = run_main(monkeypatch, ["--traces", str(traces), "--out", str(out),
                                   "--concurrency", "1", "--max_cost_usd", "0.05"])
    assert calls["n"] < 40                     # it did not judge everything
    assert "stopping at" in capsys.readouterr().out


def test_a_rerun_resumes_and_does_not_pay_twice(tmp_path, monkeypatch, capsys):
    traces, out = tmp_path / "tr.jsonl", tmp_path / "out.jsonl"
    write_traces(traces, 10)
    run_main(monkeypatch, ["--traces", str(traces), "--out", str(out),
                           "--concurrency", "1", "--max_cost_usd", "0.035"])
    first = len([l for l in out.read_text().splitlines() if l.strip()])
    assert 0 < first < 10
    calls = run_main(monkeypatch, ["--traces", str(traces), "--out", str(out),
                                   "--concurrency", "1", "--max_cost_usd", "10"])
    assert calls["n"] == 10 - first            # only the unfinished ones
    ids = [json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()]
    assert len(ids) == len(set(ids)) == 10


def test_dry_run_calls_nothing_and_shows_the_prompt(tmp_path, monkeypatch, capsys):
    traces, out = tmp_path / "tr.jsonl", tmp_path / "out.jsonl"
    write_traces(traces, 3)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = {"n": 0}
    monkeypatch.setattr(api, "call", lambda *a, **k: calls.__setitem__("n", 1))
    monkeypatch.setattr(sys, "argv", ["j", "--traces", str(traces), "--out", str(out),
                                      "--dry_run"])
    api.main()
    assert calls["n"] == 0
    assert not out.exists()
    assert "prompt tokens total" in capsys.readouterr().out


def test_an_empty_answer_is_retried_once_with_a_bigger_budget(monkeypatch):
    """A reasoning model returning nothing has run out of budget, not refused.
    The first partial run lost 15.9% of its traces this way."""
    seen = []

    def fake_call(prompt, model, key, args):
        seen.append(args.max_tokens)
        content = "" if len(seen) == 1 else "Answer: 2"
        return {"choices": [{"message": {"content": content, "reasoning": "..."}}],
                "usage": {"cost": 0.001}}

    monkeypatch.setattr(api, "call", fake_call)

    class A(Args):
        max_tokens = 1000
        retry_empty = True
    row = api.judge_one(TRACE, "m", "k", A())
    assert seen == [1000, 2000]
    assert row["parse_ok"] is True and row["first_error"] == 1
    assert row["usage"]["cost"] == pytest.approx(0.002)   # both attempts are paid


def test_a_still_empty_reply_after_the_retry_is_a_parse_failure(monkeypatch):
    monkeypatch.setattr(api, "call", lambda *a, **k: {
        "choices": [{"message": {"content": "   ", "reasoning": "..."}}],
        "usage": {"cost": 0.001}})

    class A(Args):
        retry_empty = True
    assert api.judge_one(TRACE, "m", "k", A())["parse_ok"] is False


def test_redo_failed_replaces_the_failed_rows_and_keeps_the_rest(tmp_path, monkeypatch):
    """A parse failure is usually an exhausted token budget, not a refusal, so a
    row that failed under a tighter setting deserves one more attempt. The file
    must still end with one row per trace."""
    traces, out = tmp_path / "tr.jsonl", tmp_path / "out.jsonl"
    write_traces(traces, 3)
    out.write_text("\n".join(json.dumps(r) for r in [
        {"id": "t0", "first_error": 1, "parse_ok": True, "usage": {"cost": 0.01}},
        {"id": "t1", "first_error": -1, "parse_ok": False, "usage": {"cost": 0.01}},
    ]) + "\n")
    calls = run_main(monkeypatch, ["--traces", str(traces), "--out", str(out),
                                   "--concurrency", "1", "--max_cost_usd", "10",
                                   "--redo_failed"])
    assert calls["n"] == 2                      # the failed t1 and the unseen t2
    rows = {json.loads(l)["id"]: json.loads(l)
            for l in out.read_text().splitlines() if l.strip()}
    assert set(rows) == {"t0", "t1", "t2"}
    assert rows["t1"]["parse_ok"] is True
