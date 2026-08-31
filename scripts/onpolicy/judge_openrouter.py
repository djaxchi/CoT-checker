#!/usr/bin/env python3
"""The judge as an API call, run from a machine that has internet.

Compute nodes have none, which is why the first bake-off was restricted to
models sitting in the cluster's HF cache, and why it came back at F1_PB
0.42-0.44 against 0.566 for the representation those labels would be used to
evaluate. The way out is that the labelling does not have to happen on the
cluster at all: judge the traces here, ship the labels over, and the batch job
reads a file.

That also restores ReProbe's actual recipe. They judge with DeepSeek-R1 and
report 95% agreement with PRM800K human labels; the local base models were a
substitute forced by the network, not a design choice.

**Same task, same questions, same parser.** The prompt, the outcome-told
configuration, the answer parsing and the certification metrics are imported
from judge_steps.py rather than restated, so an API judge lands in the same
bake-off table as the local ones and the comparison is between models, not
between prompts. Run it on the certification set first: a judge that does not
beat the local ones there has no claim to label anything.

Spending is bounded and resumable. Every finished trace is appended as it lands,
a rerun skips what is already in the file, and --max_cost_usd stops the run
rather than discovering the bill afterwards. --dry_run prints the first prompt
and an estimate without calling anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl  # noqa: E402
from scripts.onpolicy.judge_steps import (  # noqa: E402
    NO_ERROR, build_prompt, certify, parse_answer, trace_outcome, vote,
)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def api_key(explicit: str | None) -> str:
    key = explicit or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        path = Path.home() / ".config" / "openrouter" / "key"
        if path.exists():
            key = path.read_text().strip()
    if not key:
        raise SystemExit(
            "no API key. Set OPENROUTER_API_KEY, pass --api_key, or put the key "
            "in ~/.config/openrouter/key")
    return key


def call(prompt: str, model: str, key: str, args) -> dict:
    """One request, with backoff on the errors that are worth retrying."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "usage": {"include": True},
    }
    if args.reasoning_effort:
        body["reasoning"] = {"effort": args.reasoning_effort}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        ENDPOINT, data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/djaxchi/CoT-checker",
                 "X-Title": "CoT-checker onpolicy judge"})
    delay = args.retry_delay
    last = None
    for attempt in range(args.retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except Exception as e:                       # timeouts, resets
            last = f"{type(e).__name__}: {e}"
        if attempt < args.retries:
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(last or "request failed")


def judge_one(tr: dict, model: str, key: str, args) -> dict:
    prompt = build_prompt(
        tr["problem"], tr["steps"],
        trace_outcome(tr) if args.tell_outcome else None,
        (tr.get("gold") or tr.get("ground_truth_answer")) if args.show_gold else None,
        None, False, args.cot)
    votes, raw, usage_total = [], "", {"prompt": 0, "completion": 0, "cost": 0.0}
    for _ in range(args.n_votes):
        resp = call(prompt, model, key, args)
        msg = resp["choices"][0]["message"]
        text = (msg.get("content") or "")
        if not text.strip() and args.retry_empty:
            # A reasoning model that spends its whole budget thinking returns an
            # empty answer. That is a budget problem, not a refusal, and the
            # first partial run lost 15.9% of its traces to it. Reading the
            # verdict out of the reasoning trace instead would score a different
            # task than the local judges were scored on, so this pays for one
            # bigger attempt and counts a still-empty reply as a parse failure.
            u = resp.get("usage") or {}
            usage_total["prompt"] += int(u.get("prompt_tokens") or 0)
            usage_total["completion"] += int(u.get("completion_tokens") or 0)
            usage_total["cost"] += float(u.get("cost") or 0.0)
            bigger = argparse.Namespace(**vars(args))
            bigger.max_tokens = args.max_tokens * 2
            resp = call(prompt, model, key, bigger)
            msg = resp["choices"][0]["message"]
            text = (msg.get("content") or "")
        votes.append(parse_answer(text, len(tr["steps"])) if text.strip() else None)
        raw = raw or text[:300]
        u = resp.get("usage") or {}
        usage_total["prompt"] += int(u.get("prompt_tokens") or 0)
        usage_total["completion"] += int(u.get("completion_tokens") or 0)
        usage_total["cost"] += float(u.get("cost") or 0.0)
    label = vote(votes)
    return {"id": tr["id"], "traj_uid": tr["id"],
            "first_error": NO_ERROR if label is None else int(label),
            "parse_ok": label is not None, "votes": votes,
            "n_steps": len(tr["steps"]), "raw": raw, "usage": usage_total}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--model", default="deepseek/deepseek-r1",
                   help="OpenRouter model id. ReProbe's judge is DeepSeek-R1.")
    p.add_argument("--api_key", default=None)
    p.add_argument("--certify", action="store_true",
                   help="Score against the human labels in --traces.")
    p.add_argument("--cot", action="store_true",
                   help="Ask for an ordered check before the verdict. A reasoning "
                        "model does that on its own; this is for the others.")
    p.add_argument("--tell_outcome", dest="tell_outcome", action="store_true",
                   default=True)
    p.add_argument("--no_tell_outcome", dest="tell_outcome", action="store_false")
    p.add_argument("--show_gold", action="store_true")
    p.add_argument("--n_votes", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=8192,
                   help="A reasoning model's thinking counts against this. At "
                        "4096 the first run lost 15.9% of traces to empty "
                        "answers.")
    p.add_argument("--retry_empty", dest="retry_empty", action="store_true",
                   default=True,
                   help="Pay for one attempt at double the budget when the model "
                        "returns nothing but reasoning.")
    p.add_argument("--no_retry_empty", dest="retry_empty", action="store_false")
    p.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--retry_delay", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max_traces", type=int, default=0)
    p.add_argument("--max_cost_usd", type=float, default=5.0,
                   help="Stop once the spend passes this. The run is resumable, "
                        "so a stop is a pause, not a loss.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the first prompt and a size estimate; call nothing.")
    args = p.parse_args()

    traces = read_jsonl(args.traces)
    if args.max_traces > 0:
        traces = traces[:args.max_traces]

    done: dict[str, dict] = {}
    if args.out.exists():
        for r in read_jsonl(args.out):
            done[r["id"]] = r
        print(f"[api] resuming: {len(done)} of {len(traces)} already judged")
    todo = [t for t in traces if t["id"] not in done]

    if args.dry_run:
        t = traces[0]
        prompt = build_prompt(t["problem"], t["steps"],
                              trace_outcome(t) if args.tell_outcome else None,
                              None, None, False, args.cot)
        chars = sum(len(build_prompt(x["problem"], x["steps"], trace_outcome(x),
                                     None, None, False, args.cot)) for x in traces)
        print(prompt)
        print("-" * 60)
        print(f"{len(traces)} traces, {len(todo)} to do, ~{chars/4:,.0f} prompt "
              f"tokens total (4 chars/token), model {args.model}")
        return

    key = api_key(args.api_key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    spent = sum(float((r.get("usage") or {}).get("cost") or 0.0) for r in done.values())
    stop = threading.Event()
    fails: list[dict] = []
    t0 = time.perf_counter()

    # Accounting happens in the worker, under the lock, right after the call it
    # paid for. Doing it in the collecting thread lets a fast provider run ahead
    # of the bookkeeping and spend past the cap before anyone notices.
    with args.out.open("a") as fh, ThreadPoolExecutor(args.concurrency) as ex:
        n = [0]

        def work(tr):
            if stop.is_set():
                return
            try:
                row = judge_one(tr, args.model, key, args)
            except Exception as e:            # one bad trace must not end the run
                with lock:
                    fails.append({"id": tr["id"], "error": str(e)})
                return
            nonlocal spent
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done[row["id"]] = row
                spent += float((row.get("usage") or {}).get("cost") or 0.0)
                n[0] += 1
                if spent >= args.max_cost_usd and not stop.is_set():
                    stop.set()
                    print(f"[api] stopping at ${spent:.3f} (limit "
                          f"${args.max_cost_usd}); rerun to continue where it "
                          f"left off", flush=True)
                if n[0] % 25 == 0:
                    print(f"[api] {n[0]}/{len(todo)} "
                          f"({time.perf_counter()-t0:.0f}s) ${spent:.3f} "
                          f"failures {len(fails)}", flush=True)

        for fut in as_completed([ex.submit(work, t) for t in todo]):
            fut.result()

    rows = list(done.values())
    ok = sum(1 for r in rows if r.get("parse_ok"))
    print(f"\n[api] {len(rows)} judged, {ok} parsed, {len(fails)} request failures, "
          f"${spent:.3f} spent")
    if fails:
        print(f"[api] first failure: {fails[0]['error'][:200]}")

    if args.certify:
        by_id = {t["id"]: t for t in traces}
        scored = [r for r in rows if r["id"] in by_id]
        rep = certify(scored, [by_id[r["id"]] for r in scored])
        rep.update({"model": args.model, "n_votes": args.n_votes, "cot": args.cot,
                    "tell_outcome": args.tell_outcome, "show_gold": args.show_gold,
                    "reasoning_effort": args.reasoning_effort,
                    "cost_usd": spent, "n_request_failures": len(fails),
                    "traces": str(args.traces), "via": "openrouter",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "code_commit": git_commit()})
        j = rep["judge"]
        print(f"\n{'':<20}{'F1_PB':>9}{'Acc_err':>9}{'Acc_cor':>9}{'exact':>9}")
        print(f"{'judge':<20}{j['F1_PB']:>9.3f}{j['Acc_error']:>9.3f}"
              f"{j['Acc_correct']:>9.3f}{j['exact_match_all']:>9.3f}")
        for name, b in rep["baselines"].items():
            print(f"{name:<20}{b['F1_PB']:>9.3f}{b['Acc_error']:>9.3f}"
                  f"{b['Acc_correct']:>9.3f}{b['exact_match_all']:>9.3f}")
        print(f"\nparse failures {rep['parse_failure_rate']:.3f}")
        if "mean_relative_position" in rep:
            mp = rep["mean_relative_position"]
            print(f"relative position of the error: judge says {mp['predicted']:.2f}, "
                  f"truth is {mp['true']:.2f}")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(rep, indent=2))
            print(f"[api] wrote {args.report}")


if __name__ == "__main__":
    main()
