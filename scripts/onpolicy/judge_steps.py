#!/usr/bin/env python3
"""Stage 2: a local judge that marks the first wrong step, and its certification.

On-policy steps have no human labels, and the label F1_PB needs is the *first
error*, not the outcome. Compute nodes have no internet, so the judge is a local
model. Which one is decided by measurement, not argument: run each candidate over
a human-labelled set and report what it gets.

**The certification set is ProcessBench, not PRM800K.** ReProbe certifies on
PRM800K, and that is not reproducible here in a form that means anything.
PRM800K's annotated path advances through completions the annotators accepted, so
the rated -1 completions are alternatives *off* that path. Building a trace with
a first error out of it means gluing a rejected completion onto an accepted
prefix, which produces a trace whose error is always its final step. A judge that
answered "the last one" every time would score near-perfectly on that set and
tell us nothing. ProcessBench's traces are full model solutions with genuine
human first-error annotations and errors at their real positions, and it has the
further advantage that the judge's F1_PB there is directly comparable to the
representation leaderboard's, so the judge arrives as a row of the same table
rather than as an unanchored accuracy.

**The judge is told the outcome, not the answer.** ReProbe hands its judge the
ground-truth answer. ProcessBench carries no gold answers, only whether the
solution's final answer was right, so a judge certified with gold answers could
never be certified at all here, and a certification run under different
information than the deployment run describes a judge we are not using. Both
settings do carry the outcome: ProcessBench as `final_answer_correct`, the
on-policy traces as the grader's verdict. So the judge is told "this solution
reaches the correct/incorrect final answer, find where it first goes wrong", in
certification and in deployment alike, and the reported number describes the
judge we actually run. Gold-answer judging stays available behind --show_gold
and is untested here for exactly this reason.

Three trivial baselines are scored alongside on the same traces: always predict
no error, always predict the last step, always predict the first. An LLM judge
that has merely learned "errors come late" beats a coin flip and looks
respectable, and the only way to see that is to put the degenerate strategies in
the table next to it.

Output: one row per trace, {id, first_error, parse_ok, votes, raw}. Feed it to
scripts/onpolicy/build_pb_traces.py --labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl, write_jsonl  # noqa: E402

NO_ERROR = -1

INSTRUCTIONS = (
    "You are given a math problem and a candidate solution split into numbered "
    "steps.\n"
    "Find the FIRST step that contains a mistake. A step is wrong if it states "
    "something false, uses a wrong value, or draws a conclusion its own previous "
    "steps do not support. A step that is merely unnecessary is not wrong.\n"
    "Steps are numbered starting at 1.\n"
    "Reply with one line and nothing else:\n"
    "Answer: <number of the first wrong step, or -1 if every step is correct>"
)

# The same task, with room to work before committing. Zheng et al.'s judges reason
# before they answer; asking for the verdict on the first token makes the judge
# guess from surface features, which is the failure the always-last-step baseline
# is designed to catch.
INSTRUCTIONS_COT = (
    "You are given a math problem and a candidate solution split into numbered "
    "steps.\n"
    "Check the steps in order. For each one, say briefly whether it follows from "
    "the steps before it and whether its arithmetic and reasoning are right. Stop "
    "at the first step that is wrong.\n"
    "A step is wrong if it states something false, uses a wrong value, or draws a "
    "conclusion its own previous steps do not support. A step that is merely "
    "unnecessary is not wrong.\n"
    "Steps are numbered starting at 1. Keep the check short.\n"
    "Finish with exactly this line:\n"
    "Answer: <number of the first wrong step, or -1 if every step is correct>"
)


def render_trace(problem: str, steps: list[str], outcome: bool | None = None,
                 gold: str | None = None) -> str:
    body = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
    known = ""
    if outcome is not None:
        known += ("\nThis solution reaches the CORRECT final answer. It may still "
                  "contain a wrong step; answer -1 only if every step is sound.\n"
                  if outcome else
                  "\nThis solution reaches an INCORRECT final answer, so at least "
                  "one step is wrong.\n")
    if gold:
        known += f"Correct final answer: {gold}\n"
    return f"Problem:\n{problem}\n{known}\nCandidate solution:\n{body}\n"


def trace_outcome(tr: dict) -> bool | None:
    """The outcome field, whichever set the trace came from."""
    for k in ("final_answer_correct", "traj_correct", "correct"):
        if k in tr and tr[k] is not None:
            return bool(tr[k])
    return None


def build_prompt(problem: str, steps: list[str], outcome: bool | None = None,
                 gold: str | None = None, tokenizer=None, chat: bool = False,
                 cot: bool = False) -> str:
    """The judge's input. Chat models get the template they were tuned with; a
    base model gets the same text plus the cue it should continue from."""
    head = INSTRUCTIONS_COT if cot else INSTRUCTIONS
    user = f"{head}\n\n{render_trace(problem, steps, outcome, gold)}"
    if chat and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True)
    return f"{user}\nCheck:" if cot else f"{user}\nAnswer:"


# ---------------------------------------------------------------------------
# ReProbe protocol (arXiv:2511.06209); see docs/reprobe_label_semantics.md
# ---------------------------------------------------------------------------
# Three things differ from the first-error protocol above, and they are the
# paper's, not ours: the judge is shown the ground-truth answer, it is asked for
# the SET of steps that contain errors rather than the first one, and nothing in
# the paper says the steps after an error become negative, so they are left
# alone. Propagating them is the common PRM convention and would change the
# class balance substantially, which is exactly why it is not done here.
INSTRUCTIONS_REPROBE = (
    "You are grading a student's step-by-step solution to a math problem. You "
    "are given the problem, the correct final answer, and the student's steps.\n"
    "Examine each step to determine whether it is both logically correct and "
    "relevant. A step is faulty if it is wrong, if it does not follow from the "
    "steps before it, or if it is unnecessary or redundant reasoning that does "
    "not move toward the correct solution.\n"
    "Check the steps in order and be brief.\n"
    "Steps are numbered starting at 1.\n"
    "Finish with exactly this line, listing every faulty step:\n"
    "Faulty: <comma-separated step numbers, or NONE if every step is sound>"
)


def render_trace_reprobe(problem: str, steps: list[str], gold: str) -> str:
    body = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
    return (f"Problem:\n{problem}\n\nCorrect final answer: {gold}\n\n"
            f"Student's solution:\n{body}\n")


def build_prompt_reprobe(problem: str, steps: list[str], gold: str,
                         tokenizer=None, chat: bool = False) -> str:
    user = f"{INSTRUCTIONS_REPROBE}\n\n{render_trace_reprobe(problem, steps, gold)}"
    if chat and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True)
    return f"{user}\nCheck:"


_FAULTY = re.compile(r"faulty\s*:?\s*(.*)", re.IGNORECASE)


def parse_step_set(text: str, n_steps: int) -> list[int] | None:
    """The faulty-step numbers as 0-based indices, or None if unreadable.

    Reads the LAST `Faulty:` line, since a model that reasons first may use the
    word on the way to its verdict. NONE means every step is sound and returns an
    empty list, which is a real answer and must not be confused with a parse
    failure. Numbers outside 1..n_steps are dropped rather than clipped: a judge
    naming a step that does not exist has lost track of the solution, and
    clamping would invent a label for a step it never looked at.
    """
    hits = [m for m in _FAULTY.finditer(text or "")]
    if not hits:
        return None
    tail = hits[-1].group(1).strip()
    if not tail:
        return None
    if re.match(r"^(none|n/a|no(ne)?\b)", tail, re.IGNORECASE):
        return []
    nums = [int(x) for x in re.findall(r"\d+", tail)]
    if not nums:
        return None
    keep = sorted({v - 1 for v in nums if 1 <= v <= n_steps})
    # every number out of range: the judge was not describing this solution
    return keep if keep or not nums else None


def step_labels_from_faulty(faulty: list[int], n_steps: int) -> list[int]:
    """Binary per-step labels, 1 correct and 0 incorrect, no propagation."""
    bad = set(faulty)
    return [0 if i in bad else 1 for i in range(n_steps)]


_INT = re.compile(r"-?\d+")


def parse_answer(text: str, n_steps: int) -> int | None:
    """1-based step number (or -1) out of the model's reply, into a 0-based index.

    Returns None when the reply cannot be read, which is counted rather than
    quietly turned into a label. A literal 0 is a parse failure on purpose: the
    prompt numbers from 1, so a 0 means the model used a different convention and
    guessing which would put every label off by one.
    """
    lines = text.strip().splitlines()
    # Last answer line, not the first: with reasoning enabled the model may
    # mention the word on the way to its verdict, and the verdict is the one it
    # ends on.
    for line in reversed(lines):
        if "answer" in line.lower():
            m = _INT.findall(line)
            if m:
                v = int(m[-1])
                break
    else:
        m = _INT.findall(text)
        if not m:
            return None
        v = int(m[0])
    if v == NO_ERROR:
        return NO_ERROR
    if 1 <= v <= n_steps:
        return v - 1
    return None


def vote(labels: list[int]) -> int | None:
    """Majority over votes; a tie falls back to the earliest index, which is the
    conservative reading of "first error"."""
    labels = [x for x in labels if x is not None]
    if not labels:
        return None
    c = Counter(labels)
    top = max(c.values())
    return min(k for k, v in c.items() if v == top)


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------

def pb_metrics(pred: list[int], gold: list[int]) -> dict:
    """ProcessBench's own metric, so the judge lands on the leaderboard's scale."""
    n_err = n_cor = err_hit = cor_hit = exact = 0
    for p, g in zip(pred, gold):
        if g == NO_ERROR:
            n_cor += 1
            cor_hit += int(p == NO_ERROR)
        else:
            n_err += 1
            err_hit += int(p == g)
        exact += int(p == g)
    acc_e = err_hit / n_err if n_err else 0.0
    acc_c = cor_hit / n_cor if n_cor else 0.0
    f1 = 2 * acc_e * acc_c / (acc_e + acc_c) if (acc_e + acc_c) else 0.0
    return {"n": len(pred), "n_error": n_err, "n_correct": n_cor,
            "Acc_error": acc_e, "Acc_correct": acc_c, "F1_PB": f1,
            "exact_match_all": exact / max(1, len(pred))}


def baselines(gold: list[int], n_steps: list[int]) -> dict:
    """What a judge with no idea what a mistake is would score."""
    return {
        "always_no_error": pb_metrics([NO_ERROR] * len(gold), gold),
        "always_last_step": pb_metrics([n - 1 for n in n_steps], gold),
        "always_first_step": pb_metrics([0] * len(gold), gold),
    }


def certify(rows: list[dict], traces: list[dict]) -> dict:
    by_id = {t["id"]: t for t in traces}
    pred, gold, ns = [], [], []
    for r in rows:
        t = by_id[r["id"]]
        if t.get("label") is None:
            continue
        # An unparseable reply is scored as "no error", the answer a judge that
        # said nothing would give. Dropping those rows would flatter the judge.
        pred.append(r["first_error"] if r["parse_ok"] else NO_ERROR)
        gold.append(int(t["label"]))
        ns.append(len(t["steps"]))
    out = {"judge": pb_metrics(pred, gold), "baselines": baselines(gold, ns),
           "parse_failure_rate": 1 - sum(r["parse_ok"] for r in rows) / max(1, len(rows))}
    # Where in the trace the judge points, against where the errors are. A judge
    # that has only learned "errors come late" shows up here.
    err = [(p, g, n) for p, g, n in zip(pred, gold, ns) if g != NO_ERROR]
    if err:
        out["mean_relative_position"] = {
            "predicted": sum((p / max(1, n - 1)) for p, _, n in err if p >= 0) /
                         max(1, sum(1 for p, _, _ in err if p >= 0)),
            "true": sum(g / max(1, n - 1) for _, g, n in err) / len(err),
        }
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def judge_traces(traces, tokenizer, model, device, args) -> list[dict]:
    import torch

    rows = []
    t0 = time.perf_counter()
    for i, tr in enumerate(traces):
        prompt = build_prompt(
            tr["problem"], tr["steps"],
            trace_outcome(tr) if args.tell_outcome else None,
            (tr.get("gold") or tr.get("ground_truth_answer")) if args.show_gold else None,
            tokenizer, args.chat, args.cot)
        enc = tokenizer(prompt, return_tensors="pt")
        if enc["input_ids"].shape[1] > args.max_prompt_tokens:
            # Truncating would cut the last steps and the answer cue off the end,
            # and the judge would confidently label a trace it never saw. Refuse
            # instead, and let the parse-failure count carry it.
            rows.append({"id": tr["id"], "traj_uid": tr["id"], "first_error": NO_ERROR,
                         "parse_ok": False, "votes": [], "n_steps": len(tr["steps"]),
                         "raw": f"[skipped: prompt {enc['input_ids'].shape[1]} tokens "
                                f"> {args.max_prompt_tokens}]"})
            continue
        enc = enc.to(device)
        kw = dict(max_new_tokens=args.max_new_tokens,
                  num_return_sequences=args.n_votes,
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if args.n_votes > 1:
            kw.update(do_sample=True, temperature=args.temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        plen = enc["input_ids"].shape[1]
        texts = [tokenizer.decode(out[v, plen:], skip_special_tokens=True)
                 for v in range(out.shape[0])]
        votes = [parse_answer(t, len(tr["steps"])) for t in texts]
        label = vote(votes)
        rows.append({
            "id": tr["id"], "traj_uid": tr["id"],
            "first_error": NO_ERROR if label is None else int(label),
            "parse_ok": label is not None,
            "votes": [v if v is not None else None for v in votes],
            "n_steps": len(tr["steps"]),
            "raw": texts[0][:200],
        })
        if (i + 1) % 25 == 0 or i + 1 == len(traces):
            ok = sum(r["parse_ok"] for r in rows)
            print(f"[judge] {i+1}/{len(traces)} ({time.perf_counter()-t0:.0f}s) "
                  f"parsed {ok}/{len(rows)}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, type=Path,
                   help="ProcessBench-shaped jsonl: id, problem, steps, and for "
                        "certification a human `label`.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--tell_outcome", dest="tell_outcome", action="store_true",
                   default=True,
                   help="Tell the judge whether the final answer was right. On by "
                        "default: it is the one piece of supervision both the "
                        "certification set and the on-policy traces carry, so it "
                        "keeps the two runs the same task.")
    p.add_argument("--no_tell_outcome", dest="tell_outcome", action="store_false")
    p.add_argument("--show_gold", action="store_true",
                   help="Also show the ground-truth answer. Not available on the "
                        "certification set, so a judge run this way is uncertified.")
    p.add_argument("--cot", action="store_true",
                   help="Let the judge check the steps in order before committing "
                        "to an index, and read the verdict off its last Answer "
                        "line. Needs a larger --max_new_tokens.")
    p.add_argument("--chat", action="store_true",
                   help="Use the tokenizer's chat template (instruct models).")
    p.add_argument("--model_dtype", choices=["float16", "bfloat16", "float32"],
                   default="bfloat16")
    p.add_argument("--n_votes", type=int, default=1,
                   help="Self-consistency over the label. 1 is greedy.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--max_prompt_tokens", type=int, default=3072)
    p.add_argument("--max_traces", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--certify", action="store_true",
                   help="Score against the human labels in --traces.")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch

    traces = read_jsonl(args.traces)
    if args.max_traces > 0:
        traces = traces[:args.max_traces]
    traces = traces[args.shard_idx::args.num_shards]
    print(f"[judge] shard {args.shard_idx}/{args.num_shards}: {len(traces)} traces "
          f"from {args.traces}", flush=True)

    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.model_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only,
        torch_dtype=dtype, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    rows = judge_traces(traces, tok, model, device, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    print(f"[judge] wrote {args.out}")

    if args.certify:
        rep = certify(rows, traces)
        rep.update({"model": args.model_name_or_path, "n_votes": args.n_votes,
                    "chat": args.chat, "cot": args.cot,
                    "max_new_tokens": args.max_new_tokens,
                    "tell_outcome": args.tell_outcome,
                    "show_gold": args.show_gold, "traces": str(args.traces),
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
            print(f"[judge] wrote {args.report}")


if __name__ == "__main__":
    main()
