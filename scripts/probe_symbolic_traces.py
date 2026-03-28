#!/usr/bin/env python3
"""Diagnostic: generate CoT traces from Qwen2.5-0.5B on ProntoQA-style problems
and label each step with PropLogicSolver.

No SSAE checkpoint needed — this only uses the base LLM for generation.
Goal: understand error rate, error position, and error type before scaling.

Usage:
    python scripts/probe_symbolic_traces.py \
        --data data/prontoqa_sample.jsonl \
        --device mps \
        --n 20

Output:
    - Per-problem trace with step labels printed to stdout
    - Summary statistics: label distribution, first-error position, etc.
    - Optional JSON dump via --output
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.symbolic_logic_dataset import PropLogicSolver, label_chain

# ---------------------------------------------------------------------------
# Few-shot prompt
# ---------------------------------------------------------------------------

_FEW_SHOT = """\
You are a symbolic reasoner. Given a set of rules and a fact, derive the answer step by step.
Write each reasoning step as a single sentence. Stop when you can answer the query.

Example 1:
Rules and facts: Every fumpus is cold. Polly is a fumpus.
Query: True or false: Polly is cold.
Reasoning:
Polly is a fumpus.
Polly is cold.
Answer: True

Example 2:
Rules and facts: Every wumpus is a rompus. Every rompus is luminous. Rex is a wumpus.
Query: True or false: Rex is luminous.
Reasoning:
Rex is a wumpus.
Rex is a rompus.
Rex is luminous.
Answer: True

Example 3:
Rules and facts: Every tumpus is a zumpus. Every zumpus is large. Sam is a tumpus.
Query: True or false: Sam is large.
Reasoning:
Sam is a tumpus.
Sam is a zumpus.
Sam is large.
Answer: True

Now solve this:
Rules and facts: {question}
Query: {query}
Reasoning:
"""


def build_prompt(question: str, query: str) -> str:
    return _FEW_SHOT.format(question=question, query=query)


# ---------------------------------------------------------------------------
# Step parsing
# ---------------------------------------------------------------------------

# Match lines that look like "[Name] is [a/an] [prop]." or "[Name] is not [prop]."
_STEP_LINE_RE = re.compile(
    r"^[A-Z][a-z]*\s+is\s+(?:not\s+)?(?:a\s+|an\s+)?\w+\s*\.$"
)
_ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*:\s*(True|False)\s*$", re.IGNORECASE)


def parse_model_output(raw: str) -> tuple[list[str], str | None]:
    """Extract (steps, predicted_answer) from raw model output.

    Steps: lines matching the ProntoQA sentence pattern.
    predicted_answer: "True" / "False" / None if not found.
    """
    steps: list[str] = []
    predicted_answer: str | None = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        am = _ANSWER_LINE_RE.match(line)
        if am:
            predicted_answer = am.group(1).capitalize()
            break
        if _STEP_LINE_RE.match(line):
            steps.append(line)

    return steps, predicted_answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_problems(path: Path, n: int) -> list[dict]:
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
                if len(problems) >= n:
                    break
    return problems


def run(args):
    device = torch.device(args.device)
    print(f"Loading Qwen2.5-0.5B on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32
    ).to(device)
    model.eval()

    problems = load_problems(Path(args.data), args.n)
    print(f"Loaded {len(problems)} problems from {args.data}\n")

    results = []
    total_steps = 0
    total_correct = 0
    error_positions: list[int] = []  # 0-indexed position of first error per problem
    answer_correct = 0

    print("=" * 70)

    for i, prob in enumerate(problems):
        question = prob["question"]
        query = prob["query"]
        gold_answer = prob["answer"]
        gold_cot = prob["chain_of_thought"]
        n_hops = prob.get("meta", {}).get("n_hops", "?")

        prompt = build_prompt(question, query)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (not the prompt)
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        steps, pred_answer = parse_model_output(generated)
        labels = label_chain(question, steps) if steps else []

        n_correct = sum(labels)
        n_total = len(labels)
        first_err = next((j for j, l in enumerate(labels) if l == 0), None)

        total_steps += n_total
        total_correct += n_correct
        if first_err is not None:
            error_positions.append(first_err)
        if pred_answer == gold_answer:
            answer_correct += 1

        # Print per-problem result
        status = "OK" if all(l == 1 for l in labels) else "ERR"
        print(f"[{i+1:02d}] {status}  hops={n_hops}  gold={gold_answer}  pred={pred_answer or '?'}")
        print(f"     Q: {question}")
        print(f"     Steps parsed ({n_total}):")
        for j, (step, lbl) in enumerate(zip(steps, labels)):
            marker = "+" if lbl == 1 else "-"
            print(f"       {marker} [{j}] {step}")
        if not steps:
            print(f"       (no parseable steps — raw output: {generated[:120]!r})")
        print(f"     Gold CoT: {gold_cot}")
        print()

        results.append({
            "question": question,
            "query": query,
            "gold_answer": gold_answer,
            "gold_cot": gold_cot,
            "n_hops": n_hops,
            "generated_raw": generated,
            "steps": steps,
            "labels": labels,
            "pred_answer": pred_answer,
            "first_error_pos": first_err,
        })

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    n_problems = len(problems)
    n_with_errors = sum(1 for r in results if any(l == 0 for l in r["labels"]))
    n_no_steps = sum(1 for r in results if len(r["steps"]) == 0)

    print("=" * 70)
    print("SUMMARY")
    print(f"  Problems         : {n_problems}")
    print(f"  Answer accuracy  : {answer_correct}/{n_problems} ({100*answer_correct/n_problems:.1f}%)")
    print(f"  Total steps      : {total_steps}")
    if total_steps > 0:
        print(f"  Step label dist  : {total_correct} correct ({100*total_correct/total_steps:.1f}%)  "
              f"{total_steps - total_correct} incorrect ({100*(total_steps-total_correct)/total_steps:.1f}%)")
    print(f"  Problems with ≥1 error  : {n_with_errors}/{n_problems}")
    print(f"  Problems with no steps  : {n_no_steps}/{n_problems}")

    if error_positions:
        from collections import Counter
        pos_counts = Counter(error_positions)
        print(f"  First-error positions   : {dict(sorted(pos_counts.items()))}")

    # Break down by n_hops
    hop_stats: dict = {}
    for r in results:
        h = r["n_hops"]
        hop_stats.setdefault(h, {"total": 0, "errors": 0})
        hop_stats[h]["total"] += 1
        if any(l == 0 for l in r["labels"]):
            hop_stats[h]["errors"] += 1
    print("  Error rate by hop count:")
    for h in sorted(hop_stats, key=lambda x: (str(x))):
        s = hop_stats[h]
        print(f"    {h} hops: {s['errors']}/{s['total']} problems with errors")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/prontoqa_sample.jsonl")
    p.add_argument("--n", type=int, default=20, help="Number of problems to run")
    p.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--output", default=None, help="Path to save JSON results")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
