#!/usr/bin/env python3
"""Load ProntoQA problems from HuggingFace and generate model CoT traces.

Two phases:
  Phase A — Load problems from renma/ProntoQA (HuggingFace).
             500 problems, ~48% False, complex multi-hop chains with varied
             vocabulary. Fields are mapped to our internal format.

  Phase B — Run Qwen2.5-0.5B on the problems to produce model-generated CoT
             traces. These are what the SSAE will train on. Output: --traces-out JSONL.
             Each line: {question, query, steps: [...], pred_answer, meta}

The traces are the primary training signal for the SSAE (phase 1 reconstruction).
PropLogicSolver labels are computed on-the-fly at probe-training time.

Usage:
    # Load all 500 ProntoQA problems and generate traces
    python scripts/generate_symbolic_data.py \\
        --problems-out data/prontoqa_hf.jsonl \\
        --traces-out   data/prontoqa_traces.jsonl \\
        --device mps

    # Problems only (no LLM), useful for inspection
    python scripts/generate_symbolic_data.py \\
        --problems-only \\
        --problems-out data/prontoqa_hf.jsonl

    # Cap to N problems (for quick smoke tests)
    python scripts/generate_symbolic_data.py \\
        --max-problems 50 --problems-only
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Phase A: load from HuggingFace
# ---------------------------------------------------------------------------

def load_prontoqa(max_problems: int | None = None) -> list[dict]:
    """Load problems from renma/ProntoQA on HuggingFace.

    Maps HF fields to our internal format:
      context  → question  (rules + seed fact)
      question → query     (normalised to "True or false: X is Y.")
      answer   → answer    (A → "True", B → "False")
    """
    from datasets import load_dataset

    print("Loading renma/ProntoQA from HuggingFace …")
    ds = load_dataset("renma/ProntoQA", trust_remote_code=True)
    split = "validation"  # only split available

    problems = []
    for ex in ds[split]:
        if max_problems is not None and len(problems) >= max_problems:
            break

        # Normalise query format to match our few-shot prompt style
        query = ex["question"].replace(
            "Is the following statement true or false? ", "True or false: "
        )

        # Options are always ['A) True', 'B) False']
        answer = "True" if ex["answer"] == "A" else "False"

        problems.append({
            "question": ex["context"],
            "query": query,
            "answer": answer,
            "chain_of_thought": [],  # not provided; traces are model-generated in Phase B
            "meta": {"source": "prontoqa", "id": ex["id"]},
        })

    return problems


# ---------------------------------------------------------------------------
# Few-shot prompt + step parser
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

_STEP_RE = re.compile(r"^[A-Z][a-z]*\s+is\s+(?:not\s+)?(?:a\s+|an\s+)?\w+\s*\.$")
_ANSWER_RE = re.compile(r"^\s*Answer\s*:\s*(True|False)\s*$", re.IGNORECASE)


def parse_output(raw: str) -> tuple[list[str], str | None]:
    steps, answer = [], None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _ANSWER_RE.match(line)
        if m:
            answer = m.group(1).capitalize()
            break
        if _STEP_RE.match(line):
            steps.append(line)
    return steps, answer


# ---------------------------------------------------------------------------
# Phase B: batched LLM trace generation
# ---------------------------------------------------------------------------

def generate_traces_batched(
    problems: list[dict],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 150,
) -> list[dict]:
    """Run batched LLM generation over all problems. Returns trace records."""
    prompts = [
        _FEW_SHOT.format(question=p["question"], query=p["query"])
        for p in problems
    ]

    traces = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating traces"):
        batch_prompts = prompts[i : i + batch_size]
        batch_problems = problems[i : i + batch_size]

        tokenizer.padding_side = "left"
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,  # ProntoQA contexts are longer than synthetic
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        for prob, output_ids in zip(batch_problems, out):
            raw = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
            steps, pred_answer = parse_output(raw)
            if steps:
                traces.append({
                    "question": prob["question"],
                    "query": prob["query"],
                    "steps": steps,
                    "pred_answer": pred_answer,
                    "meta": prob.get("meta", {}),
                })

    return traces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Phase A: load from HuggingFace ---
    problems = load_prontoqa(max_problems=args.max_problems)

    n_false = sum(1 for p in problems if p["answer"] == "False")
    n_true = len(problems) - n_false
    print(f"Loaded {len(problems)} problems from ProntoQA")
    print(f"True / False split: {n_true} / {n_false} ({n_false / len(problems):.0%} False)")

    if args.problems_out:
        Path(args.problems_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.problems_out, "w") as f:
            for p in problems:
                f.write(json.dumps(p) + "\n")
        print(f"Saved {len(problems)} problems to {args.problems_out}")

    if args.problems_only:
        return

    # --- Phase B: generate model traces ---
    device = torch.device(args.device)
    print(f"\nLoading Qwen2.5-0.5B on {device} …")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32
    ).to(device)
    model.eval()

    traces = generate_traces_batched(
        problems, model, tokenizer, device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    total_steps = sum(len(t["steps"]) for t in traces)
    skipped = len(problems) - len(traces)
    print(f"\nProblems with traces : {len(traces)}/{len(problems)}")
    print(f"Skipped (no steps)   : {skipped}")
    print(f"Total steps          : {total_steps}")
    print(f"Avg steps/problem    : {total_steps / max(len(traces), 1):.1f}")

    if args.traces_out:
        Path(args.traces_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.traces_out, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        print(f"Saved {len(traces)} traces ({total_steps} steps) to {args.traces_out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max-problems", type=int, default=None,
        help="Cap number of problems loaded (default: all 500)",
    )
    p.add_argument("--problems-out", default="data/prontoqa_hf.jsonl")
    p.add_argument("--traces-out", default="data/prontoqa_traces.jsonl")
    p.add_argument(
        "--problems-only", action="store_true",
        help="Only load and save problems, skip LLM trace generation",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


if __name__ == "__main__":
    main()
