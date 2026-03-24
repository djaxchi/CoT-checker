#!/usr/bin/env python3
"""Generate synthetic ProntoQA problems and model-generated CoT traces.

Two phases:
  Phase A — Algorithmically generate N ProntoQA-style problems (no LLM needed).
             Each problem is a chain of modus-ponens rules with a deterministic
             gold CoT. Output: --problems-out JSONL.

  Phase B — Run Qwen2.5-0.5B on the problems to produce model-generated CoT
             traces. These are what the SSAE will train on. Output: --traces-out JSONL.
             Each line: {question, query, steps: [...], answer: "True/False"}

The traces are the primary training signal for the SSAE (phase 1 reconstruction).
PropLogicSolver labels are not stored here — they are computed on-the-fly at
probe-training time.

Usage:
    # Generate 14 000 problems and traces for a ~50K-step SSAE training run
    python scripts/generate_symbolic_data.py \\
        --n-problems 14000 \\
        --problems-out data/prontoqa_synthetic.jsonl \\
        --traces-out   data/prontoqa_traces.jsonl \\
        --device mps \\
        --batch-size 8

    # Problems only (no LLM), useful for inspection
    python scripts/generate_symbolic_data.py \\
        --n-problems 500 --problems-only \\
        --problems-out data/prontoqa_synthetic_small.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Synthetic problem generator
# ---------------------------------------------------------------------------

_ENTITIES = [
    "Rex", "Polly", "Sam", "Alex", "Max", "Lily", "Otto", "Zara",
    "Leo", "Nova", "Finn", "Luna", "Axel", "Iris", "Hugo", "Vera",
]
_PROPERTIES = [
    "red", "cold", "large", "metallic", "happy", "luminous",
    "warm", "blue", "heavy", "soft", "bright", "slow", "fast",
    "small", "dark", "light", "rough", "smooth", "loud", "quiet",
]
_CATEGORIES = [
    "wumpus", "fumpus", "tumpus", "rompus", "vumpus", "dumpus",
    "zumpus", "bumpus", "yumpus", "gumpus", "humpus", "sumpus",
    "numpus", "lumpus", "pumpus", "kumpus", "mumpus", "jimpus",
    "brompus", "crompus", "drompus", "frompus", "grompus", "trompus",
]

# Hop distribution matching ProntoQA's difficulty spread
_HOP_WEIGHTS = {1: 0.20, 2: 0.35, 3: 0.30, 4: 0.15}


def _sample_hop(rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for h, w in _HOP_WEIGHTS.items():
        cum += w
        if r <= cum:
            return h
    return 3


def generate_problem(rng: random.Random, n_hops: int | None = None) -> dict:
    """Generate one synthetic ProntoQA-style problem."""
    if n_hops is None:
        n_hops = _sample_hop(rng)

    entity = rng.choice(_ENTITIES)
    # Draw n_hops+1 distinct category names for the chain
    cats = rng.sample(_CATEGORIES, n_hops + 1)
    final_prop = rng.choice(_PROPERTIES)

    # Rules: cats[0]→cats[1]→...→cats[n_hops-1]→final_prop
    rules = []
    for i in range(n_hops - 1):
        art = "an" if cats[i + 1][0] in "aeiou" else "a"
        rules.append(f"Every {cats[i]} is {art} {cats[i + 1]}.")
    rules.append(f"Every {cats[n_hops - 1]} is {final_prop}.")

    # Seed fact
    art0 = "an" if cats[0][0] in "aeiou" else "a"
    fact = f"{entity} is {art0} {cats[0]}."

    # Gold CoT
    cot = [fact]
    for i in range(1, n_hops):
        art = "an" if cats[i][0] in "aeiou" else "a"
        cot.append(f"{entity} is {art} {cats[i]}.")
    cot.append(f"{entity} is {final_prop}.")

    question = " ".join(rules) + f" {fact}"
    query = f"True or false: {entity} is {final_prop}."

    return {
        "question": question,
        "query": query,
        "chain_of_thought": cot,
        "answer": "True",
        "meta": {"n_hops": n_hops, "entity": entity},
    }


def generate_problems(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return [generate_problem(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Few-shot prompt + parser (same as probe_symbolic_traces.py)
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
# Batched generation
# ---------------------------------------------------------------------------

def generate_traces_batched(
    problems: list[dict],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 120,
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

        # Left-pad for batched generation
        tokenizer.padding_side = "left"
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        for j, (prob, output_ids) in enumerate(zip(batch_problems, out)):
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
    rng = random.Random(args.seed)

    # --- Phase A: generate problems ---
    print(f"Generating {args.n_problems} synthetic problems …")
    problems = generate_problems(args.n_problems, seed=args.seed)

    hop_counts = {}
    for p in problems:
        h = p["meta"]["n_hops"]
        hop_counts[h] = hop_counts.get(h, 0) + 1
    print(f"Hop distribution: { {h: hop_counts[h] for h in sorted(hop_counts)} }")

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
        "Qwen/Qwen2.5-0.5B", dtype=torch.float32
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
    p.add_argument("--n-problems", type=int, default=14_000)
    p.add_argument("--problems-out", default="data/prontoqa_synthetic.jsonl")
    p.add_argument("--traces-out", default="data/prontoqa_traces.jsonl")
    p.add_argument("--problems-only", action="store_true",
                   help="Only generate problems, skip LLM trace generation")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
