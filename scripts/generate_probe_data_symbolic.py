#!/usr/bin/env python3
"""Generate SSAE probe data from model-generated symbolic logic traces.

Pipeline:
  1. Load ProntoQA-style problems from a JSONL file.
  2. Prompt Qwen2.5-0.5B (base LLM) to generate a chain-of-thought for each problem.
  3. Parse the model output into individual steps.
  4. Label each step with PropLogicSolver (deterministic, no model needed).
  5. Encode (context | <sep> | step) through the SSAE encoder → sparse latent h_c.
  6. Save (latents, correctness) as .npz — same format as the arithmetic pipeline.

The SSAE checkpoint used here is the *arithmetic* one (gsm8k-385k_Qwen2.5-0.5b_spar-10.pt).
That is intentional: we want to measure how well arithmetic SSAE features transfer to
symbolic logic, as a probe for task-specificity of the sparse representations.

Usage:
    python scripts/generate_probe_data_symbolic.py \\
        --checkpoint gsm8k-385k_Qwen2.5-0.5b_spar-10.pt \\
        --data data/prontoqa_sample.jsonl \\
        --output results/probe_data/symbolic_pilot.npz \\
        --device mps
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.saes.ssae import SSAE
from src.data.symbolic_logic_dataset import PropLogicSolver, label_chain

# ---------------------------------------------------------------------------
# Few-shot prompt (same as probe_symbolic_traces.py)
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

_STEP_LINE_RE = re.compile(
    r"^[A-Z][a-z]*\s+is\s+(?:not\s+)?(?:a\s+|an\s+)?\w+\s*\.$"
)
_ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*:\s*(True|False)\s*$", re.IGNORECASE)


def parse_model_output(raw: str) -> list[str]:
    steps: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if _ANSWER_LINE_RE.match(line):
            break
        if _STEP_LINE_RE.match(line):
            steps.append(line)
    return steps


# ---------------------------------------------------------------------------
# Encoding (identical to generate_probe_data.py)
# ---------------------------------------------------------------------------

def encode_batch(
    ssae: SSAE,
    tokenizer,
    contexts: list[str],
    steps: list[str],
    device,
    sep_token_id: int,
    max_len: int = 256,
) -> np.ndarray:
    batch_ids = []
    for ctx, step in zip(contexts, steps):
        ctx_ids = tokenizer.encode(ctx, max_length=max_len, truncation=True)
        step_ids = tokenizer.encode(step, max_length=max_len, truncation=True)
        seq = ctx_ids + [sep_token_id] + step_ids + [tokenizer.eos_token_id]
        batch_ids.append(seq)

    max_seq = max(len(s) for s in batch_ids)
    pad_id = tokenizer.eos_token_id
    input_ids = torch.full((len(batch_ids), max_seq), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(batch_ids), max_seq), dtype=torch.long, device=device)
    for i, seq in enumerate(batch_ids):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attn_mask[i, : len(seq)] = 1

    with torch.no_grad():
        latents = ssae.encode(input_ids, attn_mask)  # (B, 1, n_latents)
    return latents.squeeze(1).cpu().float().numpy()   # (B, n_latents)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_problems(path: Path) -> list[dict]:
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def main():
    args = parse_args()
    device = torch.device(args.device)

    # --- Phase 1: generate traces (gen model only, no SSAE on device yet) ---
    # Loading gen model + SSAE simultaneously on MPS exhausts memory and causes
    # degenerate generation output. We generate all traces first, free the gen
    # model, then load the SSAE for encoding.

    print("Loading Qwen2.5-0.5B for generation …")
    gen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    gen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", dtype=torch.float32
    ).to(device)
    gen_model.eval()

    problems = load_problems(Path(args.data))
    print(f"Loaded {len(problems)} problems from {args.data}")

    all_records: list[dict] = []
    skipped = 0

    for prob in tqdm(problems, desc="Generating traces"):
        question = prob["question"]
        query = prob.get("query", "")

        prompt = _FEW_SHOT.format(question=question, query=query)
        inputs = gen_tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = gen_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=gen_tokenizer.eos_token_id,
            )

        raw = gen_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        steps = parse_model_output(raw)

        if not steps:
            skipped += 1
            continue

        labels = label_chain(question, steps)

        for i, (step, label) in enumerate(zip(steps, labels)):
            prior = " ".join(steps[:i])
            context = f"{question} {query} {prior}".strip()
            all_records.append({"context": context, "step": step, "label": label})

    # Free gen model before loading SSAE
    del gen_model, gen_tokenizer
    if args.device == "mps":
        torch.mps.empty_cache()
    elif args.device == "cuda":
        torch.cuda.empty_cache()

    # --- Phase 2: load SSAE and encode ---
    print(f"\nLoading SSAE from {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("frozen_backbones"):
        # Slim checkpoint: backbone weights not stored, re-download from HF
        print("  Slim checkpoint detected (frozen backbones). Re-loading backbone from HF …")
        from transformers import AutoTokenizer as _AT
        tokenizer_ssae = _AT.from_pretrained("Qwen/Qwen2.5-0.5B")
        if tokenizer_ssae.sep_token is None:
            tokenizer_ssae.add_special_tokens({"sep_token": "<sep>"})
        tokenizer_ssae.sep_token_id = tokenizer_ssae.convert_tokens_to_ids(tokenizer_ssae.sep_token)
        ssae = SSAE(tokenizer=tokenizer_ssae, sparsity_factor=1, dtype=torch.float32).to(device)
        ssae.load_state_dict(ckpt["model"], strict=False)  # only sparse layers present
    else:
        ssae = SSAE.from_checkpoint(args.checkpoint, device=device)
    ssae.eval()
    ssae_tokenizer = ssae.tokenizer
    sep_tok_id = ssae_tokenizer.sep_token_id

    print(f"\nProblems skipped (no parseable steps): {skipped}")
    print(f"Steps collected: {len(all_records)}")

    if not all_records:
        print("No records collected — exiting.")
        return

    pos = sum(r["label"] for r in all_records)
    neg = len(all_records) - pos
    print(f"Correct (+): {pos}  ({pos / len(all_records):.1%})")
    print(f"Incorrect (-): {neg}  ({neg / len(all_records):.1%})")
    print(f"Majority baseline: {max(pos, neg) / len(all_records):.1%}")

    # --- Optional rebalancing ---
    if args.correct_ratio is not None:
        import random
        random.seed(args.seed)
        correct_recs = [r for r in all_records if r["label"] == 1]
        incorrect_recs = [r for r in all_records if r["label"] == 0]
        ratio = args.correct_ratio
        if len(correct_recs) / len(all_records) > ratio:
            n_cor = int(len(incorrect_recs) * ratio / (1 - ratio))
            correct_recs = random.sample(correct_recs, min(n_cor, len(correct_recs)))
        else:
            n_inc = int(len(correct_recs) * (1 - ratio) / ratio)
            incorrect_recs = random.sample(incorrect_recs, min(n_inc, len(incorrect_recs)))
        all_records = correct_recs + incorrect_recs
        random.shuffle(all_records)
        pos = sum(r["label"] for r in all_records)
        print(f"\nAfter rebalancing to {ratio:.0%}/{1-ratio:.0%}:")
        print(f"  Total: {len(all_records)}  Correct: {pos}  Incorrect: {len(all_records)-pos}")

    # --- Encode through arithmetic SSAE ---
    print(f"\nEncoding {len(all_records)} steps through arithmetic SSAE …")
    all_latents, all_labels = [], []
    bs = args.batch_size

    for i in tqdm(range(0, len(all_records), bs), desc="Encoding"):
        batch = all_records[i : i + bs]
        ctxs = [r["context"] for r in batch]
        stps = [r["step"] for r in batch]
        lbls = [r["label"] for r in batch]

        latents = encode_batch(ssae, ssae_tokenizer, ctxs, stps, device, sep_tok_id)
        all_latents.append(latents)
        all_labels.extend(lbls)

    latents_arr = np.concatenate(all_latents, axis=0).astype(np.float16)
    labels_arr = np.array(all_labels, dtype=np.int8)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, latents=latents_arr, correctness=labels_arr)
    print(f"\nSaved {latents_arr.shape} latents to {args.output}")
    print(f"Sparsity: {(latents_arr == 0).mean():.1%} of entries are zero")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to arithmetic SSAE checkpoint (.pt)")
    p.add_argument("--data", default="data/prontoqa_sample.jsonl",
                   help="ProntoQA-style JSONL problem file")
    p.add_argument("--output", default="results/probe_data/symbolic_pilot.npz")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--correct-ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
