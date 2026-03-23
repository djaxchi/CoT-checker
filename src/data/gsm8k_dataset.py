"""GSM8K step-level dataset.

Each problem's answer is split into individual reasoning steps.
Each sample is a (context, step) pair where context = problem + all prior steps.

Data format expected (JSONL, one problem per line):
    {"question": "A store...", "answer": "Step 1\nStep 2\n#### 42"}

Ported and adapted from the SSAE reference dataloader (Miaow-Lab/SSAE).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def split_answer_into_steps(answer: str) -> list[str]:
    """Split a GSM8K answer string into individual reasoning steps.

    The last two lines (usually "#### <answer>") are kept as one unit.
    """
    answer = answer.replace("\n\n", "\n")
    sentences = answer.split("\n")
    try:
        sentences = sentences[:-2] + [sentences[-2] + "\n" + sentences[-1]]
    except (IndexError, TypeError):
        sentences = answer.split(". ")
    return [s for s in sentences if s.strip()]


def gsm8k_acc_judge(ground_step: str, pred: str) -> int:
    """Return 1 if the last number in pred matches the last number in ground_step.

    Used to automatically label step correctness for GSM8K without an external LLM.
    """
    ground_nums = re.findall(r"[-+]?\d*\.\d+|\d+", ground_step)
    pred_nums = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
    if not ground_nums or not pred_nums:
        return 0
    try:
        return int(abs(float(ground_nums[-1]) - float(pred_nums[-1])) < 1e-3)
    except (ValueError, OverflowError):
        return 0


# Safe arithmetic characters after normalisation
_SAFE_EXPR_RE = re.compile(r"^[\d\s\+\-\*\/\.\(\)]+$")
# Match numbers-only multiplication: 1.5 x 2  →  we normalize to *
_MUL_X_RE = re.compile(r"(?<=[\d\s\)])\s*[xX]\s*(?=[\d\(])")
# Equation pattern: <arithmetic lhs starting with digit> = <number>
_EQUATION_RE = re.compile(r"([\d][\d\s\+\-\*\/\.\(\)]*)\s*=\s*([-+]?\d+(?:\.\d+)?)")


def _normalise(expr: str) -> str:
    """Normalize an arithmetic expression for safe eval.

    Replaces " x " used as multiplication with " * " only when it sits
    between numeric operands (avoids clobbering algebraic variable names).
    """
    return _MUL_X_RE.sub("*", expr)


def symbolic_step_judge(step: str) -> int:
    """Symbolically verify arithmetic correctness of a GSM8K reasoning step.

    Finds equations of the form "<arithmetic expr> = <number>" in the step,
    evaluates the left-hand side, and checks it matches the declared result.
    This is the paper's labeling methodology: a symbolic verifier independent
    of the SSAE or any model output.

    Returns:
        1 if all verifiable equations are arithmetically correct (or no
          verifiable equations are found — step assumed correct by default).
        0 if any equation's lhs evaluates to a value ≠ declared rhs.
    """
    # Strip <<...>> calculator annotations embedded by some GSM8K variants
    clean = re.sub(r"<<[^>]*>>", "", step)
    clean = _normalise(clean)

    equations = _EQUATION_RE.findall(clean)
    if not equations:
        return 1  # nothing to verify

    for lhs, rhs in equations:
        lhs = lhs.strip()
        if not _SAFE_EXPR_RE.match(lhs):
            continue  # skip algebraic / unparseable expressions
        try:
            computed = eval(lhs, {"__builtins__": {}})  # noqa: S307
            expected = float(rhs.strip())
            if abs(float(computed) - expected) > 1e-3:
                return 0
        except Exception:
            continue

    return 1


def corrupt_step(step: str, rng: "random.Random | None" = None) -> tuple[str, bool]:
    """Introduce an arithmetic error into a step by perturbing one result.

    Finds the last equation "<expr> = <number>" and replaces the number
    with a wrong value (+/- a small offset, never zero change).

    Returns:
        (corrupted_step, was_corrupted) — was_corrupted is False when no
        suitable equation was found and the step is returned unchanged.
    """
    if rng is None:
        rng = random

    clean = re.sub(r"<<[^>]*>>", "", step)
    normalised = _normalise(clean)
    equations = list(_EQUATION_RE.finditer(normalised))

    for m in reversed(equations):
        lhs_raw = m.group(1).strip()
        rhs_str = m.group(2).strip()
        if not _SAFE_EXPR_RE.match(lhs_raw):
            continue
        try:
            expected = float(rhs_str)
        except ValueError:
            continue

        # Pick a non-zero offset that makes the answer clearly wrong
        offset = rng.choice([-3, -2, -1, 1, 2, 3])
        wrong = expected + offset
        # Preserve integer formatting when original had no decimal point
        if "." not in rhs_str:
            wrong_str = str(int(wrong))
        else:
            wrong_str = f"{wrong:.{len(rhs_str.split('.')[-1])}f}"

        # Replace only the matched rhs span (use original step offsets)
        # Compute offset of rhs in original step
        rhs_start = m.start(2)
        rhs_end = m.end(2)
        corrupted = normalised[:rhs_start] + wrong_str + normalised[rhs_end:]
        return corrupted, True

    return step, False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_step_pairs(path: str | Path) -> list[dict]:
    """Load all (context, step) pairs from a GSM8K JSONL file."""
    samples = []
    for item in _iter_jsonl(path):
        question = item["question"]
        steps = split_answer_into_steps(item["answer"])
        for i, step in enumerate(steps):
            if not step.strip():
                continue
            context = question + " " + "".join(steps[:i])
            samples.append({"context": context.strip(), "step": step})
    return samples


class GSM8KStepDataset(Dataset):
    """One sample = one (context, step) pair from a GSM8K JSONL file.

    Args:
        file_path: Path to a JSONL file in the format
            {"question": ..., "answer": ...}.
        tokenizer: HuggingFace tokenizer (must have sep_token set to "<sep>").
        max_length: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.samples = _load_step_pairs(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        context_ids = torch.tensor(
            self.tokenizer.encode(sample["context"], max_length=self.max_length, truncation=True),
            dtype=torch.long,
        )
        step_ids = torch.tensor(
            self.tokenizer.encode(sample["step"], max_length=self.max_length, truncation=True),
            dtype=torch.long,
        )
        return {
            "context": sample["context"],
            "step": sample["step"],
            "context_ids": context_ids,
            "step_ids": step_ids,
        }


class GSM8KCollateFn:
    """Collate (context, step) pairs into padded batches.

    Builds the full input sequence: [context | <sep> | step | <eos>]
    and the hints-only sequence: [context | <sep>]

    These match the format expected by the SSAE model.
    """

    def __init__(self, eos_token_id: int, pad_token_id: int, sep_token_id: int) -> None:
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

    def __call__(self, batch: list[dict]) -> dict:
        contexts = [b["context"] for b in batch]
        steps = [b["step"] for b in batch]

        # Build full input: [context | sep | step | eos]
        full_seqs, sep_positions, seq_lengths = [], [], []
        for b in batch:
            seq = torch.cat(
                [
                    b["context_ids"],
                    torch.tensor([self.sep_token_id], dtype=torch.long),
                    b["step_ids"],
                    torch.tensor([self.eos_token_id], dtype=torch.long),
                ]
            )
            sep_positions.append(len(b["context_ids"]) + 1)  # position right after sep
            full_seqs.append(seq)
            seq_lengths.append(len(seq))

        max_len = max(seq_lengths)
        input_ids_list, attn_mask_list, loss_mask_list, hints_sep_list = [], [], [], []

        for i, seq in enumerate(full_seqs):
            pad = max_len - len(seq)
            input_id = torch.cat([seq, torch.full((pad,), self.pad_token_id, dtype=torch.long)])
            attn = torch.cat(
                [torch.ones(len(seq), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]
            )
            loss_m = torch.cat(
                [torch.ones(len(seq), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]
            )
            loss_m[: sep_positions[i]] = 0  # mask out context tokens from loss

            input_ids_list.append(input_id)
            attn_mask_list.append(attn)
            loss_mask_list.append(loss_m)
            hints_sep_list.append(input_id[: sep_positions[i]])

        max_hints_len = max(len(h) for h in hints_sep_list)
        hints_sep_ids_list, hints_sep_attn_list = [], []
        for h in hints_sep_list:
            pad = max_hints_len - len(h)
            hints_sep_ids_list.append(
                torch.cat([h, torch.full((pad,), self.pad_token_id, dtype=torch.long)])
            )
            hints_sep_attn_list.append(
                torch.cat(
                    [torch.ones(len(h), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]
                )
            )

        return {
            "contexts": contexts,
            "steps": steps,
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
            "loss_mask": torch.stack(loss_mask_list),
            "hints_sep_ids": torch.stack(hints_sep_ids_list),
            "hints_sep_attention_masks": torch.stack(hints_sep_attn_list),
            "sep_pos": sep_positions,
            "val_len": seq_lengths,
        }
