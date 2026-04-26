#!/usr/bin/env python3
"""Data audit for the Future-SSAE training pipeline.

Validates GSM8KFutureStepDataset and GSM8KFutureCollateFn before launching
training.  Reports nine sections of statistics and exits non-zero if any
failure condition is hit.

Sections:
  1. Raw problem count (total / valid / discarded with reasons)
  2. Step-pair counts and per-solution distribution
  3. Last-step exclusion verification (must be 0)
  4. Field integrity checks on a random sample
  5. Token-length statistics + truncation risk
  6. Loss-mask integrity via the collate
  7. All four prediction-context modes (length stats, truncation, prev-missing)
  8. Sample distribution by current_step index
  9. Five decoded examples for human inspection

The final line is `RESULT: PASS` or `RESULT: FAIL` so downstream tooling can
verify outcome by grep without parsing the body.

Usage:
    python scripts/audit_future_ssae_data.py \\
        --data data/gsm8k_385K_train.json \\
        --val-data data/gsm8k_385K_valid.json \\
        --model-id Qwen/Qwen2.5-0.5B \\
        --max-length 256

Fail conditions (any one triggers exit 1):
  * future_pairs == 0
  * last-step pairs found > 0
  * any pred_loss_mask or recon_loss_mask has zero target tokens
  * pred_prefix_len out of bounds, or mask invariant violated at prefix
  * recon_input or pred_input truncation rate > 5%
  * empty question/current_step/next_step in any audited sample
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k_dataset import (
    GSM8KFutureCollateFn,
    GSM8KFutureStepDataset,
    _iter_jsonl,
    split_answer_into_steps,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _percentile(values: list[int | float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def _stat_line(values: list[int | float]) -> str:
    if not values:
        return "n=0 (empty)"
    mean = sum(values) / len(values)
    return (
        f"n={len(values)}  mean={mean:.2f}  "
        f"median={_percentile(values, 50)}  "
        f"p90={_percentile(values, 90)}  "
        f"p95={_percentile(values, 95)}  "
        f"p99={_percentile(values, 99)}  "
        f"min={min(values)}  max={max(values)}"
    )


class Report:
    """Collects failures and warnings, prints structured output."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def section(self, title: str) -> None:
        bar = "=" * 76
        print(f"\n{bar}\n  {title}\n{bar}")

    def stat(self, label: str, value) -> None:
        print(f"  {label}: {value}")

    def info(self, msg: str) -> None:
        print(f"  {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  [FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")

    def passed(self) -> bool:
        return len(self.failures) == 0


def load_meta_pairs(path: str | Path) -> list[dict]:
    """Same logic as _load_future_step_pairs but tracks step index, total
    step count, and the full step list per pair (needed for cross-checks)."""
    out: list[dict] = []
    for item in _iter_jsonl(path):
        q = item.get("question", "")
        a = item.get("answer", "")
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        if not q.strip() or not a.strip():
            continue
        steps = [s for s in split_answer_into_steps(a) if s.strip()]
        if len(steps) < 2:
            continue
        for i in range(len(steps) - 1):
            out.append({
                "question": q,
                "context": (q + " " + "".join(steps[:i])).strip(),
                "current_step": steps[i],
                "next_step": steps[i + 1],
                "prev1": steps[i - 1] if i >= 1 else None,
                "prev2": steps[i - 2] if i >= 2 else None,
                "current_step_idx": i,
                "num_steps": len(steps),
                "all_steps": steps,
            })
    return out


# ===========================================================================
# Section 1: Raw problem count
# ===========================================================================

def section_1_raw_problems(path: str | Path, report: Report) -> tuple[int, int]:
    report.section(f"1. Raw problem count: {path}")
    n_total = 0
    n_valid = 0
    discards: Counter[str] = Counter()

    for item in _iter_jsonl(path):
        n_total += 1
        if not isinstance(item, dict):
            discards["non_dict_record"] += 1
            continue
        q = item.get("question", "") or ""
        a = item.get("answer", "") or ""
        if not isinstance(q, str) or not q.strip():
            discards["empty_or_missing_question"] += 1
            continue
        if not isinstance(a, str) or not a.strip():
            discards["empty_or_missing_answer"] += 1
            continue
        steps = [s for s in split_answer_into_steps(a) if s.strip()]
        if len(steps) < 2:
            discards["fewer_than_2_steps"] += 1
            continue
        n_valid += 1

    report.stat("total problems", n_total)
    report.stat("valid problems (>=2 steps)", n_valid)
    report.stat("discarded", n_total - n_valid)
    for reason, count in discards.most_common():
        report.stat(f"  discarded[{reason}]", count)

    if n_valid == 0:
        report.fail("no valid problems found in dataset")
    return n_total, n_valid


# ===========================================================================
# Section 2: Step-pair counts
# ===========================================================================

def section_2_step_pairs(meta_pairs: list[dict], report: Report) -> int:
    report.section("2. Step-pair counts")

    # One representative pair per solution: the one with current_step_idx == 0
    sol_step_counts = [s["num_steps"] for s in meta_pairs if s["current_step_idx"] == 0]
    n_solutions = len(sol_step_counts)
    total_steps = sum(sol_step_counts)
    total_future_pairs = len(meta_pairs)

    report.stat("number of valid solutions", n_solutions)
    report.stat("total steps across valid solutions", total_steps)
    report.stat("total future pairs", total_future_pairs)
    if n_solutions:
        report.stat("avg steps/solution", f"{total_steps / n_solutions:.2f}")
    report.stat("steps/solution distribution", _stat_line(sol_step_counts))

    expected = total_steps - n_solutions
    if total_future_pairs != expected:
        report.fail(
            f"invariant violation: future_pairs ({total_future_pairs}) "
            f"!= total_steps - n_solutions ({expected})"
        )
    else:
        report.stat("invariant total_future_pairs == sum(T-1)", "PASS")

    if total_future_pairs == 0:
        report.fail("zero future pairs - cannot train")

    return total_future_pairs


# ===========================================================================
# Section 3: Last-step exclusion
# ===========================================================================

def section_3_last_step(meta_pairs: list[dict], report: Report) -> int:
    report.section("3. Last-step exclusion verification")

    n_last_step = 0
    n_misaligned_next = 0
    bad_examples: list[dict] = []

    for s in meta_pairs:
        cur_idx = s["current_step_idx"]
        steps = s["all_steps"]
        if cur_idx >= s["num_steps"] - 1:
            n_last_step += 1
            if len(bad_examples) < 3:
                bad_examples.append(s)
            continue  # cannot check next-step alignment for these
        if steps[cur_idx + 1] != s["next_step"]:
            n_misaligned_next += 1

    report.stat("total pairs scanned", len(meta_pairs))
    report.stat("last-step pairs found", n_last_step)
    report.stat("next_step != steps[cur_idx + 1]", n_misaligned_next)

    if n_last_step > 0:
        report.fail(f"{n_last_step} pairs have current_step at the last position")
        for ex in bad_examples:
            report.info(
                f"  example: cur_idx={ex['current_step_idx']} "
                f"num_steps={ex['num_steps']}"
            )
    if n_misaligned_next > 0:
        report.fail(f"{n_misaligned_next} pairs have next_step not at cur_idx+1")

    return n_last_step


# ===========================================================================
# Section 4: Field integrity
# ===========================================================================

def section_4_field_integrity(
    meta_pairs: list[dict], report: Report, n_samples: int, seed: int
) -> None:
    report.section(f"4. Field integrity ({n_samples} random samples)")
    rng = random.Random(seed)
    sub = (meta_pairs if len(meta_pairs) <= n_samples
           else rng.sample(meta_pairs, n_samples))

    fail_counts: Counter[str] = Counter()
    for s in sub:
        if not s["question"].strip():
            fail_counts["empty_question"] += 1
        if not s["context"].strip():
            fail_counts["empty_context"] += 1
        if not s["current_step"].strip():
            fail_counts["empty_current_step"] += 1
        if not s["next_step"].strip():
            fail_counts["empty_next_step"] += 1
        if s["current_step"] == s["next_step"]:
            fail_counts["current_eq_next"] += 1
        if s["current_step"] not in s["all_steps"]:
            fail_counts["current_not_in_solution"] += 1
        cur_idx = s["current_step_idx"]
        if cur_idx + 1 < s["num_steps"] and s["all_steps"][cur_idx + 1] != s["next_step"]:
            fail_counts["next_misaligned"] += 1

        # prev1/prev2 invariants by step index
        if cur_idx == 0:
            if s["prev1"] is not None:
                fail_counts["first_step_prev1_not_none"] += 1
            if s["prev2"] is not None:
                fail_counts["first_step_prev2_not_none"] += 1
            if s["context"].strip() != s["question"].strip():
                fail_counts["first_step_context_neq_question"] += 1
        elif cur_idx == 1:
            if s["prev1"] is None:
                fail_counts["second_step_prev1_missing"] += 1
            if s["prev2"] is not None:
                fail_counts["second_step_prev2_not_none"] += 1
        else:
            if s["prev1"] is None:
                fail_counts["third_plus_prev1_missing"] += 1
            if s["prev2"] is None:
                fail_counts["third_plus_prev2_missing"] += 1

    if not fail_counts:
        report.stat("all checks", "PASS")
    else:
        for k, v in fail_counts.most_common():
            report.stat(f"  {k}", v)

    # Hard fails
    hard = sum(fail_counts[k] for k in (
        "empty_question", "empty_current_step", "empty_next_step",
        "next_misaligned",
        "first_step_prev1_not_none", "first_step_prev2_not_none",
        "first_step_context_neq_question",
        "second_step_prev1_missing", "second_step_prev2_not_none",
        "third_plus_prev1_missing", "third_plus_prev2_missing",
    ))
    if hard > 0:
        report.fail(f"{hard} field-integrity violations on audited subset")


# ===========================================================================
# Section 5: Token-length statistics
# ===========================================================================

def collect_token_lengths(dataset: GSM8KFutureStepDataset) -> list[dict]:
    """Single pass through the dataset to collect per-field token lengths."""
    out = []
    for idx in tqdm(range(len(dataset)), desc="tokenizing"):
        item = dataset[idx]
        out.append({
            "q": len(item["question_ids"]),
            "ctx": len(item["context_ids"]),
            "cur": len(item["current_step_ids"]),
            "nxt": len(item["next_step_ids"]),
            "p1": len(item["prev1_ids"]) if item["prev1_ids"] is not None else None,
            "p2": len(item["prev2_ids"]) if item["prev2_ids"] is not None else None,
        })
    return out


def section_5_token_lengths(
    lengths: list[dict], max_length: int, report: Report
) -> list[int]:
    report.section("5. Token-length statistics")

    q_lens = [L["q"] for L in lengths]
    ctx_lens = [L["ctx"] for L in lengths]
    cur_lens = [L["cur"] for L in lengths]
    nxt_lens = [L["nxt"] for L in lengths]
    p1_lens = [L["p1"] for L in lengths if L["p1"] is not None]
    p2_lens = [L["p2"] for L in lengths if L["p2"] is not None]

    report.stat("question_ids        ", _stat_line(q_lens))
    report.stat("context_ids         ", _stat_line(ctx_lens))
    report.stat("current_step_ids    ", _stat_line(cur_lens))
    report.stat("next_step_ids       ", _stat_line(nxt_lens))
    report.stat("prev1_ids (present) ", _stat_line(p1_lens))
    report.stat("prev2_ids (present) ", _stat_line(p2_lens))

    # Recon input length: context + SEP + current_step + EOS
    recon_lens = [c + 1 + cs + 1 for c, cs in zip(ctx_lens, cur_lens)]
    report.stat("recon_input         ", _stat_line(recon_lens))

    # Pred input length under default mode q_current
    pred_lens = [q + cu + 1 + nx + 1 for q, cu, nx in zip(q_lens, cur_lens, nxt_lens)]
    report.stat("pred_input (q_current)", _stat_line(pred_lens))

    # Truncation risk
    n = len(recon_lens)
    recon_over = sum(1 for L in recon_lens if L > max_length)
    pred_over = sum(1 for L in pred_lens if L > max_length)
    pct_recon = 100.0 * recon_over / max(n, 1)
    pct_pred = 100.0 * pred_over / max(n, 1)
    report.stat(
        f"recon_input > max_length ({max_length})", f"{recon_over} ({pct_recon:.2f}%)"
    )
    report.stat(
        f"pred_input  > max_length ({max_length})", f"{pred_over} ({pct_pred:.2f}%)"
    )

    # Per-field tokenizer-truncation hits (length == max_length is a strong signal)
    for label, vals in [
        ("question", q_lens), ("context", ctx_lens),
        ("current_step", cur_lens), ("next_step", nxt_lens),
    ]:
        hit = sum(1 for L in vals if L >= max_length)
        if hit:
            report.warn(
                f"{hit} samples have {label} >= max_length "
                f"(tokenizer truncation likely)"
            )

    if pct_recon > 5.0:
        report.fail(f"recon_input truncation rate {pct_recon:.2f}% > 5% threshold")
    if pct_pred > 5.0:
        report.fail(f"pred_input (q_current) truncation rate {pct_pred:.2f}% > 5%")

    return recon_lens


# ===========================================================================
# Section 6: Loss-mask integrity
# ===========================================================================

def section_6_mask_integrity(
    dataset: GSM8KFutureStepDataset,
    tokenizer,
    report: Report,
    n_batches: int = 5,
    batch_size: int = 8,
) -> None:
    report.section(f"6. Loss-mask integrity ({n_batches} batches x {batch_size})")

    collate = GSM8KFutureCollateFn(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        space_token_id=tokenizer._future_ssae_space_token_id,
        prediction_context_mode="q_current",
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )

    n_recon_zero = 0
    n_pred_zero = 0
    n_prefix_oob = 0
    n_prefix_token_not_marked = 0
    n_pre_prefix_marked = 0
    n_recon_first_token_not_marked = 0
    n_items = 0

    it = iter(loader)
    for b_idx in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        recon_loss = batch["recon_loss_mask"]
        pred_loss = batch["pred_loss_mask"]
        pred_prefix_lens = batch["pred_prefix_len"]
        pred_ids = batch["pred_input_ids"]
        recon_sep_pos = batch["recon_sep_pos"]

        for i in range(recon_loss.shape[0]):
            n_items += 1
            if recon_loss[i].sum().item() == 0:
                n_recon_zero += 1
            if pred_loss[i].sum().item() == 0:
                n_pred_zero += 1
            # recon: first token of the loss region is at recon_sep_pos
            if recon_loss[i, recon_sep_pos[i]].item() != 1:
                n_recon_first_token_not_marked += 1

            plen = pred_prefix_lens[i]
            if plen < 0 or plen >= pred_ids.shape[1]:
                n_prefix_oob += 1
                continue
            if pred_loss[i, plen].item() != 1:
                n_prefix_token_not_marked += 1
            if pred_loss[i, :plen].sum().item() != 0:
                n_pre_prefix_marked += 1

    report.stat("items checked", n_items)
    report.stat("recon_loss_mask sum=0", n_recon_zero)
    report.stat("pred_loss_mask sum=0", n_pred_zero)
    report.stat("pred_prefix_len out of bounds", n_prefix_oob)
    report.stat("pred_loss_mask[prefix_len] != 1", n_prefix_token_not_marked)
    report.stat("pred_loss_mask[:prefix_len] != 0", n_pre_prefix_marked)
    report.stat("recon_loss_mask[recon_sep_pos] != 1", n_recon_first_token_not_marked)

    if (n_recon_zero or n_pred_zero or n_prefix_oob
            or n_prefix_token_not_marked or n_pre_prefix_marked
            or n_recon_first_token_not_marked):
        report.fail("loss-mask integrity violation (see counts above)")
    else:
        report.stat("all mask invariants", "PASS")


# ===========================================================================
# Section 7: All prediction-context modes
# ===========================================================================

def section_7_all_modes(
    lengths: list[dict], max_length: int, report: Report
) -> None:
    report.section("7. Prediction-context modes")

    n = len(lengths)
    n_missing_p1 = sum(1 for L in lengths if L["p1"] is None)
    n_missing_p2 = sum(1 for L in lengths if L["p2"] is None)
    report.stat(
        "samples with prev1 missing",
        f"{n_missing_p1} ({100.0 * n_missing_p1 / max(n, 1):.2f}%)",
    )
    report.stat(
        "samples with prev2 missing",
        f"{n_missing_p2} ({100.0 * n_missing_p2 / max(n, 1):.2f}%)",
    )

    for mode in ["q_current", "q_prev1_current", "q_prev2_current", "full_context_current"]:
        pred_lens = []
        for L in lengths:
            q, cu, nx, ctx = L["q"], L["cur"], L["nxt"], L["ctx"]
            p1 = L["p1"] or 0
            p2 = L["p2"] or 0
            if mode == "q_current":
                prefix = q + cu + 1
            elif mode == "q_prev1_current":
                prefix = q + p1 + cu + 1
            elif mode == "q_prev2_current":
                prefix = q + p2 + p1 + cu + 1
            else:  # full_context_current
                prefix = ctx + cu + 1
            pred_lens.append(prefix + nx + 1)

        over = sum(1 for L in pred_lens if L > max_length)
        pct = 100.0 * over / max(len(pred_lens), 1)

        report.info("")
        report.info(f"  Mode: {mode}")
        report.stat("    pred_input length", _stat_line(pred_lens))
        report.stat(
            f"    pred_input > max_length ({max_length})", f"{over} ({pct:.2f}%)"
        )
        if pct > 5.0:
            report.fail(f"mode {mode}: pred_input truncation {pct:.2f}% > 5%")


# ===========================================================================
# Section 8: Distribution by step index
# ===========================================================================

def section_8_step_index_distribution(
    meta_pairs: list[dict], report: Report
) -> None:
    report.section("8. Distribution by current_step index")

    n_total = len(meta_pairs)
    idx_counts: Counter[int] = Counter(s["current_step_idx"] for s in meta_pairs)

    n_first = idx_counts.get(0, 0)
    n_first_two = idx_counts.get(0, 0) + idx_counts.get(1, 0)
    n_final_minus_one = sum(
        1 for s in meta_pairs if s["current_step_idx"] == s["num_steps"] - 2
    )

    def pct(x: int) -> str:
        return f"{x} ({100.0 * x / max(n_total, 1):.2f}%)"

    report.stat("first-step pairs (idx=0)", pct(n_first))
    report.stat("first-two-step pairs (idx<=1)", pct(n_first_two))
    report.stat("final-1-step pairs (idx == num_steps-2)", pct(n_final_minus_one))

    report.info("  Histogram (step index → count, percentage):")
    for idx in sorted(idx_counts):
        c = idx_counts[idx]
        report.stat(f"    idx={idx:>3}", f"{c} ({100.0 * c / n_total:.2f}%)")


# ===========================================================================
# Section 9: Example inspection
# ===========================================================================

def section_9_examples(
    dataset: GSM8KFutureStepDataset,
    tokenizer,
    report: Report,
    n_examples: int,
    seed: int,
) -> None:
    report.section(f"9. Example inspection ({n_examples} random)")

    rng = random.Random(seed)
    n = len(dataset)
    indices = rng.sample(range(n), min(n_examples, n))

    collate = GSM8KFutureCollateFn(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        space_token_id=tokenizer._future_ssae_space_token_id,
        prediction_context_mode="q_current",
    )

    for ex_i, idx in enumerate(indices):
        raw = dataset.samples[idx]
        item = dataset[idx]
        batch = collate([item])

        recon_ids = batch["recon_input_ids"][0]
        pred_ids = batch["pred_input_ids"][0]
        recon_attn = batch["recon_attention_mask"][0]
        pred_attn = batch["pred_attention_mask"][0]
        recon_mask = batch["recon_loss_mask"][0]
        pred_mask = batch["pred_loss_mask"][0]
        prefix_len = batch["pred_prefix_len"][0]

        # Trim padding for decoding
        recon_real = recon_ids[recon_attn.bool()].tolist()
        pred_real = pred_ids[pred_attn.bool()].tolist()
        recon_text = tokenizer.decode(recon_real, skip_special_tokens=False)
        pred_text = tokenizer.decode(pred_real, skip_special_tokens=False)

        recon_target = tokenizer.decode(
            recon_ids[recon_mask.bool()].tolist(), skip_special_tokens=False
        )
        pred_target = tokenizer.decode(
            pred_ids[pred_mask.bool()].tolist(), skip_special_tokens=False
        )

        print(f"\n  --- Example {ex_i + 1}/{n_examples} (sample idx={idx}) ---")
        print(f"  QUESTION:\n    {raw['question']!r}")
        print(f"  CONTEXT:\n    {raw['context']!r}")
        print(f"  CURRENT STEP:\n    {raw['current_step']!r}")
        print(f"  NEXT STEP:\n    {raw['next_step']!r}")
        print(f"  PRED PREFIX LEN: {prefix_len}  (token at this position is "
              f"first s_k+1 token)")
        print(f"  RECON INPUT DECODED:\n    {recon_text!r}")
        print(f"  PRED INPUT DECODED:\n    {pred_text!r}")
        print(f"  RECON MASKED TARGET:\n    {recon_target!r}")
        print(f"  PRED MASKED TARGET:\n    {pred_target!r}")


# ===========================================================================
# Orchestrator
# ===========================================================================

def audit_split(
    split_name: str, path: str | Path, args: argparse.Namespace,
    tokenizer, report: Report,
) -> None:
    print(f"\n\n{'#' * 78}")
    print(f"# SPLIT: {split_name.upper()}  ({path})")
    print(f"{'#' * 78}")

    section_1_raw_problems(path, report)

    print(f"\nLoading meta pairs from {path} ...")
    meta_pairs = load_meta_pairs(path)
    print(f"  loaded {len(meta_pairs)} pairs with metadata")

    section_2_step_pairs(meta_pairs, report)
    section_3_last_step(meta_pairs, report)
    section_4_field_integrity(
        meta_pairs, report, n_samples=args.n_samples_detailed, seed=args.seed
    )

    print(f"\nBuilding GSM8KFutureStepDataset ...")
    dataset = GSM8KFutureStepDataset(path, tokenizer, max_length=args.max_length)
    print(f"  dataset size: {len(dataset)}")
    if len(dataset) != len(meta_pairs):
        report.fail(
            f"dataset size ({len(dataset)}) != meta pairs ({len(meta_pairs)})"
        )

    print("\nCollecting token lengths (one pass) ...")
    lengths = collect_token_lengths(dataset)

    section_5_token_lengths(lengths, args.max_length, report)
    section_6_mask_integrity(dataset, tokenizer, report)
    section_7_all_modes(lengths, args.max_length, report)
    section_8_step_index_distribution(meta_pairs, report)
    section_9_examples(
        dataset, tokenizer, report, n_examples=args.n_examples, seed=args.seed
    )


def run_audit(args: argparse.Namespace) -> int:
    report = Report()

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "<sep>"})
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    if not space_ids:
        raise ValueError(
            f"Tokenizer {args.model_id} cannot encode a single space; "
            "Future-SSAE collate requires a single-token space separator."
        )
    tokenizer._future_ssae_space_token_id = space_ids[0]

    splits = [("train", args.data)]
    if args.val_data:
        splits.append(("val", args.val_data))

    for split_name, path in splits:
        audit_split(split_name, path, args, tokenizer, report)

    # ----- Final summary -----
    bar = "=" * 78
    print(f"\n\n{bar}\n  FINAL SUMMARY\n{bar}")
    print(f"  failures: {len(report.failures)}")
    print(f"  warnings: {len(report.warnings)}")

    if report.failures:
        print("\n  FAILURES:")
        for f in report.failures:
            print(f"    - {f}")
    if report.warnings:
        print("\n  WARNINGS:")
        for w in report.warnings:
            print(f"    - {w}")

    print()  # blank line before final verdict
    if report.passed():
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL")
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Future-SSAE data audit")
    p.add_argument("--data", required=True, help="Path to train JSONL")
    p.add_argument("--val-data", default=None, help="Path to val JSONL (optional)")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--pred-context-mode", default="q_current",
                   choices=list(GSM8KFutureCollateFn.MODES),
                   help="Mode used for the deep mask-integrity check (section 6)")
    p.add_argument("--n-samples-detailed", type=int, default=100,
                   help="Random sample size for section 4 field integrity")
    p.add_argument("--n-examples", type=int, default=5,
                   help="Number of decoded examples to print in section 9")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run_audit(parse_args()))
