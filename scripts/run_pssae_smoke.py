#!/usr/bin/env python3
"""Predictive Sparse Encoder (PSSAE) smoke run.

Trains a single linear enc_proj on top of a frozen Qwen2.5-0.5B backbone.
The encoder maps the last-token hidden state at step k into a sparse z_k,
which is injected as a virtual token into the decoder to predict step k+1.
Trained exclusively on positive (correct) GSM8K transitions.
After training, evaluates OOD discrimination on GSM8K-valid and ProcessBench.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.gsm8k_dataset import symbolic_step_judge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step splitting / data loading
# ---------------------------------------------------------------------------

def _split_answer_steps(answer: str) -> list[str]:
    steps = re.split(r"\n(?=Step \d+:)", answer.strip())
    if len(steps) > 1:
        return [s.strip() for s in steps if s.strip()]
    steps = [s.strip() for s in answer.split("\n") if s.strip()]
    return steps if steps else [answer.strip()]


def _load_problems(data_file: str) -> list[dict]:
    problems = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _build_transitions(problems: list[dict], positive_only: bool) -> list[dict]:
    records = []
    for prob in problems:
        question = prob.get("question", "")
        answer = prob.get("answer", "")
        steps = _split_answer_steps(answer)
        if len(steps) < 2:
            continue
        if positive_only and not all(symbolic_step_judge(s) == 1 for s in steps):
            continue
        for k in range(len(steps) - 1):
            records.append({
                "problem": question,
                "prior_steps": steps[:k],
                "step_k": steps[k],
                "step_k1": steps[k + 1],
                "label_k": symbolic_step_judge(steps[k]),
            })
    return records


# ---------------------------------------------------------------------------
# Encoder tokenisation
# ---------------------------------------------------------------------------

def _build_encoder_ids(
    tokenizer,
    problem: str,
    prior_steps: list[str],
    step_k: str,
    sep_token_id: int,
    max_seq_len: int,
) -> tuple[list[int], int]:
    """Return (token_id_list, last_token_index).

    Sequence: [problem + prior | SEP | step_k | EOS]
    Truncates context from the left when total length would exceed max_seq_len.
    """
    ctx = (problem + " " + " ".join(prior_steps)).strip()
    ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
    step_ids = tokenizer.encode(step_k, add_special_tokens=False)
    tail = [sep_token_id] + step_ids + [tokenizer.eos_token_id]
    budget = max_seq_len - len(tail)
    # Bug fix: budget=0 meant ctx_ids[-0:] == full list; use empty slice when budget<=0
    ctx_ids = ctx_ids[-budget:] if budget > 0 else []
    seq = ctx_ids + tail
    return seq, len(seq) - 1


# ---------------------------------------------------------------------------
# Batched backbone helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_batch(
    backbone,
    tokenizer,
    records: list[dict],
    sep_token_id: int,
    max_seq_len: int,
    device: str,
) -> torch.Tensor:
    """One backbone call for the whole batch.  Returns h_ks [B, d] float32."""
    all_seqs, last_idxs = [], []
    for rec in records:
        seq, last_idx = _build_encoder_ids(
            tokenizer, rec["problem"], rec.get("prior_steps", []),
            rec["step_k"], sep_token_id, max_seq_len,
        )
        all_seqs.append(seq)
        last_idxs.append(last_idx)

    max_len = max(len(s) for s in all_seqs)
    B = len(all_seqs)
    pad_id = tokenizer.eos_token_id
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(all_seqs):
        L = len(seq)
        input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=device)
        attn_mask[i, :L] = 1

    hidden = backbone(
        input_ids, attention_mask=attn_mask, output_hidden_states=True
    ).hidden_states[-1]  # [B, max_len, d]

    h_ks = torch.stack([hidden[i, last_idxs[i]] for i in range(B)])
    return h_ks.float()  # cast to float32 for enc_proj


def _build_decoder_batch(
    backbone,
    tokenizer,
    records: list[dict],
    z_ks: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Build padded (inputs_embeds, attn_mask, labels) for the decoder batch.

    Returns:
        inputs_embeds  [B, max_len, d]
        attn_mask      [B, max_len]
        labels         [B, max_len]  (-100 on prefix + padding)
        prob_lens      list[int]  number of problem tokens per item (for PPL slice)
        step_lens      list[int]  number of step_k1 tokens per item
    """
    embed = backbone.model.embed_tokens
    dtype = backbone.dtype

    items: list[tuple[torch.Tensor, torch.Tensor]] = []  # (embeds, labels) per item
    prob_lens, step_lens = [], []

    for i, rec in enumerate(records):
        prob_ids = torch.tensor(
            tokenizer.encode(rec["problem"], add_special_tokens=False),
            dtype=torch.long, device=device,
        )
        step_k1_ids = torch.tensor(
            tokenizer.encode(rec["step_k1"], add_special_tokens=False),
            dtype=torch.long, device=device,
        )
        prob_emb = embed(prob_ids)                    # [L_prob, d]
        step_emb = embed(step_k1_ids)                 # [L_step, d]
        z_tok = z_ks[i].to(dtype).unsqueeze(0)        # [1, d]

        # Sequence: [prob | z_k | step_k1]
        seq_emb = torch.cat([prob_emb, z_tok, step_emb], dim=0)  # [L_prob+1+L_step, d]

        ignore = torch.full((len(prob_ids) + 1,), -100, dtype=torch.long, device=device)
        lbl = torch.cat([ignore, step_k1_ids])        # [L_prob+1+L_step]

        items.append((seq_emb, lbl))
        prob_lens.append(len(prob_ids))
        step_lens.append(len(step_k1_ids))

    seq_lens = [e.shape[0] for e, _ in items]
    max_len = max(seq_lens)
    B = len(items)
    d = z_ks.shape[-1]

    # Build padded tensors via cat (preserves gradient graph for training)
    padded_list = []
    for emb, _ in items:
        pad_len = max_len - emb.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, d, dtype=dtype, device=device)
            padded_list.append(torch.cat([emb, padding], dim=0))
        else:
            padded_list.append(emb)
    inputs_embeds = torch.stack(padded_list)          # [B, max_len, d]

    padded_labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, (slen, (_, lbl)) in enumerate(zip(seq_lens, items)):
        padded_labels[i, :slen] = lbl
        attn_mask[i, :slen] = 1

    return inputs_embeds, attn_mask, padded_labels, prob_lens, step_lens


def _decoder_train_loss(
    backbone,
    tokenizer,
    records: list[dict],
    z_ks: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """One batched decoder forward with labels. Returns mean CE loss (has grad)."""
    inputs_embeds, attn_mask, labels, _, _ = _build_decoder_batch(
        backbone, tokenizer, records, z_ks, device
    )
    return backbone(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels).loss


@torch.no_grad()
def _decoder_perplexities(
    backbone,
    tokenizer,
    records: list[dict],
    z_ks: torch.Tensor,
    device: str,
) -> list[float]:
    """One batched decoder forward (no labels). Returns per-item perplexity list."""
    inputs_embeds, attn_mask, _, prob_lens, step_lens = _build_decoder_batch(
        backbone, tokenizer, records, z_ks, device
    )
    logits = backbone(
        inputs_embeds=inputs_embeds, attention_mask=attn_mask
    ).logits  # [B, max_len, vocab]

    ppls = []
    for i in range(len(records)):
        L_prob = prob_lens[i]
        L_step = step_lens[i]
        if L_step == 0:
            ppls.append(float("nan"))
            continue
        # logits[i, L_prob : L_prob+L_step] predict step_k1_ids[0..L_step-1]
        # (causal shift: logit at pos t predicts token at pos t+1;
        #  z_k sits at pos L_prob, so pos L_prob predicts step_k1[0])
        item_logits = logits[i, L_prob: L_prob + L_step].float()   # [L_step, vocab]
        step_k1_ids = torch.tensor(
            tokenizer.encode(records[i]["step_k1"], add_special_tokens=False),
            dtype=torch.long, device=device,
        )
        ce = F.cross_entropy(item_logits, step_k1_ids)
        ppls.append(math.exp(ce.item()))
    return ppls


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TransitionDataset(Dataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    log.info("Loading backbone: %s  (bfloat16)", args.backbone)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, cache_dir=args.hf_cache)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone,
        cache_dir=args.hf_cache,
        torch_dtype=torch.bfloat16,   # half precision: ~2x throughput, ~half memory
    ).to(args.device)
    backbone.requires_grad_(False)
    backbone.eval()

    d_model = backbone.config.hidden_size
    log.info("d_model = %d  vocab = %d", d_model, backbone.config.vocab_size)

    sep_token_id = tokenizer.eos_token_id  # Qwen has no dedicated sep token

    enc_proj = nn.Linear(d_model, d_model, bias=True).to(args.device)  # float32
    optimizer = torch.optim.Adam(enc_proj.parameters(), lr=args.lr)

    log.info("Loading training problems from %s", args.train_json)
    train_problems = _load_problems(args.train_json)
    train_problems = train_problems[: int(len(train_problems) * args.data_frac)]
    log.info("Building positive-only transitions from %d train problems...", len(train_problems))
    train_records = _build_transitions(train_problems, positive_only=True)
    if 0 < args.max_train_transitions < len(train_records):
        train_records = train_records[: args.max_train_transitions]
        log.info("Train transitions (capped): %d", len(train_records))
    else:
        log.info("Train transitions: %d", len(train_records))

    log.info("Loading validation problems from %s", args.valid_json)
    valid_problems = _load_problems(args.valid_json)
    valid_problems = valid_problems[: int(len(valid_problems) * args.data_frac)]
    log.info("Building positive-only transitions from %d valid problems...", len(valid_problems))
    valid_records = _build_transitions(valid_problems, positive_only=True)
    log.info("Valid transitions: %d", len(valid_records))

    train_loader = DataLoader(
        TransitionDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
        num_workers=0,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best_enc_proj.pt"
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        enc_proj.train()
        epoch_ce_sum = 0.0
        n_batches = 0
        t0 = time.time()

        for step_num, batch in enumerate(train_loader, 1):
            # --- ONE encoder call for the whole batch ---
            h_ks = _encode_batch(
                backbone, tokenizer, batch, sep_token_id, args.max_seq_len, args.device
            )  # [B, d] float32, no grad

            # gradient flows through enc_proj → z_ks → decoder loss
            z_ks = F.relu(enc_proj(h_ks))  # [B, d] float32

            # --- ONE decoder call for the whole batch ---
            ce_loss = _decoder_train_loss(backbone, tokenizer, batch, z_ks, args.device)
            l1_loss = z_ks.abs().mean()
            loss = ce_loss + args.l1_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_ce_sum += ce_loss.item()
            n_batches += 1

            if step_num % 50 == 0:
                log.info(
                    "Epoch %d  step %d/%d  train_CE=%.4f  elapsed=%.1fs",
                    epoch, step_num, len(train_loader),
                    epoch_ce_sum / n_batches,
                    time.time() - t0,
                )

        # --- Epoch-end validation (batched perplexity) ---
        enc_proj.eval()
        val_ces = []
        log.info("Epoch %d: validating on %d transitions...", epoch, len(valid_records))
        for start in range(0, len(valid_records), args.eval_batch_size):
            vbatch = valid_records[start: start + args.eval_batch_size]
            h_ks = _encode_batch(
                backbone, tokenizer, vbatch, sep_token_id, args.max_seq_len, args.device
            )
            z_ks = F.relu(enc_proj(h_ks.detach()))
            ppls = _decoder_perplexities(backbone, tokenizer, vbatch, z_ks, args.device)
            val_ces.extend([math.log(p) for p in ppls if not math.isnan(p)])

        mean_val_ce = float(np.mean(val_ces)) if val_ces else float("nan")
        log.info(
            "Epoch %d  train_CE=%.4f  val_CE=%.4f",
            epoch, epoch_ce_sum / max(n_batches, 1), mean_val_ce,
        )

        if mean_val_ce < best_val_loss:
            best_val_loss = mean_val_ce
            torch.save(enc_proj.state_dict(), best_ckpt)
            log.info("  => new best checkpoint (val_CE=%.4f)", best_val_loss)

    log.info("Loading best checkpoint from %s", best_ckpt)
    enc_proj.load_state_dict(torch.load(best_ckpt, map_location=args.device))
    return backbone, tokenizer, enc_proj, sep_token_id


# ---------------------------------------------------------------------------
# Eval 1: MS OOD
# ---------------------------------------------------------------------------

def eval_ms_ood(
    backbone,
    enc_proj: nn.Linear,
    tokenizer,
    sep_token_id: int,
    args: argparse.Namespace,
) -> None:
    from sklearn.metrics import roc_auc_score

    log.info("=== Eval 1: MS OOD (GSM8K-valid, all solutions) ===")
    problems = _load_problems(args.valid_json)
    problems = problems[: int(len(problems) * args.data_frac)]
    log.info("Building all-solution transitions from %d problems...", len(problems))
    records = _build_transitions(problems, positive_only=False)
    log.info("OOD eval transitions: %d", len(records))

    enc_proj.eval()
    perplexities, labels = [], []

    for start in range(0, len(records), args.eval_batch_size):
        if start % (args.eval_batch_size * 50) == 0:
            log.info("  OOD eval: %d / %d", start, len(records))
        batch = records[start: start + args.eval_batch_size]
        h_ks = _encode_batch(
            backbone, tokenizer, batch, sep_token_id, args.max_seq_len, args.device
        )
        z_ks = F.relu(enc_proj(h_ks))
        ppls = _decoder_perplexities(backbone, tokenizer, batch, z_ks, args.device)
        for rec, ppl in zip(batch, ppls):
            perplexities.append(ppl)
            labels.append(1 - rec["label_k"])  # 1 = incorrect step

    perplexities = np.array(perplexities)
    labels = np.array(labels)
    correct_ppl = perplexities[labels == 0]
    wrong_ppl = perplexities[labels == 1]

    log.info("--- MS OOD results ---")
    if len(correct_ppl):
        log.info(
            "Correct steps  n=%d  mean_ppl=%.2f  std=%.2f",
            len(correct_ppl), correct_ppl.mean(), correct_ppl.std(),
        )
    if len(wrong_ppl):
        log.info(
            "Wrong steps    n=%d  mean_ppl=%.2f  std=%.2f",
            len(wrong_ppl), wrong_ppl.mean(), wrong_ppl.std(),
        )
    if len(correct_ppl) and len(wrong_ppl):
        ratio = wrong_ppl.mean() / correct_ppl.mean()
        log.info("Ratio wrong/correct: %.3f", ratio)
        try:
            auroc = roc_auc_score(labels, perplexities)
            log.info("AUROC: %.4f", auroc)
            if auroc > 0.6:
                verdict = "supported"
            elif auroc > 0.5:
                verdict = "not supported"
            else:
                verdict = "refuted"
            log.info("Verdict: %s", verdict)
        except Exception as exc:
            log.warning("AUROC computation failed: %s", exc)
    else:
        log.warning("Skipping AUROC: need both correct and wrong steps in eval set")


# ---------------------------------------------------------------------------
# Eval 2: ProcessBench
# ---------------------------------------------------------------------------

def _pb_steps(row: dict) -> list[str]:
    if "steps" in row:
        return [s if isinstance(s, str) else s.get("text", "") for s in row["steps"]]
    if "answer" in row:
        return _split_answer_steps(row["answer"])
    return []


def eval_processbench(
    backbone,
    enc_proj: nn.Linear,
    tokenizer,
    sep_token_id: int,
    args: argparse.Namespace,
) -> None:
    log.info("=== Eval 2: ProcessBench (first %d solutions) ===", args.n_pb)
    pb_rows = []
    with open(args.pb_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                pb_rows.append(json.loads(line))
    pb_rows = pb_rows[: args.n_pb]
    log.info("Loaded %d ProcessBench solutions", len(pb_rows))

    enc_proj.eval()
    all_correct_ppls: list[float] = []
    all_wrong_ppls: list[float] = []
    n_error_sols = 0
    n_argmax_hits = 0
    step_counts_error: list[int] = []
    sol_ppls_cache: list[tuple[list[float], int]] = []  # (ppls, true_label)

    for sol_idx, row in enumerate(pb_rows):
        question = row.get("question", row.get("problem", ""))
        raw_steps = _pb_steps(row)
        if not raw_steps:
            log.warning("Sol %d: no steps found, skipping", sol_idx)
            continue

        true_label = row.get("label", -1)
        n_steps = len(raw_steps)

        # Build records for steps 0..n-2 (last step has no step_k1)
        step_records = []
        for k in range(n_steps - 1):
            step_records.append({
                "problem": question,
                "prior_steps": raw_steps[:k],
                "step_k": raw_steps[k],
                "step_k1": raw_steps[k + 1],
            })

        # Compute perplexities in one batch for this solution
        step_ppls: list[float] = [float("nan")] * n_steps
        if step_records:
            h_ks = _encode_batch(
                backbone, tokenizer, step_records, sep_token_id, args.max_seq_len, args.device
            )
            z_ks = F.relu(enc_proj(h_ks))
            ppls = _decoder_perplexities(backbone, tokenizer, step_records, z_ks, args.device)
            for k, ppl in enumerate(ppls):
                step_ppls[k] = ppl

        sol_ppls_cache.append((step_ppls, true_label))

        # Collect for aggregate stats
        for k, ppl in enumerate(step_ppls):
            if math.isnan(ppl):
                continue
            is_wrong = true_label != -1 and k >= true_label
            if is_wrong:
                all_wrong_ppls.append(ppl)
            else:
                all_correct_ppls.append(ppl)

        # Argmax prediction (excluding last-step NaN)
        valid_pairs = [(k, p) for k, p in enumerate(step_ppls) if not math.isnan(p)]
        pred_k = max(valid_pairs, key=lambda x: x[1])[0] if valid_pairs else 0

        is_error_sol = true_label != -1
        if is_error_sol:
            n_error_sols += 1
            n_argmax_hits += int(pred_k == true_label)
            step_counts_error.append(n_steps)

        result_str = (
            "HIT" if (is_error_sol and pred_k == true_label)
            else "MISS" if is_error_sol
            else "CLEAN"
        )
        log.info(
            "Sol %3d  label=%2s  steps=%d  PRED: %s (pred=%d, true=%s)",
            sol_idx, str(true_label), n_steps, result_str, pred_k, str(true_label),
        )
        for k, ppl in enumerate(step_ppls):
            is_wrong = true_label != -1 and k >= true_label
            lbl = "WRONG" if is_wrong else "correct"
            marker = ""
            if not math.isnan(ppl) and k == pred_k:
                marker += "  <- PRED max"
            if true_label != -1 and k == true_label:
                marker += "  <- TRUE first error"
            ppl_str = f"{ppl:.1f}" if not math.isnan(ppl) else "NaN"
            log.info("    k=%d  %-8s  ppl=%s%s", k, lbl, ppl_str, marker)

    # Aggregate stats
    arr_corr = np.array(all_correct_ppls)
    arr_wrong = np.array(all_wrong_ppls)

    log.info("--- ProcessBench aggregate ---")
    if len(arr_corr):
        log.info(
            "Correct steps  n=%d  mean=%.2f  std=%.2f  p95=%.2f",
            len(arr_corr), arr_corr.mean(), arr_corr.std(), np.percentile(arr_corr, 95),
        )
    if len(arr_wrong):
        log.info(
            "Wrong steps    n=%d  mean=%.2f  std=%.2f  p95=%.2f",
            len(arr_wrong), arr_wrong.mean(), arr_wrong.std(), np.percentile(arr_wrong, 95),
        )
    if len(arr_corr) and len(arr_wrong):
        log.info("Ratio wrong/correct: %.3f", arr_wrong.mean() / arr_corr.mean())

    if n_error_sols > 0:
        hit_rate = n_argmax_hits / n_error_sols
        avg_steps = float(np.mean(step_counts_error))
        rand_base = 1.0 / avg_steps if avg_steps > 0 else float("nan")
        log.info(
            "Argmax hit rate: %d/%d = %.3f  (random baseline: %.3f)",
            n_argmax_hits, n_error_sols, hit_rate, rand_base,
        )

    # PB-F1 table (thresholds from correct-step perplexity distribution)
    if len(arr_corr):
        log.info("--- PB-F1 across thresholds ---")
        log.info("%-8s  %-8s  %-8s  %-8s  %s", "pct", "P", "R", "F1", "thresh")
        for pct in [50, 75, 90, 95, 99]:
            thresh = float(np.percentile(arr_corr, pct))
            tp = fp = fn = 0
            for step_ppls, true_label in sol_ppls_cache:
                valid_ppls = [(k, p) for k, p in enumerate(step_ppls) if not math.isnan(p)]
                pred_err = next((k for k, p in valid_ppls if p > thresh), -1)
                if true_label == -1:
                    if pred_err != -1:
                        fp += 1
                else:
                    if pred_err == true_label:
                        tp += 1
                    else:
                        fn += 1
                        if pred_err != -1:
                            fp += 1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
            log.info("p%-7d  %.3f     %.3f     %.3f  %.2f", pct, prec, rec_, f1, thresh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PSSAE smoke run")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--train-json", required=True)
    p.add_argument("--valid-json", required=True)
    p.add_argument("--pb-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-frac", type=float, default=0.5)
    p.add_argument("--n-pb", type=int, default=50)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l1-coeff", type=float, default=0.01)
    p.add_argument("--max-train-transitions", type=int, default=0,
                   help="Cap train set to this many transitions (0 = no cap)")
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Args: %s", vars(args))
    backbone, tokenizer, enc_proj, sep_token_id = train(args)
    eval_ms_ood(backbone, enc_proj, tokenizer, sep_token_id, args)
    eval_processbench(backbone, enc_proj, tokenizer, sep_token_id, args)
    log.info("Done.")


if __name__ == "__main__":
    main()
