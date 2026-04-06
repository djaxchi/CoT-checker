#!/usr/bin/env python3
"""Scale experiment: 200K steps at 72/28 (correct/incorrect), threshold=0.8.

Pipeline:
  1. Stream ~200K new raw steps from Math-Shepherd (offset 110K to avoid
     overlap with existing training data and eval set)
  2. Merge with existing 57K balanced npz (28,891 correct + 28,891 incorrect)
  3. Build a 200K dataset at 72% correct / 28% incorrect by oversampling
     correct (~1.7x) and subsampling incorrect
  4. Train probe with same hyperparameters as previous experiments
  5. Evaluate on the shared 5K held-out set at threshold=0.8

Usage:
    python scripts/experiment_scale_200k.py \\
        --checkpoint <path/to/ssae.pt> \\
        --device mps
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.probes.classifier import StepCorrectnessClassifier
from src.saes.ssae import SSAE

STEP_DELIM = "\u043a\u0438"
NEW_DATA_PATH   = Path("results/probe_data/math_shepherd_new_200k.npz")
MERGED_DATA_PATH = Path("results/probe_data/math_shepherd_200k_flipped.npz")
PROBE_OUT       = Path("results/probes/correctness_probe_200k_flipped.pt")
EVAL_DATA       = Path("results/probe_data/math_shepherd_eval_5000.npz")
EXISTING_NPZ    = Path("results/probe_data/math_shepherd_100k_balanced.npz")

# ---------------------------------------------------------------------------
# Data generation (reused from generate_probe_data.py)
# ---------------------------------------------------------------------------

def parse_entry(entry: dict) -> list[dict]:
    label_str = entry["label"]
    q_match = re.search(r"^(.*?)(?=Step 1:)", label_str, re.DOTALL)
    question = q_match.group(1).strip() if q_match else ""
    step_blocks = re.findall(
        r"(Step \d+:.*?)\s*([+\-])\s*(?=Step \d+:|$)",
        label_str, re.DOTALL,
    )
    records, prior_steps = [], []
    for step_text, sign in step_blocks:
        clean = re.sub(r"<<[^>]*>>", "", step_text).strip()
        context = (question + " " + " ".join(prior_steps)).strip()
        records.append({"context": context, "text": clean, "label": 1 if sign == "+" else 0})
        prior_steps.append(clean)
    return records


def encode_batch(model, tokenizer, contexts, steps, device, sep_token_id, max_len=128):
    batch_ids = []
    for ctx, step in zip(contexts, steps):
        ctx_ids  = tokenizer.encode(ctx,  max_length=max_len, truncation=True)
        step_ids = tokenizer.encode(step, max_length=max_len, truncation=True)
        seq = ctx_ids + [sep_token_id] + step_ids + [tokenizer.eos_token_id]
        batch_ids.append(seq)
    max_seq = max(len(s) for s in batch_ids)
    pad_id = tokenizer.eos_token_id
    input_ids = torch.full((len(batch_ids), max_seq), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(batch_ids), max_seq), dtype=torch.long, device=device)
    for i, seq in enumerate(batch_ids):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attn_mask[i, :len(seq)] = 1
    with torch.no_grad():
        latents = model.encode(input_ids, attn_mask)
    return latents.squeeze(1).cpu().float().numpy()


def generate_new_data(checkpoint: str, device: str, offset: int, max_raw: int, batch_size: int):
    print(f"\n[1/4] Streaming Math-Shepherd from offset {offset:,}, up to {max_raw:,} raw steps...")
    model = SSAE.from_checkpoint(checkpoint, device=device)
    model.eval()
    tokenizer  = model.tokenizer
    sep_tok_id = tokenizer.sep_token_id

    ds = load_dataset("peiyi9979/Math-Shepherd", split="train", streaming=True)

    all_records: list[dict] = []
    raw_seen = 0
    for entry in ds:
        if entry.get("task") != "GSM8K":
            continue
        records = parse_entry(entry)
        raw_seen += len(records)
        if raw_seen <= offset:
            continue
        all_records.extend(records)
        if raw_seen >= offset + max_raw:
            break

    n_cor = sum(r["label"] for r in all_records)
    print(f"  Collected {len(all_records):,} steps  ({n_cor:,} correct {n_cor/len(all_records):.1%}, "
          f"{len(all_records)-n_cor:,} incorrect {(len(all_records)-n_cor)/len(all_records):.1%})")

    all_latents, all_labels = [], []
    checkpoint_every = 500  # save partial results every 500 batches (~4K steps)
    NEW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(0, len(all_records), batch_size), desc="  Encoding"):
        batch = all_records[i:i + batch_size]
        lats  = encode_batch(model, tokenizer, [r["context"] for r in batch],
                             [r["text"] for r in batch], device, sep_tok_id)
        all_latents.extend(lats)
        all_labels.extend(r["label"] for r in batch)
        # periodic checkpoint so a crash doesn't lose everything
        batch_idx = i // batch_size
        if batch_idx > 0 and batch_idx % checkpoint_every == 0:
            np.savez_compressed(
                NEW_DATA_PATH,
                latents=np.array(all_latents, dtype=np.float16),
                correctness=np.array(all_labels, dtype=np.int8),
            )
            print(f"  [checkpoint] {len(all_labels):,} steps saved")

    np.savez_compressed(
        NEW_DATA_PATH,
        latents=np.array(all_latents, dtype=np.float16),
        correctness=np.array(all_labels, dtype=np.int8),
    )
    print(f"  Saved {len(all_labels):,} encoded steps -> {NEW_DATA_PATH}")
    return np.array(all_latents, dtype=np.float32), np.array(all_labels, dtype=np.int8)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_merged_dataset(new_h, new_y, correct_frac: float, target_total: int, seed: int):
    print(f"\n[2/4] Merging with existing 57K balanced data...")
    d_exist = np.load(EXISTING_NPZ)
    ex_h = d_exist["latents"].astype(np.float32)
    ex_y = d_exist["correctness"]

    h_all = np.concatenate([ex_h, new_h], axis=0)
    y_all = np.concatenate([ex_y, new_y], axis=0)

    n_cor_total = (y_all == 1).sum()
    n_inc_total = (y_all == 0).sum()
    print(f"  Combined pool: {len(y_all):,} steps  "
          f"({n_cor_total:,} correct {n_cor_total/len(y_all):.1%}, "
          f"{n_inc_total:,} incorrect {n_inc_total/len(y_all):.1%})")

    rng = np.random.default_rng(seed)
    cor_idx = np.where(y_all == 1)[0]
    inc_idx = np.where(y_all == 0)[0]

    n_correct   = int(target_total * correct_frac)
    n_incorrect = target_total - n_correct

    sampled_cor = rng.choice(cor_idx, size=n_correct,   replace=len(cor_idx) < n_correct)
    sampled_inc = rng.choice(inc_idx, size=n_incorrect, replace=len(inc_idx) < n_incorrect)

    idx = np.concatenate([sampled_cor, sampled_inc])
    rng.shuffle(idx)

    h_out, y_out = h_all[idx], y_all[idx]
    oversample_ratio = n_correct / len(cor_idx)
    print(f"  Final dataset : {len(y_out):,} steps  "
          f"({n_correct:,} correct {correct_frac:.0%}, {n_incorrect:,} incorrect {1-correct_frac:.0%})")
    print(f"  Correct oversample ratio: {oversample_ratio:.2f}x  "
          f"({len(cor_idx):,} unique -> {n_correct:,})")

    np.savez_compressed(MERGED_DATA_PATH,
                        latents=h_out.astype(np.float16),
                        correctness=y_out.astype(np.int8))
    print(f"  Saved -> {MERGED_DATA_PATH}")
    return h_out, y_out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class ArrayDataset(Dataset):
    def __init__(self, h, y):
        self.latents     = torch.from_numpy(h)
        self.correctness = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return len(self.latents)
    def __getitem__(self, i): return self.latents[i], self.correctness[i]


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(y); n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = (torch.sigmoid(model(x).squeeze(-1)) >= threshold).long()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.long().cpu().tolist())
    labels_t = torch.tensor(all_labels)
    preds_t  = torch.tensor(all_preds)
    acc = (preds_t == labels_t).float().mean().item()
    results = {}
    for cls, name in [(1, "correct"), (0, "incorrect")]:
        tp = ((preds_t == cls) & (labels_t == cls)).sum().item()
        fp = ((preds_t == cls) & (labels_t != cls)).sum().item()
        fn = ((preds_t != cls) & (labels_t == cls)).sum().item()
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results[name] = {"precision": p, "recall": r, "f1": f1}
    return acc, results


def train_probe(h, y, args):
    print(f"\n[3/4] Training probe on {len(y):,} steps...")
    full_ds = ArrayDataset(h, y)
    n_train = int(len(full_ds) * args.split)
    train_ds, val_ds = random_split(
        full_ds, [n_train, len(full_ds) - n_train],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    model = StepCorrectnessClassifier(
        input_dim=h.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    best_acc, best_state = 0.0, None
    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}")
    print("─" * 28)
    for epoch in range(1, args.epochs + 1):
        loss    = train_epoch(model, train_loader, optimizer, args.device)
        val_acc, _ = evaluate(model, val_loader, args.device)
        scheduler.step()
        marker = " *" if val_acc > best_acc else ""
        print(f"{epoch:>5}  {loss:>8.4f}  {val_acc:>8.4f}{marker}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.save(PROBE_OUT)
    print(f"\n  Best val acc: {best_acc*100:.2f}%  |  Saved -> {PROBE_OUT}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def print_results(label, acc, results, majority):
    print(f"\n{'─'*54}")
    print(f"  {label}")
    print(f"{'─'*54}")
    print(f"  Majority baseline : {majority*100:.2f}%")
    print(f"  Accuracy          : {acc*100:.2f}%  (+{(acc-majority)*100:.2f}pp)")
    for cls in ["correct", "incorrect"]:
        r = results[cls]
        print(f"\n  Class '{cls}':")
        print(f"    Precision : {r['precision']:.3f}")
        print(f"    Recall    : {r['recall']:.3f}")
        print(f"    F1        : {r['f1']:.3f}")


def run_eval(args):
    print(f"\n[4/4] Evaluating on 5K held-out set (threshold={args.threshold})...")
    d = np.load(EVAL_DATA)
    eval_ds  = ArrayDataset(d["latents"].astype(np.float32), d["correctness"])
    loader   = DataLoader(eval_ds, batch_size=256, shuffle=False)
    majority = max(eval_ds.correctness.mean().item(), 1 - eval_ds.correctness.mean().item())

    model_200k = StepCorrectnessClassifier.load(PROBE_OUT, device=args.device)
    acc_200k, res_200k = evaluate(model_200k, loader, args.device, threshold=args.threshold)
    print_results(f"200K flipped probe (72/28, threshold={args.threshold})", acc_200k, res_200k, majority)

    # Compare against previous flipped 40K
    prev_path = Path("results/probes/correctness_probe_flipped.pt")
    if prev_path.exists():
        model_40k = StepCorrectnessClassifier.load(str(prev_path), device=args.device)
        acc_40k, res_40k = evaluate(model_40k, loader, args.device, threshold=args.threshold)
        print_results(f"40K flipped probe  (72/28, threshold={args.threshold})", acc_40k, res_40k, majority)

        print(f"\n{'─'*54}")
        print("  COMPARISON (threshold=0.8, 72/28 distribution)")
        print(f"{'─'*54}")
        print(f"  {'':32s} {'40K':>8}  {'200K':>8}")
        print(f"  {'Accuracy':32s} {acc_40k*100:>8.2f}%  {acc_200k*100:>8.2f}%")
        for cls in ["correct", "incorrect"]:
            for m in ["precision", "recall", "f1"]:
                print(f"  {cls+' '+m:32s} {res_40k[cls][m]:>8.3f}   {res_200k[cls][m]:>8.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to SSAE checkpoint .pt")
    p.add_argument("--device",      default="mps",  choices=["cpu", "cuda", "mps"])
    p.add_argument("--offset",      type=int,   default=110_000)
    p.add_argument("--max-raw",     type=int,   default=200_000,
                   help="Raw Math-Shepherd steps to stream for new data")
    p.add_argument("--target-total",type=int,   default=200_000)
    p.add_argument("--correct-frac",type=float, default=0.72)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--hidden-dim",  type=int,   default=1024)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--split",       type=float, default=0.8)
    p.add_argument("--threshold",   type=float, default=0.8)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip data generation if new npz already exists")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Step 1: generate new encoded data
    if args.skip_generate and NEW_DATA_PATH.exists():
        print(f"\n[1/4] Loading existing new data from {NEW_DATA_PATH}...")
        d = np.load(NEW_DATA_PATH)
        new_h, new_y = d["latents"].astype(np.float32), d["correctness"]
        print(f"  {len(new_y):,} steps  ({(new_y==1).sum():,} correct, {(new_y==0).sum():,} incorrect)")
    else:
        new_h, new_y = generate_new_data(
            args.checkpoint, args.device, args.offset, args.max_raw, args.batch_size
        )

    # Step 2: merge and build 200K dataset
    h, y = build_merged_dataset(new_h, new_y, args.correct_frac, args.target_total, args.seed)

    # Step 3: train
    model = train_probe(h, y, args)

    # Step 4: evaluate
    run_eval(args)


if __name__ == "__main__":
    main()
