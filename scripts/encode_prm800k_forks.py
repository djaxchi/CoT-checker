"""Encode PRM800K fork *items* (anchor / positive / negative) to hidden states.

Mirrors ``encode_prm800k_hidden_states.py`` exactly for positive/negative items
(last-token hidden state of ``Problem + prefix + Current step:<candidate>``), and
adds an ``anchor`` role that embeds the reasoning prefix alone: the last token of
``build_prompt_prefix(problem, prefix)`` (i.e. the model state right before the
next step is produced). The prefix text is byte-identical to the one used for
positives/negatives, so anchor and continuations live in the same activation
space.

Output (aligned to the order of rows in --items):
  {stem}_h.npy        (n_items, hidden_dim) float16/float32
  {stem}_meta.jsonl   one row per item: item_uid, fork_id, role, row index
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import (  # noqa: E402
    build_prompt_prefix,
    git_commit,
    read_jsonl,
    sha256_file,
    tokenize_example,
    write_jsonl,
)


def tokenize_anchor(tokenizer, problem: str, prefix: str, max_seq_len: int) -> tuple[list[int], int]:
    """Tokenize the prefix prompt; return (input_ids, last_token_idx)."""
    ids = tokenizer(
        build_prompt_prefix(problem, prefix),
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    if not ids:
        raise ValueError("Anchor prefix produced an empty token sequence.")
    if len(ids) > max_seq_len:
        raise ValueError(
            f"Anchor sequence length {len(ids)} exceeds max_seq_len={max_seq_len}."
        )
    return ids, len(ids) - 1


def encode_items(
    items: list[dict],
    tokenizer,
    model,
    device: torch.device,
    max_seq_len: int,
    batch_size: int,
    save_dtype: torch.dtype,
    pad_token_id: int,
) -> tuple[np.ndarray, list[dict]]:
    n = len(items)
    hidden_dim = model.config.hidden_size
    np_dtype = np.float16 if save_dtype == torch.float16 else np.float32
    all_hidden = np.zeros((n, hidden_dim), dtype=np_dtype)
    meta: list[dict] = []
    t0 = time.perf_counter()

    i = 0
    while i < n:
        batch = items[i : i + batch_size]
        batch_ids: list[list[int]] = []
        batch_last_idx: list[int] = []
        for ex in batch:
            try:
                if ex["role"] == "anchor":
                    ids, last_idx = tokenize_anchor(
                        tokenizer, ex["problem"], ex["prefix"], max_seq_len
                    )
                else:
                    ids, last_idx = tokenize_example(
                        tokenizer, ex["problem"], ex["prefix"],
                        ex["candidate_step"], max_seq_len,
                    )
            except ValueError as e:
                sys.exit(f"[encode-forks] FATAL: {ex.get('item_uid','?')}: {e}")
            batch_ids.append(ids)
            batch_last_idx.append(last_idx)

        max_len = max(len(ids) for ids in batch_ids)
        padded = [ids + [pad_token_id] * (max_len - len(ids)) for ids in batch_ids]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_ids]
        input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        attn_tensor = torch.tensor(masks, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                input_tensor, attention_mask=attn_tensor,
                output_hidden_states=True, use_cache=False,
            )
        last_layer = out.hidden_states[-1]
        del out

        for b, (ex, last_idx) in enumerate(zip(batch, batch_last_idx)):
            vec = last_layer[b, last_idx, :].detach().to(save_dtype).cpu().numpy()
            all_hidden[i + b] = vec
            meta.append({
                "row": i + b,
                "item_uid": ex["item_uid"],
                "fork_id": ex["fork_id"],
                "role": ex["role"],
                "label": ex["label"],
                "n_tokens": len(batch_ids[b]),
                "last_token_idx": last_idx,
            })
        del last_layer
        i += len(batch)
        if (i // batch_size) % 32 == 0 or i == n:
            print(f"[encode-forks] {i}/{n} ({time.perf_counter()-t0:.1f}s)", flush=True)

    return all_hidden, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Encode PRM800K fork items to hidden states.")
    p.add_argument("--items", type=Path, required=True, help="forks_*_items.jsonl")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--stem", type=str, required=True, help="output prefix, e.g. forks_train_items")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--max_seq_len", type=int, default=2048,
                   help="Hard cap; encoding fails (never truncates) on overlength. "
                        "Pass -1 to use the model's full context window.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    h_path = args.out_dir / f"{args.stem}_h.npy"
    if h_path.exists() and not args.force:
        sys.exit(f"[encode-forks] Refusing to overwrite {h_path}. Pass --force.")

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    print(f"[encode-forks] Loading {args.model_name_or_path} ...", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only
        )
        try:  # transformers >=5 uses dtype=; older uses torch_dtype=
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, local_files_only=args.local_files_only,
                dtype=dtype_map[args.model_dtype],
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, local_files_only=args.local_files_only,
                torch_dtype=dtype_map[args.model_dtype],
            )
    except OSError:
        sys.exit("Model not found locally. Pre-cache the model before running offline.")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        sys.exit("[encode-forks] Tokenizer has no pad/eos token; cannot pad.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    model_max = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        if model_max <= 0:
            sys.exit("[encode-forks] model has no max_position_embeddings; pass an explicit --max_seq_len.")
        args.max_seq_len = model_max
    if model_max and args.max_seq_len > model_max:
        sys.exit(f"[encode-forks] --max_seq_len={args.max_seq_len} exceeds model context {model_max}.")

    items = read_jsonl(args.items)
    print(f"[encode-forks] {len(items)} items from {args.items}", flush=True)

    hidden, meta = encode_items(
        items, tokenizer, model, device,
        args.max_seq_len, args.batch_size,
        dtype_map[args.save_dtype], pad_token_id,
    )

    np.save(h_path, hidden)
    write_jsonl(args.out_dir / f"{args.stem}_meta.jsonl", meta)
    manifest = {
        "run_name": args.run_name,
        "model": args.model_name_or_path,
        "stem": args.stem,
        "n_items": len(items),
        "hidden_dim": int(hidden.shape[1]),
        "layer": "last",
        "anchor_token_position": "last token of build_prompt_prefix",
        "continuation_token_position": "last token of candidate_step",
        "saved_dtype": args.save_dtype,
        "sha256_hidden": sha256_file(h_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }
    (args.out_dir / f"{args.stem}_encoding_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"[encode-forks] Saved {h_path} ({h_path.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
