"""
Encode PRM800K prestudy examples with Qwen2.5-1.5B hidden states.

Extracts the last hidden layer at the last token of candidate_step.
No truncation is ever used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"num_gpus": 0, "gpu_name": None}
    return {
        "num_gpus": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
    }


# ---------------------------------------------------------------------------
# Tokenization
#
# The prompt prefix text and candidate_step are tokenized separately so we
# can pinpoint the exact index of the last candidate token.
#
# This function MUST stay in sync with build_prompt_prefix in the builder.
# ---------------------------------------------------------------------------

def build_prompt_prefix(problem: str, prefix: str) -> str:
    prefix_section = f"Previous reasoning:\n{prefix}\n\n" if prefix else "Previous reasoning:\n\n"
    return f"Problem:\n{problem}\n\n{prefix_section}Current step:\n"


def tokenize_example(
    tokenizer,
    problem: str,
    prefix: str,
    candidate_step: str,
    max_seq_len: int,
) -> tuple[list[int], int]:
    """
    Returns (input_ids, candidate_last_token_idx).

    The prompt prefix is tokenized with BOS (add_special_tokens=True).
    The candidate is tokenized without any special tokens so concatenation
    is clean and candidate_last_token_idx is exactly len(full_ids) - 1.

    Fails hard if len(full_ids) > max_seq_len.  The dataset builder should
    have filtered all such examples already.
    """
    prompt_prefix = build_prompt_prefix(problem, prefix)

    prefix_ids: list[int] = tokenizer(
        prompt_prefix,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    candidate_ids: list[int] = tokenizer(
        candidate_step,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    if not candidate_ids:
        raise ValueError("Candidate step produced an empty token sequence.")

    full_ids = prefix_ids + candidate_ids
    candidate_last_token_idx = len(full_ids) - 1

    if len(full_ids) > max_seq_len:
        raise ValueError(
            f"Sequence length {len(full_ids)} exceeds max_seq_len={max_seq_len}. "
            "The dataset builder should have filtered this example. "
            "This is a bug — check that build_prompt_prefix is identical in "
            "both scripts and that the length filter used split tokenization."
        )

    return full_ids, candidate_last_token_idx


# ---------------------------------------------------------------------------
# Batched encoding
#
# Examples in a batch are padded on the RIGHT to the length of the longest
# example in the batch, with an attention mask so the model ignores padding.
# The hidden state is extracted at candidate_last_token_idx, which is
# computed BEFORE padding and is therefore unaffected by it.
# ---------------------------------------------------------------------------

def encode_file(
    jsonl_path: Path,
    out_dir: Path,
    stem: str,
    tokenizer,
    model,
    device: torch.device,
    max_seq_len: int,
    batch_size: int,
    save_dtype: torch.dtype,
    pad_token_id: int,
    limit: int | None = None,
) -> dict:
    """Encode one JSONL file. Returns per-file stats."""
    examples = read_jsonl(jsonl_path)
    if limit is not None:
        examples = examples[:limit]

    n = len(examples)
    print(f"[encode] {stem}: {n} examples", flush=True)

    hidden_dim = model.config.hidden_size
    np_dtype = np.float16 if save_dtype == torch.float16 else np.float32
    all_hidden = np.zeros((n, hidden_dim), dtype=np_dtype)
    all_labels = np.zeros(n, dtype=np.int32)
    all_meta: list[dict] = []

    n_overlength = 0
    t_start = time.perf_counter()

    i = 0
    while i < n:
        batch_exs = examples[i : i + batch_size]

        # Tokenize each example in the batch
        batch_input_ids: list[list[int]] = []
        batch_cand_last_idx: list[int] = []

        for ex in batch_exs:
            try:
                ids, cand_idx = tokenize_example(
                    tokenizer,
                    ex["problem"],
                    ex["prefix"],
                    ex["candidate_step"],
                    max_seq_len,
                )
                batch_input_ids.append(ids)
                batch_cand_last_idx.append(cand_idx)
            except ValueError as e:
                # Per spec §4 and §8: fail hard if any example exceeds max_seq_len.
                # The builder must have filtered all overlength examples; reaching
                # here means a builder/encoder mismatch that must be investigated.
                sys.exit(
                    f"[encode] FATAL: overlength example at encoding time. "
                    f"uid={ex.get('uid', '?')}: {e}\n"
                    "The dataset builder should have excluded this example. "
                    "Ensure build_prompt_prefix is identical in both scripts "
                    "and that filter_by_length uses split tokenization."
                )

        # Right-pad to longest sequence in batch
        max_len = max(len(ids) for ids in batch_input_ids)
        padded = []
        attn_masks = []
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [pad_token_id] * pad_len)
            attn_masks.append([1] * len(ids) + [0] * pad_len)

        input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        attn_tensor = torch.tensor(attn_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(
                input_tensor,
                attention_mask=attn_tensor,
                output_hidden_states=True,
                use_cache=False,
            )

        # Extract hidden state at the last token of candidate_step for each example
        last_layer = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        del outputs  # free GPU memory immediately

        for b_idx, (ex, cand_idx) in enumerate(zip(batch_exs, batch_cand_last_idx)):
            n_tokens = len(batch_input_ids[b_idx])
            assert n_tokens <= max_seq_len, (
                f"BUG: n_tokens={n_tokens} > max_seq_len={max_seq_len}"
            )
            assert cand_idx < n_tokens, (
                f"BUG: cand_idx={cand_idx} >= n_tokens={n_tokens}"
            )

            vec = last_layer[b_idx, cand_idx, :].detach().to(save_dtype).cpu().numpy()

            all_hidden[i + b_idx] = vec
            all_labels[i + b_idx] = ex["label"]
            all_meta.append({
                "uid": ex["uid"],
                "problem_id": ex["problem_id"],
                "solution_id": ex["solution_id"],
                "step_idx": ex["step_idx"],
                "completion_idx": ex["completion_idx"],
                "label": ex["label"],
                "rating": ex["rating"],
                "candidate_last_token_idx": cand_idx,
                "n_tokens": n_tokens,
                "was_truncated": False,
            })

        del last_layer

        i += len(batch_exs)

        # Print every 32 batches (~512 examples at batch_size=16) and at the end
        if (i // batch_size) % 32 == 0 or i == n:
            elapsed = time.perf_counter() - t_start
            print(f"[encode] {stem}: {i}/{n} done ({elapsed:.1f}s)", flush=True)

    t_end = time.perf_counter()
    total_sec = t_end - t_start
    avg_ms = (total_sec / n * 1000) if n > 0 else 0.0

    h_path = out_dir / f"{stem}_h.npy"
    y_path = out_dir / f"{stem}_y.npy"
    m_path = out_dir / f"{stem}_meta.jsonl"

    np.save(h_path, all_hidden)
    np.save(y_path, all_labels)
    write_jsonl(m_path, all_meta)

    h_sha = sha256_file(h_path)
    print(f"[encode] {stem}: saved ({h_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    return {
        "jsonl_path": str(jsonl_path),
        "hidden_path": str(h_path),
        "label_path": str(y_path),
        "meta_path": str(m_path),
        "n_examples": n,
        "sha256_hidden": h_sha,
        "n_overlength": n_overlength,
        "n_truncated": 0,
        "n_skipped": 0,
        "total_encoding_time_sec": total_sec,
        "avg_latency_ms_per_example": avg_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FILE_STEMS = [
    ("prm800k_pos_base_20k.jsonl", "pos_base_20k"),
    ("prm800k_neg_base_20k.jsonl", "neg_base_20k"),
    ("prm800k_probe_train_40k.jsonl", "probe_train_40k"),
    ("prm800k_mixed_train_40k.jsonl", "mixed_train_40k"),
    ("prm800k_val_1k.jsonl", "val_1k"),
    ("prm800k_contrastive_forks_20_flat.jsonl", "contrastive_forks_20_flat"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode PRM800K hidden states with Qwen2.5-1.5B.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit_per_file", type=int, default=None,
                        help="Debug: encode only the first N examples per file.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map[args.model_dtype]
    save_dtype = dtype_map[args.save_dtype]

    print(f"[encode] Loading tokenizer from {args.model_name_or_path} ...", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            local_files_only=args.local_files_only,
        )
    except OSError:
        sys.exit(
            "Model not found locally. This job runs offline on TamIA. "
            "Pre-cache Qwen/Qwen2.5-1.5B before submitting."
        )

    # Determine a pad token. Qwen uses EOS as pad by default; we use it
    # only for right-padding positions which are masked out in attention.
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        sys.exit("[encode] Tokenizer has no pad_token_id and no eos_token_id. Cannot pad batches.")

    print(f"[encode] Loading model from {args.model_name_or_path} ...", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            local_files_only=args.local_files_only,
            torch_dtype=model_dtype,
        )
    except OSError:
        sys.exit(
            "Model not found locally. This job runs offline on TamIA. "
            "Pre-cache Qwen/Qwen2.5-1.5B before submitting."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    hidden_dim = model.config.hidden_size
    print(f"[encode] Model loaded. hidden_dim={hidden_dim}, device={device}", flush=True)

    gpu_info = get_gpu_info()
    t_all_start = time.perf_counter()

    files_manifest: dict[str, dict] = {}
    overlength_by_file: dict[str, int] = {}
    truncated_by_file: dict[str, int] = {}
    skipped_by_file: dict[str, int] = {}

    for filename, stem in FILE_STEMS:
        jsonl_path = args.data_dir / filename
        if not jsonl_path.exists():
            sys.exit(f"Required file not found: {jsonl_path}")

        stats = encode_file(
            jsonl_path=jsonl_path,
            out_dir=args.out_dir,
            stem=stem,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            save_dtype=save_dtype,
            pad_token_id=pad_token_id,
            limit=args.limit_per_file,
        )
        files_manifest[stem] = {
            "jsonl_path": stats["jsonl_path"],
            "hidden_path": stats["hidden_path"],
            "label_path": stats["label_path"],
            "meta_path": stats["meta_path"],
            "n_examples": stats["n_examples"],
            "sha256_hidden": stats["sha256_hidden"],
        }
        overlength_by_file[stem] = stats["n_overlength"]
        truncated_by_file[stem] = stats["n_truncated"]
        skipped_by_file[stem] = stats["n_skipped"]

    t_all_end = time.perf_counter()
    total_examples = sum(v["n_examples"] for v in files_manifest.values())
    avg_ms = (
        (t_all_end - t_all_start) / total_examples * 1000
        if total_examples > 0 else 0.0
    )

    encoding_manifest = {
        "run_name": args.run_name,
        "model": args.model_name_or_path,
        "offline": True,
        "local_files_only": args.local_files_only,
        "layer": "last",
        "token_position": "last token of candidate_step",
        "length_policy": "no truncation",
        "max_seq_len": args.max_seq_len,
        "hidden_dim": hidden_dim,
        "model_dtype": args.model_dtype,
        "saved_dtype": args.save_dtype,
        "files": files_manifest,
        "num_overlength_at_encoding": overlength_by_file,
        "num_truncated": truncated_by_file,
        "num_skipped": skipped_by_file,
        "timing": {
            "total_encoding_time_sec": t_all_end - t_all_start,
            "avg_latency_ms_per_example": avg_ms,
        },
        "hardware": gpu_info,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }

    manifest_path = args.out_dir / "encoding_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(encoding_manifest, f, indent=2)

    print(f"[encode] Done. Encoding manifest: {manifest_path}", flush=True)
    print(
        f"[encode] Total time: {t_all_end - t_all_start:.1f}s, "
        f"avg {avg_ms:.1f} ms/example",
        flush=True,
    )


if __name__ == "__main__":
    main()
