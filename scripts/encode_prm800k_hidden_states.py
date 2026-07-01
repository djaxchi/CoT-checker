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
    shard_idx: int = 0,
    num_shards: int = 1,
    layer: int = -1,
) -> dict:
    """Encode one JSONL file. Returns per-file stats.

    ``layer`` selects which ``outputs.hidden_states`` index to read (0 = embeddings,
    num_hidden_layers = final). Default -1 = the final layer (L28 for Qwen2.5-7B), the
    original deployed readout; pass e.g. 20 to encode L20/last instead.

    Sharding: a deterministic ``global_index`` is assigned to every example in
    file order BEFORE sharding, so every worker agrees on the ordering. When
    ``num_shards > 1`` only examples with ``global_index % num_shards ==
    shard_idx`` are encoded; each meta row carries its ``global_index`` so
    merge_prm800k_encoded_shards.py can reconstruct the original order. With the
    default (shard_idx=0, num_shards=1) this is a no-op and behaviour is
    identical to the original S1 encoder.
    """
    all_examples = read_jsonl(jsonl_path)
    if limit is not None:
        all_examples = all_examples[:limit]

    # Assign deterministic global_index over the FULL (pre-shard) file order.
    for gi, ex in enumerate(all_examples):
        ex["global_index"] = gi
    n_total_file = len(all_examples)
    if num_shards > 1:
        examples = [ex for ex in all_examples if ex["global_index"] % num_shards == shard_idx]
        print(
            f"[encode] {stem}: shard {shard_idx}/{num_shards} -> "
            f"{len(examples)}/{n_total_file} examples",
            flush=True,
        )
    else:
        examples = all_examples

    n = len(examples)
    print(f"[encode] {stem}: {n} examples", flush=True)
    token_lengths: list[int] = []

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

        # Extract hidden state at the last token of candidate_step for each example,
        # from the requested layer (default -1 = final layer).
        last_layer = outputs.hidden_states[layer]  # (batch, seq_len, hidden_dim)
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
            token_lengths.append(n_tokens)
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
                "global_index": ex["global_index"],
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

    lt = np.asarray(token_lengths, dtype=np.int64) if token_lengths else np.zeros(0, dtype=np.int64)
    length_stats = {
        "num_examples": int(n),
        "max_observed_tokens": int(lt.max()) if lt.size else 0,
        "mean_observed_tokens": float(lt.mean()) if lt.size else 0.0,
        "p95_observed_tokens": int(np.percentile(lt, 95)) if lt.size else 0,
        "p99_observed_tokens": int(np.percentile(lt, 99)) if lt.size else 0,
    }

    return {
        "jsonl_path": str(jsonl_path),
        "hidden_path": str(h_path),
        "label_path": str(y_path),
        "meta_path": str(m_path),
        "n_examples": n,
        "n_examples_in_file": n_total_file,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "sha256_hidden": h_sha,
        "n_overlength": n_overlength,
        "n_truncated": 0,
        "n_skipped": 0,
        "length_stats": length_stats,
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


def parse_splits_arg(splits: list[str], data_dir: Path) -> list[tuple[str, str]]:
    """Parse --splits entries of the form 'input.jsonl:stem'.

    The input portion may be a basename (resolved under data_dir) or an
    absolute/relative path containing '/' (used as-is). The stem is the
    output prefix for {stem}_h.npy, {stem}_y.npy, {stem}_meta.jsonl.
    """
    parsed: list[tuple[str, str]] = []
    for entry in splits:
        if ":" not in entry:
            sys.exit(f"--splits entry malformed (expected input:stem): {entry!r}")
        inp, stem = entry.rsplit(":", 1)
        inp = inp.strip()
        stem = stem.strip()
        if not inp or not stem:
            sys.exit(f"--splits entry malformed (empty input or stem): {entry!r}")
        if "/" not in inp and not inp.startswith("."):
            # Treat as basename under data_dir
            inp_path = str((data_dir / inp).resolve())
        else:
            inp_path = inp
        parsed.append((inp_path, stem))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode PRM800K hidden states with Qwen2.5-1.5B.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Hard cap on tokenized sequence length. Encoding "
                             "FAILS (never truncates) on any example exceeding "
                             "this. Pass -1 to use the model's full context "
                             "window (model.config.max_position_embeddings).")
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="Worker shard index in [0, num_shards). Examples "
                             "with global_index %% num_shards == shard_idx are "
                             "encoded. Merge with merge_prm800k_encoded_shards.py.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards (e.g. 4 for 4-GPU encoding).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--layer", type=int, default=-1,
                        help="hidden_states index to read (0=embeddings .. "
                             "num_hidden_layers=final). Default -1 = final layer (the "
                             "original L28 readout). Pass 20 to encode L20/last.")
    parser.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit_per_file", type=int, default=None,
                        help="Debug: encode only the first N examples per file.")
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Optional list of <input.jsonl>:<stem> pairs. If omitted, the "
             "original 40k FILE_STEMS layout is used. The input may be a "
             "basename (resolved under --data_dir) or a path with '/'.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing {stem}_h.npy outputs. Default refuses.",
    )
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
        # transformers >=5 renamed `torch_dtype` to `dtype`; try the new kwarg
        # first and fall back to the old one for older transformers.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                local_files_only=args.local_files_only,
                dtype=model_dtype,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                local_files_only=args.local_files_only,
                torch_dtype=model_dtype,
            )
    except OSError:
        sys.exit(
            "Model not found locally. This job runs offline on TamIA. "
            "Pre-cache the model before submitting."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    hidden_dim = model.config.hidden_size
    model_max = int(getattr(model.config, "max_position_embeddings", 0) or 0)

    # Resolve the no-truncation length cap. -1 => the model's full context
    # window. Any positive value is honoured but must not exceed the model
    # context (otherwise we could request positions the model cannot encode).
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        if model_max <= 0:
            sys.exit("[encode] model has no max_position_embeddings; pass an explicit --max_seq_len.")
        resolved_max_seq_len = model_max
    else:
        resolved_max_seq_len = args.max_seq_len
    if model_max and resolved_max_seq_len > model_max:
        sys.exit(
            f"[encode] --max_seq_len={resolved_max_seq_len} exceeds model "
            f"context window max_position_embeddings={model_max}. Refusing: "
            "the assertion encoded_length <= max_position_embeddings would be unsafe."
        )
    args.max_seq_len = resolved_max_seq_len
    print(
        f"[encode] Model loaded. hidden_dim={hidden_dim}, "
        f"max_position_embeddings={model_max}, max_seq_len={resolved_max_seq_len}, "
        f"device={device}",
        flush=True,
    )

    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        sys.exit(
            f"[encode] invalid shard config: shard_idx={args.shard_idx} "
            f"num_shards={args.num_shards}"
        )

    n_layers = int(model.config.num_hidden_layers)
    if not (args.layer == -1 or 0 <= args.layer <= n_layers):
        sys.exit(f"[encode] --layer {args.layer} out of range [-1, {n_layers}]")
    print(f"[encode] reading hidden_states[{args.layer}] "
          f"({'final layer' if args.layer == -1 else f'L{args.layer}'})", flush=True)

    gpu_info = get_gpu_info()
    t_all_start = time.perf_counter()

    files_manifest: dict[str, dict] = {}
    overlength_by_file: dict[str, int] = {}
    truncated_by_file: dict[str, int] = {}
    skipped_by_file: dict[str, int] = {}

    if args.splits is not None:
        split_specs = [(Path(p), stem) for p, stem in parse_splits_arg(args.splits, args.data_dir)]
    else:
        split_specs = [(args.data_dir / fn, stem) for fn, stem in FILE_STEMS]

    for jsonl_path, stem in split_specs:
        if not jsonl_path.exists():
            sys.exit(f"Required file not found: {jsonl_path}")

        existing_h = args.out_dir / f"{stem}_h.npy"
        if existing_h.exists() and not args.force:
            sys.exit(
                f"[encode] Refusing to overwrite existing {existing_h}. "
                "Pass --force to overwrite."
            )

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
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
            layer=args.layer,
        )
        files_manifest[stem] = {
            "jsonl_path": stats["jsonl_path"],
            "hidden_path": stats["hidden_path"],
            "label_path": stats["label_path"],
            "meta_path": stats["meta_path"],
            "n_examples": stats["n_examples"],
            "n_examples_in_file": stats["n_examples_in_file"],
            "sha256_hidden": stats["sha256_hidden"],
            "length_stats": stats["length_stats"],
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

    # Aggregate no-truncation audit across all encoded files. Because encoding
    # FAILS hard on any overlength example (see tokenize_example), reaching this
    # point proves zero truncation occurred.
    agg_max = max((files_manifest[s]["length_stats"]["max_observed_tokens"]
                   for s in files_manifest), default=0)
    agg_p95 = max((files_manifest[s]["length_stats"]["p95_observed_tokens"]
                   for s in files_manifest), default=0)
    agg_p99 = max((files_manifest[s]["length_stats"]["p99_observed_tokens"]
                   for s in files_manifest), default=0)

    encoding_manifest = {
        "run_name": args.run_name,
        "model_name": args.model_name_or_path,
        "tokenizer_name": args.model_name_or_path,
        "offline": True,
        "local_files_only": args.local_files_only,
        "layer": "last" if args.layer == -1 else int(args.layer),
        "token_position": "last token of candidate_step",
        "length_policy": "no truncation (fail-hard on overlength)",
        "max_seq_len": args.max_seq_len,
        "model_max_position_embeddings": model_max,
        "max_observed_tokens": agg_max,
        "p95_observed_tokens": agg_p95,
        "p99_observed_tokens": agg_p99,
        "num_examples": total_examples,
        "num_steps": total_examples,
        "num_truncated_examples": 0,
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
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
