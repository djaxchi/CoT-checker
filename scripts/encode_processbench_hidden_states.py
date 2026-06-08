"""
Encode ProcessBench-GSM8K steps with Qwen2.5-1.5B hidden states.

For each trace, each step is encoded as:

    Problem:
    {problem}

    Previous reasoning:
    {steps[0] \\n\\n steps[1] \\n\\n ... \\n\\n steps[k-1]}    <- empty for k=0

    Current step:
    {steps[k]}

The hidden state is extracted at the last token of steps[k], using the
same split-tokenization approach as encode_prm800k_hidden_states.py.

Output:
    pb_gsm8k_step_h.npy          shape (N_steps, hidden_dim), float16
    pb_gsm8k_step_meta.jsonl     N_steps rows, one per step
    encoding_manifest.json

Meta row format (matches what train_easy_probe_method.py expects):
    {"id": "gsm8k-5", "step_idx": 2, "label": 2, "n_steps": 4,
     "candidate_last_token_idx": ..., "n_tokens": ..., "was_truncated": false}
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
# Utilities
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


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"num_gpus": 0, "gpu_name": None}
    return {"num_gpus": torch.cuda.device_count(), "gpu_name": torch.cuda.get_device_name(0)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Prompt format -- must match encode_prm800k_hidden_states.py exactly
# ---------------------------------------------------------------------------

def build_prompt_prefix(problem: str, prefix: str) -> str:
    prefix_section = f"Previous reasoning:\n{prefix}\n\n" if prefix else "Previous reasoning:\n\n"
    return f"Problem:\n{problem}\n\n{prefix_section}Current step:\n"


# ---------------------------------------------------------------------------
# Tokenization (identical logic to PRM800K encoder)
# ---------------------------------------------------------------------------

def tokenize_step(
    tokenizer,
    problem: str,
    prefix: str,
    current_step: str,
    max_seq_len: int,
) -> tuple[list[int], int]:
    """
    Returns (full_input_ids, candidate_last_token_idx).
    Raises ValueError if sequence exceeds max_seq_len.
    """
    prompt_prefix = build_prompt_prefix(problem, prefix)

    prefix_ids: list[int] = tokenizer(
        prompt_prefix, add_special_tokens=True, truncation=False,
    )["input_ids"]

    step_ids: list[int] = tokenizer(
        current_step, add_special_tokens=False, truncation=False,
    )["input_ids"]

    if not step_ids:
        raise ValueError("Step produced an empty token sequence.")

    full_ids = prefix_ids + step_ids
    if len(full_ids) > max_seq_len:
        raise ValueError(
            f"Sequence length {len(full_ids)} exceeds max_seq_len={max_seq_len}."
        )
    return full_ids, len(full_ids) - 1


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def load_trace_file(path: Path) -> list[dict]:
    """Load ProcessBench traces from either a JSON list or a JSONL file."""
    text = path.read_text(encoding="utf-8")
    s = text.lstrip()
    if s.startswith("["):
        return json.loads(text)
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def encode_all_steps(
    traces: list[dict],
    tokenizer,
    model,
    device: torch.device,
    max_seq_len: int,
    batch_size: int,
    save_dtype: torch.dtype,
    pad_token_id: int,
    shard_idx: int = 0,
    num_shards: int = 1,
    fail_on_overlength: bool = False,
    subset_name: str | None = None,
) -> tuple[np.ndarray, list[dict], int, list[int]]:
    """
    Flatten all steps from all traces, encode in batches.

    Returns (hidden_states, labels_per_step, meta_rows, n_skipped).
    hidden_states shape: (N_steps, hidden_dim).

    When num_shards > 1, only steps whose deterministic global_step_index
    satisfies (global_step_index % num_shards == shard_idx) are encoded,
    and each meta row carries its global_step_index so the shard merger
    can reconstruct the original order.
    """
    # Flatten: one entry per (trace, step_idx), then assign a deterministic
    # global_step_index BEFORE sharding so every worker agrees on the ordering.
    flat: list[dict] = []
    for trace in traces:
        problem = trace["problem"]
        steps = trace["steps"]
        trace_label = trace["label"]
        n_steps = len(steps)
        prefix_parts: list[str] = []
        for k, step_text in enumerate(steps):
            prefix = "\n\n".join(prefix_parts)
            flat.append({
                "id": trace["id"],
                "step_idx": k,
                "label": trace_label,
                "n_steps": n_steps,
                "problem": problem,
                "prefix": prefix,
                "current_step": step_text,
            })
            prefix_parts.append(step_text)
    for gi, rec in enumerate(flat):
        rec["global_step_index"] = gi
    if num_shards > 1:
        flat = [r for r in flat if r["global_step_index"] % num_shards == shard_idx]
        print(
            f"[encode] shard {shard_idx}/{num_shards}: {len(flat)} step(s) selected",
            flush=True,
        )

    hidden_dim = model.config.hidden_size
    np_dtype = np.float16 if save_dtype == torch.float16 else np.float32
    n = len(flat)
    all_hidden = np.zeros((n, hidden_dim), dtype=np_dtype)
    all_meta: list[dict] = []
    n_skipped = 0
    token_lengths: list[int] = []

    t_start = time.perf_counter()
    i = 0
    while i < n:
        batch = flat[i : i + batch_size]
        batch_ids: list[list[int]] = []
        batch_cand_idx: list[int] = []
        batch_valid: list[bool] = []

        for ex in batch:
            try:
                ids, cand_idx = tokenize_step(
                    tokenizer, ex["problem"], ex["prefix"],
                    ex["current_step"], max_seq_len,
                )
                batch_ids.append(ids)
                batch_cand_idx.append(cand_idx)
                batch_valid.append(True)
                # carry subset tag through if present
                ex.setdefault("pb_subset", ex.get("pb_subset"))
            except ValueError as e:
                if fail_on_overlength:
                    sys.exit(
                        "[encode_pb] FATAL: overlength step under the "
                        "no-truncation contract. "
                        f"subset={subset_name or ex.get('pb_subset')} "
                        f"id={ex['id']} step_idx={ex['step_idx']} "
                        f"global_step_index={ex.get('global_step_index')}: {e}\n"
                        "Raise --max_seq_len (or -1 for the model context "
                        "window); never truncate."
                    )
                print(
                    f"[encode] SKIP overlength step: id={ex['id']} step={ex['step_idx']}: {e}",
                    file=sys.stderr, flush=True,
                )
                batch_ids.append([pad_token_id])  # placeholder
                batch_cand_idx.append(0)
                batch_valid.append(False)
                n_skipped += 1

        max_len = max(len(ids) for ids in batch_ids)
        padded, masks = [], []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [pad_token_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        attn_tensor = torch.tensor(masks, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(
                input_tensor,
                attention_mask=attn_tensor,
                output_hidden_states=True,
                use_cache=False,
            )
        last_layer = outputs.hidden_states[-1]
        del outputs

        for b_idx, (ex, cand_idx, valid) in enumerate(
            zip(batch, batch_cand_idx, batch_valid)
        ):
            n_tokens = len(batch_ids[b_idx])
            if valid:
                vec = last_layer[b_idx, cand_idx, :].detach().to(save_dtype).cpu().numpy()
                all_hidden[i + b_idx] = vec
                token_lengths.append(n_tokens)
            meta_row = {
                "id": ex["id"],
                "step_idx": ex["step_idx"],
                "label": ex["label"],
                "n_steps": ex["n_steps"],
                "candidate_last_token_idx": cand_idx if valid else -1,
                "n_tokens": n_tokens if valid else -1,
                "was_truncated": False,
                "skipped": not valid,
                "global_step_index": ex.get("global_step_index"),
            }
            if ex.get("pb_subset"):
                meta_row["pb_subset"] = ex["pb_subset"]
            all_meta.append(meta_row)

        del last_layer
        i += len(batch)

        elapsed = time.perf_counter() - t_start
        if i % (batch_size * 16) == 0 or i >= n:
            print(f"[encode] {i}/{n} steps done ({elapsed:.1f}s)", flush=True)

    return all_hidden, all_meta, n_skipped, token_lengths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Encode ProcessBench-GSM8K steps with Qwen2.5-1.5B hidden states."
    )
    p.add_argument("--raw_file", type=Path, required=True,
                   help="Path to gsm8k.json from the ProcessBench dataset.")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Directory for output files (pb_gsm8k_step_h.npy, etc.).")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--max_seq_len", type=int, default=2048,
                   help="Hard cap on tokenized sequence length. Pass -1 to use "
                        "the model's full context window "
                        "(model.config.max_position_embeddings).")
    p.add_argument("--fail_on_overlength", action="store_true",
                   help="No-truncation contract: abort (instead of skipping) if "
                        "any step exceeds --max_seq_len, logging "
                        "subset/id/step_idx/length.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--subset_name", type=str, default=None,
                   help="If set, write a 'pb_subset' field on every meta row.")
    p.add_argument("--output_layout", choices=["legacy", "generic"], default="legacy",
                   help="legacy=pb_gsm8k_step_*; generic=pb_step_* (new layout).")
    p.add_argument("--shard_idx", type=int, default=0,
                   help="Worker shard index in [0, num_shards). Each shard "
                        "encodes steps whose deterministic global_step_index "
                        "satisfies (gi %% num_shards == shard_idx).")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of shards (e.g. 4 for 4-GPU encoding).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output files.")
    args = p.parse_args()

    if not args.raw_file.exists():
        sys.exit(
            f"ProcessBench raw file not found: {args.raw_file}\n"
            "This job does not download datasets. Provide --raw_file."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map[args.model_dtype]
    save_dtype = dtype_map[args.save_dtype]

    traces: list[dict] = load_trace_file(args.raw_file)
    print(f"[encode_pb] Loaded {len(traces)} traces from {args.raw_file}", flush=True)
    if args.subset_name:
        for t in traces:
            t.setdefault("pb_subset", args.subset_name)

    # Validate expected fields
    required = {"id", "problem", "steps", "label"}
    missing_fields = required - set(traces[0].keys())
    if missing_fields:
        sys.exit(f"ProcessBench record missing fields: {missing_fields}")

    print(f"[encode_pb] Loading tokenizer from {args.model_name_or_path} ...", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
        )
    except OSError:
        sys.exit(
            "Model not found locally. This job runs offline on TamIA. "
            "Pre-cache Qwen/Qwen2.5-1.5B before submitting."
        )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        sys.exit("[encode_pb] Tokenizer has no pad_token_id and no eos_token_id.")

    print(f"[encode_pb] Loading model from {args.model_name_or_path} ...", flush=True)
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
    model_max = int(getattr(model.config, "max_position_embeddings", 0) or 0)

    if args.max_seq_len is not None and args.max_seq_len <= 0:
        if model_max <= 0:
            sys.exit("[encode_pb] model has no max_position_embeddings; pass an explicit --max_seq_len.")
        args.max_seq_len = model_max
    if model_max and args.max_seq_len > model_max:
        sys.exit(
            f"[encode_pb] --max_seq_len={args.max_seq_len} exceeds model context "
            f"window max_position_embeddings={model_max}. Refusing."
        )
    print(
        f"[encode_pb] Model loaded. hidden_dim={hidden_dim}, "
        f"max_position_embeddings={model_max}, max_seq_len={args.max_seq_len}, "
        f"device={device}",
        flush=True,
    )

    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        sys.exit(
            f"[encode_pb] invalid shard config: shard_idx={args.shard_idx} "
            f"num_shards={args.num_shards}"
        )
    print(
        f"[encode_pb] shard config: shard_idx={args.shard_idx} "
        f"num_shards={args.num_shards}",
        flush=True,
    )
    t_start = time.perf_counter()
    all_hidden, all_meta, n_skipped, token_lengths = encode_all_steps(
        traces=traces,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        save_dtype=save_dtype,
        pad_token_id=pad_token_id,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        fail_on_overlength=args.fail_on_overlength,
        subset_name=args.subset_name,
    )
    t_end = time.perf_counter()

    n_total = len(all_meta)
    if args.output_layout == "legacy":
        h_path = args.out_dir / "pb_gsm8k_step_h.npy"
        meta_path = args.out_dir / "pb_gsm8k_step_meta.jsonl"
    else:
        h_path = args.out_dir / "pb_step_h.npy"
        meta_path = args.out_dir / "pb_step_meta.jsonl"
    if h_path.exists() and not args.force:
        sys.exit(f"[encode_pb] Refusing to overwrite {h_path}. Pass --force.")

    np.save(h_path, all_hidden)
    write_jsonl(meta_path, all_meta)

    h_sha = sha256_file(h_path)
    total_sec = t_end - t_start
    avg_ms = total_sec / n_total * 1000 if n_total > 0 else 0.0

    lt = np.asarray(token_lengths, dtype=np.int64) if token_lengths else np.zeros(0, dtype=np.int64)
    manifest = {
        "run_name": args.run_name,
        "model_name": args.model_name_or_path,
        "tokenizer_name": args.model_name_or_path,
        "offline": True,
        "local_files_only": args.local_files_only,
        "source_file": str(args.raw_file),
        "subset_name": args.subset_name,
        "n_traces": len(traces),
        "n_steps_total": n_total,
        "n_skipped": n_skipped,
        "layer": "last",
        "token_position": "last token of current_step",
        "length_policy": (
            "no truncation (fail-hard on overlength)"
            if args.fail_on_overlength else "skip overlength (not fail)"
        ),
        "fail_on_overlength": args.fail_on_overlength,
        "max_seq_len": args.max_seq_len,
        "model_max_position_embeddings": model_max,
        "num_examples": len(traces),
        "num_steps": n_total,
        "max_observed_tokens": int(lt.max()) if lt.size else 0,
        "mean_observed_tokens": float(lt.mean()) if lt.size else 0.0,
        "p95_observed_tokens": int(np.percentile(lt, 95)) if lt.size else 0,
        "p99_observed_tokens": int(np.percentile(lt, 99)) if lt.size else 0,
        "num_truncated_examples": 0,
        "hidden_dim": hidden_dim,
        "model_dtype": args.model_dtype,
        "saved_dtype": args.save_dtype,
        "files": {
            "hidden": str(h_path),
            "meta": str(meta_path),
            "sha256_hidden": h_sha,
        },
        "timing": {
            "total_encoding_time_sec": total_sec,
            "avg_latency_ms_per_step": avg_ms,
        },
        "hardware": get_gpu_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }
    with open(args.out_dir / "encoding_manifest_pb.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[encode_pb] Done. {n_total} steps encoded ({n_skipped} skipped).", flush=True)
    print(f"[encode_pb] Total: {total_sec:.1f}s, avg {avg_ms:.1f} ms/step", flush=True)
    print(f"[encode_pb] Output: {h_path}", flush=True)


if __name__ == "__main__":
    main()
