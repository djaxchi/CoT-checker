"""Encode PRM800K steps at MULTIPLE layers x MULTIPLE token positions in one pass.

Pre-pays for the layer and token-position ablations: a single forward pass per
example captures, for each selected layer, the hidden state at the FIRST and LAST
token of the candidate step. All future ablation becomes a local array re-slice
(no cluster trips).

Mirrors encode_prm800k_hidden_states.py (same prompt, tokenization, no truncation,
sharding by global_index) and the multi-layer pattern of
encode_processbench_multilayer.py (output_hidden_states, loop selected layers).

Output in --out_dir (per shard):
  {stem}_h.npy       float16, shape (n, L, T=2, H)   T order = [first, last]
  {stem}_y.npy       int32   (n,)   label (1 = incorrect, 0 = correct)
  {stem}_meta.jsonl  uid, problem_id, solution_id, step_idx, completion_idx,
                     label, rating, n_tokens, first_idx, last_idx, global_index
  {stem}_manifest.json  layer_indices, layer_fracs, token_order, hidden_size, ...

Usage (one shard):
    python scripts/encode_prm800k_multitoken_multilayer.py \
      --data_dir <dir> --out_dir <dir>/shard_00 \
      --model_name_or_path Qwen/Qwen2.5-7B --local_files_only \
      --splits prm800k_heldout_test.jsonl:prm800k_heldout_test \
      --layers 11 17 20 22 25 28 --max_seq_len -1 \
      --shard_idx 0 --num_shards 4 --batch_size 16
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
sys.path.insert(0, str(ROOT / "scripts"))

from encode_prm800k_hidden_states import (  # noqa: E402
    build_prompt_prefix, read_jsonl, write_jsonl,
)

TOKEN_ORDER = ["first", "last"]


def tokenize_span(tokenizer, problem, prefix, candidate_step, max_seq_len):
    """Returns (full_ids, first_idx, last_idx) for the candidate-step token span."""
    prefix_ids = tokenizer(build_prompt_prefix(problem, prefix),
                           add_special_tokens=True, truncation=False)["input_ids"]
    cand_ids = tokenizer(candidate_step, add_special_tokens=False,
                         truncation=False)["input_ids"]
    if not cand_ids:
        raise ValueError("empty candidate step")
    full = prefix_ids + cand_ids
    if len(full) > max_seq_len:
        raise ValueError(f"len {len(full)} > max_seq_len {max_seq_len}")
    first_idx = len(prefix_ids)            # first token of the step
    last_idx = len(full) - 1               # last token of the step
    return full, first_idx, last_idx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--splits", nargs="+", required=True,
                   help="<input.jsonl>:<stem> (basename resolved under --data_dir)")
    p.add_argument("--layers", type=int, nargs="+", default=[11, 17, 20, 22, 25, 28],
                   help="hidden_states indices (0=embeddings .. num_layers=final)")
    p.add_argument("--max_seq_len", type=int, default=-1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--limit_per_file", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        sys.exit(f"invalid shard config: {args.shard_idx}/{args.num_shards}")

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            dtype=dtype_map[args.model_dtype])
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            torch_dtype=dtype_map[args.model_dtype])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    num_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    if args.max_seq_len <= 0:
        args.max_seq_len = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    for li in args.layers:
        if not (0 <= li <= num_layers):
            sys.exit(f"layer {li} out of range [0,{num_layers}]")
    np_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    save_dtype = dtype_map[args.save_dtype]
    L, T = len(args.layers), len(TOKEN_ORDER)

    for spec in args.splits:
        if ":" not in spec:
            sys.exit(f"--splits entry malformed (expected input:stem): {spec!r}")
        inp, stem = spec.rsplit(":", 1)
        jsonl_path = (args.data_dir / inp) if ("/" not in inp) else Path(inp)
        if not jsonl_path.exists():
            sys.exit(f"not found: {jsonl_path}")
        if (args.out_dir / f"{stem}_h.npy").exists() and not args.force:
            sys.exit(f"refusing overwrite {args.out_dir}/{stem}_h.npy; pass --force")

        examples = read_jsonl(jsonl_path)
        if args.limit_per_file is not None:
            examples = examples[: args.limit_per_file]
        for gi, ex in enumerate(examples):
            ex["global_index"] = gi
        n_total = len(examples)
        if args.num_shards > 1:
            examples = [e for e in examples
                        if e["global_index"] % args.num_shards == args.shard_idx]
        n = len(examples)
        print(f"[mtml] {stem}: shard {args.shard_idx}/{args.num_shards} -> {n}/{n_total} "
              f"| layers={args.layers} tokens={TOKEN_ORDER}", flush=True)

        H = np.zeros((n, L, T, hidden), dtype=np_dtype)
        y = np.zeros(n, dtype=np.int32)
        meta: list[dict] = []
        t0 = time.perf_counter(); i = 0
        while i < n:
            batch = examples[i:i + args.batch_size]
            ids_list, firsts, lasts = [], [], []
            for ex in batch:
                try:
                    full, fi, la = tokenize_span(tok, ex["problem"], ex["prefix"],
                                                 ex["candidate_step"], args.max_seq_len)
                except ValueError as e:
                    sys.exit(f"[mtml] FATAL overlength uid={ex.get('uid','?')}: {e}")
                ids_list.append(full); firsts.append(fi); lasts.append(la)
            mx = max(len(x) for x in ids_list)
            inp_t = torch.tensor([x + [pad] * (mx - len(x)) for x in ids_list],
                                 dtype=torch.long, device=device)
            att = torch.tensor([[1] * len(x) + [0] * (mx - len(x)) for x in ids_list],
                               dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(inp_t, attention_mask=att, output_hidden_states=True,
                            use_cache=False)
            hs = out.hidden_states
            for b, ex in enumerate(batch):
                for lj, li in enumerate(args.layers):
                    H[i + b, lj, 0] = hs[li][b, firsts[b], :].to(save_dtype).cpu().numpy()
                    H[i + b, lj, 1] = hs[li][b, lasts[b], :].to(save_dtype).cpu().numpy()
                y[i + b] = int(ex["label"])
                meta.append({
                    "uid": ex["uid"], "problem_id": ex["problem_id"],
                    "solution_id": ex["solution_id"], "step_idx": ex["step_idx"],
                    "completion_idx": ex["completion_idx"], "label": int(ex["label"]),
                    "rating": ex.get("rating"), "n_tokens": len(ids_list[b]),
                    "first_idx": firsts[b], "last_idx": lasts[b],
                    "global_index": ex["global_index"],
                })
            del out, hs
            i += len(batch)
            if (i // max(args.batch_size, 1)) % 16 == 0 or i == n:
                print(f"[mtml] {stem}: {i}/{n} ({time.perf_counter()-t0:.1f}s)", flush=True)

        np.save(args.out_dir / f"{stem}_h.npy", H)
        np.save(args.out_dir / f"{stem}_y.npy", y)
        write_jsonl(args.out_dir / f"{stem}_meta.jsonl", meta)
        (args.out_dir / f"{stem}_manifest.json").write_text(json.dumps({
            "model_name": args.model_name_or_path, "stem": stem,
            "num_hidden_layers": num_layers, "hidden_size": hidden,
            "layer_indices": args.layers,
            "layer_fracs": [round(li / num_layers, 4) for li in args.layers],
            "token_order": TOKEN_ORDER, "shape": list(H.shape),
            "n_steps": n, "n_total_in_file": n_total,
            "shard_idx": args.shard_idx, "num_shards": args.num_shards,
            "max_seq_len": args.max_seq_len, "saved_dtype": args.save_dtype,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        print(f"[mtml] {stem} done -> {args.out_dir} shape={H.shape}", flush=True)


if __name__ == "__main__":
    main()
