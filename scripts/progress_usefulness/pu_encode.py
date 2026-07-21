"""progress_usefulness_v0 P2 encode: last/mean/max pooling over the step span.

One forward pass per candidate captures, for each selected layer, three poolings
of the candidate-step token span:
  last  -> hidden state at the final step token
  mean  -> mean over all step tokens [first_idx .. last_idx]
  max   -> elementwise max over all step tokens

Mirrors encode_prm800k_multitoken_multilayer.py (same build_prompt_prefix format,
tokenization, no truncation, sharding by global_index) so the representation is
directly comparable to the correctness probe work; the only change is pooling
last/mean/max over the whole step span instead of storing first+last positions.

Consumes the progress/neutral items from pu_build_pairs (anchors skipped). Labels
are progress_label (1 = progress/+1, 0 = neutral/0).

Output in --out_dir (per shard):
  {stem}_h.npy       float16 (n, L, P=3, H)   P order = [last, mean, max]
  {stem}_y.npy       int32   (n,)   progress_label
  {stem}_meta.jsonl  item_uid, fork_id, role, progress_label, rating, problem_id,
                     step_idx, n_tokens, first_idx, last_idx, global_index
  {stem}_manifest.json

Usage (one shard):
  python scripts/progress_usefulness/pu_encode.py \
    --data_dir data/pu --out_dir runs/pu/enc/shard_00 \
    --model_name_or_path Qwen/Qwen2.5-7B --local_files_only \
    --splits pu_train_items.jsonl:pu_train pu_val_items.jsonl:pu_val \
    --layers 12 20 26 28 --shard_idx 0 --num_shards 4 --batch_size 16
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.encode_prm800k_hidden_states import read_jsonl, write_jsonl  # noqa: E402
from scripts.encode_prm800k_multitoken_multilayer import tokenize_span  # noqa: E402

POOL_ORDER = ["last", "mean", "max"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--splits", nargs="+", required=True,
                   help="<input.jsonl>:<stem> (basename resolved under --data_dir)")
    p.add_argument("--layers", type=int, nargs="+", default=[12, 20, 26, 28])
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
    L, P = len(args.layers), len(POOL_ORDER)

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
        # progress/neutral only; anchors carry no candidate step.
        examples = [e for e in examples if e.get("role") != "anchor"
                    and e.get("candidate_step")]
        if args.limit_per_file is not None:
            examples = examples[: args.limit_per_file]
        for gi, ex in enumerate(examples):
            ex["global_index"] = gi
        n_total = len(examples)
        if args.num_shards > 1:
            examples = [e for e in examples
                        if e["global_index"] % args.num_shards == args.shard_idx]
        n = len(examples)
        print(f"[pu-enc] {stem}: shard {args.shard_idx}/{args.num_shards} -> {n}/{n_total} "
              f"| layers={args.layers} pools={POOL_ORDER}", flush=True)

        H = np.zeros((n, L, P, hidden), dtype=np_dtype)
        y = np.zeros(n, dtype=np.int32)
        meta: list[dict] = []
        t0 = time.perf_counter()
        i = 0
        while i < n:
            batch = examples[i:i + args.batch_size]
            ids_list, firsts, lasts = [], [], []
            for ex in batch:
                try:
                    full, fi, la = tokenize_span(tok, ex["problem"], ex["prefix"],
                                                 ex["candidate_step"], args.max_seq_len)
                except ValueError as e:
                    sys.exit(f"[pu-enc] FATAL overlength uid={ex.get('item_uid','?')}: {e}")
                ids_list.append(full)
                firsts.append(fi)
                lasts.append(la)
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
                fi, la = firsts[b], lasts[b]
                for lj, li in enumerate(args.layers):
                    span = hs[li][b, fi:la + 1, :]                 # (T_step, H)
                    H[i + b, lj, 0] = hs[li][b, la, :].to(save_dtype).cpu().numpy()
                    H[i + b, lj, 1] = span.mean(0).to(save_dtype).cpu().numpy()
                    H[i + b, lj, 2] = span.max(0).values.to(save_dtype).cpu().numpy()
                y[i + b] = int(ex["progress_label"])
                meta.append({
                    "item_uid": ex["item_uid"], "fork_id": ex["fork_id"],
                    "role": ex["role"], "progress_label": int(ex["progress_label"]),
                    "rating": ex.get("rating"), "problem_id": ex["problem_id"],
                    "step_idx": ex["step_idx"], "n_tokens": len(ids_list[b]),
                    "first_idx": fi, "last_idx": la, "global_index": ex["global_index"],
                })
            del out, hs
            i += len(batch)
            if (i // max(args.batch_size, 1)) % 16 == 0 or i == n:
                print(f"[pu-enc] {stem}: {i}/{n} ({time.perf_counter()-t0:.1f}s)", flush=True)

        np.save(args.out_dir / f"{stem}_h.npy", H)
        np.save(args.out_dir / f"{stem}_y.npy", y)
        write_jsonl(args.out_dir / f"{stem}_meta.jsonl", meta)
        (args.out_dir / f"{stem}_manifest.json").write_text(json.dumps({
            "model_name": args.model_name_or_path, "stem": stem,
            "num_hidden_layers": num_layers, "hidden_size": hidden,
            "layer_indices": args.layers,
            "layer_fracs": [round(li / num_layers, 4) for li in args.layers],
            "pool_order": POOL_ORDER, "shape": list(H.shape),
            "n_items": n, "n_total_in_file": n_total,
            "shard_idx": args.shard_idx, "num_shards": args.num_shards,
            "max_seq_len": args.max_seq_len, "saved_dtype": args.save_dtype,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        print(f"[pu-enc] {stem} done -> {args.out_dir} shape={H.shape}", flush=True)


if __name__ == "__main__":
    main()
