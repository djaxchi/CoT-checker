"""Encode one ProcessBench subset at MULTIPLE layer depths (one forward pass).

Mirrors encode_processbench_hidden_states.py (same flatten + last-token-of-step
extraction, no truncation) but saves the step hidden state at several layer
fractions so per-layer F1_PB can be computed alongside the PRM800K probe sweep.

Output in --out_dir:
  pb_step_L{idx}_h.npy   (n_steps, hidden) float16   one per selected layer
  pb_step_meta.jsonl     id, step_idx, label, n_steps   (matches the F1_PB evaluator)
  pb_multilayer_manifest.json
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

from encode_processbench_hidden_states import (  # noqa: E402
    build_prompt_prefix, load_trace_file, write_jsonl,
)
from encode_prm800k_multilayer import resolve_layer_indices  # noqa: E402


def tokenize_step(tokenizer, problem, prefix, step, max_seq_len):
    pre = tokenizer(build_prompt_prefix(problem, prefix), add_special_tokens=True, truncation=False)["input_ids"]
    st = tokenizer(step, add_special_tokens=False, truncation=False)["input_ids"]
    if not st:
        raise ValueError("empty step")
    full = pre + st
    if len(full) > max_seq_len:
        raise ValueError(f"len {len(full)} > max_seq_len {max_seq_len}")
    return full, len(full) - 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_file", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--subset_name", type=str, required=True)
    p.add_argument("--layer_fracs", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--max_seq_len", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only, dtype=dtype_map[args.model_dtype])
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only, torch_dtype=dtype_map[args.model_dtype])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    num_layers = int(model.config.num_hidden_layers)
    if args.max_seq_len <= 0:
        args.max_seq_len = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    layer_map = resolve_layer_indices(args.layer_fracs, num_layers)
    layer_idxs = [li for _, li in layer_map]

    traces = load_trace_file(args.raw_file)
    flat = []
    for tr in traces:
        steps = tr["steps"]
        for k, step in enumerate(steps):
            flat.append({"id": tr["id"], "step_idx": k, "label": int(tr["label"]),
                         "n_steps": len(steps), "problem": tr["problem"],
                         "prefix": "\n\n".join(steps[:k]), "step": step})
    n = len(flat)
    hidden = model.config.hidden_size
    np_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    per_layer = {li: np.zeros((n, hidden), dtype=np_dtype) for li in layer_idxs}
    meta = []
    print(f"[pbml] {args.subset_name}: {len(traces)} traces, {n} steps, layers={layer_map}", flush=True)

    save_dtype = dtype_map[args.save_dtype]
    t0 = time.perf_counter(); i = 0
    while i < n:
        batch = flat[i:i + args.batch_size]
        ids_list, cand_list = [], []
        for ex in batch:
            try:
                ids, cand = tokenize_step(tok, ex["problem"], ex["prefix"], ex["step"], args.max_seq_len)
            except ValueError as e:
                sys.exit(f"[pbml] FATAL overlength id={ex['id']} step={ex['step_idx']}: {e}")
            ids_list.append(ids); cand_list.append(cand)
        mx = max(len(x) for x in ids_list)
        inp = torch.tensor([x + [pad] * (mx - len(x)) for x in ids_list], dtype=torch.long, device=device)
        att = torch.tensor([[1] * len(x) + [0] * (mx - len(x)) for x in ids_list], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inp, attention_mask=att, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        for b, (ex, cand) in enumerate(zip(batch, cand_list)):
            for li in layer_idxs:
                per_layer[li][i + b] = hs[li][b, cand, :].detach().to(save_dtype).cpu().numpy()
            meta.append({"id": ex["id"], "step_idx": ex["step_idx"], "label": ex["label"], "n_steps": ex["n_steps"]})
        del out, hs
        i += len(batch)
        if (i // args.batch_size) % 16 == 0 or i == n:
            print(f"[pbml] {args.subset_name}: {i}/{n} ({time.perf_counter()-t0:.1f}s)", flush=True)

    if (args.out_dir / "pb_step_meta.jsonl").exists() and not args.force:
        sys.exit(f"[pbml] refusing overwrite in {args.out_dir}; pass --force")
    write_jsonl(args.out_dir / "pb_step_meta.jsonl", meta)
    for li in layer_idxs:
        np.save(args.out_dir / f"pb_step_L{li}_h.npy", per_layer[li])
    (args.out_dir / "pb_multilayer_manifest.json").write_text(json.dumps({
        "model_name": args.model_name_or_path, "subset": args.subset_name,
        "num_hidden_layers": num_layers, "hidden_size": hidden,
        "layer_indices": {f"{f:.2f}": li for f, li in layer_map},
        "n_steps": n, "n_traces": len(traces), "num_truncated_examples": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"[pbml] {args.subset_name} done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
