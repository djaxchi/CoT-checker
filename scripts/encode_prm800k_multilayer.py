"""Encode PRM800K examples at MULTIPLE layer depths in one forward pass.

Same tokenization and last-token extraction as encode_prm800k_hidden_states.py
(reused directly), but saves the candidate's hidden state at several layer
fractions (deciles by default) so a per-layer correctness probe sweep can find
where the signal lives. No truncation: fails on any overlength example.

Output (per split stem), in --out_dir:
  {stem}_L{idx}_h.npy   (n, hidden_dim) float16   one file per selected layer
  {stem}_y.npy          (n,) int32                shared labels
  {stem}_meta.jsonl     n rows (uid, step_idx, label, global_index, n_tokens)
  multilayer_manifest.json   layer fractions -> indices, length audit, model meta
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
    build_prompt_prefix, git_commit, get_gpu_info, read_jsonl, sha256_file,
    tokenize_example, write_jsonl,
)


def resolve_layer_indices(fracs: list[float], num_layers: int) -> list[tuple[float, int]]:
    """Map depth fractions to hidden_states indices (0=embeddings, num_layers=final)."""
    out = []
    seen = set()
    for f in fracs:
        idx = max(1, min(num_layers, round(f * num_layers)))
        if idx not in seen:
            seen.add(idx)
            out.append((f, idx))
    return out


def encode_file(jsonl_path, out_dir, stem, tokenizer, model, device, max_seq_len,
                batch_size, save_dtype, pad_token_id, layer_idxs, force):
    examples = read_jsonl(jsonl_path)
    for gi, ex in enumerate(examples):
        ex["global_index"] = gi
    n = len(examples)
    hidden_dim = model.config.hidden_size
    np_dtype = np.float16 if save_dtype == torch.float16 else np.float32
    per_layer = {li: np.zeros((n, hidden_dim), dtype=np_dtype) for li in layer_idxs}
    labels = np.zeros(n, dtype=np.int32)
    meta: list[dict] = []
    token_lengths: list[int] = []
    print(f"[ml] {stem}: {n} examples, layers={layer_idxs}", flush=True)

    t0 = time.perf_counter()
    i = 0
    while i < n:
        batch = examples[i:i + batch_size]
        ids_list, cand_idx_list = [], []
        for ex in batch:
            try:
                ids, cand = tokenize_example(tokenizer, ex["problem"], ex["prefix"],
                                             ex["candidate_step"], max_seq_len)
            except ValueError as e:
                sys.exit(f"[ml] FATAL overlength uid={ex.get('uid','?')}: {e}")
            ids_list.append(ids); cand_idx_list.append(cand)
        mx = max(len(x) for x in ids_list)
        padded = [x + [pad_token_id] * (mx - len(x)) for x in ids_list]
        masks = [[1] * len(x) + [0] * (mx - len(x)) for x in ids_list]
        inp = torch.tensor(padded, dtype=torch.long, device=device)
        att = torch.tensor(masks, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inp, attention_mask=att, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple len num_layers+1
        for b, (ex, cand) in enumerate(zip(batch, cand_idx_list)):
            labels[i + b] = ex["label"]
            token_lengths.append(len(ids_list[b]))
            for li in layer_idxs:
                per_layer[li][i + b] = hs[li][b, cand, :].detach().to(save_dtype).cpu().numpy()
            meta.append({"uid": ex["uid"], "step_idx": ex["step_idx"], "label": ex["label"],
                         "global_index": ex["global_index"], "n_tokens": len(ids_list[b])})
        del out, hs
        i += len(batch)
        if (i // batch_size) % 32 == 0 or i == n:
            print(f"[ml] {stem}: {i}/{n} ({time.perf_counter()-t0:.1f}s)", flush=True)

    y_path = out_dir / f"{stem}_y.npy"
    np.save(y_path, labels)
    write_jsonl(out_dir / f"{stem}_meta.jsonl", meta)
    for li in layer_idxs:
        np.save(out_dir / f"{stem}_L{li}_h.npy", per_layer[li])
    lt = np.asarray(token_lengths)
    return {"n": n, "max_tokens": int(lt.max()), "p95": int(np.percentile(lt, 95)),
            "p99": int(np.percentile(lt, 99)), "num_truncated_examples": 0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--splits", nargs="+", required=True, help="<input.jsonl>:<stem> pairs.")
    p.add_argument("--layer_fracs", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--max_seq_len", type=int, default=-1, help="-1 = model context window.")
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
    model_max = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if args.max_seq_len <= 0:
        args.max_seq_len = model_max
    layer_map = resolve_layer_indices(args.layer_fracs, num_layers)
    layer_idxs = [li for _, li in layer_map]
    print(f"[ml] model={args.model_name_or_path} num_layers={num_layers} "
          f"hidden={model.config.hidden_size} layers={layer_map}", flush=True)

    files = {}
    for spec in args.splits:
        inp, stem = spec.rsplit(":", 1)
        jp = args.data_dir / inp if "/" not in inp else Path(inp)
        if not jp.exists():
            sys.exit(f"[ml] missing split file: {jp}")
        if (args.out_dir / f"{stem}_y.npy").exists() and not args.force:
            sys.exit(f"[ml] refusing overwrite {stem}_y.npy; pass --force")
        files[stem] = encode_file(jp, args.out_dir, stem, tok, model, device,
                                  args.max_seq_len, args.batch_size, dtype_map[args.save_dtype],
                                  pad, layer_idxs, args.force)

    manifest = {
        "run_name": args.run_name, "model_name": args.model_name_or_path,
        "num_hidden_layers": num_layers, "hidden_size": int(model.config.hidden_size),
        "model_max_position_embeddings": model_max, "max_seq_len": args.max_seq_len,
        "layer_fracs": args.layer_fracs,
        "layer_indices": {f"{f:.2f}": li for f, li in layer_map},
        "files": files, "saved_dtype": args.save_dtype,
        "num_truncated_examples": 0,
        "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": git_commit(),
    }
    (args.out_dir / "multilayer_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ml] done -> {args.out_dir/'multilayer_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
