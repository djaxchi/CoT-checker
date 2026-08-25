#!/usr/bin/env python3
"""Store ALL last-layer token states of every ProcessBench step into the repstore.

ProcessBench sibling of encode_prm800k_token_store.py. Flattens each trace into
(step, prior-steps-as-prefix) entries (identical to the dense PB encoder), stores
the full last-layer token sequence per step as a token_seq item, plus the offsets
and ProcessBench meta needed to derive dense_last / delta / pooled offline:

    meta: id, step_idx, label (trace first-error idx), n_steps, pb_subset,
          step_start_idx, pre_step_boundary_idx, n_tokens, global_index

One store split per subset. Whole-node H100:4, 4 in-node shards; read back with
repstore.ShardedRepSplit. Serves both dense_last and delta for all subsets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.encode_prm800k_hidden_states import build_prompt_prefix  # noqa: E402
from src.repstore.store import TOKEN_SEQ, RepSpec  # noqa: E402


def load_traces(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def flatten(traces: list[dict], subset: str) -> list[dict]:
    flat = []
    for tr in traces:
        steps = tr["steps"]
        lbl = int(tr["label"])
        n = len(steps)
        for k, step in enumerate(steps):
            flat.append({
                "id": tr["id"], "step_idx": k, "label": lbl, "n_steps": n,
                "pb_subset": subset, "problem": tr["problem"],
                "prefix": "\n\n".join(steps[:k]), "current_step": step,
            })
    for gi, r in enumerate(flat):
        r["global_index"] = gi
    return flat


def encode_subset(raw_file, subset, rep_root, tokenizer, model, device, layer,
                  max_seq_len, batch_size, pad_id, shard_idx, num_shards, backbone,
                  span_only=False):
    flat = flatten(load_traces(raw_file), subset)
    shard = [r for r in flat if r["global_index"] % num_shards == shard_idx]

    toks = []
    for ex in shard:
        prefix_ids = tokenizer(build_prompt_prefix(ex["problem"], ex["prefix"]),
                               add_special_tokens=True, truncation=False)["input_ids"]
        step_ids = tokenizer(ex["current_step"], add_special_tokens=False, truncation=False)["input_ids"]
        ids = prefix_ids + step_ids
        if not step_ids or len(ids) > max_seq_len:
            # ProcessBench steps can be long; skip overlength to match dense encoder policy.
            print(f"[pb_tokstore] skip {ex['id']} step {ex['step_idx']} len={len(ids)}", flush=True)
            continue
        toks.append((ids, len(prefix_ids), ex))

    if span_only and any(start < 1 for _, start, _ in toks):
        raise ValueError("span_only needs a non-empty prefix (step_start_idx >= 1)")
    lengths = np.array(
        [len(ids) - start + 1 if span_only else len(ids) for ids, start, _ in toks],
        dtype=np.int32)
    total = int(lengths.sum())
    d = model.config.hidden_size
    out_dir = rep_root / subset / f"shard_{shard_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    h_mm = np.lib.format.open_memmap(out_dir / "h.npy", mode="w+", dtype=np.float16, shape=(total, d))
    print(f"[pb_tokstore] {subset} shard {shard_idx}: {len(toks)} items {total:,} rows "
          f"{total*d*2/1e9:.2f}GB", flush=True)

    y = np.zeros(len(toks), dtype=np.int8)
    meta = []
    cur = 0
    t0 = time.perf_counter()
    for i in range(0, len(toks), batch_size):
        batch = toks[i:i + batch_size]
        maxlen = max(len(t[0]) for t in batch)
        padded = [t[0] + [pad_id] * (maxlen - len(t[0])) for t in batch]
        mask = [[1] * len(t[0]) + [0] * (maxlen - len(t[0])) for t in batch]
        inp = torch.tensor(padded, dtype=torch.long, device=device)
        att = torch.tensor(mask, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inp, attention_mask=att, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[layer]; del out
        for b, (ids, start, ex) in enumerate(batch):
            nt = len(ids)
            lo = start - 1 if span_only else 0
            keep = nt - lo
            h_mm[cur:cur + keep] = hs[b, lo:nt, :].detach().to(torch.float16).cpu().numpy()
            y[i + b] = int(ex["step_idx"] == ex["label"])
            row = {
                "id": ex["id"], "step_idx": ex["step_idx"], "label": ex["label"],
                "n_steps": ex["n_steps"], "pb_subset": subset,
                "n_tokens": nt, "step_start_idx": start, "pre_step_boundary_idx": start - 1,
                "global_index": ex["global_index"],
            }
            if span_only:
                row.update({
                    "orig_n_tokens": nt, "orig_step_start_idx": start,
                    "orig_pre_step_boundary_idx": start - 1,
                    "n_tokens": keep, "step_start_idx": 1, "pre_step_boundary_idx": 0,
                })
            meta.append(row)
            cur += keep
        del hs
    h_mm.flush()
    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "y.npy", y)
    with (out_dir / "meta.jsonl").open("w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    spec = RepSpec(
        name=rep_root.name, kind=TOKEN_SEQ, dim=d, layer=layer, backbone=backbone,
        readout="step_span_with_boundary" if span_only else "token_all_last_layer",
        source_split=subset)
    (out_dir / "spec.json").write_text(spec.to_json())
    print(f"[pb_tokstore] {subset} shard {shard_idx}: done {cur:,} rows ({time.perf_counter()-t0:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_specs", nargs="+", required=True, help="subset:rawfile.jsonl pairs")
    p.add_argument("--rep_root", required=True, type=Path)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--span_only", action="store_true",
                   help="Store only the pre-step boundary row plus the step's own "
                        "tokens (byte-identical to encoding in full then compacting).")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only, torch_dtype=torch.float16,
    ).to(device).eval()

    for spec in args.raw_specs:
        subset, raw = spec.split(":", 1)
        encode_subset(Path(raw), subset, args.rep_root, tok, model, device, args.layer,
                      args.max_seq_len, args.batch_size, pad_id, args.shard_idx, args.num_shards,
                      Path(args.model_name_or_path).name, args.span_only)


if __name__ == "__main__":
    main()
