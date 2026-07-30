#!/usr/bin/env python3
"""Encode each ProcessBench solution as ONE full causal pass and store all
last-layer token states, with per-step boundaries.

Unlike encode_processbench_token_store.py (which encodes each step separately with
the sequence ending at that step, so future steps are never attended to), this
runs the whole solution once. A step-k token state is identical to the per-step
encode (causal masking, same left context), but now the store ALSO holds the
downstream step states, which DID attend over step k. That is what a future-aware
(lookahead) representation needs: for judging step k we can pool step k's own
tokens (current), the pre-step boundary state (clean past), and the following
steps' tokens (future).

One item per trace: item = (n_tokens, d) last-layer states of the full solution.
meta per trace: id, pb_subset, label (first-error idx, -1 = all correct), n_steps,
step_starts, step_ends (token [start, end) of each step's content), n_tokens,
global_index. Solutions over --max_seq_len are skipped (reported).

Solution text framing (BOS on the prefix only, steps concatenated with blank
lines, exact additive boundaries):
    "Problem:\n{problem}\n\nSolution:\n" + steps[0] + "\n\n" + steps[1] + ...
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
from src.repstore.store import TOKEN_SEQ, RepSpec, write_split  # noqa: E402


def load_traces(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def tokenize_solution(tokenizer, problem: str, steps: list[str]):
    """Return (ids, step_starts, step_ends) with exact additive boundaries."""
    base = f"Problem:\n{problem}\n\nSolution:\n"
    ids = list(tokenizer(base, add_special_tokens=True)["input_ids"])
    sep_ids = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    step_starts, step_ends = [], []
    for j, step in enumerate(steps):
        if j > 0:
            ids += sep_ids
        s = len(ids)
        ids += tokenizer(step, add_special_tokens=False)["input_ids"]
        step_starts.append(s)
        step_ends.append(len(ids))
    return ids, step_starts, step_ends


def encode_subset(raw_file, subset, rep_root, tokenizer, model, device, layer,
                  max_seq_len, batch_size, pad_id, backbone):
    traces = load_traces(raw_file)
    toks = []
    skipped = 0
    for gi, tr in enumerate(traces):
        steps = tr["steps"]
        ids, ss, se = tokenize_solution(tokenizer, tr["problem"], steps)
        if len(ids) > max_seq_len:
            skipped += 1
            continue
        toks.append((ids, {
            "id": tr["id"], "pb_subset": subset, "label": int(tr["label"]),
            "n_steps": len(steps), "step_starts": ss, "step_ends": se,
            "n_tokens": len(ids), "global_index": gi,
        }))

    d = model.config.hidden_size
    items: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict] = []
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
        for b, (ids, m) in enumerate(batch):
            nt = len(ids)
            items.append(hs[b, :nt, :].detach().to(torch.float16).cpu().numpy())
            labels.append(m["label"])
            meta.append(m)
        del hs
    out_dir = rep_root / subset / "shard_00"
    spec = RepSpec(name=rep_root.name, kind=TOKEN_SEQ, dim=d, layer=layer,
                   backbone=backbone, readout="full_solution_tokens", source_split=subset)
    write_split(out_dir, items, labels, meta, spec)
    tot = sum(it.shape[0] for it in items)
    print(f"[pb_full] {subset}: {len(items)} traces, {tot:,} rows, "
          f"{tot*d*2/1e9:.2f}GB, skipped {skipped} ({time.perf_counter()-t0:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_specs", nargs="+", required=True, help="subset:rawfile.jsonl pairs")
    p.add_argument("--rep_root", required=True, type=Path)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=8)
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
                      args.max_seq_len, args.batch_size, pad_id, Path(args.model_name_or_path).name)


if __name__ == "__main__":
    main()
