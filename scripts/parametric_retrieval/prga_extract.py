"""parametric_retrieval_access_v1 stage 3: all-layer multi-position residual
extraction for the extraction set (sharded TamIA worker).

For every instance in extraction_set.json the rendered chat prompt is
re-tokenized with character offsets (must reproduce the stage-1 ids exactly;
asserted against prompt_token_count and greedy_ids), teacher-forced together
with the FIRST greedy generated token, and hidden states are stored at ALL
hidden_states indices 0..num_layers (spec: do not restrict layers before
localization) for the positions computed by compute_access_positions:

  entity_first, entity_last, entity_mean (mean over the entity mention),
  question_last, answer_prefix, final_prompt_token (pre-generation),
  first_generated_token (auxiliary, post-decision).

Outputs (in --out_dir/hidden_states_v1):
  hs_meta{suf}.parquet    one row per instance x position
  h_hs{K}{suf}.npy        (n_rows, hidden) float16, K = 0..num_layers
Merge writes layer_{K:02d}.safetensors (key "h") + hs_meta.parquet.

  python scripts/parametric_retrieval/prga_extract.py \
      --out_dir runs/parametric_retrieval_access_v1 \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only \
      --shard_idx $i --num_shards 4
  python scripts/parametric_retrieval/prga_extract.py --merge --num_shards 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.parametric_retrieval.prg_generate import (  # noqa: E402
    git_commit,
    shard_suffix,
)
from scripts.parametric_retrieval.prga_generate import (  # noqa: E402
    render_prompt_text,
)
from src.analysis.parametric_retrieval_access import (  # noqa: E402
    compute_access_positions,
)

META_COLS = ["row_id", "instance_id", "fact_id", "direction", "seed_variant",
             "template_id", "split", "position_name", "token_start",
             "token_end", "prompt_token_count", "is_correct"]


def merge_shards(args) -> None:
    from safetensors.numpy import save_file

    hs_dir = args.out_dir / "hidden_states_v1"
    first = pd.read_parquet(hs_dir / f"hs_meta{shard_suffix(0, args.num_shards)}.parquet")
    n_layers = json.loads(
        (args.out_dir / f"extract_manifest{shard_suffix(0, args.num_shards)}.json"
         ).read_text())["n_hs_indices"]
    metas, arrays = [], {k: [] for k in range(n_layers)}
    for s in range(args.num_shards):
        suf = shard_suffix(s, args.num_shards)
        metas.append(pd.read_parquet(hs_dir / f"hs_meta{suf}.parquet"))
        for k in range(n_layers):
            arrays[k].append(np.load(hs_dir / f"h_hs{k}{suf}.npy"))
    meta = pd.concat(metas, ignore_index=True)
    cat = {k: np.concatenate(a, axis=0) for k, a in arrays.items()}
    order = meta.sort_values(["instance_id", "token_start"],
                             kind="stable").index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)
    meta.to_parquet(hs_dir / "hs_meta.parquet", index=False)
    for k in range(n_layers):
        assert cat[k].shape[0] == len(meta)
        save_file({"h": cat[k][order]},
                  str(hs_dir / f"layer_{k:02d}.safetensors"))
    print(f"[extract] merged {args.num_shards} shards -> {len(meta)} rows x "
          f"{n_layers} layers", flush=True)
    _ = first  # schema sanity only


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--model_name_or_path", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--model_dtype",
                    choices=["bfloat16", "float16", "float32"],
                    default="bfloat16")
    ap.add_argument("--fwd_batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge_shards(args)
        return
    if not (0 <= args.shard_idx < args.num_shards):
        sys.exit(f"invalid shard config {args.shard_idx}/{args.num_shards}")

    hs_dir = args.out_dir / "hidden_states_v1"
    hs_dir.mkdir(parents=True, exist_ok=True)
    suf = shard_suffix(args.shard_idx, args.num_shards)
    meta_out = hs_dir / f"hs_meta{suf}.parquet"
    if meta_out.exists() and not args.force:
        sys.exit(f"refusing to overwrite {meta_out}; pass --force")

    ext = json.loads((args.out_dir / "extraction_set.json").read_text())
    keep = set(ext["instance_ids"])
    inst = pd.read_parquet(args.out_dir / "metadata.parquet")
    inst = inst[inst.instance_id.isin(keep)].reset_index(drop=True)
    grading = {}
    gens = {}
    for ln in (args.out_dir / "generations.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r["instance_id"] in keep:
                gens[r["instance_id"]] = r
    for ln in (args.out_dir / "grading.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r["instance_id"] in keep:
                grading[r["instance_id"]] = bool(r["is_correct"])
    facts = sorted(inst.fact_id.unique())
    shard_facts = {f for i, f in enumerate(facts)
                   if i % args.num_shards == args.shard_idx}
    inst = inst[inst.fact_id.isin(shard_facts)].reset_index(drop=True)
    if args.limit is not None:
        inst = inst.iloc[: args.limit].reset_index(drop=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    dtype = getattr(torch, args.model_dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            torch_dtype=dtype)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    num_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    hs_indices = list(range(num_layers + 1))  # 0 = embeddings

    # ---- rows + sequences ---------------------------------------------------
    rows: list[dict] = []
    sequences: list[list[int]] = []
    n_entity_missing = 0
    for r in inst.itertuples():
        gen = gens.get(r.instance_id)
        if gen is None:
            sys.exit(f"no generation record for {r.instance_id}")
        rendered = render_prompt_text(tok, r.user_message)
        enc = tok(rendered, add_special_tokens=False,
                  return_offsets_mapping=True)
        p_ids = enc["input_ids"]
        if len(p_ids) != gen["prompt_token_count"]:
            sys.exit(f"tokenization drift for {r.instance_id}: "
                     f"{len(p_ids)} vs {gen['prompt_token_count']}")
        first_gen = gen["greedy_ids"][:1]
        if not first_gen:
            continue  # empty generation: nothing to teacher-force
        positions = compute_access_positions(
            rendered, enc["offset_mapping"], r.question, r.entity,
            r.user_message)
        if not any(p["position_name"] == "entity_first" for p in positions):
            n_entity_missing += 1
        for p in positions:
            rows.append({
                "row_id": f"{r.instance_id}::{p['position_name']}",
                "instance_id": r.instance_id, "fact_id": r.fact_id,
                "direction": r.direction, "seed_variant": r.seed_variant,
                "template_id": r.template_id, "split": r.split,
                "position_name": p["position_name"],
                "token_start": p["token_start"],
                "token_end": p["token_end"],
                "prompt_token_count": len(p_ids),
                "is_correct": grading.get(r.instance_id),
                "_seq": len(sequences),
            })
        sequences.append(list(p_ids) + list(first_gen))

    print(f"[extract] shard {args.shard_idx}/{args.num_shards}: "
          f"{len(inst)} instances -> {len(rows)} rows x "
          f"{len(hs_indices)} layers; entity missing in "
          f"{n_entity_missing} instances", flush=True)

    out = {k: np.empty((len(rows), hidden), dtype=np.float16)
           for k in hs_indices}
    by_seq: dict[int, list[int]] = {}
    for ri, row in enumerate(rows):
        by_seq.setdefault(row["_seq"], []).append(ri)
    seq_order = sorted(by_seq, key=lambda s: len(sequences[s]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0, done = time.perf_counter(), 0
    import torch  # noqa: F811
    for start in range(0, len(seq_order), args.fwd_batch_size):
        sidx = seq_order[start:start + args.fwd_batch_size]
        ids_list = [sequences[s] for s in sidx]
        mx = max(len(x) for x in ids_list)
        inp = torch.tensor([x + [pad] * (mx - len(x)) for x in ids_list],
                           dtype=torch.long, device=device)
        att = torch.tensor([[1] * len(x) + [0] * (mx - len(x))
                            for x in ids_list], dtype=torch.long,
                           device=device)
        with torch.no_grad():
            o = model(inp, attention_mask=att, output_hidden_states=True,
                      use_cache=False)
        hs = [h.to(torch.float32) for h in o.hidden_states]
        for b, s in enumerate(sidx):
            for ri in by_seq[s]:
                ts, te = rows[ri]["token_start"], rows[ri]["token_end"]
                assert te < len(sequences[s]), rows[ri]["row_id"]
                for k in hs_indices:
                    vec = hs[k][b, ts] if ts == te \
                        else hs[k][b, ts:te + 1].mean(dim=0)
                    out[k][ri] = vec.to(torch.float16).cpu().numpy()
        done += len(sidx)
        if (start // args.fwd_batch_size) % 20 == 0:
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"[extract] {done}/{len(seq_order)} sequences "
                  f"({rate:.2f}/s)", flush=True)

    meta = pd.DataFrame(rows, columns=META_COLS + ["_seq"]).drop(
        columns=["_seq"])
    meta.to_parquet(meta_out, index=False)
    for k in hs_indices:
        np.save(hs_dir / f"h_hs{k}{suf}.npy", out[k])
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_name_or_path": args.model_name_or_path,
        "model_dtype": args.model_dtype,
        "n_hs_indices": len(hs_indices),
        "hidden_size": hidden,
        "n_instances": int(len(inst)),
        "n_rows": len(rows),
        "n_entity_missing": n_entity_missing,
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
    }
    (args.out_dir / f"extract_manifest{suf}.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[extract] shard done: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
