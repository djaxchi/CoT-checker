"""parametric_retrieval_component_v1 stage 3b: per-layer attention- and
MLP-contribution extraction at the final prompt token (sharded TamIA worker).

Companion to prga_extract.py. That job stored the residual stream (the SUM of
all component contributions) at every layer; this job stores the two additive
sub-contributions of each decoder layer separately, so a downstream patch can
ask which component (attention output vs MLP output) carries the fact-specific
rescue, not just the whole residual.

Qwen2DecoderLayer computes, per layer L:
    h = h + self_attn(ln1(h))     <- attention contribution
    h = h + mlp(ln2(h))           <- MLP contribution
so hooking self_attn / mlp output captures exactly those two vectors. We store
them only at the final prompt token (index prompt_len - 1), for every instance
in extraction_set.json, at all decoder layers L = 0 .. num_layers-1.

Outputs (in --out_dir/component_states_v1):
  comp_meta{suf}.parquet   one row per instance (instance_id, fact_id,
                           direction, is_correct, split)
  attn_L{K:02d}{suf}.npy   (n_inst, hidden) float16, K = 0 .. num_layers-1
  mlp_L{K:02d}{suf}.npy    (n_inst, hidden) float16
Merge writes attn_L{K:02d}.safetensors / mlp_L{K:02d}.safetensors (key "h")
plus comp_meta.parquet.

  python scripts/parametric_retrieval/prgc_extract_components.py \
      --out_dir runs/parametric_retrieval_access_v1 \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only \
      --shard_idx $i --num_shards 4
  python scripts/parametric_retrieval/prgc_extract_components.py --merge \
      --num_shards 4
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

META_COLS = ["instance_id", "fact_id", "direction", "seed_variant",
             "template_id", "split", "is_correct"]


def merge_shards(args) -> None:
    from safetensors.numpy import save_file

    cs_dir = args.out_dir / "component_states_v1"
    n_layers = json.loads(
        (args.out_dir / f"component_manifest{shard_suffix(0, args.num_shards)}"
         ".json").read_text())["n_layers"]
    metas, attn, mlp = [], {k: [] for k in range(n_layers)}, \
        {k: [] for k in range(n_layers)}
    for s in range(args.num_shards):
        suf = shard_suffix(s, args.num_shards)
        metas.append(pd.read_parquet(cs_dir / f"comp_meta{suf}.parquet"))
        for k in range(n_layers):
            attn[k].append(np.load(cs_dir / f"attn_L{k:02d}{suf}.npy"))
            mlp[k].append(np.load(cs_dir / f"mlp_L{k:02d}{suf}.npy"))
    meta = pd.concat(metas, ignore_index=True)
    order = meta.sort_values("instance_id", kind="stable").index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)
    meta.to_parquet(cs_dir / "comp_meta.parquet", index=False)
    for k in range(n_layers):
        a = np.concatenate(attn[k], axis=0)[order]
        m = np.concatenate(mlp[k], axis=0)[order]
        assert a.shape[0] == len(meta) and m.shape[0] == len(meta)
        save_file({"h": a}, str(cs_dir / f"attn_L{k:02d}.safetensors"))
        save_file({"h": m}, str(cs_dir / f"mlp_L{k:02d}.safetensors"))
    print(f"[components] merged {args.num_shards} shards -> {len(meta)} "
          f"instances x {n_layers} layers x 2 components", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
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

    cs_dir = args.out_dir / "component_states_v1"
    cs_dir.mkdir(parents=True, exist_ok=True)
    suf = shard_suffix(args.shard_idx, args.num_shards)
    meta_out = cs_dir / f"comp_meta{suf}.parquet"
    if meta_out.exists() and not args.force:
        sys.exit(f"refusing to overwrite {meta_out}; pass --force")

    ext = json.loads((args.out_dir / "extraction_set.json").read_text())
    keep = set(ext["instance_ids"])
    inst = pd.read_parquet(args.out_dir / "metadata.parquet")
    inst = inst[inst.instance_id.isin(keep)].reset_index(drop=True)
    grading = {}
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

    # ---- rows + token ids ---------------------------------------------------
    rows: list[dict] = []
    seqs: list[list[int]] = []
    for r in inst.itertuples():
        rendered = render_prompt_text(tok, r.user_message)
        p_ids = tok(rendered, add_special_tokens=False)["input_ids"]
        if not p_ids:
            continue
        rows.append({"instance_id": r.instance_id, "fact_id": r.fact_id,
                     "direction": r.direction, "seed_variant": r.seed_variant,
                     "template_id": r.template_id, "split": r.split,
                     "is_correct": grading.get(r.instance_id),
                     "_seq": len(seqs)})
        seqs.append(list(p_ids))
    print(f"[components] shard {args.shard_idx}/{args.num_shards}: "
          f"{len(rows)} instances x {num_layers} layers", flush=True)

    attn_out = {k: np.empty((len(rows), hidden), dtype=np.float16)
                for k in range(num_layers)}
    mlp_out = {k: np.empty((len(rows), hidden), dtype=np.float16)
               for k in range(num_layers)}

    # forward hooks stash each submodule output for the current batch
    cache: dict[tuple, "torch.Tensor"] = {}

    def mk_hook(layer: int, kind: str):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            cache[(kind, layer)] = h.detach()
        return hook

    handles = []
    for L in range(num_layers):
        handles.append(model.model.layers[L].self_attn.register_forward_hook(
            mk_hook(L, "attn")))
        handles.append(model.model.layers[L].mlp.register_forward_hook(
            mk_hook(L, "mlp")))

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(len(rows)), key=lambda i: len(seqs[i]))
    t0, done = time.perf_counter(), 0
    for start in range(0, len(order), args.fwd_batch_size):
        sidx = order[start:start + args.fwd_batch_size]
        ids_list = [seqs[i] for i in sidx]
        mx = max(len(x) for x in ids_list)
        inp = torch.tensor([x + [pad] * (mx - len(x)) for x in ids_list],
                           dtype=torch.long, device=device)
        att = torch.tensor([[1] * len(x) + [0] * (mx - len(x))
                            for x in ids_list], dtype=torch.long,
                           device=device)
        cache.clear()
        with torch.no_grad():
            model(inp, attention_mask=att, use_cache=False)
        for b, i in enumerate(sidx):
            last = len(seqs[i]) - 1
            for L in range(num_layers):
                attn_out[L][i] = cache[("attn", L)][b, last].to(
                    torch.float16).cpu().numpy()
                mlp_out[L][i] = cache[("mlp", L)][b, last].to(
                    torch.float16).cpu().numpy()
        done += len(sidx)
        if (start // args.fwd_batch_size) % 20 == 0:
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"[components] {done}/{len(order)} ({rate:.2f}/s)",
                  flush=True)
    for h in handles:
        h.remove()

    meta = pd.DataFrame(rows, columns=META_COLS + ["_seq"]).drop(
        columns=["_seq"])
    meta.to_parquet(meta_out, index=False)
    for k in range(num_layers):
        np.save(cs_dir / f"attn_L{k:02d}{suf}.npy", attn_out[k])
        np.save(cs_dir / f"mlp_L{k:02d}{suf}.npy", mlp_out[k])
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_name_or_path": args.model_name_or_path,
        "model_dtype": args.model_dtype,
        "n_layers": num_layers,
        "hidden_size": hidden,
        "n_instances": len(rows),
        "position": "final_prompt_token",
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
    }
    (args.out_dir / f"component_manifest{suf}.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[components] shard done: {len(rows)} instances", flush=True)


if __name__ == "__main__":
    main()
