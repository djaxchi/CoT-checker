"""parametric_retrieval_minimal_v1 stage 3c: MLP intermediate-activation
extraction at the final prompt token (sharded TamIA worker).

For the neuron-level decomposition we need g = silu(gate(x)) * up(x), the
pre-down_proj intermediate activation, because the MLP output is linear in it:
    mlp_out = down_proj.weight @ g          (down_proj.weight: hidden x inter)
so swapping neuron i's activation from recipient to donor adds exactly
    (g_donor_i - g_recip_i) * down_proj.weight[:, i]
to the residual. Storing g for donors and recipients lets prgm_minimal.py
reconstruct any neuron-subset injection without a second forward.

We extract only at the MLP-selected layer(s) (from expE/selection.json, the
component that carried the rescue; default 27), final prompt token, for every
instance in extraction_set.json.

Outputs (in --out_dir/neuron_states_v1):
  neuron_meta{suf}.parquet   instance_id, fact_id, direction, is_correct, split
  g_L{K:02d}{suf}.npy        (n_inst, intermediate) float16
Merge writes g_L{K:02d}.safetensors (key "h") + neuron_meta.parquet.

  python scripts/parametric_retrieval/prgm_extract_neurons.py \
      --out_dir runs/parametric_retrieval_access_v1 \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only \
      --shard_idx $i --num_shards 4
  python scripts/parametric_retrieval/prgm_extract_neurons.py --merge \
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


def selected_layers(out_dir: Path, override):
    if override:
        return list(override)
    sel = out_dir / "expE" / "selection.json"
    if sel.exists():
        return [int(json.loads(sel.read_text())["mlp"]["layer"])]
    return [27]


def merge_shards(args) -> None:
    from safetensors.numpy import save_file

    ns_dir = args.out_dir / "neuron_states_v1"
    layers = json.loads(
        (args.out_dir / f"neuron_manifest{shard_suffix(0, args.num_shards)}"
         ".json").read_text())["layers"]
    metas, arr = [], {k: [] for k in layers}
    for s in range(args.num_shards):
        suf = shard_suffix(s, args.num_shards)
        metas.append(pd.read_parquet(ns_dir / f"neuron_meta{suf}.parquet"))
        for k in layers:
            arr[k].append(np.load(ns_dir / f"g_L{k:02d}{suf}.npy"))
    meta = pd.concat(metas, ignore_index=True)
    order = meta.sort_values("instance_id", kind="stable").index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)
    meta.to_parquet(ns_dir / "neuron_meta.parquet", index=False)
    for k in layers:
        h = np.concatenate(arr[k], axis=0)[order]
        assert h.shape[0] == len(meta)
        save_file({"h": h}, str(ns_dir / f"g_L{k:02d}.safetensors"))
    print(f"[neurons] merged {args.num_shards} shards -> {len(meta)} "
          f"instances x layers {layers}", flush=True)


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
    ap.add_argument("--layers", type=int, nargs="+", default=None)
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

    ns_dir = args.out_dir / "neuron_states_v1"
    ns_dir.mkdir(parents=True, exist_ok=True)
    suf = shard_suffix(args.shard_idx, args.num_shards)
    meta_out = ns_dir / f"neuron_meta{suf}.parquet"
    if meta_out.exists() and not args.force:
        sys.exit(f"refusing to overwrite {meta_out}; pass --force")

    layers = selected_layers(args.out_dir, args.layers)
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
    inter = int(model.config.intermediate_size)

    rows, seqs = [], []
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
    print(f"[neurons] shard {args.shard_idx}/{args.num_shards}: "
          f"{len(rows)} instances x layers {layers} (inter={inter})",
          flush=True)

    out = {k: np.empty((len(rows), inter), dtype=np.float16) for k in layers}
    cache: dict[int, "torch.Tensor"] = {}

    def mk_hook(L):
        def hook(module, hook_args):
            cache[L] = hook_args[0].detach()   # down_proj input = g
        return hook

    handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(
        mk_hook(L)) for L in layers]

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
            for L in layers:
                out[L][i] = cache[L][b, last].to(torch.float16).cpu().numpy()
        done += len(sidx)
        if (start // args.fwd_batch_size) % 20 == 0:
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"[neurons] {done}/{len(order)} ({rate:.2f}/s)", flush=True)
    for h in handles:
        h.remove()

    meta = pd.DataFrame(rows, columns=META_COLS + ["_seq"]).drop(
        columns=["_seq"])
    meta.to_parquet(meta_out, index=False)
    for L in layers:
        np.save(ns_dir / f"g_L{L:02d}{suf}.npy", out[L])
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_name_or_path": args.model_name_or_path,
        "layers": layers, "intermediate_size": inter,
        "n_instances": len(rows), "position": "final_prompt_token",
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
    }
    (args.out_dir / f"neuron_manifest{suf}.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[neurons] shard done: {len(rows)} instances", flush=True)


if __name__ == "__main__":
    main()
