"""parametric_retrieval_access_v1 stage 1: greedy generation (sharded TamIA
worker).

Per instance in metadata.parquet: ONE greedy run (T=0, no sampling), direct
paraphrases at --direct_max_new_tokens, canonical CoT instances at
--cot_max_new_tokens. No hidden states here: extraction (stage 3) runs after
grading has selected the extraction set, so generation stays cheap.

Prompt rendering: the chat template is applied as TEXT
(apply_chat_template(tokenize=False)) and the rendered text is tokenized with
the fast tokenizer (add_special_tokens=False). Stage 3 re-renders the same
way to recover identical token ids WITH character offsets, so
prompt_token_count and greedy_ids recorded here are its consistency check.

Outputs (in --out_dir):
  generations{suf}.jsonl        one record per instance (text + token ids)
  generate_manifest{suf}.json

Sharded usage (4 GPUs then merge):
  python scripts/parametric_retrieval/prga_generate.py \
      --out_dir runs/parametric_retrieval_access_v1 \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only \
      --shard_idx $i --num_shards 4
  python scripts/parametric_retrieval/prga_generate.py --merge --num_shards 4 \
      --out_dir runs/parametric_retrieval_access_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.parametric_retrieval.prg_generate import (  # noqa: E402
    batched_generate,
    git_commit,
    shard_suffix,
)


def render_prompt_text(tok, user_message: str) -> str:
    """Rendered chat-template TEXT for a single user turn (no system turn)."""
    return tok.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True, tokenize=False)


def encode_prompt(tok, user_message: str) -> list[int]:
    """Token ids of the rendered chat text; identical to what stage 3
    recovers with offsets (add_special_tokens=False: the template already
    contains the special tokens)."""
    return tok(render_prompt_text(tok, user_message),
               add_special_tokens=False)["input_ids"]


def merge_shards(args) -> None:
    lines = []
    for s in range(args.num_shards):
        p = args.out_dir / f"generations{shard_suffix(s, args.num_shards)}.jsonl"
        if not p.exists():
            sys.exit(f"missing shard output {p}")
        lines.extend(p.read_text().splitlines())
    recs = [json.loads(ln) for ln in lines if ln.strip()]
    recs.sort(key=lambda r: r["instance_id"])
    with open(args.out_dir / "generations.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[generate] merged {args.num_shards} shards -> {len(recs)} records",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=None,
                    help="defaults to <out_dir>/metadata.parquet")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--model_name_or_path", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--model_dtype",
                    choices=["bfloat16", "float16", "float32"],
                    default="bfloat16")
    ap.add_argument("--direct_max_new_tokens", type=int, default=24)
    ap.add_argument("--cot_max_new_tokens", type=int, default=256)
    ap.add_argument("--gen_batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only the first N instances of the shard (smoke)")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge_shards(args)
        return
    if not (0 <= args.shard_idx < args.num_shards):
        sys.exit(f"invalid shard config {args.shard_idx}/{args.num_shards}")

    suf = shard_suffix(args.shard_idx, args.num_shards)
    gen_out = args.out_dir / f"generations{suf}.jsonl"
    if gen_out.exists() and not args.force:
        sys.exit(f"refusing to overwrite {gen_out}; pass --force")

    meta_path = args.metadata or (args.out_dir / "metadata.parquet")
    inst = pd.read_parquet(meta_path)
    facts = sorted(inst.fact_id.unique())
    shard_facts = {f for i, f in enumerate(facts)
                   if i % args.num_shards == args.shard_idx}
    inst = inst[inst.fact_id.isin(shard_facts)].reset_index(drop=True)
    if args.limit is not None:
        inst = inst.iloc[: args.limit].reset_index(drop=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed + args.shard_idx)
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
    stop_ids = {t for t in [tok.eos_token_id,
                            tok.convert_tokens_to_ids("<|im_end|>")]
                if t is not None and t >= 0}

    prompt_ids = [encode_prompt(tok, m) for m in inst.user_message]
    is_cot = (inst.prompt_mode == "cot").to_numpy()
    d_idx = [i for i in range(len(inst)) if not is_cot[i]]
    c_idx = [i for i in range(len(inst)) if is_cot[i]]
    print(f"[generate] shard {args.shard_idx}/{args.num_shards}: "
          f"{len(d_idx)} direct + {len(c_idx)} cot instances", flush=True)

    bs = max(1, args.gen_batch_size)
    d_out = batched_generate(model, tok, device,
                             [prompt_ids[i] for i in d_idx], bs,
                             args.direct_max_new_tokens, True, 1, 0.0, 1.0,
                             stop_ids, "direct greedy")
    c_out = batched_generate(model, tok, device,
                             [prompt_ids[i] for i in c_idx], max(1, bs // 4),
                             args.cot_max_new_tokens, True, 1, 0.0, 1.0,
                             stop_ids, "cot greedy")
    gen_of: dict[int, list[int]] = {}
    for j, i in enumerate(d_idx):
        gen_of[i] = d_out[j][0]
    for j, i in enumerate(c_idx):
        gen_of[i] = c_out[j][0]

    gold_tok_cache: dict[str, int] = {}

    def gold_n_tokens(gold: str) -> int:
        if gold not in gold_tok_cache:
            gold_tok_cache[gold] = len(
                tok(" " + gold, add_special_tokens=False)["input_ids"])
        return gold_tok_cache[gold]

    with open(gen_out, "w") as f:
        for i, r in enumerate(inst.itertuples()):
            g_ids = gen_of[i]
            rec = {
                "instance_id": r.instance_id, "fact_id": r.fact_id,
                "direction": r.direction, "prompt_mode": r.prompt_mode,
                "paraphrase_id": r.paraphrase_id,
                "gold_answer": r.gold_answer,
                "gold_n_tokens": gold_n_tokens(str(r.gold_answer)),
                "prompt_token_count": len(prompt_ids[i]),
                "greedy_text": tok.decode(g_ids, skip_special_tokens=True),
                "greedy_ids": list(map(int, g_ids)),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[generate] wrote {gen_out} ({len(inst)} records)", flush=True)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_name_or_path": args.model_name_or_path,
        "model_dtype": args.model_dtype,
        "direct_max_new_tokens": args.direct_max_new_tokens,
        "cot_max_new_tokens": args.cot_max_new_tokens,
        "decoding": "greedy_only",
        "n_instances": int(len(inst)),
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
    }
    (args.out_dir / f"generate_manifest{suf}.json").write_text(
        json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
