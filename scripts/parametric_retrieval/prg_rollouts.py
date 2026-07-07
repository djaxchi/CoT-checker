"""parametric_retrieval_geometry_v0 exp 2: within-instance CoT rollout
divergence (attacking reasoning_unlocked directly).

reasoning_unlocked is invisible in the static prompt geometry. The phenomenon
is dynamic: thinking sometimes unlocks recall. So instead of comparing across
instances at the prompt, we compare WITHIN one instance: sampled CoT rollouts
that succeed vs that fail. Same fact, same prompt, only the reasoning path
differs, so the fact is perfectly controlled.

This script selects instances whose sampled CoT is MIXED (at least one success
and one failure among the greedy + K sampled rollouts), generates K CoT
rollouts each (temperature 0.7), grades each rollout, and saves per-rollout
hidden states at trajectory checkpoints:

  final_prompt_token, first_generated_token, sentence_end (each), and the
  token before the final answer.

Selection pool = instances flagged cot_unstable or reasoning_unlocked_soft in
grading (that is where success and failure coexist). Sharded by fact.

Outputs under <out_dir>/rollouts/ :
  rollouts{suf}.jsonl              one row per rollout (text, success, checkpoints)
  h_hs{K}{suf}.npy                 (n_checkpoints, hidden) f16, aligned
  ckpt_meta{suf}.parquet           one row per saved checkpoint
Merge with --merge (writes layer_{K:02d}.safetensors, ckpt_meta.parquet,
rollouts.jsonl).

  python scripts/parametric_retrieval/prg_rollouts.py \
      --out_dir runs/parametric_retrieval_geometry_v0 \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only \
      --layers 20 24 --k 8 --shard_idx $i --num_shards 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.parametric_retrieval import (  # noqa: E402
    build_user_message,
    compute_positions,
    extract_cot_final_answer,
    grade_answer,
)

CKPT_COLS = ["row_id", "question_id", "fact_id", "rollout_idx", "success",
             "position_name", "position_rank", "token_index", "n_gen_tokens"]


def suffix(s, n):
    return "" if n == 1 else f"_shard{s:02d}"


def merge(args):
    from safetensors.numpy import save_file
    rd = args.out_dir / "rollouts"
    metas, arrs = [], {k: [] for k in args.layers}
    lines = []
    for s in range(args.num_shards):
        suf = suffix(s, args.num_shards)
        metas.append(pd.read_parquet(rd / f"ckpt_meta{suf}.parquet"))
        lines += (rd / f"rollouts{suf}.jsonl").read_text().splitlines()
        for k in args.layers:
            arrs[k].append(np.load(rd / f"h_hs{k}{suf}.npy"))
    meta = pd.concat(metas, ignore_index=True)
    cat = {k: np.concatenate(a) for k, a in arrs.items()}
    for k in args.layers:
        assert len(meta) == len(cat[k])
    meta.to_parquet(rd / "ckpt_meta.parquet", index=False)
    for k in args.layers:
        save_file({"h": cat[k]}, str(rd / f"layer_{k:02d}.safetensors"))
    (rd / "rollouts.jsonl").write_text("\n".join(lines))
    print(f"[roll] merged {args.num_shards} shards -> {len(meta)} checkpoints "
          f"/ {len(lines)} rollouts", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 24])
    ap.add_argument("--k", type=int, default=8, help="rollouts per instance")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_instances", type=int, default=None)
    ap.add_argument("--gen_batch_size", type=int, default=32)
    ap.add_argument("--fwd_batch_size", type=int, default=16)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pool_all", action="store_true",
                    help="smoke: ignore the mixed-outcome filter, take any QA")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir_rd = args.out_dir / "rollouts"
    args.out_dir_rd.mkdir(exist_ok=True)
    if args.merge:
        merge(args)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed + args.shard_idx)
    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    md = pd.read_parquet(args.out_dir / "metadata.parquet")
    # mixed-outcome pool: cot greedy or sampled disagree -> success+failure live
    if args.pool_all:
        pool = grading[~grading.is_control]
    else:
        pool = grading[(~grading.is_control)
                       & (grading.cot_unstable.fillna(False)
                          | grading.reasoning_unlocked_soft.fillna(False))]
    pool = pool.merge(md[["question_id", "question"]], on="question_id")
    facts = sorted(pool.fact_id.unique())
    keep = {f for i, f in enumerate(facts)
            if i % args.num_shards == args.shard_idx}
    pool = pool[pool.fact_id.isin(keep)].reset_index(drop=True)
    if args.max_instances:
        pool = pool.iloc[: args.max_instances].reset_index(drop=True)
    print(f"[roll] shard {args.shard_idx}/{args.num_shards}: "
          f"{len(pool)} mixed-outcome instances x k={args.k}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            torch_dtype=torch.bfloat16)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    hidden = int(model.config.hidden_size)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stop = {t for t in [tok.eos_token_id,
                        tok.convert_tokens_to_ids("<|im_end|>")]
            if t is not None and t >= 0}

    def render(q, fam):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": build_user_message(q, fam, "cot")}],
            add_generation_prompt=True, tokenize=True, return_dict=False)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    prompts = [render(r.question, r.family) for r in pool.itertuples()]

    # ---- generate k rollouts per instance (batched over instances) --------
    roll_ids = [[None] * args.k for _ in range(len(pool))]
    order = sorted(range(len(pool)), key=lambda i: len(prompts[i]))
    t0 = time.perf_counter()
    for s in range(0, len(order), args.gen_batch_size):
        idxs = order[s:s + args.gen_batch_size]
        ids_list = [prompts[i] for i in idxs]
        mx = max(len(x) for x in ids_list)
        inp = torch.tensor([[pad] * (mx - len(x)) + x for x in ids_list],
                           device=device)
        att = torch.tensor([[0] * (mx - len(x)) + [1] * len(x)
                            for x in ids_list], device=device)
        with torch.no_grad():
            seq = model.generate(inp, attention_mask=att,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, num_return_sequences=args.k,
                                 pad_token_id=pad)
        gen = seq[:, mx:].tolist()
        for j, i in enumerate(idxs):
            for r in range(args.k):
                g = gen[j * args.k + r]
                for c, t in enumerate(g):
                    if t in stop:
                        g = g[:c]
                        break
                roll_ids[i][r] = g
        if (s // args.gen_batch_size) % 5 == 0:
            print(f"[roll] gen {s + len(idxs)}/{len(pool)} instances "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- grade + build checkpoint rows ------------------------------------
    ckpt_rows, sequences, roll_records = [], [], []
    for i, r in enumerate(pool.itertuples()):
        p_ids = prompts[i]
        for ridx in range(args.k):
            g_ids = roll_ids[i][ridx]
            text = tok.decode(g_ids, skip_special_tokens=True)
            ans, _ = extract_cot_final_answer(text)
            ok = grade_answer(ans, r.gold_answer)[0]
            roll_records.append({
                "question_id": r.question_id, "fact_id": r.fact_id,
                "rollout_idx": ridx, "success": bool(ok),
                "final_answer": ans, "n_gen_tokens": len(g_ids)})
            # cumulative offsets for CoT position finding
            offs, prev = [], 0
            for c in range(len(g_ids)):
                cur = max(len(tok.decode(g_ids[:c + 1],
                                         skip_special_tokens=True)), prev)
                offs.append(cur)
                prev = cur
            for p in compute_positions(len(p_ids), len(g_ids), text, offs,
                                       "cot"):
                ckpt_rows.append({
                    "row_id": f"{r.question_id}::r{ridx}::"
                              f"{p['position_name']}{p['position_rank']}",
                    "question_id": r.question_id, "fact_id": r.fact_id,
                    "rollout_idx": ridx, "success": bool(ok),
                    "position_name": p["position_name"],
                    "position_rank": p["position_rank"],
                    "token_index": p["token_index"],
                    "n_gen_tokens": len(g_ids), "_seq": len(sequences)})
                sequences.append(p_ids + list(g_ids))

    suf = suffix(args.shard_idx, args.num_shards)
    (args.out_dir_rd / f"rollouts{suf}.jsonl").write_text(
        "\n".join(json.dumps(x) for x in roll_records))

    # ---- one forward per (rollout) sequence, read checkpoint tokens -------
    n = len(ckpt_rows)
    out = {k: np.empty((n, hidden), np.float16) for k in args.layers}
    by_seq = {}
    for ri, row in enumerate(ckpt_rows):
        by_seq.setdefault(row["_seq"], []).append(ri)
    seq_order = sorted(by_seq, key=lambda s: len(sequences[s]))
    t0 = time.perf_counter()
    for s in range(0, len(seq_order), args.fwd_batch_size):
        sidx = seq_order[s:s + args.fwd_batch_size]
        ids_list = [sequences[q] for q in sidx]
        mx = max(len(x) for x in ids_list)
        inp = torch.tensor([x + [pad] * (mx - len(x)) for x in ids_list],
                           device=device)
        att = torch.tensor([[1] * len(x) + [0] * (mx - len(x))
                            for x in ids_list], device=device)
        with torch.no_grad():
            o = model(inp, attention_mask=att, output_hidden_states=True,
                      use_cache=False)
        for b, q in enumerate(sidx):
            for ri in by_seq[q]:
                ti = min(ckpt_rows[ri]["token_index"], len(sequences[q]) - 1)
                for k in args.layers:
                    out[k][ri] = (o.hidden_states[k][b, ti]
                                  .float().cpu().numpy())
        if (s // args.fwd_batch_size) % 20 == 0:
            print(f"[roll] fwd {s + len(sidx)}/{len(seq_order)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    meta = pd.DataFrame(ckpt_rows, columns=CKPT_COLS + ["_seq"]).drop(
        columns=["_seq"])
    meta.to_parquet(args.out_dir_rd / f"ckpt_meta{suf}.parquet", index=False)
    for k in args.layers:
        np.save(args.out_dir_rd / f"h_hs{k}{suf}.npy", out[k])
    succ = pd.DataFrame(roll_records)
    print(f"[roll] shard done: {len(pool)} instances, {len(roll_records)} "
          f"rollouts (success rate {succ.success.mean():.2f}), "
          f"{n} checkpoints", flush=True)


if __name__ == "__main__":
    main()
