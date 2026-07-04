"""parametric_retrieval_geometry_v0 stage 1: generation + hidden-state
extraction (sharded TamIA worker).

Per prompt instance (from metadata.parquet):
  direct greedy   T=0,   max_new_tokens=32
  direct sampled  T=0.7, top_p=0.95, n=4, max_new_tokens=32
  cot greedy      T=0,   max_new_tokens=256          (QA families only)
  cot sampled     T=0.7, top_p=0.95, n=4, 256 tokens (QA families only)

Sampled generations exist only for behavioral labeling. Hidden states are
extracted from the two greedy runs via one teacher-forced forward pass over
prompt + greedy completion (identical states to the ones during generation),
at the hidden_states tuple indices --hs_indices (hidden_states[0]=embeddings,
hidden_states[k]=post block k-1, so hs_idx 20 = block_idx 19).

Positions: final_prompt_token = last token of the rendered Qwen chat prompt
(chat template applied, never raw tokenization). Direct adds
first_generated_token + final_answer_token; CoT adds sentence_end ranks,
token_before_final_answer, first_final_answer_token, final_answer_token
(marker positions only when 'Final answer:' is present).

Outputs (in --out_dir):
  generations{suf}.jsonl                    one record per instance
  hidden_states/hs_meta{suf}.parquet        one row per extracted position
  hidden_states/h_hs{K}{suf}.npy            (n_rows, hidden) float16
  generate_manifest{suf}.json

Sharded usage (4 GPUs then merge):
  python scripts/parametric_retrieval/prg_generate.py \
      --out_dir runs/parametric_retrieval_geometry_v0 \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only \
      --shard_idx $i --num_shards 4
  python scripts/parametric_retrieval/prg_generate.py --merge --num_shards 4 \
      --out_dir runs/parametric_retrieval_geometry_v0
Merge writes hidden_states/layer_{K:02d}.safetensors (key "h"), the merged
hs_meta.parquet and generations.jsonl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.parametric_retrieval import (  # noqa: E402
    HS_INDICES,
    block_idx,
    build_user_message,
    compute_positions,
)

HS_META_COLS = ["row_id", "question_id", "fact_id", "family", "is_control",
                "prompt_mode", "position_name", "position_rank", "token_index",
                "prompt_token_count", "n_gen_tokens", "has_final_marker"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            text=True).strip()
    except Exception:
        return "unknown"


def shard_suffix(shard_idx: int, num_shards: int) -> str:
    return "" if num_shards == 1 else f"_shard{shard_idx:02d}"


def cumulative_offsets(tok, gen_ids: list[int]) -> tuple[str, list[int]]:
    """gen_text plus cumulative decoded char length after each token
    (monotonic-clamped; incremental decode, exact by construction)."""
    offsets, prev = [], 0
    for i in range(len(gen_ids)):
        cur = len(tok.decode(gen_ids[: i + 1], skip_special_tokens=True))
        cur = max(cur, prev)
        offsets.append(cur)
        prev = cur
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return text, offsets


def strip_generation(ids: list[int], stop_ids: set[int]) -> list[int]:
    """Cut the generated ids at the first stop/eos token."""
    for i, t in enumerate(ids):
        if t in stop_ids:
            return ids[:i]
    return ids


# ---------------------------------------------------------------------------
# merge mode
# ---------------------------------------------------------------------------

def merge_shards(args) -> None:
    from safetensors.numpy import save_file

    hs_dir = args.out_dir / "hidden_states"
    metas, arrays = [], {k: [] for k in args.hs_indices}
    gen_lines = []
    for s in range(args.num_shards):
        suf = shard_suffix(s, args.num_shards)
        meta_p = hs_dir / f"hs_meta{suf}.parquet"
        gen_p = args.out_dir / f"generations{suf}.jsonl"
        if not meta_p.exists() or not gen_p.exists():
            sys.exit(f"missing shard outputs for shard {s} ({meta_p})")
        metas.append(pd.read_parquet(meta_p))
        gen_lines.extend(gen_p.read_text().splitlines())
        for k in args.hs_indices:
            arrays[k].append(np.load(hs_dir / f"h_hs{k}{suf}.npy"))
    meta = pd.concat(metas, ignore_index=True)
    cat = {k: np.concatenate(a, axis=0) for k, a in arrays.items()}
    for k in args.hs_indices:
        assert len(meta) == cat[k].shape[0], f"row/array mismatch at hs{k}"
    order = meta.sort_values(
        ["question_id", "prompt_mode", "token_index"],
        kind="stable").index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)
    meta.to_parquet(hs_dir / "hs_meta.parquet", index=False)
    for k in args.hs_indices:
        save_file({"h": cat[k][order]},
                  str(hs_dir / f"layer_{k:02d}.safetensors"))
    recs = [json.loads(ln) for ln in gen_lines if ln.strip()]
    recs.sort(key=lambda r: r["question_id"])
    with open(args.out_dir / "generations.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[generate] merged {args.num_shards} shards -> {len(meta)} hs rows, "
          f"{len(recs)} generation records", flush=True)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def batched_generate(model, tok, device, prompt_ids: list[list[int]],
                     batch_size: int, max_new_tokens: int, greedy: bool,
                     num_return: int, temperature: float, top_p: float,
                     stop_ids: set[int], desc: str) -> list[list[list[int]]]:
    """Left-padded batched generation. Returns, per prompt, a list of
    num_return generated-id lists (stop-stripped)."""
    import torch

    n = len(prompt_ids)
    out: list[list[list[int]] | None] = [None] * n
    order = sorted(range(n), key=lambda i: len(prompt_ids[i]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0, done = time.perf_counter(), 0
    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        ids_list = [prompt_ids[i] for i in idxs]
        mx = max(len(x) for x in ids_list)
        inp = torch.tensor([[pad] * (mx - len(x)) + x for x in ids_list],
                           dtype=torch.long, device=device)
        att = torch.tensor([[0] * (mx - len(x)) + [1] * len(x)
                            for x in ids_list], dtype=torch.long, device=device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens,
                          pad_token_id=pad,
                          num_return_sequences=num_return)
        if greedy:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature,
                              top_p=top_p)
        with torch.no_grad():
            seqs = model.generate(inp, attention_mask=att, **gen_kwargs)
        gen_part = seqs[:, mx:].tolist()
        for j, i in enumerate(idxs):
            outs = gen_part[j * num_return:(j + 1) * num_return]
            out[i] = [strip_generation(g, stop_ids) for g in outs]
        done += len(idxs)
        if (start // batch_size) % 10 == 0:
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"[generate] {desc}: {done}/{n} prompts ({rate:.2f}/s)",
                  flush=True)
    return out  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=None,
                    help="defaults to <out_dir>/metadata.parquet")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--model_name_or_path", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--model_dtype",
                    choices=["bfloat16", "float16", "float32"],
                    default="bfloat16")
    ap.add_argument("--hs_indices", type=int, nargs="+", default=HS_INDICES,
                    help="hidden_states tuple indices (0=embeddings)")
    ap.add_argument("--direct_max_new_tokens", type=int, default=32)
    ap.add_argument("--cot_max_new_tokens", type=int, default=256)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--gen_batch_size", type=int, default=32,
                    help="prompts per greedy direct batch; sampled and cot "
                         "batches are scaled down from this")
    ap.add_argument("--fwd_batch_size", type=int, default=16)
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

    meta_path = args.metadata or (args.out_dir / "metadata.parquet")
    hs_dir = args.out_dir / "hidden_states"
    hs_dir.mkdir(parents=True, exist_ok=True)
    suf = shard_suffix(args.shard_idx, args.num_shards)
    gen_out = args.out_dir / f"generations{suf}.jsonl"
    if gen_out.exists() and not args.force:
        sys.exit(f"refusing to overwrite {gen_out}; pass --force")

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
    num_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    for k in args.hs_indices:
        if not (1 <= k <= num_layers):
            sys.exit(f"hs_idx {k} out of range [1,{num_layers}] "
                     f"(hidden_states tuple has {num_layers + 1} entries)")
    stop_ids = {t for t in [tok.eos_token_id,
                            tok.convert_tokens_to_ids("<|im_end|>")]
                if t is not None and t >= 0}

    # ---- render chat prompts ----------------------------------------------
    def render(question: str, family: str, mode: str) -> list[int]:
        msg = build_user_message(question, family, mode)
        ids = tok.apply_chat_template([{"role": "user", "content": msg}],
                                      add_generation_prompt=True,
                                      tokenize=True, return_dict=False)
        if not isinstance(ids, list):  # BatchEncoding on some versions
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    direct_prompts = [render(r.question, r.family, "direct")
                      for r in inst.itertuples()]
    qa_mask = (~inst.is_control).to_numpy()
    qa_rows = np.flatnonzero(qa_mask)
    cot_prompts = [render(inst.question.iloc[i], inst.family.iloc[i], "cot")
                   for i in qa_rows]
    print(f"[generate] shard {args.shard_idx}/{args.num_shards}: "
          f"{len(inst)} instances ({len(qa_rows)} QA with CoT arm), "
          f"hs_indices={args.hs_indices}", flush=True)

    # ---- generation --------------------------------------------------------
    bs = max(1, args.gen_batch_size)
    n_s = args.n_samples
    d_greedy = batched_generate(model, tok, device, direct_prompts, bs,
                                args.direct_max_new_tokens, True, 1, 0.0, 1.0,
                                stop_ids, "direct greedy")
    d_sampled = batched_generate(model, tok, device, direct_prompts,
                                 max(1, bs // n_s),
                                 args.direct_max_new_tokens, False, n_s,
                                 args.temperature, args.top_p, stop_ids,
                                 "direct sampled")
    c_greedy = batched_generate(model, tok, device, cot_prompts,
                                max(1, bs // 2),
                                args.cot_max_new_tokens, True, 1, 0.0, 1.0,
                                stop_ids, "cot greedy")
    c_sampled = batched_generate(model, tok, device, cot_prompts,
                                 max(1, bs // (2 * n_s)),
                                 args.cot_max_new_tokens, False, n_s,
                                 args.temperature, args.top_p, stop_ids,
                                 "cot sampled")
    cot_of = {int(g): j for j, g in enumerate(qa_rows)}

    # ---- generations.jsonl -------------------------------------------------
    records = []
    for i, r in enumerate(inst.itertuples()):
        rec = {
            "question_id": r.question_id, "fact_id": r.fact_id,
            "family": r.family, "is_control": bool(r.is_control),
            "gold_answer": r.gold_answer,
            "direct": {
                "prompt_token_count": len(direct_prompts[i]),
                "greedy_text": tok.decode(d_greedy[i][0],
                                          skip_special_tokens=True),
                "greedy_n_tokens": len(d_greedy[i][0]),
                "sampled_texts": [tok.decode(g, skip_special_tokens=True)
                                  for g in d_sampled[i]],
            },
            "cot": None,
        }
        if i in cot_of:
            j = cot_of[i]
            rec["cot"] = {
                "prompt_token_count": len(cot_prompts[j]),
                "greedy_text": tok.decode(c_greedy[j][0],
                                          skip_special_tokens=True),
                "greedy_n_tokens": len(c_greedy[j][0]),
                "sampled_texts": [tok.decode(g, skip_special_tokens=True)
                                  for g in c_sampled[j]],
            }
        records.append(rec)
    with open(gen_out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[generate] wrote {gen_out} ({len(records)} records)", flush=True)

    # ---- hidden-state rows (greedy runs only) ------------------------------
    rows: list[dict] = []
    sequences: list[list[int]] = []
    for i, r in enumerate(inst.itertuples()):
        runs = [("direct", direct_prompts[i], d_greedy[i][0])]
        if i in cot_of:
            j = cot_of[i]
            runs.append(("cot", cot_prompts[j], c_greedy[j][0]))
        for mode, p_ids, g_ids in runs:
            gen_text, offsets = cumulative_offsets(tok, g_ids)
            positions = compute_positions(len(p_ids), len(g_ids), gen_text,
                                          offsets, mode)
            has_marker = any(p["position_name"] == "first_final_answer_token"
                             for p in positions)
            for p in positions:
                rows.append({
                    "row_id": (f"{r.question_id}::{mode}::"
                               f"{p['position_name']}{p['position_rank']}"),
                    "question_id": r.question_id, "fact_id": r.fact_id,
                    "family": r.family, "is_control": bool(r.is_control),
                    "prompt_mode": mode,
                    "position_name": p["position_name"],
                    "position_rank": p["position_rank"],
                    "token_index": p["token_index"],
                    "prompt_token_count": len(p_ids),
                    "n_gen_tokens": len(g_ids),
                    "has_final_marker": has_marker,
                    "_seq": len(sequences),
                })
            sequences.append(p_ids + list(g_ids))

    n_rows = len(rows)
    out = {k: np.empty((n_rows, hidden), dtype=np.float16)
           for k in args.hs_indices}
    by_seq: dict[int, list[int]] = {}
    for ri, row in enumerate(rows):
        by_seq.setdefault(row["_seq"], []).append(ri)
    seq_order = sorted(by_seq, key=lambda s: len(sequences[s]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0, done = time.perf_counter(), 0
    for start in range(0, len(seq_order), args.fwd_batch_size):
        sidx = seq_order[start:start + args.fwd_batch_size]
        ids_list = [sequences[s] for s in sidx]
        mx = max(len(x) for x in ids_list)
        inp = torch.tensor([x + [pad] * (mx - len(x)) for x in ids_list],
                           dtype=torch.long, device=device)
        att = torch.tensor([[1] * len(x) + [0] * (mx - len(x))
                            for x in ids_list], dtype=torch.long, device=device)
        with torch.no_grad():
            o = model(inp, attention_mask=att, output_hidden_states=True,
                      use_cache=False)
        for b, s in enumerate(sidx):
            for ri in by_seq[s]:
                ti = rows[ri]["token_index"]
                assert ti < len(sequences[s]), \
                    f"position {ti} out of range for {rows[ri]['row_id']}"
                for k in args.hs_indices:
                    out[k][ri] = (o.hidden_states[k][b, ti]
                                  .detach().to(torch.float16).cpu().numpy())
        done += len(sidx)
        if (start // args.fwd_batch_size) % 20 == 0:
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"[extract] {done}/{len(seq_order)} sequences "
                  f"({rate:.2f}/s)", flush=True)

    meta = pd.DataFrame(rows, columns=HS_META_COLS + ["_seq"])
    meta = meta.drop(columns=["_seq"])
    meta.to_parquet(hs_dir / f"hs_meta{suf}.parquet", index=False)
    for k in args.hs_indices:
        np.save(hs_dir / f"h_hs{k}{suf}.npy", out[k])

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_name_or_path": args.model_name_or_path,
        "model_dtype": args.model_dtype,
        "num_hidden_layers": num_layers,
        "hidden_size": hidden,
        "hs_indices": args.hs_indices,
        "block_indices": [block_idx(k) for k in args.hs_indices],
        "generation": {
            "direct_max_new_tokens": args.direct_max_new_tokens,
            "cot_max_new_tokens": args.cot_max_new_tokens,
            "n_samples": args.n_samples,
            "temperature": args.temperature, "top_p": args.top_p,
        },
        "seed": args.seed + args.shard_idx,
        "n_instances": len(inst),
        "n_hs_rows": n_rows,
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
    }
    (args.out_dir / f"generate_manifest{suf}.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[generate] shard done: {n_rows} hs rows x "
          f"{len(args.hs_indices)} layers", flush=True)


if __name__ == "__main__":
    main()
