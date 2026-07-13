"""attention_routing_v0: attention-feature extraction for matched
correct/incorrect forks (runs/contrib_cluster/forks.jsonl).

For each fork, two teacher-forced passes over
    "\n".join([question, *prefix_steps, candidate])
with candidate in {correct, wrong} (same text assembly as
s4_contrib_extract_forks.py). For every layer and query head, routing
features of the candidate-token attention rows are computed (region masses,
entropy, mean distance, top-5 concentration) at three read positions
(first/last/mean candidate token). No probe is trained here.

Stage-1 validation is built in: on the first processed pass the script
asserts attention tensor shape (query heads under GQA), row normalization,
causal masking, and candidate-span alignment, and records the measured
values in the manifest.

Outputs (in --out_dir, default runs/attention_routing/forks_attn/):
  metadata.parquet   one row per (fork, role), roles correct/wrong
  features.npy       (n_rows, n_layers, n_heads, n_features, n_reads) float16
  inspect/<fork_id>_<role>.npz   head-mean candidate attention rows + tokens
                                 for the first --n_inspect forks
  extract_manifest.json

Sharded like s4_contrib_extract_forks.py (whole forks per shard, --merge to
combine). Requires attn_implementation="eager"; batch size is fixed at 1
because full attention tensors for all layers are materialized per pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_prm800k_prestudy import git_commit, read_jsonl  # noqa: E402
from src.analysis.attention_routing import (  # noqa: E402
    FEATURES,
    N_FEATURES,
    N_READS,
    READS,
    assign_token_regions,
    attention_step_features,
    build_fork_segments,
    candidate_token_span,
    check_causal,
    check_row_normalized,
    count_numbers,
    count_operators,
    region_token_counts,
)

ROLES = ("correct", "wrong")
META_COLS = [
    "row_id", "fork_id", "role", "label", "question_hash", "step_index",
    "n_prefix_steps", "seq_len", "cand_token_len", "prefix_token_len",
    "question_token_len", "prev1_token_len", "older_token_len",
    "other_token_len", "cand_char_len", "n_numbers", "n_operators",
    "mean_logprob", "mean_pred_entropy",
]


def shard_suffix(shard_idx: int, num_shards: int) -> str:
    return "" if num_shards == 1 else f"_shard{shard_idx:02d}"


def h16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def merge_shards(args) -> None:
    metas, arrays = [], []
    for s in range(args.num_shards):
        suf = shard_suffix(s, args.num_shards)
        meta_p = args.out_dir / f"metadata{suf}.parquet"
        if not meta_p.exists():
            sys.exit(f"missing shard output {meta_p}")
        metas.append(pd.read_parquet(meta_p))
        arrays.append(np.load(args.out_dir / f"features{suf}.npy"))
    meta = pd.concat(metas, ignore_index=True)
    feats = np.concatenate(arrays, axis=0)
    assert len(meta) == feats.shape[0], "row/array mismatch"
    meta = meta.copy()
    meta["_r"] = meta["role"].map({r: i for i, r in enumerate(ROLES)})
    order = meta.sort_values(["fork_id", "_r"]).index.to_numpy()
    meta = meta.loc[order].drop(columns="_r").reset_index(drop=True)
    meta["row_id"] = np.arange(len(meta))
    meta.to_parquet(args.out_dir / "metadata.parquet", index=False)
    np.save(args.out_dir / "features.npy", feats[order])
    print(f"[ar-extract] merged {args.num_shards} shards -> {len(meta)} rows "
          f"({meta['fork_id'].nunique()} forks)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forks", type=Path,
                    default=Path("runs/contrib_cluster/forks.jsonl"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/attention_routing/forks_attn"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--max_seq_len", type=int, default=4096,
                    help="skip forks whose longest role exceeds this")
    ap.add_argument("--model_dtype",
                    choices=["bfloat16", "float16", "float32"],
                    default="bfloat16",
                    help="bfloat16 by default: Qwen2.5-7B overflows float16 "
                         "in the last attention layer and at the lm_head")
    ap.add_argument("--n_inspect", type=int, default=8,
                    help="save head-mean attention rows for the first N forks")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge_shards(args)
        return
    if not (0 <= args.shard_idx < args.num_shards):
        sys.exit(f"invalid shard config {args.shard_idx}/{args.num_shards}")

    suf = shard_suffix(args.shard_idx, args.num_shards)
    meta_path = args.out_dir / f"metadata{suf}.parquet"
    if meta_path.exists() and not args.force:
        sys.exit(f"refusing to overwrite {meta_path}; pass --force")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    if not tok.is_fast:
        sys.exit("fast tokenizer required (offset_mapping)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only,
        dtype=dtype_map[args.model_dtype], attn_implementation="eager")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device).eval()
    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    n_heads = int(cfg.num_attention_heads)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))

    forks = read_jsonl(args.forks)
    if args.limit is not None:
        forks = forks[: args.limit]
    inspect_ids = {f["fork_id"] for f in forks[: args.n_inspect]}
    if args.num_shards > 1:  # whole forks per shard
        forks = [f for i, f in enumerate(forks)
                 if i % args.num_shards == args.shard_idx]

    inspect_dir = args.out_dir / "inspect"
    inspect_dir.mkdir(exist_ok=True)

    rows: list[dict] = []
    feats_all: list[np.ndarray] = []
    n_skipped = 0
    stage1: dict | None = None
    t0 = time.time()

    for fi, fk in enumerate(forks):
        prepared = {}
        too_long = False
        for role in ROLES:
            cand = fk[role]
            text, segments = build_fork_segments(
                fk["question"], fk["prefix_steps"], cand)
            enc = tok(text, return_offsets_mapping=True,
                      return_tensors="np", add_special_tokens=True)
            ids = enc["input_ids"][0]
            if ids.shape[0] > args.max_seq_len:
                too_long = True
                break
            regions = assign_token_regions(
                [tuple(o) for o in enc["offset_mapping"][0]], segments)
            c0, c1 = candidate_token_span(regions)
            prepared[role] = (text, ids, regions, c0, c1)
        if too_long:
            n_skipped += 1
            continue

        for role in ROLES:
            text, ids, regions, c0, c1 = prepared[role]
            seq_len = int(ids.shape[0])
            input_ids = torch.tensor(ids[None], device=device)
            with torch.no_grad():
                out = model(input_ids, output_attentions=True, use_cache=False)

            feats = np.empty((n_layers, n_heads, N_FEATURES, N_READS),
                             dtype=np.float16)
            inspect_rows = [] if fk["fork_id"] in inspect_ids else None
            for li, layer_attn in enumerate(out.attentions):
                a = layer_attn[0, :, c0:c1, :].float().cpu().numpy()
                if not np.isfinite(a).all():
                    raise RuntimeError(
                        f"non-finite attention weights: fork "
                        f"{fk['fork_id']} role {role} layer {li} "
                        f"(dtype {args.model_dtype}; float16 overflows on "
                        "Qwen2.5-7B, use bfloat16 or float32)")
                if stage1 is None:
                    stage1 = {
                        "n_layers": len(out.attentions),
                        "attn_shape": list(layer_attn.shape),
                        "n_query_heads": int(layer_attn.shape[1]),
                        "n_kv_heads": n_kv_heads,
                        "max_row_sum_err": check_row_normalized(a),
                        "max_future_mass": check_causal(a, c0),
                        "fork_id": fk["fork_id"], "role": role,
                    }
                    assert stage1["n_layers"] == n_layers
                    assert stage1["n_query_heads"] == n_heads, \
                        "attention tensors must be per QUERY head under GQA"
                    assert stage1["max_row_sum_err"] < 2e-2, stage1
                    assert stage1["max_future_mass"] < 1e-4, stage1
                feats[li] = attention_step_features(a, regions, c0)
                if inspect_rows is not None:
                    inspect_rows.append(a.mean(axis=0).astype(np.float16))
            logits = out.logits[0].float()  # (L, V)
            tgt = torch.tensor(ids[c0:c1], device=device, dtype=torch.long)
            lp = torch.log_softmax(logits[c0 - 1:c1 - 1], dim=-1)
            mean_logprob = float(lp.gather(1, tgt[:, None]).mean())
            mean_pred_entropy = float(-(lp.exp() * lp).sum(-1).mean())
            del out, logits, lp
            if not (np.isfinite(mean_logprob)
                    and np.isfinite(mean_pred_entropy)):
                raise RuntimeError(
                    f"non-finite candidate logprob/entropy: fork "
                    f"{fk['fork_id']} role {role} "
                    f"(dtype {args.model_dtype}; float16 overflows on "
                    "Qwen2.5-7B, use bfloat16 or float32)")

            counts = region_token_counts(regions)
            rows.append({
                "row_id": len(rows),
                "fork_id": fk["fork_id"],
                "role": role,
                "label": 0 if role == "correct" else 1,
                "question_hash": h16(fk["question"]),
                "step_index": int(fk["step_index"]),
                "n_prefix_steps": len(fk["prefix_steps"]),
                "seq_len": seq_len,
                "cand_token_len": int(c1 - c0),
                "prefix_token_len": int(c0),
                "question_token_len": counts["question"],
                "prev1_token_len": counts["prev1"],
                "older_token_len": counts["older"],
                "other_token_len": counts["other"],
                "cand_char_len": len(fk[role]),
                "n_numbers": count_numbers(fk[role]),
                "n_operators": count_operators(fk[role]),
                "mean_logprob": mean_logprob,
                "mean_pred_entropy": mean_pred_entropy,
            })
            feats_all.append(feats)

            if inspect_rows is not None:
                np.savez_compressed(
                    inspect_dir / f"{fk['fork_id']}_{role}.npz",
                    attn_headmean=np.stack(inspect_rows),  # (layers, cand, L)
                    regions=regions,
                    c0=c0, c1=c1,
                    tokens=np.array([tok.decode([t]) for t in ids],
                                    dtype=str),
                )

        if (fi + 1) % 20 == 0:
            rate = (fi + 1) / (time.time() - t0)
            print(f"[ar-extract] shard {args.shard_idx}: {fi + 1}/{len(forks)} "
                  f"forks ({rate:.2f} forks/s, skipped {n_skipped})",
                  flush=True)

    if not rows:
        sys.exit("no forks processed")
    meta = pd.DataFrame(rows, columns=META_COLS)
    meta.to_parquet(meta_path, index=False)
    np.save(args.out_dir / f"features{suf}.npy", np.stack(feats_all))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "model": args.model_name_or_path,
        "n_layers": n_layers, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
        "features": FEATURES, "reads": READS,
        "n_rows": len(meta), "n_forks": int(meta["fork_id"].nunique()),
        "n_skipped_too_long": n_skipped,
        "stage1_checks": stage1,
    }
    with open(args.out_dir / f"extract_manifest{suf}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ar-extract] shard {args.shard_idx}: wrote {len(meta)} rows "
          f"({meta['fork_id'].nunique()} forks, {n_skipped} skipped) "
          f"-> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
