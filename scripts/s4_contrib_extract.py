"""S4 contrib-cluster stage 2: hidden-state extraction over cumulative prefixes.

For every sampled trajectory (from s4_contrib_audit.py) build the prefixes

  p_0 = question
  p_i = question + "\n" + step_1 + "\n" + ... + step_i

run one forward pass per prefix (batched, sorted by token length to limit
padding) with output_hidden_states=True, and keep the hidden state of the last
non-padding token at each selected layer. hidden_states[l] is the residual
stream AFTER transformer layer l (index 0 = embeddings), matching the L20/L28
convention used everywhere else in this repo.

Outputs (in --out_dir, default runs/contrib_cluster/hidden_states/):
  metadata.parquet     one row per prefix (p_0 and each step), sorted by
                       (trajectory_id, step_index)
  h_layer_<L>.npy      (n_rows, hidden) float16, aligned with metadata
  extract_manifest.json

Sharded usage (TamIA, 4 GPUs then merge):
  python scripts/s4_contrib_extract.py --trajectories runs/contrib_cluster/trajectories.jsonl \
    --out_dir runs/contrib_cluster/hidden_states --model_name_or_path Qwen/Qwen2.5-7B \
    --local_files_only --layers 20 28 --shard_idx $i --num_shards 4
  python scripts/s4_contrib_extract.py --merge --num_shards 4 \
    --out_dir runs/contrib_cluster/hidden_states --layers 20 28
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_prm800k_prestudy import git_commit, read_jsonl  # noqa: E402

from src.analysis.contrib_cluster import build_prefixes, fit_steps_to_length  # noqa: E402

META_COLS = ["row_id", "trajectory_id", "step_index", "prefix_type", "question",
             "step_text", "prefix_hash", "prefix_char_len", "token_count",
             "num_steps_in_trajectory"]


def shard_suffix(shard_idx: int, num_shards: int) -> str:
    return "" if num_shards == 1 else f"_shard{shard_idx:02d}"


def sort_rows(meta: pd.DataFrame, arrays: dict[int, np.ndarray]):
    order = meta.sort_values(["trajectory_id", "step_index"]).index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)
    return meta, {li: a[order] for li, a in arrays.items()}


def merge_shards(args) -> None:
    metas, arrays = [], {li: [] for li in args.layers}
    for s in range(args.num_shards):
        suf = shard_suffix(s, args.num_shards)
        meta_p = args.out_dir / f"metadata{suf}.parquet"
        if not meta_p.exists():
            sys.exit(f"missing shard output {meta_p}")
        metas.append(pd.read_parquet(meta_p))
        for li in args.layers:
            arrays[li].append(np.load(args.out_dir / f"h_layer_{li}{suf}.npy"))
    meta = pd.concat(metas, ignore_index=True)
    cat = {li: np.concatenate(a, axis=0) for li, a in arrays.items()}
    for li in args.layers:
        assert len(meta) == cat[li].shape[0], f"row/array mismatch at layer {li}"
    meta, cat = sort_rows(meta, cat)
    meta.to_parquet(args.out_dir / "metadata.parquet", index=False)
    for li in args.layers:
        np.save(args.out_dir / f"h_layer_{li}.npy", cat[li])
    print(f"[extract] merged {args.num_shards} shards -> {len(meta)} rows "
          f"({meta['trajectory_id'].nunique()} trajectories)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", type=Path,
                    default=Path("runs/contrib_cluster/trajectories.jsonl"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/contrib_cluster/hidden_states"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 28],
                    help="hidden_states indices (project standard L20/L28 for the "
                         "7B; last=28); --all_layers overrides")
    ap.add_argument("--all_layers", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only the first N trajectories (smoke runs)")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true",
                    help="merge shard outputs instead of encoding")
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
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            dtype=dtype_map[args.model_dtype])
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            torch_dtype=dtype_map[args.model_dtype])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    num_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    if args.all_layers:
        args.layers = list(range(num_layers + 1))
    for li in args.layers:
        if not (0 <= li <= num_layers):
            sys.exit(f"layer {li} out of range [0,{num_layers}]")

    trajectories = read_jsonl(args.trajectories)
    if args.limit is not None:
        trajectories = trajectories[: args.limit]
    if args.num_shards > 1:  # shard by trajectory so each stays whole
        trajectories = [t for i, t in enumerate(trajectories)
                        if i % args.num_shards == args.shard_idx]

    # ---- build one row per prefix ----------------------------------------
    rows: list[dict] = []
    encodings: list[list[int]] = []
    n_skipped_overlength = 0
    n_truncated_overlength = 0
    for traj in trajectories:
        steps = traj["steps"]
        prefixes = build_prefixes(traj["question"], steps)
        enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prefixes]
        if len(enc[-1]) > args.max_seq_len:
            # Fallback (audit should already have fitted): drop trailing steps
            # until p_T fits; the kept prefixes stay a continuous trajectory.
            fitted = fit_steps_to_length(
                lambda p: len(tok(p, add_special_tokens=False)["input_ids"]),
                traj["question"], steps, args.max_seq_len)
            if fitted is None:
                n_skipped_overlength += 1
                continue
            n_truncated_overlength += 1
            steps = fitted
            prefixes = prefixes[: len(steps) + 1]
            enc = enc[: len(steps) + 1]
        for i, (p, ids) in enumerate(zip(prefixes, enc)):
            rows.append({
                "row_id": f"{traj['trajectory_id']}::p{i}",
                "trajectory_id": traj["trajectory_id"],
                "step_index": i,
                "prefix_type": "p0" if i == 0 else "step",
                "question": traj["question"],
                "step_text": "" if i == 0 else steps[i - 1],
                "prefix_hash": hashlib.sha1(p.encode("utf-8")).hexdigest()[:16],
                "prefix_char_len": len(p),
                "token_count": len(ids),
                "num_steps_in_trajectory": len(steps),
            })
            encodings.append(ids)
    n = len(rows)
    print(f"[extract] shard {args.shard_idx}/{args.num_shards}: "
          f"{len(trajectories)} trajectories -> {n} prefixes "
          f"({n_truncated_overlength} truncated / {n_skipped_overlength} skipped "
          f"overlength) layers={args.layers}", flush=True)

    # ---- batch by token length to limit padding ---------------------------
    order = sorted(range(n), key=lambda i: len(encodings[i]))
    out = {li: np.empty((n, hidden), dtype=np.float16) for li in args.layers}
    t0 = time.perf_counter()
    done = 0
    for start in range(0, n, args.batch_size):
        idxs = order[start:start + args.batch_size]
        ids_list = [encodings[i] for i in idxs]
        mx = max(len(x) for x in ids_list)
        inp = torch.tensor([x + [pad] * (mx - len(x)) for x in ids_list],
                           dtype=torch.long, device=device)
        att = torch.tensor([[1] * len(x) + [0] * (mx - len(x)) for x in ids_list],
                           dtype=torch.long, device=device)
        with torch.no_grad():
            o = model(inp, attention_mask=att, output_hidden_states=True,
                      use_cache=False)
        last = att.sum(dim=1) - 1  # last non-padding token per row
        for li in args.layers:
            hs = o.hidden_states[li]
            vec = hs[torch.arange(len(idxs), device=device), last]
            out[li][idxs] = vec.detach().to(torch.float16).cpu().numpy()
        done += len(idxs)
        if (start // args.batch_size) % 50 == 0:
            rate = done / (time.perf_counter() - t0)
            print(f"[extract] {done}/{n} prefixes ({rate:.1f}/s)", flush=True)

    meta = pd.DataFrame(rows, columns=META_COLS)
    meta, out = sort_rows(meta, out)
    meta.to_parquet(meta_path, index=False)
    for li in args.layers:
        np.save(args.out_dir / f"h_layer_{li}{suf}.npy", out[li])

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_name_or_path": args.model_name_or_path,
        "model_dtype": args.model_dtype,
        "num_hidden_layers": num_layers,
        "hidden_size": hidden,
        "layers": args.layers,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "n_trajectories": len(trajectories),
        "n_prefix_rows": n,
        "n_truncated_overlength": n_truncated_overlength,
        "n_skipped_overlength": n_skipped_overlength,
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (args.out_dir / f"extract_manifest{suf}.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[extract] wrote {meta_path} + {len(args.layers)} layer arrays "
          f"in {manifest['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
