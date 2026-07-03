"""S4 contrib-cluster: hidden-state extraction for matched correct/incorrect
forks (from s4_contrib_forks.py).

For each fork, four prefixes are encoded (last non-padding token, selected
layers), mirroring the trajectory extraction format exactly:

  p0      = question
  prefix  = question + "\n" + prefix_steps        (== p0 for step-1 forks)
  correct = prefix + "\n" + correct_step
  wrong   = prefix + "\n" + wrong_step

so state/qres/contribution for both continuations use the same h_p0/h_prefix.

Outputs (in --out_dir, default runs/contrib_cluster/forks_hidden/):
  metadata.parquet   one row per (fork, role), roles p0/prefix/correct/wrong
  h_layer_<L>.npy    (n_rows, hidden) float16, aligned with metadata
  extract_manifest.json

Sharded like s4_contrib_extract.py (whole forks per shard, --merge to combine).
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_prm800k_prestudy import git_commit, read_jsonl  # noqa: E402

ROLES = ("p0", "prefix", "correct", "wrong")
META_COLS = ["row_id", "fork_id", "role", "step_index", "question", "step_text",
             "n_prefix_steps", "token_count"]


def shard_suffix(shard_idx: int, num_shards: int) -> str:
    return "" if num_shards == 1 else f"_shard{shard_idx:02d}"


def fork_texts(fk: dict) -> dict[str, str]:
    base = "\n".join([fk["question"], *fk["prefix_steps"]])
    return {
        "p0": fk["question"],
        "prefix": base,
        "correct": base + "\n" + fk["correct"],
        "wrong": base + "\n" + fk["wrong"],
    }


def sort_rows(meta: pd.DataFrame, arrays: dict[int, np.ndarray]):
    meta = meta.copy()
    meta["_r"] = meta["role"].map({r: i for i, r in enumerate(ROLES)})
    order = meta.sort_values(["fork_id", "_r"]).index.to_numpy()
    meta = meta.loc[order].drop(columns="_r").reset_index(drop=True)
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
    print(f"[forks-extract] merged {args.num_shards} shards -> {len(meta)} rows "
          f"({meta['fork_id'].nunique()} forks)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forks", type=Path, default=Path("runs/contrib_cluster/forks.jsonl"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/contrib_cluster/forks_hidden"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 28])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
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
    for li in args.layers:
        if not (0 <= li <= num_layers):
            sys.exit(f"layer {li} out of range [0,{num_layers}]")

    forks = read_jsonl(args.forks)
    if args.limit is not None:
        forks = forks[: args.limit]
    if args.num_shards > 1:  # whole forks per shard
        forks = [f for i, f in enumerate(forks)
                 if i % args.num_shards == args.shard_idx]

    rows: list[dict] = []
    encodings: list[list[int]] = []
    n_skipped_overlength = 0
    for fk in forks:
        texts = fork_texts(fk)
        enc = {r: tok(texts[r], add_special_tokens=False)["input_ids"] for r in ROLES}
        if max(len(v) for v in enc.values()) > args.max_seq_len:
            n_skipped_overlength += 1  # builder already filtered; safety net
            continue
        for r in ROLES:
            rows.append({
                "row_id": f"{fk['fork_id']}::{r}",
                "fork_id": fk["fork_id"],
                "role": r,
                "step_index": int(fk["step_index"]),
                "question": fk["question"],
                "step_text": {"p0": "", "prefix": "",
                              "correct": fk["correct"], "wrong": fk["wrong"]}[r],
                "n_prefix_steps": len(fk["prefix_steps"]),
                "token_count": len(enc[r]),
            })
            encodings.append(enc[r])
    n = len(rows)
    print(f"[forks-extract] shard {args.shard_idx}/{args.num_shards}: "
          f"{len(forks)} forks -> {n} rows "
          f"({n_skipped_overlength} skipped overlength) layers={args.layers}",
          flush=True)

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
        last = att.sum(dim=1) - 1
        for li in args.layers:
            vec = o.hidden_states[li][torch.arange(len(idxs), device=device), last]
            out[li][idxs] = vec.detach().to(torch.float16).cpu().numpy()
        done += len(idxs)
        if (start // args.batch_size) % 50 == 0:
            rate = done / (time.perf_counter() - t0)
            print(f"[forks-extract] {done}/{n} rows ({rate:.1f}/s)", flush=True)

    meta = pd.DataFrame(rows, columns=META_COLS)
    meta, out = sort_rows(meta, out)
    meta.to_parquet(meta_path, index=False)
    for li in args.layers:
        np.save(args.out_dir / f"h_layer_{li}{suf}.npy", out[li])
    (args.out_dir / f"extract_manifest{suf}.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_name_or_path": args.model_name_or_path,
        "layers": args.layers,
        "seed": args.seed,
        "n_forks": len(forks),
        "n_rows": n,
        "n_skipped_overlength": n_skipped_overlength,
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }, indent=2))
    print(f"[forks-extract] wrote {meta_path} in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
