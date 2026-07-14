"""transition_operator_v0 Stage 1b: extract baseline representations per transition.

One teacher-forced pass per branch (correct, wrong) of each fork, storing the raw
boundary/step activations the Stage-1 baselines are built from, at L20 (primary) and
L28. Per transition and layer:
  S_prev   h_L at the boundary "\n" before the step (S_{t-1})
  S_last   h_L at the last step token (S_t)
  H_mean   mean over step-token h_L
  H_max    max over step-token h_L

The five baselines in the plan are derived downstream from these:
  S_t | S_t - S_{t-1} | [S_{t-1}; S_t] | mean-pool(H_t) | max-pool(H_t).

Output (--out_dir, default runs/transition_operator/stage1): reps_layer_<L>.npz with
arrays S_prev/S_last/H_mean/H_max (n_transitions, hidden) in the SAME row order as
rows.parquet (fork_id, branch, question), matching step_labels.parquet on (fork_id,
branch).

Sharded like s4_contrib_extract_forks: --shard_idx / --num_shards over transitions,
then --merge.

  python scripts/transition_operator/to_extract_baselines.py \
    --forks runs/transition_operator/forks.jsonl --layers 20 28 --shard_idx 0 --num_shards 4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.transition_operator import sep_join_ids  # noqa: E402


def transitions(forks: list[dict]):
    for fk in forks:
        for branch in ("correct", "wrong"):
            yield fk, branch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forks", type=Path,
                    default=Path("runs/transition_operator/forks.jsonl"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/transition_operator/stage1"))
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 28])
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        _merge(args)
        return

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype,
        local_files_only=args.local_files_only).to(device).eval()

    forks = [json.loads(l) for l in open(args.forks) if l.strip()]
    items = list(transitions(forks))
    shard = items[args.shard_idx::args.num_shards]
    print(f"[extract] shard {args.shard_idx}/{args.num_shards}: {len(shard)} "
          f"transitions on {device}", flush=True)

    meta = []
    acc = {L: {k: [] for k in ("S_prev", "S_last", "H_mean", "H_max")}
           for L in args.layers}
    for j, (fk, branch) in enumerate(shard):
        pre_ids = sep_join_ids(tok, [fk["question"], *fk["prefix_steps"]])
        step_core = tok(fk[branch], add_special_tokens=False)["input_ids"]
        if not step_core:
            step_core = [tok.eos_token_id]
        full = pre_ids + step_core
        if len(full) > args.max_seq_len:
            full = full[-args.max_seq_len:]
            # re-derive boundary/step spans after left-truncation
            b = len(full) - len(step_core) - 1
        else:
            b = len(pre_ids) - 1
        step_lo, step_hi = b + 1, len(full)  # step tokens [step_lo, step_hi)
        ids = torch.tensor([full], device=device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        for L in args.layers:
            h = out.hidden_states[L][0].float().cpu()  # (seq, hidden)
            step = h[step_lo:step_hi]
            acc[L]["S_prev"].append(h[b].numpy())
            acc[L]["S_last"].append(h[step_hi - 1].numpy())
            acc[L]["H_mean"].append(step.mean(0).numpy())
            acc[L]["H_max"].append(step.max(0).values.numpy())
        meta.append({"fork_id": fk["fork_id"], "branch": branch,
                     "question": fk["question"], "step_index": fk["step_index"]})
        if (j + 1) % 100 == 0:
            print(f"[extract] {j + 1}/{len(shard)}", flush=True)

    tag = f"shard{args.shard_idx}of{args.num_shards}"
    for L in args.layers:
        np.savez(args.out_dir / f"reps_layer_{L}_{tag}.npz",
                 **{k: np.stack(v).astype(np.float32) for k, v in acc[L].items()})
    (args.out_dir / f"rows_{tag}.json").write_text(json.dumps(meta))
    print(f"[extract] wrote shard {tag}", flush=True)


def _merge(args) -> None:
    import glob
    for L in args.layers:
        parts = sorted(glob.glob(str(args.out_dir / f"reps_layer_{L}_shard*of*.npz")))
        n_shards = int(parts[0].split("of")[-1].split(".")[0])
        # interleave shards back into original transition order
        shard_arrays = [np.load(p) for p in parts]
        rows_parts = [json.loads((args.out_dir /
                      f"rows_shard{i}of{n_shards}.json").read_text())
                      for i in range(n_shards)]
        keys = ("S_prev", "S_last", "H_mean", "H_max")
        total = sum(a[keys[0]].shape[0] for a in shard_arrays)
        hidden = shard_arrays[0][keys[0]].shape[1]
        merged = {k: np.zeros((total, hidden), np.float32) for k in keys}
        rows: list = [None] * total
        for i, (arr, rp) in enumerate(zip(shard_arrays, rows_parts)):
            for local, glob_i in enumerate(range(i, total, n_shards)):
                for k in keys:
                    merged[k][glob_i] = arr[k][local]
                rows[glob_i] = rp[local]
        np.savez(args.out_dir / f"reps_layer_{L}.npz", **merged)
        (args.out_dir / "rows.json").write_text(json.dumps(rows))
    (args.out_dir / "extract_manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name_or_path, "layers": args.layers,
        "n_transitions": total}, indent=2))
    print(f"[merge] wrote reps_layer_*.npz ({total} transitions)")


if __name__ == "__main__":
    main()
