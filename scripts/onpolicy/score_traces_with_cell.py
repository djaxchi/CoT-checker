#!/usr/bin/env python3
"""Score every step of already-generated traces with a trained cell.

tts_roster_v1's offline arm compares selection rules driven by our verifier
against ones driven by token confidence and by free orderings. Job A produces
the traces and the logprobs; this produces the missing third thing, the
verifier's per-step scores, without the representation-store pipeline that the
leaderboard harness needs.

It is the same shortcut that made the Phase 1 gate cheap (REPORT.md §20.16): the
trace text is on disk, so a forward pass recovers what is needed. Here the pass
runs through `online_bon.Checker`, which is the object the online arm already
scores with, so the offline and online numbers come from one implementation
rather than two that can drift.

Two details that would produce plausible wrong numbers if skipped, both
inherited from Checker and both deliberate:

**Scoring uses the verifier template, not the generation context.** These cells
were fitted on states taken under `verifier_prefix`, so that is what they are
applied under, whatever prompt the sampler ran. The generation-context variant
is a different experiment (§20.2 reports both) and is not what the offline
leaderboard rules were measured with.

**Rescaling statistics are required.** The head was fitted on rescaled states,
so `--prm_store` or `--stats` is mandatory; online_bon refuses without them, and
this refuses for the same reason.

Steps are split with the sampler's own `split_into_steps`, so a step here is the
step the generator wrote and the step the confidence encoder spanned.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl, write_jsonl  # noqa: E402
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from scripts.onpolicy.online_bon import Checker, cell_stats  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p.add_argument("--cell_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--model_dtype", choices=["float16", "bfloat16", "float32"],
                   default="bfloat16")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--prm_store", type=Path, default=None)
    p.add_argument("--stats", type=Path, default=None)
    p.add_argument("--stats_cache", type=Path, default=None)
    p.add_argument("--train_stem", type=str, default=None)
    p.add_argument("--assume_rescale", choices=["none", "zscore", "whiten"],
                   default="zscore")
    p.add_argument("--max_traces", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.out.exists() and not a.force:
        sys.exit(f"[score] refusing to overwrite {a.out}. Pass --force.")
    if not (a.prm_store or a.stats):
        sys.exit("[score] pass --prm_store or --stats. The head was fitted on "
                 "rescaled states; scoring without the statistics applies it to "
                 "unscaled ones and the numbers would be wrong without looking wrong.")

    rows: list[dict] = []
    for path in a.trajectories:
        rows.extend(read_jsonl(path))
    rows = [r for r in rows if r.get("gradeable") and (r.get("solution") or "").strip()]
    if a.max_traces:
        rows = rows[:a.max_traces]
    mine = rows[a.shard_idx::a.num_shards]
    print(f"[score] shard {a.shard_idx}/{a.num_shards}: {len(mine)} of {len(rows)} "
          f"traces, cell={a.cell_dir.name}", flush=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[a.model_dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path,
                                        local_files_only=a.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    backbone = AutoModelForCausalLM.from_pretrained(
        a.model_name_or_path, torch_dtype=dtype,
        local_files_only=a.local_files_only).to(device).eval()

    if a.stats:
        stats = json.loads(a.stats.read_text())
    else:
        res = json.loads((a.cell_dir / "results.json").read_text())
        stats = cell_stats(res, a.prm_store, None, a.stats_cache, {},
                           train_stem_override=a.train_stem,
                           assume_rescale=a.assume_rescale)
    checker = Checker(a.cell_dir, backbone, tok, a.layer, stats, device)

    out: list[dict] = []
    t0 = time.perf_counter()
    for i, r in enumerate(mine):
        steps = split_into_steps(r["solution"])
        if not steps:
            continue
        scores = [float(checker.score_steps(r["problem"], steps[:k], [steps[k]])[0])
                  for k in range(len(steps))]
        out.append({"traj_uid": r["traj_uid"], "problem_id": r.get("fork_id"),
                    "dataset": r.get("dataset"), "split": r.get("split"),
                    "correct": bool(r["correct"]), "n_steps": len(steps),
                    "cell": a.cell_dir.name, "scores": scores})
        if (i + 1) % 200 == 0 or i + 1 == len(mine):
            print(f"[score] {i+1}/{len(mine)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out, out)
    manifest = {"cell": str(a.cell_dir), "layer": a.layer, "n_traces": len(out),
                "shard_idx": a.shard_idx, "num_shards": a.num_shards,
                "model": a.model_name_or_path, "assume_rescale": a.assume_rescale,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "code_commit": git_commit()}
    a.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[score] wrote {a.out}  traces={len(out)}")


if __name__ == "__main__":
    main()
