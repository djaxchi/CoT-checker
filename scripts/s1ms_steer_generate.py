#!/usr/bin/env python3
"""S3 Stage-5 Tier-1: steer the residual stream DURING generation and grade the outcome.

For each steering condition (layer, direction, signed alpha) we generate the model's own
solutions to a set of problems and grade each by final-answer match. A forward hook on
decoder block (layer-1) adds ``alpha * s_layer * unit_dir`` to that layer's residual at every
token, every decode step. Directions come from build_steering_directions.py and are oriented
toward-correct for the correctness treatments, so:

    alpha > 0  -> steer toward correct   (does it REPAIR wrong trajectories?)
    alpha < 0  -> steer toward incorrect (does it CORRUPT good trajectories?)

The alpha=0 baseline is a single shared, hook-free condition; with a fixed per-problem seed it
is the matched control every steered condition is paired against (compare by sample_idx).

This is the data-parallel worker: it processes only the work units (condition, problem) whose
flat index mod --n_shards equals --shard_idx, so the launcher runs 4 copies (one per H100) and
merges the shard JSONLs before analysis.

Outputs (per shard):
  {stem}_shard{k}.jsonl        one row per generated trajectory:
      cond_id, layer, direction, alpha, fork_id, sample_idx, pred, correct, gradeable,
      gen_len, n_steps, solution
  {stem}_shard{k}_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl  # noqa: E402
from scripts.generate_onpolicy_steps import (  # noqa: E402
    build_prompt, split_into_steps, unique_problems,
)
from scripts.s1ms_steer_forks import get_decoder_layers  # noqa: E402
from src.eval.math_grade import grade  # noqa: E402


def build_conditions(npz_by_layer: dict, directions: list[str], alphas: list[float]) -> list[dict]:
    """Flat condition list: one shared baseline + (layer, direction, signed nonzero alpha)."""
    conds = [{"cond_id": "baseline", "layer": None, "direction": "baseline", "alpha": 0.0}]
    nz = sorted({a for a in alphas if abs(a) > 1e-9})
    for L in sorted(npz_by_layer):
        names = npz_by_layer[L]["names"]
        for d in directions:
            if d not in names:
                print(f"[gen] WARN direction {d!r} absent at L{L}; skipping", flush=True)
                continue
            for a in nz:
                conds.append({"cond_id": f"L{L}:{d}:{a:g}", "layer": L,
                              "direction": d, "alpha": float(a)})
    return conds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fork_items", type=Path, required=True, help="problem source (forks jsonl)")
    p.add_argument("--directions_npz", type=Path, nargs="+", required=True,
                   help="one directions_L{idx}.npz per injection layer")
    p.add_argument("--directions", type=str, nargs="+", default=None,
                   help="subset of direction names to run (default: all in the npz)")
    p.add_argument("--alphas", type=float, nargs="+", required=True,
                   help="signed alphas; 0 is always added as the shared baseline")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--stem", type=str, default="steer_gen")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_problems", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    import torch
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.stem}_shard{args.shard_idx}.jsonl"
    if out_path.exists() and not args.force:
        sys.exit(f"[gen] refusing to overwrite {out_path}; pass --force")

    # ---- directions per layer -------------------------------------------
    npz_by_layer: dict[int, dict] = {}
    for path in args.directions_npz:
        z = np.load(path, allow_pickle=True)
        L = int(z["layer_index"])
        npz_by_layer[L] = {
            "names": [str(s) for s in z["names"]],
            "vectors": z["vectors"].astype(np.float32),
            "s_layer": float(z["s_layer"]),
            "by_name": {str(s): i for i, s in enumerate(z["names"])},
        }
    all_names = sorted({n for d in npz_by_layer.values() for n in d["names"]})
    directions = args.directions or all_names
    conds = build_conditions(npz_by_layer, directions, args.alphas)

    # ---- model + steering hook ------------------------------------------
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.model_dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    layers = get_decoder_layers(model)

    steer = {"vec": None}

    def hook(_m, _i, out):
        if steer["vec"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + steer["vec"],) + tuple(out[1:])
        return out + steer["vec"]

    hook_state = {"layer": None, "handle": None}

    def set_condition(cond: dict) -> None:
        L = cond["layer"]
        if L != hook_state["layer"]:
            if hook_state["handle"] is not None:
                hook_state["handle"].remove()
            hook_state["handle"] = (None if L is None
                                    else layers[L - 1].register_forward_hook(hook))
            hook_state["layer"] = L
        if L is None or cond["alpha"] == 0.0:
            steer["vec"] = None
        else:
            info = npz_by_layer[L]
            u = info["vectors"][info["by_name"][cond["direction"]]]
            v = (cond["alpha"] * info["s_layer"]) * u
            steer["vec"] = torch.tensor(v, device=device, dtype=dtype)

    # ---- problems + sharded work units ----------------------------------
    problems = unique_problems(read_jsonl(args.fork_items))
    if args.max_problems > 0:
        problems = problems[:args.max_problems]
    # flat unit index over (condition, problem); this shard takes idx % n_shards == shard_idx
    units = [(ci, pi) for ci in range(len(conds)) for pi in range(len(problems))]
    mine = [(ci, pi) for k, (ci, pi) in enumerate(units)
            if k % args.n_shards == args.shard_idx]
    # group by condition to set the hook once per condition
    by_cond: dict[int, list[int]] = {}
    for ci, pi in mine:
        by_cond.setdefault(ci, []).append(pi)
    print(f"[gen] shard {args.shard_idx}/{args.n_shards}: {len(conds)} conds x "
          f"{len(problems)} problems -> {len(mine)} units ({args.n_samples} samples each)",
          flush=True)

    gen_base = dict(max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.n_samples,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if args.temperature > 0:
        gen_base.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_base.update(do_sample=False)

    rows: list[dict] = []
    t0 = time.perf_counter()
    done = 0
    for ci in sorted(by_cond):
        cond = conds[ci]
        set_condition(cond)
        for pi in by_cond[ci]:
            prob = problems[pi]
            enc = tok(build_prompt(prob["problem"]), return_tensors="pt").to(device)
            plen = enc["input_ids"].shape[1]
            torch.manual_seed(args.seed + pi)        # fixed per-problem -> paired across conds
            with torch.no_grad():
                gen = model.generate(**enc, **gen_base)
            for s in range(gen.shape[0]):
                new_ids = gen[s, plen:]
                text = tok.decode(new_ids, skip_special_tokens=True)
                g = grade(text, prob["ground_truth_answer"])
                rows.append({
                    "cond_id": cond["cond_id"], "layer": cond["layer"],
                    "direction": cond["direction"], "alpha": cond["alpha"],
                    "fork_id": prob["fork_id"], "sample_idx": s,
                    "pred": g["pred"], "correct": bool(g["correct"]),
                    "gradeable": bool(g["gradeable"]),
                    "gen_len": int((new_ids != tok.pad_token_id).sum().item()),
                    "n_steps": len(split_into_steps(text)),
                    "solution": text,
                })
            done += 1
            if done % 25 == 0:
                gr = np.mean([r["correct"] for r in rows]) if rows else 0.0
                print(f"[gen] {done}/{len(mine)} units  ({time.perf_counter()-t0:.0f}s)  "
                      f"rolling correct-rate={gr:.3f}", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "run_name": args.run_name, "model": args.model_name_or_path,
        "shard_idx": args.shard_idx, "n_shards": args.n_shards,
        "n_conditions": len(conds), "n_problems": len(problems),
        "n_samples": args.n_samples, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "alphas": sorted(set(args.alphas) | {0.0}),
        "directions": directions, "layers": sorted(npz_by_layer),
        "n_rows": len(rows), "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }
    (args.out_dir / f"{args.stem}_shard{args.shard_idx}_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[gen] wrote {out_path}  rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
