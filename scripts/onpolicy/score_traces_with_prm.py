#!/usr/bin/env python3
"""Score saved trajectories with Qwen2.5-Math-PRM-7B, in our own score schema.

Every consumer in this codebase reads `scores` as **suspicion**: higher means the
step is more likely wrong, and `tts_build_frontier.py` hardcodes
`quality_from(vals, higher_is_better=False)`. Qwen2.5-Math-PRM-7B emits
P(step correct). This writes `1 - P(correct)`, so the PRM drops into the existing
frontier with no change anywhere downstream, and the reward a weighted vote wants
falls out as `1 - suspicion = P(correct)`.

Getting that backwards produces a mirror-image curve that looks perfectly
ordinary, which is the §20.16 failure mode, so the orientation gate below runs
before any scoring is written.

**The step segmentation is not ours to choose.** The PRM was trained with its own
chat template and an `<extra_0>` separator, but it has to score *the steps the
probe scored* or the head-to-head is comparing two different objects. Steps come
from `split_into_steps`, the same function `score_traces_with_cell.py` uses, and
`--verify_against` asserts a per-trace 1:1 match against an existing probe score
file before anything is written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import (git_commit, read_jsonl,  # noqa: E402
                                                  write_jsonl)
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402

SEP = "<extra_0>"
SYSTEM = ("Please reason step by step, and put your final answer within "
          "\\boxed{}.")


def load_expected_steps(paths: list[Path]) -> dict[str, int]:
    """traj_uid -> n_steps over every given probe score file.

    Reading one shard only would arm the gate for about a quarter of the pool
    while reporting a clean zero, because probe shards and PRM shards stride the
    trajectories differently.
    """
    out: dict[str, int] = {}
    for path in paths:
        for r in read_jsonl(path):
            out[r["traj_uid"]] = int(r["n_steps"])
    return out


def step_rewards(logits: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """P(correct) at each separator position.

    The model exposes a 2-way head; the released reference implementation takes
    softmax over the last dimension and reads channel 1 as the positive class.
    """
    probs = torch.softmax(logits, dim=-1) * mask.unsqueeze(-1)
    return probs[mask.bool()][:, 1].tolist()


@torch.no_grad()
def score_one(model, tok, problem: str, steps: list[str], device: str) -> list[float]:
    """P(correct) per step, under the PRM's own template."""
    convo = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": SEP.join(steps) + SEP},
    ]
    text = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    out = model(input_ids=ids)
    logits = out[0] if isinstance(out, tuple) else out.logits
    sep_id = tok.encode(SEP)[0]
    mask = (ids == sep_id)
    return step_rewards(logits, mask)[: len(steps)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p.add_argument("--prm_name_or_path", default="Qwen/Qwen2.5-Math-PRM-7B")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--model_dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell_name", default="prm_qwen25_math_7b",
                   help="appears in the frontier's scorer id, so keep it stable")
    p.add_argument("--verify_against", type=Path, nargs="+", default=None,
                   help="probe score files (pass every shard: probe shards and "
                        "PRM shards stride the pool differently); asserts the "
                        "same n_steps per trace")
    p.add_argument("--orientation_traces", type=int, default=20,
                   help="traces used for the sign gate before scoring starts")
    p.add_argument("--max_traces", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    if a.out.exists() and not a.force:
        sys.exit(f"[prm] refusing to overwrite {a.out}. Pass --force.")

    rows = [r for path in a.trajectories for r in read_jsonl(path)]
    rows = [r for r in rows if r.get("gradeable") and (r.get("solution") or "").strip()]
    if a.max_traces:
        rows = rows[: a.max_traces]
    mine = rows[a.shard_idx:: a.num_shards]
    print(f"[prm] shard {a.shard_idx}/{a.num_shards}: {len(mine)} of {len(rows)} traces",
          flush=True)

    expect: dict[str, int] = load_expected_steps(a.verify_against or [])
    if a.verify_against:
        covered = sum(r["traj_uid"] in expect for r in mine)
        print(f"[prm] segmentation gate armed against {len(expect)} probe-scored "
              f"traces; covers {covered}/{len(mine)} of this shard", flush=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[a.model_dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.prm_name_or_path,
                                        local_files_only=a.local_files_only)
    model = AutoModel.from_pretrained(
        a.prm_name_or_path, torch_dtype=dtype, trust_remote_code=True,
        local_files_only=a.local_files_only).to(device).eval()

    # --- gate 1: orientation. A PRM wired backwards, or fed a malformed
    # template, fails this in seconds and costs nothing. ---
    pos, neg = [], []
    for r in mine[: a.orientation_traces * 4]:
        steps = split_into_steps(r["solution"])
        if not steps:
            continue
        m = sum(score_one(model, tok, r["problem"], steps, device)) / len(steps)
        (pos if r["correct"] else neg).append(m)
        if len(pos) >= a.orientation_traces and len(neg) >= a.orientation_traces:
            break
    if not pos or not neg:
        sys.exit("[prm] orientation gate needs both correct and incorrect traces")
    mp, mn = sum(pos) / len(pos), sum(neg) / len(neg)
    print(f"[prm] orientation: mean P(correct) {mp:.4f} on {len(pos)} correct traces "
          f"vs {mn:.4f} on {len(neg)} incorrect", flush=True)
    if mp <= mn:
        sys.exit("[prm] FAILED: the PRM does not rate correct traces higher. "
                 "Template or head channel is wrong; nothing written.")

    out, mismatch = [], 0
    t0 = time.perf_counter()
    for i, r in enumerate(mine):
        steps = split_into_steps(r["solution"])
        if not steps:
            continue
        rewards = score_one(model, tok, r["problem"], steps, device)
        if len(rewards) != len(steps):
            mismatch += 1
            continue
        want = expect.get(r["traj_uid"])
        if want is not None and want != len(steps):
            mismatch += 1
            continue
        # Suspicion, to match every other scorer in this codebase.
        out.append({"traj_uid": r["traj_uid"], "problem_id": r.get("fork_id"),
                    "dataset": r.get("dataset"), "split": r.get("split"),
                    "correct": bool(r["correct"]), "n_steps": len(steps),
                    "cell": a.cell_name,
                    "scores": [1.0 - float(x) for x in rewards]})
        if (i + 1) % 200 == 0 or i + 1 == len(mine):
            print(f"[prm] {i+1}/{len(mine)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # --- gate 2: segmentation. A nonzero rate means the PRM and the probe are
    # reading different objects and the head-to-head is confounded. ---
    rate = mismatch / max(1, len(mine))
    print(f"[prm] segmentation mismatch {mismatch}/{len(mine)} = {rate:.4%}", flush=True)
    if mismatch:
        sys.exit(f"[prm] FAILED: {mismatch} traces segment differently. Nothing written.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out, out)
    (a.out.parent / f"{a.out.stem}_manifest.json").write_text(json.dumps({
        "prm": a.prm_name_or_path, "cell": a.cell_name, "n_traces": len(out),
        "orientation": {"mean_correct": mp, "mean_incorrect": mn},
        "segmentation_mismatch": mismatch,
        "shard_idx": a.shard_idx, "num_shards": a.num_shards,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit()}, indent=1))
    print(f"[prm] wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
