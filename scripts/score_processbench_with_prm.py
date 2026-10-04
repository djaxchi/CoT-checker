#!/usr/bin/env python3
"""Qwen2.5-Math-PRM-7B as a stand-alone scorer on ProcessBench.

The reference the probes on its activations have to be read against: does a
probe on the PRM's residual stream add anything over the PRM's own head?

The PRM reads each trace in its native format (system prompt, the problem as the
user turn, the steps joined by `<extra_0>` with one after the last step), the
same scoring path `scripts/onpolicy/score_traces_with_prm.py` uses downstream.
ProcessBench gives the steps, so there is no segmentation to choose.

Output is the probe cells' per-trace layout (`pb_step_scores_{subset}.jsonl`:
id, label, n_steps, scores), with scores as **suspicion** = 1 - P(correct), so
`scripts/merge_rep_grid_leaderboard.py`'s oracle and calib-20 machinery reads it
unchanged. Qwen's published protocol (first step with P(correct) < 0.5) is the
fixed threshold 0.5 on this suspicion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.encode_processbench_token_store import load_traces  # noqa: E402
from scripts.onpolicy.score_traces_with_prm import score_one_details  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True, help="processbench_<subset>.jsonl")
    p.add_argument("--subset", required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--prm_name_or_path", default="Qwen/Qwen2.5-Math-PRM-7B")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    traces = load_traces(a.raw)
    if a.limit:
        traces = traces[: a.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.prm_name_or_path, local_files_only=a.local_files_only)
    model, info = AutoModel.from_pretrained(
        a.prm_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=a.local_files_only, output_loading_info=True)
    if info.get("missing_keys"):
        sys.exit(f"[prm] FAILED: {len(info['missing_keys'])} weights not loaded")
    model = model.to(device).eval()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"pb_step_scores_{a.subset}.jsonl"
    t0, n_bad = time.perf_counter(), 0
    with out.open("w") as f:
        for i, tr in enumerate(traces):
            steps = tr["steps"]
            d = score_one_details(model, tok, tr["problem"], steps, device)
            if len(d["rewards"]) != len(steps):
                # a step containing the separator string would shift every
                # later reward; refuse rather than misalign
                n_bad += 1
                continue
            f.write(json.dumps({
                "id": tr["id"], "label": int(tr["label"]), "n_steps": len(steps),
                "scores": [1.0 - r for r in d["rewards"]],
                "log_odds_incorrect": d["logits"],
            }) + "\n")
            if i % 200 == 0:
                print(f"[prm:{a.subset}] {i+1}/{len(traces)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"[prm:{a.subset}] wrote {out} ({len(traces)-n_bad} traces, "
          f"{n_bad} skipped for step-count mismatch)", flush=True)
    if n_bad:
        sys.exit(f"[prm:{a.subset}] FAILED: {n_bad} traces misaligned")


if __name__ == "__main__":
    main()
