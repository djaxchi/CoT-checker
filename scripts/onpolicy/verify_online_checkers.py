#!/usr/bin/env python3
"""Gate: do the online checkers reproduce the offline scores on the same text?

The offline pool was scored in one pass per finished trace; online, a step is
scored when it is written, with only the steps before it in context. Under a
causal model those are the same states, so for stored traces the online score of
step k must match the offline score of step k up to numerics. A systematic
offset (wrong span, wrong context, wrong orientation) would move every step and
shows in the median, so the median is gated; the tail is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from scripts.onpolicy.online_checkers import GenStateChecker, PRMChecker  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--stem", default="tts_math500")
    p.add_argument("--gen_cells", type=Path, nargs="+", required=True)
    p.add_argument("--prm_name_or_path", default=None)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--n_traces", type=int, default=12)
    p.add_argument("--max_steps", type=int, default=6)
    p.add_argument("--tol_median", type=float, default=0.02)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path, local_files_only=True)
    bb = AutoModelForCausalLM.from_pretrained(a.model_name_or_path, torch_dtype=torch.bfloat16,
                                              local_files_only=True).to(dev).eval()
    gen = GenStateChecker(a.gen_cells, bb, tok, 35, dev, "chat")
    prm = PRMChecker(a.prm_name_or_path, dev) if a.prm_name_or_path else None

    tr = [json.loads(l) for f in sorted(a.pool.glob(f"{a.stem}.shard*_trajectories.jsonl"))
          for l in f.read_text().splitlines() if l.strip()]
    tr = [r for r in tr if r.get("gradeable")][: a.n_traces]
    offline = {}
    names = [c.name for c in gen.cells] + ([prm.name] if prm else [])
    for n in names:
        suffix = "" if n == "prm_qwen25_math_7b" else "__gen"
        offline[n] = {}
        for f in sorted((a.pool / "scores").glob(f"{a.stem}__{n}{suffix}.shard*.jsonl")):
            for l in f.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); offline[n][r["traj_uid"]] = r["scores"]
    diffs = {n: [] for n in names}
    for r in tr:
        steps = split_into_steps(r["solution"])
        for k in range(min(len(steps), a.max_steps)):
            got = gen.score_all(r["problem"], steps[:k], steps[k])
            if prm:
                got[prm.name] = prm.score(r["problem"], steps[:k], steps[k])
            for n in names:
                ref = offline[n].get(r["traj_uid"])
                if ref and k < len(ref):
                    diffs[n].append(abs(got[n] - ref[k]))
    ok = True
    for n, d in diffs.items():
        d = np.asarray(d)
        if d.size == 0:
            print(f"[verify] {n}: no offline scores to compare"); ok = False; continue
        med = float(np.median(d))
        print(f"[verify] {n:50s} steps {d.size}  median |diff| {med:.4f}  p95 "
              f"{np.percentile(d, 95):.4f}  max {d.max():.4f}")
        ok &= med <= a.tol_median
    print("[verify]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
