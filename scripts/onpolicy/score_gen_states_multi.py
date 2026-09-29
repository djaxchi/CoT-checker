#!/usr/bin/env python3
"""Score saved trajectories with many cells from one generation-state pass.

`score_traces_generation_states.py` scores one step_tokens cell per backbone
pass. The downstream comparison needs every leaderboard representation, so this
runs the teacher-forced pass over prompt + solution once per trajectory and
hands the same states to every cell.

Each step is rebuilt in the layout the cells were trained on: row 0 is the
pre-step boundary state (the last token before the step, which for step 0 is
the last prompt token) and rows 1.. are the step's own tokens, all rounded
through float16 as the training store was. Vector readouts then go through
`_reduce_item`, the exact function that derived the training vectors, and
sequence cells see the step tokens truncated to their last `t_max`, as the
training loader truncated them. Only raw-state cells are accepted (every
Instruct leaderboard cell is `rescale=none`), and lengthfree_geom is refused
because its training-fitted length transform is not applied here.

Output per cell: `<out_dir>/<stem>__<cell>__gen.shard<i>.jsonl`, the frontier's
score format. The manifest times the backbone and each head separately, because
in deployment the states already exist and only the heads are paid for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from derive_delta_from_token_store import _reduce_item  # noqa: E402
from scripts.encode_prm800k_hidden_states import git_commit, read_jsonl  # noqa: E402
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from src.harness.learners import build_learner, is_sequence  # noqa: E402
from src.onpolicy.prompts import context_from_row  # noqa: E402
from src.onpolicy.spans import step_token_spans, verify_spans_cover  # noqa: E402

READOUT = {"last_token": "last", "step_mean": "mean", "step_delta": "delta",
           "step_stats": "multistat", "boundary_stats": "boundary_stats"}


def step_blocks(h: np.ndarray, n_prompt: int, spans: list[tuple[int, int]]) -> list[np.ndarray]:
    """[boundary; step tokens] per step, float16, from one pass's (T, d) states."""
    out = []
    for a, b in spans:
        if b <= a:                       # empty span: its boundary token stands in
            a, b = max(0, a - 1), max(1, a)
        lo = n_prompt + a
        out.append(h[lo - 1: n_prompt + b])
    return out


def readout(block: np.ndarray, rep: str) -> np.ndarray:
    """One step's vector, through the function that derived the training vectors."""
    L = block.shape[0] - 1
    if rep == "step_delta":
        return (block[L].astype(np.float32) - block[0].astype(np.float32)).astype(np.float16)
    return _reduce_item(block, 0, L + 1, 1, L, READOUT[rep]).astype(np.float16)


class Cell:
    def __init__(self, d: Path, device: str):
        res = json.loads((d / "results.json").read_text())
        if (res["protocol"].get("rescale") or "none") != "none":
            raise ValueError(f"{d.name}: rescale={res['protocol'].get('rescale')}, raw only")
        if res["rep"] not in READOUT and res["rep"] != "step_tokens":
            raise ValueError(f"{d.name}: representation {res['rep']} is not supported here")
        self.name, self.rep, self.learner = d.name, res["rep"], res["learner"]
        self.t_max = int(res["protocol"].get("t_max", 512))
        self.model = build_learner(self.learner, int(res["dim"]), t_max=self.t_max)
        self.model.load_state_dict(torch.load(d / "model.pt", map_location=device))
        self.model.to(device).eval()
        self.device, self.seconds = device, 0.0

    @torch.no_grad()
    def score(self, blocks: list[np.ndarray]) -> list[float]:
        t0 = time.perf_counter()
        if is_sequence(self.learner):
            seqs = [b[1:][-self.t_max:] for b in blocks]
            T = max(s.shape[0] for s in seqs)
            x = torch.zeros(len(seqs), T, seqs[0].shape[1], device=self.device)
            m = torch.zeros(len(seqs), T, device=self.device)
            for i, s in enumerate(seqs):
                x[i, : s.shape[0]] = torch.from_numpy(s.astype(np.float32))
                m[i, : s.shape[0]] = 1.0
            logits = self.model(x, m)
        else:
            X = np.stack([readout(b, self.rep) for b in blocks]).astype(np.float32)
            logits = self.model(torch.from_numpy(X).to(self.device), None)
        out = torch.sigmoid(logits.reshape(-1)).float().cpu().tolist()
        self.seconds += time.perf_counter() - t0
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p.add_argument("--cells", type=Path, nargs="+", required=True)
    p.add_argument("--stem", required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--layer", type=int, default=35)
    p.add_argument("--min_span_coverage", type=float, default=0.95)
    p.add_argument("--max_traces", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells = [Cell(d, device) for d in a.cells]
    rows = [r for f in a.trajectories for r in read_jsonl(f)]
    rows = [r for r in rows if r.get("gradeable") and (r.get("solution") or "").strip()]
    if a.max_traces:
        rows = rows[: a.max_traces]
    mine = rows[a.shard_idx:: a.num_shards]
    print(f"[multi] shard {a.shard_idx}/{a.num_shards}: {len(mine)} traces x {len(cells)} cells",
          flush=True)

    tok = AutoTokenizer.from_pretrained(a.model_name_or_path, local_files_only=a.local_files_only)
    backbone = AutoModelForCausalLM.from_pretrained(
        a.model_name_or_path, torch_dtype=torch.bfloat16,
        local_files_only=a.local_files_only).to(device).eval()
    n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731

    outs = {c.name: [] for c in cells}
    bad, t_backbone, t0 = 0, 0.0, time.perf_counter()
    for i, r in enumerate(mine):
        steps = split_into_steps(r["solution"])
        if not steps:
            continue
        prompt = context_from_row(r)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        s_ids = tok(r["solution"], add_special_tokens=False)["input_ids"]
        spans = [(x, min(y, len(s_ids))) for x, y in step_token_spans(prompt, steps, n_tok)]
        ok = verify_spans_cover(spans, len(s_ids), tol=3) and all(y > x for x, y in spans)
        bad += int(not ok)
        tb = time.perf_counter()
        with torch.no_grad():
            h = backbone(input_ids=torch.tensor([p_ids + s_ids], device=device),
                         output_hidden_states=True).hidden_states[a.layer][0]
            h = h.to(torch.float16).cpu().numpy()
        t_backbone += time.perf_counter() - tb
        blocks = step_blocks(h, len(p_ids), spans)
        for c in cells:
            outs[c.name].append({"traj_uid": r["traj_uid"], "problem_id": r.get("fork_id"),
                                 "correct": bool(r["correct"]), "n_steps": len(steps),
                                 "cell": f"{c.name}__gen", "span_ok": ok,
                                 "scores": c.score(blocks)})
        if (i + 1) % 200 == 0 or i + 1 == len(mine):
            print(f"[multi] {i+1}/{len(mine)} ({time.perf_counter()-t0:.0f}s) bad_spans={bad}",
                  flush=True)

    coverage = 1 - bad / max(1, len(mine))
    if coverage < a.min_span_coverage:
        sys.exit(f"[multi] FAILED: span coverage {coverage:.4f}; nothing written")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for c in cells:
        f = a.out_dir / f"{a.stem}__{c.name}__gen.shard{a.shard_idx:02d}.jsonl"
        f.write_text("\n".join(json.dumps(x) for x in outs[c.name]) + "\n")
    (a.out_dir / f"{a.stem}__gen_multi.shard{a.shard_idx:02d}_manifest.json").write_text(json.dumps({
        "cells": [str(d) for d in a.cells], "layer": a.layer, "context": "generation",
        "n_traces": len(mine), "flagged_bad_spans": bad, "seconds_backbone": t_backbone,
        "seconds_head": {c.name: c.seconds for c in cells},
        "model": a.model_name_or_path, "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit()}, indent=1))
    print(f"[multi] span coverage {coverage:.4f}; backbone {t_backbone:.0f}s; heads "
          f"{sum(c.seconds for c in cells):.1f}s over {len(cells)} cells")


if __name__ == "__main__":
    main()
