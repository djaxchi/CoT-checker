"""Concatenate a step-span store into one token array per collection (step_bert_probe_v1, Arm A).

Each stored item is one step under `step_span_with_boundary`: the pre-step
boundary token then the step's span tokens (meta.pre_step_boundary_idx,
meta.step_start_idx). Steps are ordered by (trace_id, step_idx), the same order
as the v0 pooled files, and each step contributes [boundary, span...] to

    <out>/X.npy      (n_tokens_total, 4096) float16
    <out>/steps.npz  trace_id, problem_id, step_idx, n_steps, y (1 = incorrect),
                     first_error_idx, n_tokens, tok_off (n_steps_total + 1,)

Token t belongs to step k when tok_off[k] <= t < tok_off[k + 1]; the first token
of every step is its boundary token.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.repstore.store import ShardedRepSplit  # noqa: E402


def step_tokens(item: np.ndarray, meta: dict) -> np.ndarray:
    b = int(meta["pre_step_boundary_idx"])
    s = int(meta["step_start_idx"])
    if s >= len(item):
        raise ValueError(f"empty step span for {meta.get('id')} step {meta.get('step_idx')}")
    return np.concatenate([item[b:b + 1], item[s:]], 0)


def build(roots: list[Path], out: Path) -> dict:
    splits = [ShardedRepSplit(r) for r in roots]
    rows = []
    for si, split in enumerate(splits):
        y = split.y
        for k, m in enumerate(split.meta()):
            rows.append((m["id"], int(m["step_idx"]), si, k, int(y[k]), m))
    rows.sort(key=lambda r: (r[0], r[1]))
    # lengths without reading data: item length is meta n_tokens (checked per step below)
    n_tok = np.array([1 + int(m["n_tokens"]) - int(m["step_start_idx"]) for *_, m in rows], dtype=np.int64)
    tok_off = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(n_tok, out=tok_off[1:])
    out.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float16,
                                  shape=(int(tok_off[-1]), splits[0].spec.dim))
    t0 = time.time()
    for i, (_, _, si, k, _, m) in enumerate(rows):
        tok = step_tokens(splits[si].item(k), m)
        if len(tok) != n_tok[i]:
            raise ValueError(f"length mismatch {m['id']} step {m['step_idx']}: {len(tok)} vs {n_tok[i]}")
        X[tok_off[i]:tok_off[i + 1]] = tok
        if i % 5000 == 0:
            print(f"{out.name} {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    X.flush()
    steps = {
        "trace_id": np.array([r[0] for r in rows]),
        "problem_id": np.array([r[5].get("problem_id", r[0]) for r in rows]),
        "step_idx": np.array([r[1] for r in rows], dtype=np.int32),
        "n_steps": np.array([int(r[5]["n_steps"]) for r in rows], dtype=np.int32),
        "y": np.array([r[4] for r in rows], dtype=np.int8),
        "first_error_idx": np.array([int(r[5]["label"]) for r in rows], dtype=np.int32),
        "n_tokens": np.array([int(r[5]["n_tokens"]) for r in rows], dtype=np.int32),
        "tok_off": tok_off,
    }
    np.savez(out / "steps.npz", **steps)
    info = {"out": str(out), "steps": len(rows), "tokens": int(tok_off[-1]),
            "traces": len(set(steps["trace_id"].tolist())), "incorrect_frac": float(steps["y"].mean()),
            "sec": round(time.time() - t0, 1)}
    (out / "info.json").write_text(json.dumps(info))
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store_roots", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.store_roots, args.out)))


if __name__ == "__main__":
    main()
