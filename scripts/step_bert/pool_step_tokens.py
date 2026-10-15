"""Pool a token_seq step-span store into three vectors per step (step_bert_probe_v0, S0).

Each stored item is one step under the `step_span_with_boundary` readout: the
pre-step boundary token followed by the step's span tokens, indexed by
meta.pre_step_boundary_idx and meta.step_start_idx. Per step this writes

    b  (4096,) fp16  the pre-step boundary state
    m  (4096,) fp16  mean over the step span
    l  (4096,) fp16  the last span token

plus trace id, step index, label (store y: 1 = incorrect) and the trace's
first-error index, sorted by (trace_id, step_idx). A deterministic trace subset
(--max_traces, hash of the trace id) keeps whole traces, which matters because the
sharded stores spread one trace's steps across shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.repstore.store import ShardedRepSplit  # noqa: E402


def trace_rank(trace_id: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{trace_id}".encode()).hexdigest()[:12], 16)


def select_traces(trace_ids: list[str], max_traces: int | None, seed: int) -> set[str]:
    uniq = sorted(set(trace_ids))
    if max_traces is None or max_traces >= len(uniq):
        return set(uniq)
    return set(sorted(uniq, key=lambda t: trace_rank(t, seed))[:max_traces])


def pool_item(item: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = item[int(meta["pre_step_boundary_idx"])]
    span = item[int(meta["step_start_idx"]):]
    if len(span) == 0:
        raise ValueError(f"empty step span for {meta.get('id')} step {meta.get('step_idx')}")
    return b, span.mean(0), span[-1]


def pool_store(roots: list[Path], max_traces: int | None, seed: int) -> dict[str, np.ndarray]:
    splits = [ShardedRepSplit(r) for r in roots]
    metas = [s.meta() for s in splits]
    keep = select_traces([m["id"] for ms in metas for m in ms], max_traces, seed)

    rows = []
    for si, (split, ms) in enumerate(zip(splits, metas)):
        y = split.y
        for k, m in enumerate(ms):
            if m["id"] in keep:
                rows.append((m["id"], int(m["step_idx"]), si, k, int(y[k]), m))
    rows.sort(key=lambda r: (r[0], r[1]))

    n, d = len(rows), splits[0].spec.dim
    b = np.empty((n, d), dtype=np.float16)
    mm = np.empty((n, d), dtype=np.float16)
    ll = np.empty((n, d), dtype=np.float16)
    for i, (_, _, si, k, _, m) in enumerate(rows):
        b[i], mm[i], ll[i] = pool_item(splits[si].item(k), m)

    return {
        "b": b,
        "m": mm,
        "l": ll,
        "trace_id": np.array([r[0] for r in rows]),
        "problem_id": np.array([r[5].get("problem_id", r[0]) for r in rows]),
        "step_idx": np.array([r[1] for r in rows], dtype=np.int32),
        "n_steps": np.array([int(r[5]["n_steps"]) for r in rows], dtype=np.int32),
        "y": np.array([r[4] for r in rows], dtype=np.int8),
        "first_error_idx": np.array([int(r[5]["label"]) for r in rows], dtype=np.int32),
        "n_tokens": np.array([int(r[5]["n_tokens"]) for r in rows], dtype=np.int32),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store_roots", type=Path, nargs="+", required=True,
                   help="split dirs holding shard_NN/ (e.g. .../onpolicy_train_spans/train .../val)")
    p.add_argument("--out", type=Path, required=True, help="output .npz")
    p.add_argument("--max_traces", type=int, default=None)
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()

    out = pool_store(args.store_roots, args.max_traces, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(json.dumps({"out": str(args.out), "steps": int(len(out["y"])),
                      "traces": int(len(set(out["trace_id"].tolist()))),
                      "incorrect_frac": float(out["y"].mean())}))


if __name__ == "__main__":
    main()
