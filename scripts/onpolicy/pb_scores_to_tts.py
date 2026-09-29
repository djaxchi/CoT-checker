#!/usr/bin/env python3
"""Turn per-cell `pb_step_scores_<split>.jsonl` into the TTS frontier's score files.

`score_cells_on_split.py` writes one row per trace (`id` = traj_uid, `scores` =
per-step P(incorrect)). The frontier reads `scores/<stem>__<cell>.shard*.jsonl`
rows keyed by `traj_uid`. This converts one into the other and refuses to write
anything the frontier would silently misread: every gradeable trajectory of the
pool must be present, with exactly as many step scores as `split_into_steps`
gives its solution, because a trace with missing steps would have its worst
step computed over the steps the encoder happened to keep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402


def convert(pb_rows: list[dict], trajectories: list[dict], cell: str) -> list[dict]:
    want = {t["traj_uid"]: t for t in trajectories if t.get("gradeable")}
    got = {r["id"]: r for r in pb_rows}
    missing = sorted(set(want) - set(got))
    if missing:
        raise ValueError(f"{cell}: {len(missing)} gradeable trajectories have no scores, "
                         f"e.g. {missing[:3]}")
    out = []
    for uid, t in want.items():
        n = len(split_into_steps(t["solution"]))
        scores = [float(x) for x in got[uid]["scores"]]
        if len(scores) != n:
            raise ValueError(f"{cell}: {uid} has {len(scores)} step scores for {n} steps")
        out.append({"traj_uid": uid, "problem_id": t.get("fork_id"), "correct": bool(t["correct"]),
                    "n_steps": n, "cell": cell, "scores": scores})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells_root", type=Path, required=True,
                   help="directory holding one subdirectory per scored cell")
    p.add_argument("--split", required=True, help="pb_step_scores_<split>.jsonl to read")
    p.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p.add_argument("--stem", required=True, help="frontier stem, e.g. tts_gsm8k")
    p.add_argument("--out_dir", type=Path, required=True)
    a = p.parse_args()

    trajs = [json.loads(l) for f in a.trajectories for l in f.read_text().splitlines() if l.strip()]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(a.cells_root.glob(f"*/pb_step_scores_{a.split}.jsonl")):
        cell = f.parent.name
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        out = convert(rows, trajs, cell)
        dst = a.out_dir / f"{a.stem}__{cell}.shard00.jsonl"
        dst.write_text("\n".join(json.dumps(r) for r in out) + "\n")
        n += 1
    if n == 0:
        sys.exit(f"[convert] no pb_step_scores_{a.split}.jsonl under {a.cells_root}")
    print(f"[convert] {n} cells -> {a.out_dir} ({a.stem})")


if __name__ == "__main__":
    main()
