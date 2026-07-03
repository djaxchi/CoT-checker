"""S4 contrib-cluster (post-hoc): attach PRM800K golden-path ratings to the
sampled steps, for VISUALIZATION ONLY. The clustering pipeline never reads
these; this exists so the explorer can color points by correctness.

Downloads the raw PRM800K jsonl files from tasksource/PRM800K (identical to the
TamIA raw dir: phase1+phase2 train/test, 101,599 sessions) and parses them
directly (datasets' schema casting chokes on the mixed-type 'generation'
column). Reconstructs each session's golden path with ratings and matches our
steps by (sha1(question), step_index, sha1(step_text)). Sessions can repeat a
question+step with different ratings; conflicting matches are marked
'ambiguous'.

Output: runs/contrib_cluster/step_ratings.parquet
  row_id, rating in {"1","0","-1","human","ambiguous","unmatched"}

Usage:
  python scripts/analysis/s4_contrib_ratings.py --run_dir runs/contrib_cluster
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.prm800k_trajectories import reconstruct_trajectory  # noqa: E402


def h16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def step_key(question: str, step_index: int, step_text: str) -> tuple:
    return (h16(question), int(step_index), h16(step_text.strip()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/contrib_cluster"))
    ap.add_argument("--dataset", type=str, default="tasksource/PRM800K")
    ap.add_argument("--files", type=str, nargs="+",
                    default=["phase1_test.jsonl", "phase1_train.jsonl",
                             "phase2_test.jsonl", "phase2_train.jsonl"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = args.run_dir / "step_ratings.parquet"
    if out_path.exists() and not args.force:
        sys.exit(f"refusing to overwrite {out_path}; pass --force")

    meta = pd.read_parquet(args.run_dir / "reprs" / "step_metadata.parquet")
    wanted_questions = {h16(q) for q in meta["question"].unique()}
    print(f"[ratings] {len(meta)} steps over {len(wanted_questions)} distinct questions")

    # key -> set of ratings seen across annotation sessions
    seen: dict[tuple, set] = {}
    counters = {k: 0 for k in ("malformed_samples", "missing_problem",
                               "missing_steps", "truncated_paths", "too_few_steps")}

    from huggingface_hub import hf_hub_download
    n_rows = n_hit_traj = 0
    for fname in args.files:
        path = hf_hub_download(args.dataset, fname, repo_type="dataset")
        print(f"[ratings] scanning {fname} ...", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_rows += 1
                if n_rows % 20000 == 0:
                    print(f"[ratings] scanned {n_rows} rows, matched {n_hit_traj} "
                          "sessions", flush=True)
                q = row.get("question")
                problem = q.get("problem") if isinstance(q, dict) else None
                if not problem or h16(problem) not in wanted_questions:
                    continue
                traj = reconstruct_trajectory(row, n_rows, counters, with_ratings=True)
                if traj is None:
                    continue
                n_hit_traj += 1
                for i, (text, rating) in enumerate(zip(traj["steps"],
                                                       traj["ratings"]), 1):
                    if rating is None:
                        continue
                    seen.setdefault(step_key(traj["question"], i, text),
                                    set()).add(rating)

    def resolve(row) -> str:
        ratings = seen.get(step_key(row.question, row.step_index, row.step_text))
        if not ratings:
            return "unmatched"
        if len(ratings) > 1:
            return "ambiguous"
        return str(next(iter(ratings)))

    ratings = [resolve(r) for r in meta.itertuples()]
    df = pd.DataFrame({"row_id": meta["row_id"], "rating": ratings})
    df.to_parquet(out_path, index=False)

    dist = Counter(ratings)
    (args.run_dir / "step_ratings_manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "files": args.files,
        "n_hub_rows_scanned": n_rows,
        "n_matched_sessions": n_hit_traj,
        "rating_distribution": dict(dist),
        "note": "visualization-only; the clustering pipeline never reads ratings",
    }, indent=2))
    print(f"[ratings] wrote {out_path}")
    for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"[ratings]   {k:<10} {v:6d}  ({v / len(df):.1%})")


if __name__ == "__main__":
    main()
