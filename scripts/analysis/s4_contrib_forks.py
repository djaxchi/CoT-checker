"""S4 contrib-cluster: build matched correct/incorrect fork pairs from raw
PRM800K for the correct-vs-incorrect geometry view.

Golden-path trajectories contain almost no incorrect steps (annotators advance
through correct completions), so the balanced comparison uses FORKS instead:
steps where the same prefix has both the chosen +1 completion and an unflagged
-1 sibling. Each fork yields one correct and one incorrect continuation of the
identical context, so the two classes are balanced by construction and every
representation (state/qres/contribution) is computed from the same h_0/h_prefix.

Reads the raw jsonl from tasksource/PRM800K (HF cache; identical to TamIA raw),
dedups by (question, prefix, correct, wrong), filters to <= --max_seq_len
tokens, and samples --n_forks with a fixed seed.

Output: runs/contrib_cluster/forks.jsonl  (one row per fork)
  fork_id, question, prefix_steps, step_index, correct, wrong
plus forks_manifest.json.

Usage:
  python scripts/analysis/s4_contrib_forks.py --n_forks 4000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.prm800k_trajectories import extract_fork_pairs  # noqa: E402


def h16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/contrib_cluster"))
    ap.add_argument("--dataset", type=str, default="tasksource/PRM800K")
    ap.add_argument("--files", type=str, nargs="+",
                    default=["phase1_test.jsonl", "phase1_train.jsonl",
                             "phase2_test.jsonl", "phase2_train.jsonl"])
    ap.add_argument("--n_forks", type=int, default=4000)
    ap.add_argument("--max_prefix_steps", type=int, default=9,
                    help="cap golden prefix length (mirrors the 10-step traj cap)")
    ap.add_argument("--tokenizer_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B",
                    help="Qwen2.5 family shares one tokenizer; 0.5B is cached locally")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = args.run_dir / "forks.jsonl"
    if out_path.exists() and not args.force:
        sys.exit(f"refusing to overwrite {out_path}; pass --force")

    from huggingface_hub import hf_hub_download
    rng = random.Random(args.seed)
    counters = {"skipped_sessions": 0, "forks_found": 0}
    candidates: dict[str, dict] = {}
    n_rows = 0
    for fname in args.files:
        path = hf_hub_download(args.dataset, fname, repo_type="dataset")
        print(f"[forks] scanning {fname} ...", flush=True)
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
                for fk in extract_fork_pairs(row, counters):
                    if len(fk["prefix_steps"]) > args.max_prefix_steps:
                        continue
                    wrong = rng.choice(fk["wrongs"])
                    key = h16("\x1e".join([fk["question"], *fk["prefix_steps"],
                                           fk["correct"], wrong]))
                    if key not in candidates:
                        candidates[key] = {
                            "fork_id": key,
                            "question": fk["question"],
                            "prefix_steps": fk["prefix_steps"],
                            "step_index": fk["step_index"],
                            "correct": fk["correct"],
                            "wrong": wrong,
                        }
    print(f"[forks] {n_rows} sessions -> {counters['forks_found']} fork steps, "
          f"{len(candidates)} unique (question,prefix,correct,wrong) pairs", flush=True)

    # deterministic order before sampling
    pool = [candidates[k] for k in sorted(candidates)]
    rng.shuffle(pool)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,
                                        local_files_only=True)

    def n_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    kept: list[dict] = []
    n_overlength = 0
    for fk in pool:
        if len(kept) >= args.n_forks:
            break
        base = "\n".join([fk["question"], *fk["prefix_steps"]])
        longest = base + "\n" + max(fk["correct"], fk["wrong"], key=len)
        if n_tokens(longest) > args.max_seq_len:
            n_overlength += 1
            continue
        kept.append(fk)

    with open(out_path, "w", encoding="utf-8") as f:
        for fk in kept:
            f.write(json.dumps(fk, ensure_ascii=False) + "\n")

    prefix_lens = [len(f["prefix_steps"]) for f in kept]
    (args.run_dir / "forks_manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "n_sessions_scanned": n_rows,
        "n_fork_steps_found": counters["forks_found"],
        "n_unique_pairs": len(candidates),
        "n_sampled": len(kept),
        "n_dropped_overlength": n_overlength,
        "max_prefix_steps": args.max_prefix_steps,
        "max_seq_len": args.max_seq_len,
        "prefix_steps_mean": (sum(prefix_lens) / len(prefix_lens)) if kept else 0,
        "note": "one correct + one wrong continuation per fork, same prefix; "
                "labels are for the pairs view only, main pipeline stays unsupervised",
    }, indent=2))
    print(f"[forks] wrote {out_path} ({len(kept)} forks, "
          f"{n_overlength} dropped overlength)")


if __name__ == "__main__":
    main()
