"""Build a fresh, problem-disjoint PRM800K held-out TEST set (never seen in
train or val).

val_1k was used for threshold selection and was drawn from the same curated pool
as probe_train, so it is not a clean held-out evaluation of the correctness
signal. This script samples a brand-new balanced step-classification test set from
PRM800K problems that appear in NONE of the existing artifacts.

It reuses build_prm800k_prestudy.py verbatim (candidate extraction, labeling,
prompt format, length filter) so every field matches what the encoder expects;
the ONLY new logic is excluding seen problem_ids and balanced sampling.

Seen problems = union of problem_id over the --seen_jsonl files you pass (the
originals on TAMIA: prm800k_pos_base_20k.jsonl, prm800k_neg_base_20k.jsonl,
prm800k_val_1k.jsonl). probe_train/mixed are derived from the base files, so
excluding the bases covers them.

Label convention (unchanged): rating +1 -> label 0 (correct),
rating -1 -> label 1 (incorrect), rating 0 dropped.

Output:
  <out_dir>/<stem>.jsonl              one candidate step per line (encoder-ready)
  <out_dir>/<stem>_build_manifest.json

Usage (on TAMIA, where raw PRM800K + tokenizer are cached):
    python scripts/build_prm800k_heldout_test.py \
      --raw_file $SCRATCH/cot_mech/prestudy_v1/<raw_prm800k>.jsonl \
      --seen_jsonl $SCRATCH/cot_mech/prestudy_v1/data/prm800k_pos_base_20k.jsonl \
                   $SCRATCH/cot_mech/prestudy_v1/data/prm800k_neg_base_20k.jsonl \
                   $SCRATCH/cot_mech/prestudy_v1/data/prm800k_val_1k.jsonl \
      --out_dir $SCRATCH/cot_mech/prestudy_v1/data \
      --tokenizer_name_or_path Qwen/Qwen2.5-7B --local_files_only \
      --stem prm800k_heldout_test_6k --n_per_class 3000 --max_seq_len 2048
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_prm800k_prestudy import (  # noqa: E402
    build_candidates, filter_by_length, load_raw_prm800k, write_jsonl,
)


def seen_problem_ids(paths: list[Path]) -> set[str]:
    seen: set[str] = set()
    for p in paths:
        if not p.exists():
            sys.exit(f"--seen_jsonl not found: {p}")
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                pid = json.loads(line).get("problem_id")
                if pid is not None:
                    seen.add(pid)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--raw_dir", type=Path)
    g.add_argument("--raw_file", type=Path)
    ap.add_argument("--seen_jsonl", type=Path, nargs="+", required=True,
                    help="existing artifact jsonls whose problem_ids are excluded")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--stem", type=str, default="prm800k_heldout_test_6k")
    ap.add_argument("--tokenizer_name_or_path", type=str, default=None,
                    help="only needed when --max_seq_len > 0 (length filter)")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--n_per_class", type=int, default=3000,
                    help="balanced: this many correct AND this many incorrect")
    ap.add_argument("--full", action="store_true",
                    help="write ALL unseen candidates at the natural class "
                         "distribution (no balancing); ignores --n_per_class. Use "
                         "for the literature-comparable full PRM800K test set.")
    ap.add_argument("--max_seq_len", type=int, default=2048,
                    help="length filter cap (needs transformers); <=0 SKIPS the "
                         "filter (no tokenizer needed) since we encode at full "
                         "context and PRM800K steps never approach it")
    ap.add_argument("--max_per_problem", type=int, default=0,
                    help="0 = no cap; else keep at most N examples per problem_id "
                         "to avoid a few problems dominating")
    ap.add_argument("--seed", type=int, default=20260618)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print("[build] loading raw PRM800K ...")
    samples = load_raw_prm800k(args.raw_dir, args.raw_file)
    counters: dict = defaultdict(int)
    all_examples, _ = build_candidates(samples, counters)
    print(f"[build] {len(all_examples)} candidate steps extracted")

    seen = seen_problem_ids(args.seen_jsonl)
    print(f"[build] excluding {len(seen)} seen problem_ids "
          f"(from {len(args.seen_jsonl)} artifact files)")
    unseen = [e for e in all_examples if e["problem_id"] not in seen]
    print(f"[build] {len(unseen)} candidates on unseen problems "
          f"({len({e['problem_id'] for e in unseen})} problems)")

    if args.max_seq_len and args.max_seq_len > 0:
        if not args.tokenizer_name_or_path:
            sys.exit("--max_seq_len > 0 requires --tokenizer_name_or_path for the "
                     "length filter (or pass --max_seq_len -1 to skip it)")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, local_files_only=args.local_files_only)
        unseen = filter_by_length(unseen, tok, args.max_seq_len, counters)
        print(f"[build] {len(unseen)} after length filter "
              f"(<= {args.max_seq_len} tokens; dropped {counters['candidate_overlength']})")
    else:
        print("[build] length filter SKIPPED (--max_seq_len <= 0); encode at full "
              "context handles any length")

    if args.max_per_problem > 0:
        by_pid: dict[str, list[dict]] = defaultdict(list)
        rng.shuffle(unseen)
        capped = []
        for e in unseen:
            if len(by_pid[e["problem_id"]]) < args.max_per_problem:
                by_pid[e["problem_id"]].append(e)
                capped.append(e)
        print(f"[build] {len(capped)} after max_per_problem={args.max_per_problem} cap")
        unseen = capped

    if args.full:
        test = list(unseen)
        rng.shuffle(test)
        nc = sum(1 for e in test if e["label"] == 0)
        print(f"[build] FULL natural test: {len(test)} steps "
              f"({nc} correct / {len(test) - nc} incorrect, no balancing)")
    else:
        pos = [e for e in unseen if e["label"] == 0]   # correct
        neg = [e for e in unseen if e["label"] == 1]   # incorrect
        rng.shuffle(pos); rng.shuffle(neg)
        k = args.n_per_class
        if len(pos) < k or len(neg) < k:
            sys.exit(f"[build] not enough unseen examples: correct={len(pos)} "
                     f"incorrect={len(neg)} < n_per_class={k}. Lower --n_per_class.")
        test = pos[:k] + neg[:k]
        rng.shuffle(test)

    out_path = args.out_dir / f"{args.stem}.jsonl"
    write_jsonl(out_path, test)

    pid_set = {e["problem_id"] for e in test}
    manifest = {
        "stem": args.stem,
        "mode": "full_natural" if args.full else f"balanced_{args.n_per_class}",
        "n_total": len(test),
        "n_correct": sum(1 for e in test if e["label"] == 0),
        "n_incorrect": sum(1 for e in test if e["label"] == 1),
        "n_problems": len(pid_set),
        "n_seen_problems_excluded": len(seen),
        "disjoint_from_seen": pid_set.isdisjoint(seen),
        "rating_balance": dict(Counter(e["rating"] for e in test)),
        "max_seq_len": args.max_seq_len,
        "max_per_problem": args.max_per_problem,
        "tokenizer": args.tokenizer_name_or_path,
        "seed": args.seed,
        "seen_jsonl": [str(p) for p in args.seen_jsonl],
        "raw_source": str(args.raw_file or args.raw_dir),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / f"{args.stem}_build_manifest.json").write_text(json.dumps(manifest, indent=2))
    assert manifest["disjoint_from_seen"], "BUG: test set overlaps seen problems"
    print(f"[build] wrote {out_path} ({manifest['n_total']} steps, "
          f"{manifest['n_correct']} correct / {manifest['n_incorrect']} incorrect, "
          f"{manifest['n_problems']} unseen problems)")
    print(f"[build] manifest -> {args.out_dir}/{args.stem}_build_manifest.json")


if __name__ == "__main__":
    main()
