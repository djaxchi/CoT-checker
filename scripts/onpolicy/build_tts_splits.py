#!/usr/bin/env python3
"""Materialise the GSM8K and MATH500 problem sets for tts_roster_v1.

Splits are written to JSON before any GPU job so every arm reads the same frozen
set, per docs/tts_roster_v1_plan.md and the project's standing rule after the
ablation that did not.

Runs on the login node, which is the only place with internet. GSM8K's gold
answer lives after a "#### " marker in its `answer` field; MATH500 carries an
`answer` column directly and also `level` and `subject`, which the plan's
difficulty stratification needs, so both are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GSM8K_GOLD = re.compile(r"####\s*(.+?)\s*$")


def canon(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def qhash(text: str) -> str:
    return hashlib.sha1(canon(text).encode()).hexdigest()[:16]


def load_gsm8k(split: str = "test") -> list[dict]:
    from datasets import load_dataset
    rows = []
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split=split)):
        m = GSM8K_GOLD.search(r["answer"])
        if not m:
            continue
        rows.append({"problem_id": f"gsm8k_{split}_{i:05d}", "dataset": "gsm8k",
                     "problem": r["question"].strip(),
                     "ground_truth_answer": m.group(1).replace(",", "").strip(),
                     "question_hash": qhash(r["question"])})
    return rows


def load_math500() -> list[dict]:
    from datasets import load_dataset
    rows = []
    for i, r in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test")):
        rows.append({"problem_id": f"math500_{i:05d}", "dataset": "math",
                     "problem": r["problem"].strip(),
                     "ground_truth_answer": str(r["answer"]).strip(),
                     "level": r.get("level"), "subject": r.get("subject"),
                     "question_hash": qhash(r["problem"])})
    return rows


def assign_split(rows: list[dict], conf_frac: float, seed: int) -> None:
    """Exploratory / confirmatory by question hash, so a repeat cannot straddle.

    Hash-based rather than shuffled so the assignment is reproducible from the
    question text alone, without carrying an index.
    """
    for r in rows:
        h = int(hashlib.sha1(f"{seed}:{r['question_hash']}".encode()).hexdigest(), 16)
        r["split"] = "conf" if (h % 1000) / 1000.0 < conf_frac else "expl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--conf_frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--online_gsm8k", type=int, default=200)
    p.add_argument("--online_math", type=int, default=100)
    p.add_argument("--smoke", type=int, default=0,
                   help="If set, also write a smoke set of this many problems "
                        "per dataset, drawn from the exploratory half.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gsm, math = load_gsm8k(), load_math500()
    for rows in (gsm, math):
        assign_split(rows, args.conf_frac, args.seed)

    manifest = {"seed": args.seed, "conf_frac": args.conf_frac, "sets": {}}
    for name, rows in (("gsm8k", gsm), ("math500", math)):
        path = args.out_dir / f"{name}_full.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["sets"][name] = {
            "n": len(rows), "path": str(path),
            "n_conf": sum(r["split"] == "conf" for r in rows),
            "n_expl": sum(r["split"] == "expl" for r in rows),
            "n_unique_questions": len({r["question_hash"] for r in rows}),
            "sha1": hashlib.sha1(path.read_bytes()).hexdigest()[:16]}
        print(f"[splits] {name}: {len(rows)} problems, "
              f"{manifest['sets'][name]['n_unique_questions']} unique questions")

    # The online subset: the rejection arms are the expensive ones, so they run
    # on a fixed sample rather than the full sets. Drawn deterministically by
    # question hash so it is reproducible and disjoint-checkable.
    online = (sorted(gsm, key=lambda r: r["question_hash"])[:args.online_gsm8k]
              + sorted(math, key=lambda r: r["question_hash"])[:args.online_math])
    op = args.out_dir / "online_subset.jsonl"
    op.write_text("".join(json.dumps(r) + "\n" for r in online))
    manifest["sets"]["online_subset"] = {"n": len(online), "path": str(op),
                                         "sha1": hashlib.sha1(op.read_bytes()).hexdigest()[:16]}
    print(f"[splits] online_subset: {len(online)} problems")

    if args.smoke:
        smoke = (sorted(gsm, key=lambda r: r["question_hash"])[-args.smoke:]
                 + sorted(math, key=lambda r: r["question_hash"])[-args.smoke:])
        sp = args.out_dir / "smoke.jsonl"
        sp.write_text("".join(json.dumps(r) + "\n" for r in smoke))
        manifest["sets"]["smoke"] = {"n": len(smoke), "path": str(sp)}
        print(f"[splits] smoke: {len(smoke)} problems "
              f"(taken from the far end of the hash order, disjoint from online_subset)")

    (args.out_dir / "splits_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[splits] wrote {args.out_dir/'splits_manifest.json'}")


if __name__ == "__main__":
    main()
