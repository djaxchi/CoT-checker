"""parametric_retrieval_access_v1 stage 0: materialize paraphrase instances
and fact-disjoint splits from the full WikiProfile table.

Per fact x direction (direct: gold=object, reverse: gold=subject):
  2 seed questions x 6 instruction wrappers = 12 direct-mode paraphrases
  (all ending "Question: ...\nAnswer:") + 1 canonical greedy-CoT instance.
Instances whose question leaks the gold answer are dropped and counted.
Splits are 60/20/20 fact-disjoint, stratified by gbc_bin x category; both
directions of a fact share the split.

Outputs (in --out_dir):
  metadata.parquet       one row per prompt instance (frozen; every
                         downstream job reads this)
  build_manifest.json    seed, git hash, counts, leak audit

Run locally once:
  python scripts/parametric_retrieval/prga_build_prompts.py \
      --out_dir runs/parametric_retrieval_access_v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.parametric_retrieval.prg_sample_facts import (  # noqa: E402
    REQUIRED_COLS,
    load_wikiprofile,
)
from src.analysis.parametric_retrieval import gbc_bins  # noqa: E402
from src.analysis.parametric_retrieval_access import (  # noqa: E402
    DIRECTIONS,
    PARAPHRASE_WRAPPERS,
    SEED_VARIANTS,
    assign_fact_splits,
    build_access_instances,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--csv_cache", type=Path,
                    default=Path("data/wikiprofile/wikiprofile.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_meta = args.out_dir / "metadata.parquet"
    if out_meta.exists() and not args.force:
        sys.exit(f"refusing to overwrite {out_meta}; pass --force")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    facts = load_wikiprofile(args.csv_cache)
    missing = [c for c in REQUIRED_COLS if c not in facts.columns]
    if missing:
        sys.exit(f"WikiProfile schema drift, missing columns: {missing}")
    n_bad = facts[REQUIRED_COLS].isna().any(axis=1).sum()
    if n_bad:
        print(f"[build] dropping {n_bad} facts with missing fields",
              flush=True)
        facts = facts.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    facts = facts.copy()
    facts["fact_id"] = facts.fact_id.astype(str)
    facts["gbc_bin"] = gbc_bins(facts["gbc"])

    inst = build_access_instances(facts)
    splits = assign_fact_splits(facts, seed=args.seed)
    inst["split"] = inst.fact_id.map(splits)
    assert inst.split.notna().all()

    n_expected = (len(facts) * len(DIRECTIONS)
                  * (len(SEED_VARIANTS["direct"]) * len(PARAPHRASE_WRAPPERS)
                     + 1))
    n_dropped = n_expected - len(inst)
    inst.to_parquet(out_meta, index=False)

    qa = inst[inst.prompt_mode == "direct"]
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset": "google/WikiProfile",
        "seed": args.seed,
        "n_facts": int(len(facts)),
        "n_instances": int(len(inst)),
        "n_direct_paraphrases": int(len(qa)),
        "n_cot_instances": int((inst.prompt_mode == "cot").sum()),
        "n_leak_dropped": int(n_dropped),
        "wrappers": PARAPHRASE_WRAPPERS,
        "seed_variants": SEED_VARIANTS,
        "split_counts_facts": pd.Series(splits).value_counts().to_dict(),
        "split_counts_instances": inst.split.value_counts().to_dict(),
        "paraphrases_per_group": (qa.groupby(["fact_id", "direction"])
                                  .size().describe().round(2).to_dict()),
    }
    (args.out_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[build] wrote {out_meta}: {len(inst)} instances "
          f"({len(qa)} direct paraphrases, "
          f"{manifest['n_cot_instances']} cot) from {len(facts)} facts; "
          f"{n_dropped} leaked instances dropped", flush=True)
    print(f"[build] fact splits: {manifest['split_counts_facts']}", flush=True)


if __name__ == "__main__":
    main()
