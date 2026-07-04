"""parametric_retrieval_geometry_v0 stage 0: sample WikiProfile facts and
materialize prompt instances.

Downloads the WikiProfile fact table (google/WikiProfile, single CSV of 2,150
facts), assigns quartile popularity bins on the FULL table, samples --n_facts
facts stratified by gbc_bin x category (largest-remainder proportional
allocation), and explodes each sampled fact into 5 prompt instances:

  4 closed-book QA forms (direct, direct_natural, reverse, reverse_natural)
      -> the core four-class retrieval geometry (3,200 instances at 800 facts)
  1 completion form (direct-only auxiliary control, graded against object)

MC (direct_choices / reverse_choices: answer in prompt) and contextual
(not closed-book) are excluded by design.

Outputs (in --out_dir):
  metadata.parquet        one row per prompt instance
  sampling_manifest.json  seed, git hash, bin edges, per-cell allocation

Run locally once; the parquet is the frozen split every downstream job reads
(materialize-before-extraction convention).

  python scripts/parametric_retrieval/prg_sample_facts.py \
      --out_dir runs/parametric_retrieval_geometry_v0 --n_facts 800
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.parametric_retrieval import (  # noqa: E402
    QA_FAMILIES,
    build_prompt_instances,
    gbc_bins,
    stratified_fact_sample,
)

CSV_URL = ("https://huggingface.co/datasets/google/WikiProfile/resolve/main/"
           "wikiprofile.csv")
REQUIRED_COLS = ["fact_id", "page_title", "item_id", "gbc", "category",
                 "subject", "object", "subject_type", "object_type",
                 "completion", "direct", "direct_answer", "direct_natural",
                 "reverse", "reverse_answer", "reverse_natural"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            text=True).strip()
    except Exception:
        return "unknown"


def load_wikiprofile(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    print(f"[sample] downloading {CSV_URL}", flush=True)
    with urllib.request.urlopen(CSV_URL) as r:
        raw = r.read()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    return pd.read_csv(io.BytesIO(raw))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--csv_cache", type=Path,
                    default=Path("data/wikiprofile/wikiprofile.csv"))
    ap.add_argument("--n_facts", type=int, default=800)
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
        print(f"[sample] dropping {n_bad} facts with missing fields", flush=True)
        facts = facts.dropna(subset=REQUIRED_COLS).reset_index(drop=True)

    facts = facts.copy()
    facts["gbc_bin"] = gbc_bins(facts["gbc"])
    sampled = stratified_fact_sample(facts, args.n_facts, seed=args.seed)
    instances = build_prompt_instances(sampled)
    instances.to_parquet(out_meta, index=False)

    alloc = (sampled.groupby(["gbc_bin", "category"], observed=True)
             .size().reset_index(name="n_facts"))
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset": "google/WikiProfile",
        "csv_url": CSV_URL,
        "n_facts_total": int(len(facts)),
        "n_facts_sampled": int(args.n_facts),
        "seed": args.seed,
        "qa_families": QA_FAMILIES,
        "n_qa_instances": int((~instances.is_control).sum()),
        "n_completion_instances": int(instances.is_control.sum()),
        "gbc_bin_edges": {
            b: [int(facts.loc[facts.gbc_bin == b, "gbc"].min()),
                int(facts.loc[facts.gbc_bin == b, "gbc"].max())]
            for b in ["low", "mid", "high", "very_high"]},
        "allocation": alloc.to_dict(orient="records"),
    }
    (args.out_dir / "sampling_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[sample] wrote {out_meta}: {len(instances)} instances "
          f"({manifest['n_qa_instances']} QA + "
          f"{manifest['n_completion_instances']} completion) "
          f"from {args.n_facts} facts", flush=True)


if __name__ == "__main__":
    main()
