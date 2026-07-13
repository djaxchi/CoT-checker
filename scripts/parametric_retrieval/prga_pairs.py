"""parametric_retrieval_access_v1 stage 2: grade greedy generations, select
same-fact mixed-outcome paraphrase groups, build matched pairs, candidate
sets, and the extraction set. Local, no GPU.

Grading is the deterministic v0 grader (grade_answer); ambiguous counts as
incorrect but is stored. CoT answers are read after the last 'Final answer:'
marker. Per fact x direction:
  p_direct   fraction of the (up to 12) greedy paraphrases correct
  cot_correct  greedy CoT correctness on the canonical prompt (0/1)
  is_mixed   >= --min_success successes AND >= --min_fail failures

Extraction set = all direct paraphrases of mixed groups + --nonmixed_per_group
paraphrases (w0 of each seed) for every non-mixed group (decoder training and
cross-fact comparisons need pure-success/pure-fail groups too).

Outputs (in --out_dir):
  grading.jsonl              per-instance grades
  group_outcomes.parquet     per fact x direction aggregates + split
  pairs.parquet              matched (success, fail) instance pairs
  candidates.json            per mixed/extracted group: gold + hard negatives
  extraction_set.json        instance_ids for stage 3
  pair_summary.csv           counts by split / direction / gbc_bin

  python scripts/parametric_retrieval/prga_pairs.py \
      --out_dir runs/parametric_retrieval_access_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.parametric_retrieval.prg_grade import first_line  # noqa: E402
from scripts.parametric_retrieval.prg_sample_facts import (  # noqa: E402
    load_wikiprofile,
)
from src.analysis.parametric_retrieval import (  # noqa: E402
    extract_cot_final_answer,
    gbc_bins,
    grade_answer,
)
from src.analysis.parametric_retrieval_access import (  # noqa: E402
    build_candidate_set,
    build_pairs,
    group_outcomes,
)


def grade_generations(gen_path: Path) -> pd.DataFrame:
    rows = []
    for ln in gen_path.read_text().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        if rec["prompt_mode"] == "direct":
            ans = first_line(rec["greedy_text"])
            marker = None
        else:
            ans, marker = extract_cot_final_answer(rec["greedy_text"])
        ok, status = grade_answer(ans, rec["gold_answer"])
        rows.append({
            "instance_id": rec["instance_id"], "fact_id": rec["fact_id"],
            "direction": rec["direction"], "prompt_mode": rec["prompt_mode"],
            "paraphrase_id": rec["paraphrase_id"],
            "gold_answer": rec["gold_answer"],
            "gold_n_tokens": rec.get("gold_n_tokens"),
            "predicted_answer": ans, "cot_marker_found": marker,
            "is_correct": ok, "grading_status": status,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--csv_cache", type=Path,
                    default=Path("data/wikiprofile/wikiprofile.csv"))
    ap.add_argument("--min_success", type=int, default=2)
    ap.add_argument("--min_fail", type=int, default=2)
    ap.add_argument("--max_pairs_per_group", type=int, default=8)
    ap.add_argument("--candidates_k", type=int, default=32)
    ap.add_argument("--nonmixed_per_group", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    graded = grade_generations(args.out_dir / "generations.jsonl")
    meta = pd.read_parquet(args.out_dir / "metadata.parquet")
    graded = graded.merge(
        meta[["instance_id", "seed_variant", "template_id", "split",
              "gbc_bin", "category"]],
        on="instance_id", how="left", validate="one_to_one")
    with open(args.out_dir / "grading.jsonl", "w") as f:
        for row in graded.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- per fact x direction outcomes -------------------------------------
    groups = group_outcomes(graded, min_success=args.min_success,
                            min_fail=args.min_fail)
    cot = graded[graded.prompt_mode == "cot"][
        ["fact_id", "direction", "is_correct"]].rename(
        columns={"is_correct": "cot_correct"})
    groups = groups.merge(cot, on=["fact_id", "direction"], how="left")
    gmeta = (meta.drop_duplicates(["fact_id", "direction"])
             [["fact_id", "direction", "split", "gbc_bin", "category"]])
    groups = groups.merge(gmeta, on=["fact_id", "direction"], how="left",
                          validate="one_to_one")
    groups["delta_cot"] = groups.cot_correct.astype(float) - groups.p_direct
    groups.to_parquet(args.out_dir / "group_outcomes.parquet", index=False)

    # ---- pairs --------------------------------------------------------------
    pairs = build_pairs(graded, groups,
                        max_pairs_per_group=args.max_pairs_per_group,
                        seed=args.seed)
    pairs = pairs.merge(gmeta, on=["fact_id", "direction"], how="left")
    pairs.to_parquet(args.out_dir / "pairs.parquet", index=False)

    # ---- extraction set ------------------------------------------------------
    d = graded[graded.prompt_mode == "direct"].merge(
        groups[["fact_id", "direction", "is_mixed"]],
        on=["fact_id", "direction"], how="left")
    mixed_ids = d.loc[d.is_mixed, "instance_id"]
    nonmixed = d[~d.is_mixed].sort_values("instance_id")
    nonmixed = nonmixed[nonmixed.template_id == "w0"]
    nonmixed_ids = (nonmixed.groupby(["fact_id", "direction"], sort=True)
                    .head(args.nonmixed_per_group).instance_id)
    extraction = sorted(set(mixed_ids) | set(nonmixed_ids))
    (args.out_dir / "extraction_set.json").write_text(json.dumps({
        "n_instances": len(extraction),
        "n_mixed_instances": int(len(mixed_ids)),
        "n_nonmixed_instances": int(len(nonmixed_ids)),
        "instance_ids": extraction,
    }, indent=2))

    # ---- candidate sets (every extracted group) -----------------------------
    facts = load_wikiprofile(args.csv_cache)
    facts = facts.copy()
    facts["fact_id"] = facts.fact_id.astype(str)
    facts["gbc_bin"] = gbc_bins(facts["gbc"])
    keys = (d[d.instance_id.isin(extraction)]
            [["fact_id", "direction"]].drop_duplicates()
            .sort_values(["fact_id", "direction"]))
    cands = [build_candidate_set(facts, fid, direction,
                                 k=args.candidates_k, seed=args.seed)
             for fid, direction in keys.itertuples(index=False)]
    (args.out_dir / "candidates.json").write_text(
        json.dumps(cands, ensure_ascii=False, indent=1))

    # ---- summary -------------------------------------------------------------
    mixed = groups[groups.is_mixed]
    parts = []
    for by in ["split", "direction", "gbc_bin", "category"]:
        t = mixed.groupby(by, observed=True).size().reset_index(name="n_mixed_groups")
        t = t.rename(columns={by: "group"})
        t.insert(0, "group_by", by)
        parts.append(t)
    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(args.out_dir / "pair_summary.csv", index=False)

    qa = graded[graded.prompt_mode == "direct"]
    print(f"[pairs] direct greedy accuracy: {qa.is_correct.mean():.3f} "
          f"({len(qa)} paraphrases)")
    print(f"[pairs] groups: {len(groups)} fact x direction, "
          f"{int(groups.is_mixed.sum())} mixed "
          f"({groups.is_mixed.mean():.1%})")
    print("[pairs] mixed by split:")
    print(mixed.split.value_counts().to_string())
    print(f"[pairs] pairs: {len(pairs)}; extraction set: {len(extraction)} "
          f"instances ({int(len(mixed_ids))} mixed + "
          f"{int(len(nonmixed_ids))} non-mixed)")
    print(f"[pairs] candidate sets: {len(cands)} groups, k<={args.candidates_k}",
          flush=True)


if __name__ == "__main__":
    main()
