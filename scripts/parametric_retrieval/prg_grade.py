"""parametric_retrieval_geometry_v0 stage 2: deterministic grading and
behavioral retrieval-class labels.

Reads generations.jsonl (merged) + metadata.parquet, grades every generation
with pure string matching (grade_answer: exact | containment |
normalized_number | ambiguous | failed; ambiguous counts as incorrect but is
kept for inspection), derives

  direct_greedy_correct, direct_pass_at_4 (over the 4 sampled generations),
  cot_greedy_correct, cot_pass_at_4

and assigns the hard mutually-exclusive retrieval_class (priority chain) plus
soft flags. Completion controls get ctrl_* labels instead (direct-only arm).

Outputs (in --out_dir):
  grading.jsonl
  geometry/class_counts.csv    counts by class x family and class x gbc_bin

  python scripts/parametric_retrieval/prg_grade.py \
      --out_dir runs/parametric_retrieval_geometry_v0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.parametric_retrieval import (  # noqa: E402
    assign_retrieval_class,
    completion_control_class,
    extract_cot_final_answer,
    grade_answer,
    soft_flags,
)


def first_line(text: str) -> str:
    for ln in text.strip().splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def grade_record(rec: dict) -> dict:
    gold = rec["gold_answer"]
    d = rec["direct"]
    d_ans = first_line(d["greedy_text"])
    d_ok, d_status = grade_answer(d_ans, gold)
    d_sampled = [grade_answer(first_line(t), gold)[0]
                 for t in d["sampled_texts"]]
    d_pass4 = any(d_sampled)

    row = {
        "question_id": rec["question_id"], "fact_id": rec["fact_id"],
        "family": rec["family"], "is_control": rec["is_control"],
        "gold_answer": gold,
        "direct_greedy_answer": d_ans,
        "direct_greedy_correct": d_ok,
        "direct_greedy_status": d_status,
        "direct_sampled_correct": d_sampled,
        "direct_pass_at_4": d_pass4,
        "cot_greedy_final_answer": None, "cot_marker_found": None,
        "cot_greedy_correct": None, "cot_greedy_status": None,
        "cot_sampled_correct": None, "cot_pass_at_4": None,
    }
    if rec["cot"] is None:
        row["retrieval_class"] = completion_control_class(d_ok, d_pass4)
        row.update({"reasoning_unlocked_soft": None, "direct_unstable": None,
                    "cot_unstable": None})
        return row

    c = rec["cot"]
    c_ans, c_marker = extract_cot_final_answer(c["greedy_text"])
    c_ok, c_status = grade_answer(c_ans, gold)
    c_sampled = [grade_answer(extract_cot_final_answer(t)[0], gold)[0]
                 for t in c["sampled_texts"]]
    c_pass4 = any(c_sampled)
    row.update({
        "cot_greedy_final_answer": c_ans, "cot_marker_found": c_marker,
        "cot_greedy_correct": c_ok, "cot_greedy_status": c_status,
        "cot_sampled_correct": c_sampled, "cot_pass_at_4": c_pass4,
        "retrieval_class": assign_retrieval_class(d_ok, d_pass4, c_ok, c_pass4),
        **soft_flags(d_ok, d_pass4, c_ok, c_pass4),
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    args = ap.parse_args()

    gen_path = args.out_dir / "generations.jsonl"
    records = [json.loads(ln) for ln in gen_path.read_text().splitlines()
               if ln.strip()]
    rows = [grade_record(r) for r in records]
    grading = pd.DataFrame(rows)

    meta = pd.read_parquet(args.out_dir / "metadata.parquet")
    grading = grading.merge(
        meta[["question_id", "gbc_bin", "category"]], on="question_id",
        how="left", validate="one_to_one")

    with open(args.out_dir / "grading.jsonl", "w") as f:
        for row in grading.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    geo_dir = args.out_dir / "geometry"
    geo_dir.mkdir(exist_ok=True)
    parts = []
    for by in ["family", "gbc_bin", "category"]:
        t = (grading.groupby(["retrieval_class", by], observed=True)
             .size().reset_index(name="n"))
        t = t.rename(columns={by: "group"})
        t.insert(1, "group_by", by)
        parts.append(t)
    total = (grading.groupby("retrieval_class", observed=True)
             .size().reset_index(name="n"))
    total.insert(1, "group_by", "all")
    total.insert(2, "group", "all")
    counts = pd.concat([total] + parts, ignore_index=True)
    counts.to_csv(geo_dir / "class_counts.csv", index=False)

    qa = grading[~grading.is_control]
    print("[grade] QA class counts:")
    print(qa.retrieval_class.value_counts().to_string())
    print("[grade] soft reasoning_unlocked_soft:",
          int(qa.reasoning_unlocked_soft.sum()))
    print("[grade] completion control counts:")
    print(grading[grading.is_control].retrieval_class
          .value_counts().to_string())
    print("[grade] direct greedy accuracy by family:")
    print(grading.groupby("family").direct_greedy_correct.mean()
          .round(3).to_string())
    amb = int((grading.direct_greedy_status == "ambiguous").sum())
    print(f"[grade] wrote grading.jsonl ({len(grading)} rows, "
          f"{amb} ambiguous direct gradings) + class_counts.csv", flush=True)


if __name__ == "__main__":
    main()
