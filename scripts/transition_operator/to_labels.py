"""transition_operator_v0 Stage 1a: operation labels + coverage report.

One transition = one branch (correct or wrong) of a fork. For each transition's
step text this assigns:
  op_symbolic   ADD/SUB/MUL/DIV/POW or None (precision-first parser)
  op_verified   bool (numeric identity checked)
  tag_top       single rarest-matching keyword tag (recall-oriented fallback)
plus the multi-hot keyword tags. Writes step_labels.parquet and a coverage report:
symbolic coverage, verified fraction, per-op distribution, tag coverage, and the
agreement between the symbolic op and the keyword tags on the overlap. Also dumps a
200-step sample (label + text) for manual spot-checking, since the plan requires a
manual-agreement pass on ~200 steps before trusting the labels.

Local, no GPU:
  python scripts/transition_operator/to_labels.py --run_dir runs/transition_operator
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.contrib_cluster import TAG_NAMES, assign_top_tag, tag_step  # noqa: E402
from src.analysis.transition_operator_ops import symbolic_operation  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--sample_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    forks = [json.loads(l) for l in open(args.run_dir / "forks.jsonl")
             if l.strip()]
    rows = []
    for fk in forks:
        for branch in ("correct", "wrong"):
            text = fk[branch]
            lab = symbolic_operation(text)
            tags = tag_step(text)
            rows.append({
                "fork_id": fk["fork_id"],
                "question": fk["question"],
                "branch": branch,
                "step_index": fk["step_index"],
                "text": text,
                "op_symbolic": lab.op,
                "op_verified": lab.verified,
                **{f"tag_{t}": tags[t] for t in TAG_NAMES},
            })
    df = pd.DataFrame(rows)
    tag_matrix = df[[f"tag_{t}" for t in TAG_NAMES]].to_numpy(dtype=bool)
    df["tag_top"] = assign_top_tag(tag_matrix)

    n = len(df)
    has_sym = df["op_symbolic"].notna()
    report: dict = {
        "n_transitions": n,
        "n_forks": len(forks),
        "symbolic_coverage": float(has_sym.mean()),
        "verified_fraction_of_symbolic": float(
            df.loc[has_sym, "op_verified"].mean()) if has_sym.any() else 0.0,
        "op_distribution": {k: int(v) for k, v in
                            Counter(df.loc[has_sym, "op_symbolic"]).items()},
        "tag_coverage_nonempty": float((df["tag_top"] != "NONE").mean()),
        "tag_top_distribution": {k: int(v) for k, v in
                                 Counter(df["tag_top"]).most_common()},
    }
    # agreement: on the symbolic-labeled subset, do keyword tags fire an arithmetic
    # tag? (weak cross-check; the two label systems are not one-to-one)
    arith_tags = [c for c in df.columns
                  if c in ("tag_ARITHMETIC", "tag_EQUATION_SOLVE",
                           "tag_EQUATION_TRANSFORM")]
    if arith_tags and has_sym.any():
        fires = df.loc[has_sym, arith_tags].any(axis=1)
        report["symbolic_subset_arith_tag_rate"] = float(fires.mean())

    out_dir = args.run_dir / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "step_labels.parquet")
    (out_dir / "labels_coverage.json").write_text(json.dumps(report, indent=2))

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=min(args.sample_n, n), replace=False)
    sample = df.iloc[idx][["op_symbolic", "op_verified", "tag_top", "text"]]
    sample.to_json(out_dir / "labels_sample_200.jsonl", orient="records", lines=True)

    print(json.dumps(report, indent=2))
    print(f"[labels] wrote {out_dir}/step_labels.parquet, labels_coverage.json, "
          f"labels_sample_200.jsonl")


if __name__ == "__main__":
    main()
