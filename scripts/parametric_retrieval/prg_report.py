"""parametric_retrieval_geometry_v0: print the headline geometry readout.

Read-only over geometry/ CSVs + grading.jsonl; answers the five v0 tests
numerically (the explorer gives the visual counterpart):

  Test 1  class separation before generation (direct/final_prompt_token CH +
          shuffle p, centroid separations, reasoning_unlocked betweenness)
  Test 2  CoT trajectory movement toward the direct_retrieval centroid
  Test 3  prompt-token vs answer-token separation (answer-artifact check)
  Test 4  reverse vs direct question families (accuracy + class mix)
  Test 5  popularity: class mix per gbc bin (geometry-within-bin lives in the
          explorer's gbc filter; per-bin CH needs hidden states, not run here)

  python scripts/parametric_retrieval/prg_report.py \
      --out_dir runs/parametric_retrieval_geometry_v0 [--hs 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--hs", type=int, default=20,
                    help="focus layer for centroid/trajectory tables")
    args = ap.parse_args()
    geo = args.out_dir / "geometry"

    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    qa = grading[~grading.is_control]

    print("=" * 78)
    print("QA class counts (n=%d)" % len(qa))
    print(qa.retrieval_class.value_counts().to_string())

    bw = read_csv_safe(geo / "between_within_ratio.csv")
    cd = read_csv_safe(geo / "centroid_distances.csv")
    if bw.empty or cd.empty:
        print("\n(no separation stats: geometry cells were skipped, "
              "needs >=2 classes per cell)")
        return
    print("\n" + "=" * 78)
    print("TEST 1+3 — CH between/within by layer x view "
          "(higher = more class-separated)")
    print(bw.pivot_table(index="hs_idx", columns=["prompt_mode", "position"],
                         values="ch_between_within").round(2).to_string())
    bad = bw[bw.shuffle_p > 0.01]
    print(f"\ncells NOT beating the 1000-label-shuffle null (p>0.01): "
          f"{len(bad)} of {len(bw)}")
    if len(bad):
        print(bad[["hs_idx", "prompt_mode", "position", "ch_between_within",
                   "shuffle_p"]].to_string(index=False))

    print("\n" + "=" * 78)
    print(f"TEST 1 — hs{args.hs} direct/final_prompt_token centroid "
          "separations (euclid / pooled within-dispersion)")
    m = cd[(cd.prompt_mode == "direct")
           & (cd.position == "final_prompt_token") & (cd.hs_idx == args.hs)]
    print(m[["class_a", "class_b", "separation", "centroid_cosine_sim"]]
          .round(3).to_string(index=False))

    def sep(a, b):
        r = m[((m.class_a == a) & (m.class_b == b))
              | ((m.class_a == b) & (m.class_b == a))]
        return float(r.separation.iloc[0]) if len(r) else float("nan")

    dr_nr = sep("direct_retrieval", "non_retrieved")
    ru_dr = sep("direct_retrieval", "reasoning_unlocked")
    ru_nr = sep("reasoning_unlocked", "non_retrieved")
    print(f"\nreasoning_unlocked betweenness: d(ru,dr)={ru_dr:.3f} "
          f"d(ru,nr)={ru_nr:.3f} vs d(dr,nr)={dr_nr:.3f}")
    print("  -> 'between' reading: both ru distances < dr-nr distance"
          f" : {ru_dr < dr_nr and ru_nr < dr_nr}")

    print("\n" + "=" * 78)
    print(f"TEST 3 — same layer, prompt vs answer token (direct mode, "
          f"hs{args.hs})")
    t3 = bw[(bw.prompt_mode == "direct") & (bw.hs_idx == args.hs)]
    print(t3[["position", "ch_between_within", "shuffle_p"]].round(2)
          .to_string(index=False))

    traj_p = geo / "trajectory_summary.csv"
    if traj_p.exists():
        traj = pd.read_csv(traj_p)
        t = traj[(traj.hs_idx == args.hs)
                 & (traj.centroid_source == "cot_prompt")]
        print("\n" + "=" * 78)
        print(f"TEST 2 — hs{args.hs} CoT trajectory: toward_retrieval = "
              "(d_nonretrieved - d_directretrieval)/centroid_gap")
        print("(rising along the trajectory = moving toward retrieval "
              "geometry)")
        piv = t.pivot_table(index="traj_order", columns="retrieval_class",
                            values="toward_retrieval")
        piv.index = t.drop_duplicates("traj_order").set_index("traj_order") \
            .loc[piv.index, "traj_step"]
        print(piv.round(3).to_string())

    print("\n" + "=" * 78)
    print("TEST 4 — family difficulty + class mix")
    print(qa.groupby("family").direct_greedy_correct.mean().round(3)
          .to_string())
    mix = pd.crosstab(qa.family, qa.retrieval_class, normalize="index")
    print(mix.round(3).to_string())

    print("\n" + "=" * 78)
    print("TEST 5 — class mix per gbc bin (geometry-within-bin: use the "
          "explorer gbc filter)")
    mix = pd.crosstab(qa.gbc_bin, qa.retrieval_class, normalize="index")
    print(mix.reindex(["low", "mid", "high", "very_high"]).round(3)
          .to_string())


if __name__ == "__main__":
    main()
