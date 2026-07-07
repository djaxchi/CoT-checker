"""parametric_retrieval_geometry_v0: figure for exp 1 (steering) + exp 2
(within-instance rollout success geometry).

Panel A: direct_retrieval retention vs steering alpha per direction arm (all
arms degrade symmetrically = gauge, not lever; non_retrieved stays ~0).
Panel B: instance-demeaned CV AUROC predicting per-rollout success by
trajectory checkpoint (chance at prompt/gen0, rising through reasoning).

Reads sae/steer_results.csv and rollouts/rollout_success_geometry.csv.

  python scripts/parametric_retrieval/prg_exp_plot.py \
      --out_dir runs/parametric_retrieval_geometry_v0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ARM_COL = {"sae_dec_58264": "#3987e5", "sae_dec_88965": "#199e70",
           "dense_diff": "#cc7a00", "random": "#8a8a86"}
STEP_LABEL = {"final_prompt_token": "prompt", "first_generated_token": "gen0",
              "se_q1": "se1", "se_q2": "se2", "se_q3": "se3", "se_q4": "se4",
              "token_before_final_answer": "pre-ans",
              "first_final_answer_token": "ans0"}
LAYER_COL = {20: "#3987e5", 24: "#9467bd"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steer = pd.read_csv(args.out_dir / "sae" / "steer_results.csv")
    roll = pd.read_csv(args.out_dir / "rollouts"
                       / "rollout_success_geometry.csv")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    fig.patch.set_facecolor("white")

    base = steer[steer.arm == "baseline"]
    base_dr = float(base[base.start_class == "direct_retrieval"].retrieval_rate)
    a = ax[0]
    for arm in ARM_COL:
        sub = steer[(steer.arm == arm)
                    & (steer.start_class == "direct_retrieval")]
        xs = [-40, -20, -10, 0, 10, 20, 40]
        ys = []
        for al in xs:
            if al == 0:
                ys.append(base_dr)
            else:
                r = sub[sub.alpha == al]
                ys.append(float(r.retrieval_rate) if len(r) else None)
        a.plot(xs, ys, marker="o", ms=4, lw=2, color=ARM_COL[arm], label=arm,
               ls="--" if arm == "random" else "-")
    a.axhline(base_dr, color="#ccc", lw=1, zorder=0)
    a.set_title("A  Steering the answer-commitment feature at the decision "
                "point\ndirect_retrieval retention vs alpha "
                "(non_retrieved stays ~0 for every arm)", fontsize=10,
                loc="left")
    a.set_xlabel("steering alpha (residual norm ~188)", fontsize=9)
    a.set_ylabel("retrieval rate, direct_retrieval", fontsize=9)
    a.legend(fontsize=8, frameon=False)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)
    a.annotate("SAE feature curves track random:\nsymmetric degradation, no "
               "directional\nlift on failing classes -> GAUGE",
               xy=(0.5, 0.45), xycoords="axes fraction", fontsize=8.5,
               color="#555", ha="center")

    b = ax[1]
    b.axhline(0.5, color="#ccc", lw=1, zorder=0)
    for K in sorted(roll.hs_idx.unique()):
        sub = roll[roll.hs_idx == K].sort_values("step_order")
        b.plot(range(len(sub)), sub.auc_within, marker="o", ms=5, lw=2,
               color=LAYER_COL.get(K, "#555"), label=f"hs{K} (within-instance)")
        labels = [STEP_LABEL.get(s, s) for s in sub.step]
    b.set_xticks(range(len(labels)))
    b.set_xticklabels(labels, fontsize=8)
    b.set_title("B  Predicting which CoT rollout succeeds within one "
                "instance\nfact/prompt info removed (instance-demeaned)",
                fontsize=10, loc="left")
    b.set_ylabel("fact-grouped CV AUROC", fontsize=9)
    b.set_xlabel("trajectory checkpoint", fontsize=9)
    b.legend(fontsize=8, frameon=False)
    b.spines[["top", "right"]].set_visible(False)
    b.tick_params(labelsize=8)
    b.annotate("chance at prompt/gen0 -> fate not set early;\n"
               "signal builds during reasoning",
               xy=(0.03, 0.88), xycoords="axes fraction", fontsize=8.5,
               color="#555")

    fig.tight_layout()
    out = args.out_dir / "sae" / "plots" / "exp_steer_and_rollout.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
