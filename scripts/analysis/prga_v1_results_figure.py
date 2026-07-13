"""parametric_retrieval_access_v1: 4-panel results figure.

A: first-token logit-lens gold hits@1 across layers (final_prompt_token,
   TEST split, mixed groups), successful vs failed paraphrases vs chance.
B: success-vs-fail accessibility probe AUC across layers per position,
   against the confound-only baseline (Experiment B null).
C: same-fact patching, TEST deltas of the gold-minus-distractor margin with
   fact-bootstrap CIs per condition (Experiment C).
D: access-subspace steering, TEST delta logP(gold) with CIs, learned
   direction vs random controls (Experiment D null vs random).

  python scripts/analysis/prga_v1_results_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = Path("runs/parametric_retrieval_access_v1")
OUT = Path("results/parametric_retrieval_access_v1")
BLUE, RED = "#2a78d6", "#e34948"
POS_COLORS = {"final_prompt_token": "#2a78d6", "answer_prefix": "#1baf7a",
              "question_last": "#eda100", "entity_last": "#4a3aa7"}
GRAY, INK, MUT = "#b5b4ac", "#0b0b0b", "#52514e"


def style(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d8d7d0")
    ax.tick_params(colors=MUT, labelsize=8)
    ax.grid(axis="y", color="#ecebe4", lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")

    # ---- A: lens hits@1 by layer, success vs fail --------------------------
    ax = axes[0, 0]
    s = pd.read_parquet(R / "logitlens" / "scores.parquet")
    groups = pd.read_parquet(R / "group_outcomes.parquet")
    groups["fact_id"] = groups.fact_id.astype(str)
    mixed = set(zip(groups[groups.is_mixed].fact_id,
                    groups[groups.is_mixed].direction))
    t = s[(s.split == "test") & (s.position_name == "final_prompt_token")
          & s.is_correct.notna()]
    t = t[[k in mixed for k in zip(t.fact_id.astype(str), t.direction)]]
    for corr, color, label in [(True, BLUE, "successful paraphrase"),
                               (False, RED, "failed paraphrase")]:
        h = (t[t.is_correct == corr].groupby("hs_idx")
             .gold_rank.apply(lambda r: (r == 1).mean()))
        ax.plot(h.index, h.values, color=color, lw=2, label=label)
    ax.axhline(1 / 32, color=MUT, lw=1, ls="--")
    ax.text(14, 1 / 32 + 0.11, "chance (1/32)", fontsize=7.5, color=MUT)
    ax.annotate("", xy=(15.5, 1 / 32 + 0.01), xytext=(15.5, 1 / 32 + 0.10),
                arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8))
    ax.set_xlabel("hidden_states index (final prompt token)", fontsize=8.5,
                  color=MUT)
    ax.set_ylabel("gold hits@1 among 32 candidates", fontsize=8.5, color=MUT)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax, "A  Answer identity is decodable pre-generation,\n"
              "    also when the model fails (test, first-token lens)")

    # ---- B: probe AUC vs confounds ------------------------------------------
    ax = axes[0, 1]
    b = pd.read_csv(R / "expB" / "probe_auc.csv")
    conf = float(b.loc[b.position == "confounds_only", "auc"].iloc[0])
    for pos, g in b[b.hs_idx >= 0].groupby("position"):
        g = g.sort_values("hs_idx")
        ax.plot(g.hs_idx, g.auc, color=POS_COLORS[pos], lw=1.8)
        ax.annotate(pos.replace("_", " "), (g.hs_idx.iloc[-1], g.auc.iloc[-1]),
                    textcoords="offset points", xytext=(4, 0), fontsize=7,
                    color=POS_COLORS[pos], va="center")
    ax.axhline(conf, color=MUT, lw=1.2, ls="--")
    ax.text(0.5, conf + 0.004, f"confound-only probe ({conf:.2f})",
            fontsize=7.5, color=MUT)
    ax.axhline(0.5, color="#d8d7d0", lw=1)
    ax.set_xlim(0, 34)
    ax.set_xlabel("hidden_states index", fontsize=8.5, color=MUT)
    ax.set_ylabel("success-vs-fail AUC (fact-grouped CV)", fontsize=8.5,
                  color=MUT)
    style(ax, "B  No accessibility signal beyond surface confounds\n"
              "    (mixed groups; residualized AUC 0.50)")

    # ---- C: patching deltas --------------------------------------------------
    ax = axes[1, 0]
    c = pd.read_csv(R / "expC" / "results.csv")
    c = c[c.phase == "test"]
    order = ["matched", "noop", "random_noise", "mismatched_type",
             "mismatched_rand", "reverse"]
    c = c.set_index("condition").loc[order].reset_index()
    ypos = range(len(c))[::-1]
    colors = [BLUE if x == "matched" else GRAY for x in c.condition]
    ax.barh(list(ypos), c.d_margin, color=colors, height=0.62, zorder=3)
    ax.errorbar(c.d_margin, list(ypos),
                xerr=[c.d_margin - c.d_margin_lo,
                      c.d_margin_hi - c.d_margin],
                fmt="none", ecolor=INK, elinewidth=1.1, capsize=2, zorder=4)
    for y, v, cond, em in zip(ypos, c.d_margin, c.condition,
                              c.exact_match_rate):
        note = f"{v:+.2f}"
        if cond == "matched":
            note += f"   rescues {c[c.condition == 'matched'].exact_flip_up.iloc[0]:.0%} exact"
        if cond == "reverse":
            note = (f"{v:+.2f}   breaks "
                    f"{c[c.condition == 'reverse'].exact_flip_down.iloc[0]:.0%} exact")
        hi = c.d_margin_hi.iloc[list(c.condition).index(cond)]
        ax.text(max(v, hi, 0) + 0.35, y, note, va="center", fontsize=7.5,
                color=INK)
    ax.axvline(0, color=MUT, lw=1)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([x.replace("_", " ") for x in c.condition],
                       fontsize=8.5)
    ax.set_xlabel("Δ gold-minus-distractor margin vs baseline "
                  "(test, fact-bootstrap 95% CI)", fontsize=8.5, color=MUT)
    ax.set_xlim(-7.2, 6.5)
    style(ax, "C  Same-fact patching rescues; mismatched donors destroy\n"
              "    (hs26, α=1.0, final prompt token; copying ≈ 0)")

    # ---- D: steering vs random ----------------------------------------------
    ax = axes[1, 1]
    d = pd.read_csv(R / "expD" / "results.csv")
    order_d = ["lda", "random_0", "random_1"]
    d = d.set_index("direction_name").loc[order_d].reset_index()
    labels = ["LDA access direction\n(train facts, residualized)",
              "random direction 1", "random direction 2"]
    colors = [BLUE, GRAY, GRAY]
    x = range(len(d))
    ax.bar(list(x), d.d_logp_gold, color=colors, width=0.55, zorder=3)
    ax.errorbar(list(x), d.d_logp_gold,
                yerr=[d.d_logp_gold - d.d_logp_gold_lo,
                      d.d_logp_gold_hi - d.d_logp_gold],
                fmt="none", ecolor=INK, elinewidth=1.1, capsize=2, zorder=4)
    for xi, v, hi, em in zip(x, d.d_logp_gold, d.d_logp_gold_hi,
                             d.exact_match):
        ax.text(xi, hi + 0.25, f"{v:+.2f}  exact {em:.0%}", ha="center",
                fontsize=7.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ logP(gold) vs baseline (test, 95% CI)", fontsize=8.5,
                  color=MUT)
    ax.set_ylim(0, 8.2)
    style(ax, "D  No fact-independent access direction: learned ≈ random\n"
              "    (hs28, α=4; margin NEGATIVE for all: a gauge, not a lever)")

    fig.suptitle("parametric_retrieval_access_v1: the answer is there, and only "
                 "same-fact content patches unlock it",
                 fontsize=12, color=INK, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = OUT / "prga_v1_results.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
