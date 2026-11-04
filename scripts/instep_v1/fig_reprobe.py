#!/usr/bin/env python3
"""InStep placed inside the ReProbe paper's own result tables (arXiv 2511.06209 v5).

Every ReProbe-paper row is drawn with its printed value (paper_numbers.py, Tier B). InStep (and
our passive linear baseline, where shown) are our measurements on the same released test traces
and DeepSeek-R1 labels; our scoring reproduces the printed ReProbe rows within 0.002 per set
(checked in collect_reprobe_figdata.py, listed in the JSON under "repro_check").

  fig_reprobe_table1   Table 1 (Qwen3-8B) as a figure: per-set PR-AUC grid in the paper's row
                       order, InStep rows added, overall PR-AUC alongside.
  fig_reprobe_cost     quality against what each verifier adds to the generator: (a) overall
                       PR-AUC against parameters added (all Table 1 rows), (b) measured latency.
  fig_reprobe_backbones Tables 9 (Phi-4, five released sets) and 5 (Qwen3-32B) with InStep added.

    python scripts/instep_v1/fig_reprobe.py --ours results/instep_v1/figures_v2/ours.json \
        --final final.json --out results/instep_v1/figures_v2

  --paper draws the same three figures for the LaTeX paper: no baked-in title, subtitle or notes
  (they move to the caption), drawn at PAPER_WIDTH so they are included at their natural size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.instep_v1 import paper_numbers as PN  # noqa: E402

INK, INK2, MUTED, GRID = "#1F1F1F", "#555555", "#8C8C8C", "#E8E8E8"
FAM = {"UQ": "#BDBDBD", "PRM": "#009E73", "ReProbe": "#D55E00", "InStep": "#0072B2", "passive": "#E69F00"}
MARK = {"UQ": "o", "PRM": "^", "ReProbe": "D", "InStep": "o", "passive": "s"}
SETS = PN.TEST
SET_LBL = {"math": "MATH", "gsm8k": "GSM8K", "proofnet": "ProofNet", "trips": "Trips", "meetings": "Meetings",
           "calendar": "Calendar", "strategyqa": "StrQA", "scienceqa": "SciQA"}

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False, "legend.frameon": False,
                     "figure.dpi": 150, "pdf.fonttype": 42, "savefig.bbox": "tight"})

# Font scale. 1.0 for the standalone figures; the paper version (--paper) is drawn with larger
# text so that it stays readable once scaled down to the LaTeX text width.
FONT_SCALE = 1.0
PAPER_WIDTH = 6.5  # inches: \textwidth of the paper (article class with 1in margins)


def _f(size: float) -> float:
    return size * FONT_SCALE


# short display names, the paper's own wording
NAME = {
    PN.RANDOM: "Random", PN.MAXPROB: "MaxProb", PN.MAXENT: "MaxEntropy", PN.PPL: "Perplexity",
    PN.SELFCERT: "Self-Certainty", PN.CCP: "CCP", PN.PTRUE: "P(True)", PN.SEMENT: "Semantic Entropy",
    PN.LEXSIM: "Lexical Similarity", PN.DEGMAT: "Degree Matrix", PN.SKYWORK: "Skywork-PRM-1.5B",
    PN.H4: "H4-Qwen2.5-PRM-1.5B", PN.SHEPHERD: "Math-Shepherd-PRM-7B", PN.RLHF_DS: "RLHFlow-PRM-Deepseek-8B",
    PN.RLHF_MI: "RLHFlow-PRM-Mistral-8B", PN.UNIVERSAL: "Universal-PRM-7B", PN.PRM800K: "Qwen2.5-Math-7B-PRM800K",
    PN.PRM7B: "Qwen2.5-Math-PRM-7B", PN.RP_AL_SELF: "ReProbe Attn+Logit, self-anno",
    PN.RP_AL_DS: "ReProbe Attn+Logit, DeepSeek-anno", PN.RP_HS_SELF: "ReProbe hidden states, self-anno",
    PN.RP_HS_DS: "ReProbe hidden states, DeepSeek-anno", PN.RP_HS_OSS: "ReProbe hidden states, GPT-OSS-anno",
    PN.RP_AL_Q3: "ReProbe Attn+Logit, Qwen3-8B-anno",
}
# parameters a method adds next to the generator at check time. PRMs: their backbone size.
# ReProbe: the released hidden-state head (8,140,801, counted from the checkpoint); the Attn+Logit
# head is drawn at the same size (not released separately, marked hollow). InStep: embedding +
# 36 per-layer offsets + start marker + linear head = 307,201. UQ scores train nothing.
PARAMS = {PN.SKYWORK: 1.5e9, PN.H4: 1.5e9, PN.SHEPHERD: 7.2e9, PN.RLHF_DS: 8.0e9, PN.RLHF_MI: 8.0e9,
          PN.UNIVERSAL: 7.6e9, PN.PRM800K: 7.6e9, PN.PRM7B: 7.6e9, PN.RP_HS_SELF: 8.14e6, PN.RP_HS_DS: 8.14e6,
          PN.RP_AL_SELF: 8.14e6, PN.RP_AL_DS: 8.14e6}
INSTEP_PARAMS, PASSIVE_PARAMS = 307201, 5 * 4096 + 1
SECTION = [("Uncertainty scores (no training)", "UQ"), ("PRMs, 1.5B", "PRM15"), ("PRMs, 7-8B", "PRM78"),
           ("ReProbe (paper)", "ReProbe")]


def _section(m: str) -> str:
    g = PN.GROUP[m]
    if g == "PRM":
        return "PRM15" if m in (PN.SKYWORK, PN.H4) else "PRM78"
    return g


def _fam(sec: str) -> str:
    return "PRM" if sec.startswith("PRM") else sec


def _save(fig, out: Path, name: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=220 if ext == "png" else None)
        paths.append(str(p))
    plt.close(fig)
    return paths


# ------------------------------------------------------------------------------- Table 1 figure
def fig_table1(ours: dict, out: Path, paper: bool = False) -> list[str]:
    o = ours["qwen3_8b"]
    rows = []  # (label, section, {set: v, overall: v}, is_ours)
    for m in PN.methods("qwen3_8b"):
        rows.append((NAME[m], _section(m), PN.per_set("qwen3_8b", m), False))
    rows.append(("InStep, self-anno (ours)", "InStep", o["instep_self"], True))
    rows.append(("InStep, DeepSeek-anno (ours)", "InStep", o["instep_ds"], True))
    for r in rows:  # ID / OOD for our rows as the paper defines them (3 math sets / 5 others)
        if "id" not in r[2]:
            r[2]["id"] = float(np.mean([r[2][s] for s in SETS[:3]]))
            r[2]["ood"] = float(np.mean([r[2][s] for s in SETS[3:]]))
    GRIDC = list(SETS) + ["id", "ood"]
    XPOS = list(range(8)) + [8.35, 9.35]
    cols = GRIDC + ["overall"]
    V = np.array([[r[2][c] for c in cols] for r in rows])
    rnd = V[0]  # Random row: column floor
    best = V.max(axis=0)
    n = len(rows)

    # row y positions with a gap between paper sections
    ys, y, prev = [], 0.0, None
    for r in rows:
        if prev is not None and r[1] != prev:
            y += 0.7
        ys.append(y)
        y += 1.0
        prev = r[1]
    ys = np.array(ys)

    if paper:  # drawn at the LaTeX text width (PAPER_WIDTH), column headers rotated to fit
        fig = plt.figure(figsize=(PAPER_WIDTH, 5.7))
        ax = fig.add_axes([0.305, 0.075, 0.49, 0.82])
        axb = fig.add_axes([0.81, 0.075, 0.145, 0.82], sharey=ax)
    else:
        fig = plt.figure(figsize=(7.9, 6.6))
        ax = fig.add_axes([0.255, 0.115, 0.535, 0.79])
        axb = fig.add_axes([0.80, 0.115, 0.185, 0.79], sharey=ax)
    cmap = plt.get_cmap("Greys")
    for i, r in enumerate(rows):
        for j, c in enumerate(GRIDC):
            v, xj = V[i, j], XPOS[j]
            t = np.clip((v - rnd[j]) / max(best[j] - rnd[j], 1e-9), 0, 1)
            fc = cmap(0.04 + 0.42 * t)
            ax.add_patch(Rectangle((xj - 0.48, ys[i] - 0.46), 0.96, 0.92, facecolor=fc, edgecolor="white", lw=0.8))
            isbest = abs(v - best[j]) < 1e-9
            ax.text(xj, ys[i], f"{v:.3f}"[1:], ha="center", va="center", fontsize=_f(6.6), color=INK,
                    fontweight="bold" if isbest else "normal")
        if r[3]:
            ax.add_patch(Rectangle((-0.5, ys[i] - 0.5), XPOS[-1] + 1.0, 1.0, fill=False, edgecolor=FAM["InStep"],
                                   lw=1.3))
    ax.set_xlim(-0.55, XPOS[-1] + 0.55)
    ax.set_ylim(ys[-1] + 0.8, -1.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=_f(7))
    for tl, r in zip(ax.get_yticklabels(), rows):
        if r[3]:
            tl.set_color(FAM["InStep"]); tl.set_fontweight("bold")
    ax.set_xticks(XPOS)
    if paper:
        ax.set_xticklabels([SET_LBL[s] for s in SETS] + ["ID", "OOD"], fontsize=_f(6.6), rotation=40, ha="left",
                           rotation_mode="anchor")
    else:
        ax.set_xticklabels([SET_LBL[s] for s in SETS] + ["ID", "OOD"], fontsize=_f(6.4))
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(1.0, -1.55, "in-domain (math)", ha="center", fontsize=_f(7), color=INK2)
    ax.text(5.0, -1.55, "out-of-domain", ha="center", fontsize=_f(7), color=INK2)
    ax.text(8.85, -1.55, "means", ha="center", fontsize=_f(7), color=INK2)
    ax.plot([7.9, 9.8], [-1.3, -1.3], color=MUTED, lw=0.8)
    ax.plot([-0.45, 2.45], [-1.3, -1.3], color=MUTED, lw=0.8)
    ax.plot([2.55, 7.45], [-1.3, -1.3], color=MUTED, lw=0.8)
    # section labels in the left margin
    for title, key in SECTION + [("Ours", "InStep")]:
        idx = [i for i, r in enumerate(rows) if r[1] == key]
        if not idx:
            continue
        ax.text(-0.62, ys[idx[0]] - 0.85, title, ha="right", va="center", fontsize=_f(7), fontstyle="italic",
                color=FAM["InStep"] if key == "InStep" else INK2, transform=ax.transData, clip_on=False)

    # overall PR-AUC bars aligned to the rows
    ov = V[:, -1]
    best_paper = max(ov[i] for i, r in enumerate(rows) if not r[3])
    for i, r in enumerate(rows):
        fam = _fam(r[1])
        axb.barh(ys[i], ov[i], height=0.72, color=FAM[fam], edgecolor="none")
        axb.text(ov[i] + 0.01, ys[i], f"{ov[i]:.3f}"[1:], va="center", fontsize=_f(6.6),
                 color=FAM["InStep"] if r[3] else INK2, fontweight="bold" if r[3] else "normal", zorder=3,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.4) if paper else None)
    axb.axvline(best_paper, color=INK2, lw=0.8, ls=(0, (3, 2)))
    axb.text(best_paper, -1.45, f"best in paper\n{best_paper:.3f}"[0:], ha="center", fontsize=_f(6.4), color=INK2)
    axb.text(0.0, -3.2 if paper else -2.35, "Overall", ha="left", fontsize=_f(7.2), color=INK, fontweight="bold")
    axb.set_xlim(0, 0.80)
    axb.set_xticks([0, 0.2, 0.4, 0.6])
    axb.tick_params(axis="y", left=False, labelleft=False)
    axb.tick_params(axis="x", labelsize=_f(6.5))
    axb.spines["left"].set_visible(False)
    axb.set_xlabel("overall PR-AUC\n(mean of 8 sets)" if paper else "overall PR-AUC (mean of 8 sets)", fontsize=_f(7))
    axb.xaxis.set_label_position("bottom")

    if not paper:  # title, subtitle and notes go in the LaTeX caption
        ours_best = [j for j in range(len(SETS)) if max(V[-2:, j]) >= best[j] - 1e-9]
        fig.text(0.02, 0.985, "ReProbe benchmark, Qwen3-8B: the paper's Table 1 with InStep added", ha="left", va="top",
                 fontsize=_f(9.5), fontweight="bold", color=INK)
        fig.text(0.02, 0.958, f"InStep has the best overall PR-AUC ({max(ov[-2:]):.3f} against {best_paper:.3f}) and "
                 f"the best value on {len(ours_best)} of the 8 sets.", ha="left", va="top", fontsize=_f(7.6), color=INK2)
        fig.text(0.02, 0.005,
                 "Step-level PR-AUC for detecting incorrect steps. Paper rows: values printed in ReProbe Table 1. InStep: "
                 "mean of 3 seeds on the released test traces and labels\n(our scoring reproduces the printed ReProbe "
                 "rows within 0.002 per set). Shading runs from the Random row (white) to the column best (dark); bold = "
                 "column best.\nThe paper's UQ rows on MATH, GSM8K, ProofNet and StrQA do not match the released labels "
                 "(printed Random .061 on GSM8K vs a released positive rate of .086).",
                 fontsize=_f(6.2), color=INK2, va="bottom")
    return _save(fig, out, "fig_reprobe_table1")


# ------------------------------------------------------------------------------- cost figure
def fig_cost(ours: dict, out: Path, paper: bool = False) -> list[str]:
    o = ours["qwen3_8b"]
    if paper:  # drawn at the LaTeX text width, labels placed to avoid overlaps at that size
        fig, (ax, axl) = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 3.0),
                                      gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.1})
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.9)
    else:
        fig, (ax, axl) = plt.subplots(1, 2, figsize=(7.4, 3.4),
                                      gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.12})
    ZERO = 10 ** 2.2  # where "no trained parameters" is drawn on the log axis
    pts = []
    for m in PN.methods("qwen3_8b"):
        sec = _section(m)
        pts.append((NAME[m], _fam(sec), PARAMS.get(m, 0), PN.per_set("qwen3_8b", m)["overall"], m))
    for name, fam, p, v, m in pts:
        x = p if p > 0 else ZERO
        hollow = m in (PN.RP_AL_SELF, PN.RP_AL_DS)
        ax.scatter(x, v, s=26 if fam != "UQ" else 16, marker=MARK[fam], zorder=3,
                   facecolor="white" if hollow else FAM[fam], edgecolor=FAM[fam], linewidth=1.0)
    # ours
    ax.scatter(PASSIVE_PARAMS, o["passive_full"]["overall"], s=34, marker="s", facecolor=FAM["passive"],
               edgecolor="white", linewidth=0.8, zorder=4)
    for k in ("instep_self", "instep_ds"):
        ax.scatter(INSTEP_PARAMS, o[k]["overall"], s=60, marker="o", facecolor=FAM["InStep"], edgecolor="white",
                   linewidth=0.9, zorder=5)
    # Pareto frontier (upper-left staircase) over every point
    allp = [((p if p > 0 else ZERO), v) for _, _, p, v, _ in pts]
    allp += [(PASSIVE_PARAMS, o["passive_full"]["overall"]), (INSTEP_PARAMS, o["instep_self"]["overall"])]
    allp.sort()
    fr, bestv = [], -1
    for x, v in allp:
        if v > bestv:
            fr.append((x, v)); bestv = v
    fx, fv = [], []
    for i, (x, v) in enumerate(fr):
        fx += [x]; fv += [v]
        nx = fr[i + 1][0] if i + 1 < len(fr) else 1e11
        fx += [nx]; fv += [v]
    ax.plot(fx, fv, color=FAM["InStep"], lw=0.9, alpha=0.35, zorder=2, drawstyle="default")

    def lab(x, y, t, dx=0, dy=0, ha="left", col=INK2, bold=False):
        ax.annotate(t, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=_f(6.6),
                    color=col, fontweight="bold" if bold else "normal")
    lab(INSTEP_PARAMS, o["instep_self"]["overall"], "InStep (ours)\n.652 self / .647 DeepSeek labels", dx=-6, dy=13,
        ha="center", col=FAM["InStep"], bold=True)
    lab(PASSIVE_PARAMS, o["passive_full"]["overall"], "linear probe on\ngenerator states\n(our baseline) .602",
        dx=-8, dy=-20, ha="center")
    if paper:  # under the hollow diamonds, clear of the PRM label
        lab(8.14e6, PN.per_set("qwen3_8b", PN.RP_AL_DS)["overall"], "ReProbe .604\n(best in paper)", dy=-16,
            ha="center")
    else:
        lab(8.14e6, PN.per_set("qwen3_8b", PN.RP_HS_SELF)["overall"], "ReProbe .604\n(best in paper)", dx=8, dy=4,
            ha="left")
    lab(7.6e9, PN.per_set("qwen3_8b", PN.PRM7B)["overall"], "best PRM\n.565", dx=0, dy=16, ha="center")
    if paper:
        lab(1.5e9, PN.per_set("qwen3_8b", PN.SKYWORK)["overall"], "1.5B PRMs", dx=7, dy=-5, ha="left")
        lab(7.6e9, 0.42, "7-8B PRMs", dx=-9, dy=0, ha="right")
    else:
        lab(1.5e9, PN.per_set("qwen3_8b", PN.SKYWORK)["overall"], "1.5B PRMs", dx=-7, dy=-6, ha="right")
        lab(7.6e9, 0.42, "7-8B PRMs", dx=10, dy=0, ha="left")
    lab(ZERO, PN.per_set("qwen3_8b", PN.MAXPROB)["overall"], "uncertainty scores\n(no training)", dx=7, dy=-12)
    ax.set_xscale("log")
    ax.set_xlim(10 ** 1.8, 1e11)
    ax.set_ylim(0.24, 0.72 if paper else 0.70)
    ax.set_xticks([ZERO, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10])
    ax.set_xticklabels(["0", "10k", "100k", "1M", "10M", "100M", "1B", "10B"], fontsize=_f(6.8))
    ax.tick_params(axis="y", labelsize=_f(6.8))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("parameters added next to the generator (log)", fontsize=_f(7.2))
    ax.set_ylabel("overall PR-AUC, ReProbe benchmark", fontsize=_f(7.2))
    ax.set_title("a  Quality against what each verifier adds", loc="left", fontsize=_f(8), fontweight="bold", color=INK)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=MARK[f], ls="", markerfacecolor=FAM[f], markeredgecolor=FAM[f], markersize=5,
                 label=lbl) for f, lbl in (("UQ", "uncertainty score"), ("PRM", "PRM (separate model)"),
                                          ("ReProbe", "ReProbe head"), ("passive", "linear probe (ours)"),
                                          ("InStep", "InStep (ours)"))]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.38 if paper else 0.36, 0.0), fontsize=_f(6.2), handletextpad=0.2,
              borderaxespad=0.2, labelspacing=0.35)

    # (b) latency: one timing job (TamIA 658276, scripts/analysis/inline_speed.py --reprobe_release, H100,
    # 200 MATH-500 Qwen3-8B trajectories; 190 in groups of 10), MEAN ms per trajectory (total time over
    # trajectories, the cost that adds up), every arm from the same run. Medians, in the caption: 1 per call
    # PRM 22.0, ReProbe 1.25, InStep 8.75; 10 per call PRM 17.2, ReProbe 1.47, InStep 3.0. ReProbe's head
    # attends over every token of the trace for every step, so its cost grows with length (heavy tail).
    # InStep at 10 per call is the CUDA-graphed read (the fused kernel failed at batch 10 in this run).
    items = [("Qwen2.5-Math-PRM-7B: a second 7.6B model re-reads the text", 29.9, 43.7, "PRM"),
             ("ReProbe head over the stored generator states", 9.4, 14.0, "ReProbe"),
             ("InStep: read slots in the generator's own cache", 9.8, 6.6, "InStep")]
    if paper:  # same wording, wrapped to the narrower panel
        wrap = {0: "Qwen2.5-Math-PRM-7B: a second 7.6B\nmodel re-reads the text",
                1: "ReProbe head over the stored\ngenerator states", 2: "InStep: read slots in the\ngenerator's own cache"}
        items = [(wrap[i],) + it[1:] for i, it in enumerate(items)]
    yb = np.arange(len(items))[::-1] * (1.25 if paper else 1.0)
    for (t, b1, b10, fam), y in zip(items, yb):
        if paper:
            axl.text(0, y + 0.40, t, ha="left", va="bottom", fontsize=_f(6.5), color=INK, linespacing=1.1)
        else:
            axl.text(0, y + 0.52, t, ha="left", va="center", fontsize=_f(6.5), color=INK)
        axl.barh(y + 0.19, b1, height=0.34, color=FAM[fam], edgecolor="none")
        axl.text(b1 + 0.5, y + 0.19, f"{b1:.1f} ms", va="center", fontsize=_f(6.6), color=INK2)
        if b10 is not None:
            axl.barh(y - 0.19, b10, height=0.34, color=FAM[fam], alpha=0.45, edgecolor="none")
            axl.text(b10 + 0.5, y - 0.19, f"{b10:.1f} ms", va="center", fontsize=_f(6.6), color=INK2)
        else:
            axl.text(0.4, y - 0.19, "not measured", va="center", fontsize=_f(6.2), color=MUTED, fontstyle="italic")
    axl.set_yticks([])
    axl.set_ylim(-0.55, yb[0] + (1.0 if paper else 0.8))
    axl.spines["left"].set_color(MUTED)
    axl.set_xlim(0, 52)
    axl.tick_params(axis="x", labelsize=_f(6.8))
    axl.grid(axis="x", color=GRID, lw=0.6)
    axl.set_axisbelow(True)
    axl.set_xlabel("added ms per trajectory (mean)", fontsize=_f(7.2))
    axl.set_title("b  Added latency", loc="left", fontsize=_f(8), fontweight="bold", color=INK)
    from matplotlib.patches import Patch
    axl.legend(handles=[Patch(color=INK2, label="1 trajectory per call"),
                        Patch(color=INK2, alpha=0.45, label="10 per call, per candidate")],
               loc="lower right", fontsize=_f(6.0), bbox_to_anchor=(1.02, -0.01) if paper else (1.0, 0.04))
    if not paper:  # title, subtitle and notes go in the LaTeX caption
        fig.text(0.01, 1.0, "InStep beats every verifier in the ReProbe paper with 26x fewer parameters than "
                 "ReProbe and no second model;\nits latency is ReProbe's at one trajectory per call and about 3x "
                 "below a PRM", ha="left", va="bottom", fontsize=_f(8.6),
                 fontweight="bold", color=INK)
        fig.text(0.01, -0.07,
                 "a: paper methods at their printed Table 1 values; InStep and the linear probe measured on the same test "
                 "sets and labels (InStep: 3 seeds; the linear probe reads 5 late layers, fit on all\n167k training "
                 "steps). The line traces the best quality reachable at each size. ReProbe Attn+Logit rows (hollow) are "
                 "drawn at the hidden-state head's size. b: one timing run, Qwen3-8B on 200 MATH-500\ntrajectories, H100, "
                 "mean ms. Medians (1 / 10 per call): PRM 22.0 / 17.2, ReProbe 1.25 / 1.47, InStep 8.8 / 3.0; ReProbe's "
                 "cost grows with trace length. Per decode step during generation\nthe InStep read adds 0.27 ms and the "
                 "ReProbe step 0.37 ms. Scoring a finished trace offline costs a full generator pass plus the read "
                 "(InStep 46.9 ms, ReProbe 47.9 ms, mean).",
                 fontsize=_f(6.0), color=INK2, va="top")
    return _save(fig, out, "fig_reprobe_cost")


# ------------------------------------------------------------------------------- other backbones
def _dots(ax, items: list, title: str, xlabel: str, xlim: tuple, paper: bool = False):
    """items: (label, family, value, hollow_alt or None, is_ours). Sorted by value, best on top."""
    items = sorted(items, key=lambda r: r[2])
    for i, (t, fam, v, alt, mine) in enumerate(items):
        ax.plot([xlim[0], v], [i, i], color=GRID, lw=0.8, zorder=1)
        ax.scatter(v, i, s=46 if mine else 26, marker=MARK[fam], facecolor=FAM[fam], edgecolor="white",
                   linewidth=0.7, zorder=3)
        vkw = dict(va="center", fontsize=_f(6.6), color=FAM["InStep"] if fam == "InStep" else INK2,
                   fontweight="bold" if fam == "InStep" else "normal")
        if paper:  # offset in points so the label clears the marker at any axis scale
            ax.annotate(f"{v:.3f}"[1:], (v, i), xytext=(4.5 if mine else 3.5, 0), textcoords="offset points", **vkw)
        else:
            ax.text(v + 0.008, i, f"{v:.3f}"[1:], **vkw)
        if alt is not None:
            ax.scatter(alt[0], i, s=26, marker=MARK[fam], facecolor="white", edgecolor=FAM[fam], linewidth=1.0, zorder=3)
            ax.annotate("", (alt[0], i), xytext=(v, i), arrowprops=dict(arrowstyle="->", color=FAM[fam], lw=0.7))
            if paper:  # wrapped under the hollow marker, clear of the row labels
                ax.text(alt[0], i - 0.25, alt[1].replace(" the ", " the\n", 1), ha="center", va="top",
                        fontsize=_f(6.0), color=INK2, linespacing=1.05)
            else:
                ax.text(alt[0] - 0.008, i - 0.42, alt[1], ha="right", va="center", fontsize=_f(5.8), color=INK2)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([r[0] for r in items], fontsize=_f(6.8))
    for tl, r in zip(ax.get_yticklabels(), items):
        if r[1] == "InStep":
            tl.set_color(FAM["InStep"]); tl.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(*xlim)
    if paper:
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.tick_params(axis="x", labelsize=_f(6.8))
    ax.set_xlabel(xlabel, fontsize=_f(7.2))
    if paper:  # panel title starts over the row labels, the plot area is too narrow for it
        ax.set_title(title, loc="left", x=-1.55, fontsize=_f(8), fontweight="bold", color=INK)
    else:
        ax.set_title(title, loc="left", fontsize=_f(8), fontweight="bold", color=INK)


def fig_backbones(ours: dict, finalj: dict | None, out: Path, paper: bool = False) -> list[str]:
    if paper:  # drawn at the LaTeX text width
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 3.2), gridspec_kw={"wspace": 1.87})
        fig.subplots_adjust(left=0.285, right=0.955, bottom=0.14, top=0.92)
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"wspace": 0.95})
    # Phi-4: Table 9 on the five released sets (planning traces are not released)
    it = []
    for m in PN.methods("phi4"):
        if m == PN.RANDOM:
            continue
        it.append((NAME[m], _fam(_section(m)), PN.per_set("phi4", m)["t9"], None, False))
    p = ours["phi4"]
    t9 = lambda d: float(np.mean([d[s] for s in PN.T9]))  # noqa: E731
    it.append(("InStep (ours)", "InStep", t9(p["instep"]), None, True))
    it.append(("linear probe (ours)", "passive", t9(p["passive"]), None, True))
    _dots(a1, it, "a  Phi-4 (paper Table 9)",
          "PR-AUC, mean of\nthe 5 released sets" if paper else "PR-AUC, mean of the 5 released sets", (0.05, 0.47), paper)
    # Qwen3-32B: Table 5
    q = ours["qwen3_32b"]
    it = []
    alt7 = None
    if finalj is not None:
        st = finalj["backbones"]["qwen3_32b"]["metrics"].get("prm7b_stored", {}).get("overall", {}).get("ap")
        if st is not None:
            alt7 = (st, f"{st:.3f}"[1:] + " from the released scores")
    for m in PN.methods("qwen3_32b"):
        it.append((NAME[m], _fam(_section(m)), PN.per_set("qwen3_32b", m)["overall"],
                   alt7 if m == PN.PRM7B else None, False))
    it.append(("InStep (ours)", "InStep", q["instep_ep4"]["overall"], None, True))
    it.append(("linear probe (ours)", "passive", q["passive"]["overall"], None, True))
    _dots(a2, it, "b  Qwen3-32B (paper Table 5)",
          "overall PR-AUC\n(mean of 8 sets)" if paper else "overall PR-AUC (mean of 8 sets)", (0.25, 0.70), paper)
    if not paper:  # title, subtitle and notes go in the LaTeX caption
        fig.text(0.02, 1.0, "The other two backbones of the ReProbe paper, with InStep added", ha="left",
                 fontsize=_f(9.5), fontweight="bold", color=INK)
        fig.text(0.02, 0.955, "InStep tops both tables. On Qwen3-32B a plain linear probe on the generator's states ties it "
                 "(.629, difference +.007, CI [-.029, +.032])", ha="left", fontsize=_f(7.6), color=INK2)
        fig.text(0.01, -0.06,
                 "Paper rows: printed values (Table 5 Meetings/Calendar and StrQA/SciQA columns un-swapped to match the "
                 "released scores). InStep: 3 seeds; Qwen3-32B trained\nfor 4 epochs (2,040 optimizer steps). Linear probe: "
                 "one layer of the generator's step-end states, picked on validation (Phi-4 block 19, Qwen3-32B block 55),\n"
                 "40k training steps. Table 5's GSM8K column for the PRMs does not match ReProbe's released scores "
                 "(Qwen2.5-Math-PRM-7B prints .930, the released\nscores give .552); the hollow marker is that row's "
                 "overall recomputed from the released scores.",
                 fontsize=_f(6.0), color=INK2, va="top")
    return _save(fig, out, "fig_reprobe_backbones")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", type=Path, required=True, help="JSON of our per-set PR-AUC (collect_reprobe_figdata.py)")
    p.add_argument("--final", type=Path, default=None, help="final.json of paper_v1.py (stored PRM scores, 32B)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--paper", action="store_true",
                   help="paper version: no baked-in title, subtitle or notes (they go in the LaTeX caption)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ours = json.loads(a.ours.read_text())
    finalj = json.loads(a.final.read_text()) if a.final else None
    paths = (fig_table1(ours, a.out, a.paper) + fig_cost(ours, a.out, a.paper)
             + fig_backbones(ours, finalj, a.out, a.paper))
    print("\n".join(paths))


if __name__ == "__main__":
    main()
