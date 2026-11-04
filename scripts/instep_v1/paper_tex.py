#!/usr/bin/env python3
"""LaTeX sources (booktabs tables, pgfplots figure) for the InStep paper's ReProbe comparison.

Same numbers as fig_reprobe.py: ReProbe-paper rows as printed (paper_numbers.py), our rows from
the per-set seed-mean JSON (results/instep_v1/figures_v2/ours.json). Writes, under --out:

  tables/body/reprobe_qwen3_8b.tex     Table 1 of the ReProbe paper (Qwen3-8B) with our rows added
  tables/body/reprobe_phi4.tex         Phi-4 (Table 9, mean of the five released sets) with our rows
  tables/body/reprobe_qwen3_32b.tex    Qwen3-32B (Table 5, overall) with our rows
  figures/plots/reprobe_cost.tex       (a) overall PR-AUC vs parameters added, (b) added latency

Only the bodies are generated. The float wrappers with captions (tables/tab_*.tex,
figures/fig_*.tex) are written by hand in the paper repo and \\input these files.

    python scripts/instep_v1/paper_tex.py --ours results/instep_v1/figures_v2/ours.json \
        --final final.json --out ~/Code/InStep

The paper needs \\usepackage{booktabs, pgfplots, subcaption}, \\usepackage[table]{xcolor} and the colors
instep (main tone, fills), instepdark (same hue, marks and lines) and oursrow (shaded InStep rows).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.instep_v1 import paper_numbers as PN  # noqa: E402

SETS = PN.TEST
HEAD = ["MATH", "GSM8K", "ProofNet", "Trips", "Meet.", "Cal.", "StrQA", "SciQA", "ID", "OOD", "All"]
COLS = list(SETS) + ["id", "ood", "overall"]

# table names: short, in the style of the ReProbe paper; HS = hidden states, A+L = attention + logit
TEX = {
    PN.RANDOM: "Random", PN.MAXPROB: "MaxProb", PN.MAXENT: "MaxEntropy", PN.PPL: "Perplexity",
    PN.SELFCERT: "Self-Certainty", PN.CCP: "CCP", PN.PTRUE: "P(True)", PN.SEMENT: "Semantic Entropy",
    PN.LEXSIM: "Lexical Similarity", PN.DEGMAT: "Degree Matrix", PN.SKYWORK: "Skywork-PRM-1.5B",
    PN.H4: "H4-Qwen2.5-PRM-1.5B", PN.SHEPHERD: "Math-Shepherd-PRM-7B", PN.RLHF_DS: "RLHFlow-PRM-DeepSeek-8B",
    PN.RLHF_MI: "RLHFlow-PRM-Mistral-8B", PN.UNIVERSAL: "Universal-PRM-7B", PN.PRM800K: "Qwen2.5-Math-7B-PRM800K",
    PN.PRM7B: "Qwen2.5-Math-PRM-7B", PN.RP_AL_SELF: "ReProbe$_{\\text{A+L}}$ (self)",
    PN.RP_AL_DS: "ReProbe$_{\\text{A+L}}$ (DS)", PN.RP_HS_SELF: "ReProbe$_{\\text{HS}}$ (self)",
    PN.RP_HS_DS: "ReProbe$_{\\text{HS}}$ (DS)", PN.RP_HS_OSS: "ReProbe$_{\\text{HS}}$ (GPT-OSS)",
    PN.RP_AL_Q3: "ReProbe$_{\\text{A+L}}$ (Qwen3-8B)",
}
GROUP_TITLE = {"UQ": "Uncertainty scores (no training)", "PRM": "Process reward models (separate model)",
               "ReProbe": "ReProbe (probe on stored generator states)",
               "step": "STEP (last-layer probe, retrained on the same data)", "ours": "Ours"}
# parameters each verifier adds next to the generator (see fig_reprobe.py PARAMS)
PARAMS = {PN.SKYWORK: 1.5e9, PN.H4: 1.5e9, PN.SHEPHERD: 7.2e9, PN.RLHF_DS: 8.0e9, PN.RLHF_MI: 8.0e9,
          PN.UNIVERSAL: 7.6e9, PN.PRM800K: 7.6e9, PN.PRM7B: 7.6e9, PN.RP_HS_SELF: 8.14e6, PN.RP_HS_DS: 8.14e6}
# STEP baseline (Liang et al. 2026; scripts/instep_v1/step_probe.py): a 2-layer MLP (d -> 512 -> 1)
# on the last-layer state at each step's last token, trained on the full training split of every
# backbone, mean of 3 seeds; ours.json key "step" (add_step below, from its summary.json).
INSTEP_PARAMS, STEP_PARAMS = 307201, 4096 * 512 + 512 + 512 + 1
OURS_ROW = "\\rowcolor{oursrow}"
# Mean added ms per trajectory, Qwen3-8B on 200 MATH-500 trajectories, H100, given the generation
# states; (1 trajectory per call, per candidate at 10 per call). All six from one job
# (reprobe_cmp_v2/speed/rt_self-658276.json, sec_per_trace_mean): prm, reprobe_head_graph, and InStep
# read_fused at 1 per call; at 10 per call read_fused hit OOM in that job, so InStep is read_graph
# there (the slower of the two read paths, so conservative). Means throughout, never mixed with
# medians: ReProbe's median at 1 per call is 1.25 ms against a 9.38 ms mean (heavy tail).
LATENCY = {"PRM": (29.88, 43.68), "ReProbe": (9.38, 14.03), "InStep": (9.77, 6.61)}


def fmt(v: float) -> str:
    return f"{v:.3f}"[1:] if v < 1 else f"{v:.3f}"


def _marks(values: list[float | None]) -> list[str]:
    """Bold the best, underline the second best (ties share the mark); '--' for missing."""
    present = sorted({round(v, 3) for v in values if v is not None}, reverse=True)
    out = []
    for v in values:
        if v is None:
            out.append("--")
            continue
        s = fmt(v)
        if present and round(v, 3) == present[0]:
            s = f"\\textbf{{{s}}}"
        elif len(present) > 1 and round(v, 3) == present[1]:
            s = f"\\underline{{{s}}}"
        out.append(s)
    return out


def _with_means(d: dict) -> dict:
    d = dict(d)
    d.setdefault("id", float(np.mean([d[s] for s in SETS[:3]])))
    d.setdefault("ood", float(np.mean([d[s] for s in SETS[3:]])))
    return d


# --------------------------------------------------------------------------------- Table 1
def table_qwen3_8b(ours: dict) -> str:
    o = ours["qwen3_8b"]
    rows = [(TEX[m], PN.GROUP[m], PN.per_set("qwen3_8b", m)) for m in PN.methods("qwen3_8b")]
    rows += [("STEP", "step", _with_means(o["step"])),
             ("\\textbf{InStep} (self)", "ours", _with_means(o["instep_self"])),
             ("\\textbf{InStep} (DS)", "ours", _with_means(o["instep_ds"]))]
    cells = [_marks([r[2][c] for r in rows]) for c in COLS]  # column-wise marks
    n = len(COLS) + 1
    L = ["\\begin{tabular}{l ccc ccccc cc c}", "\\toprule",
         " & \\multicolumn{3}{c}{In-domain (math)} & \\multicolumn{5}{c}{Out-of-domain} & "
         "\\multicolumn{3}{c}{Mean} \\\\",
         "\\cmidrule(lr){2-4} \\cmidrule(lr){5-9} \\cmidrule(lr){10-12}",
         "Method & " + " & ".join(HEAD) + " \\\\"]
    prev = None
    for i, (name, grp, _) in enumerate(rows):
        if grp != prev:
            L.append("\\midrule")
            L.append(f"\\multicolumn{{{n}}}{{l}}{{\\textit{{{GROUP_TITLE[grp]}}}}} \\\\")
            prev = grp
        pre = OURS_ROW if "InStep" in name else ""
        L.append(pre + name + " & " + " & ".join(cells[j][i] for j in range(len(COLS))) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------- Phi-4 / 32B
def table_backbone(ours: dict, backbone: str, finalj: dict | None = None) -> str:
    """One backbone, exactly the rows the ReProbe paper reports for it, plus ours.
    Phi-4: mean of the five released sets (Table 9). Qwen3-32B: overall mean of 8 sets (Table 5)."""
    if backbone == "phi4":
        val = lambda m: PN.per_set("phi4", m)["t9"]  # noqa: E731
        mine = lambda d: float(np.mean([d[s] for s in PN.T9]))  # noqa: E731
        o = ours["phi4"]
        arms = [("STEP", "step", mine(o["step"])), ("\\textbf{InStep}", "ours", mine(o["instep"]))]
    else:
        val = lambda m: PN.per_set(backbone, m)["overall"]  # noqa: E731
        o = ours[backbone]
        arms = [("STEP", "step", o["step"]["overall"]), ("\\textbf{InStep}", "ours", o["instep_ep4"]["overall"])]
    grp_order = {"UQ": 0, "PRM": 1, "ReProbe": 2}
    ms = sorted(PN.methods(backbone), key=lambda m: grp_order[PN.GROUP[m]])
    rows = [(TEX[m], PN.GROUP[m], val(m), m) for m in ms] + [(n, g, v, None) for n, g, v in arms]
    marks = _marks([r[2] for r in rows])
    stored = None
    if finalj is not None and backbone == "qwen3_32b":
        stored = finalj["backbones"]["qwen3_32b"]["metrics"].get("prm7b_stored", {}).get("overall", {}).get("ap")
    L = ["\\begin{tabular}{l c}", "\\toprule", "Method & PR-AUC \\\\"]
    prev = None
    for i, (name, grp, _, m) in enumerate(rows):
        if grp != prev:
            L.append("\\midrule")
            L.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{GROUP_TITLE[grp].split(' (')[0]}}}}} \\\\")
            prev = grp
        v = marks[i] + ("$^\\dagger$" if m == PN.PRM7B and stored is not None else "")
        pre = OURS_ROW if "InStep" in name else ""
        L.append(f"{pre}{name} & {v} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------- cost figure
def figure_cost(ours: dict) -> str:
    o = ours["qwen3_8b"]
    prm = [(PARAMS[m], PN.per_set("qwen3_8b", m)["overall"]) for m in PN.methods("qwen3_8b")
           if PN.GROUP[m] == "PRM"]
    rp = [(PARAMS[m], PN.per_set("qwen3_8b", m)["overall"]) for m in (PN.RP_HS_SELF, PN.RP_HS_DS)]
    best_prm = max(prm, key=lambda t: t[1])
    best_rp = max(rp, key=lambda t: t[1])
    ins = (INSTEP_PARAMS, o["instep_self"]["overall"])
    lin = (STEP_PARAMS, o["step"]["overall"])
    co = lambda pts: " ".join(f"({x:.4g},{y:.3f})" for x, y in pts)  # noqa: E731
    def lab(pt, name, where, style=""):
        return (f"\\node[font=\\footnotesize{style}, anchor={where}, inner sep=5pt] "
                f"at (axis cs:{pt[0]:.4g},{pt[1]:.3f}) {{{name} ({fmt(pt[1])})}};")

    labels = "\n".join([lab(ins, "\\textbf{InStep}", "west", ", text=instepdark"),
                        lab(lin, "STEP", "north"), lab(best_rp, "ReProbe", "west"),
                        lab(best_prm, "best PRM", "east")])
    # One point per verifier: InStep, the STEP probe on the same states, the best ReProbe probe,
    # and every PRM (light, one label). Direct labels, no legend, no frontier lines.
    return f"""% generated by scripts/instep_v1/paper_tex.py
\\begin{{subfigure}}[b]{{0.55\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=\\linewidth, height=5.4cm,
  xmode=log, xmin=1e5, xmax=3e10, ymin=0.30, ymax=0.70, xminorticks=false,
  xtick={{1e5,1e6,1e7,1e8,1e9,1e10}},
  xticklabels={{100K,1M,10M,100M,1B,10B}},
  ytick={{0.3,0.4,0.5,0.6,0.7}},
  yticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1}},
  xlabel={{Parameters added to the generator}}, ylabel={{Overall PR-AUC}},
  tick label style={{font=\\footnotesize}}, label style={{font=\\small}},
  ymajorgrids, grid style={{gray!20}},
  clip=false,
]
\\addplot[only marks, mark=triangle*, mark size=2.4pt, color=gray!55] coordinates {{{co(prm)}}};
\\addplot[only marks, mark=diamond*, mark size=3pt, color=black!80] coordinates {{{co([best_rp])}}};
\\addplot[only marks, mark=square*, mark size=2.2pt, color=gray!80!black] coordinates {{{co([lin])}}};
\\addplot[only marks, mark=*, mark size=3.2pt, color=instepdark] coordinates {{{co([ins])}}};
{labels}
\\node[font=\\footnotesize, gray!80!black, anchor=east] at (axis cs:1.3e9,0.40) {{PRMs}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Quality against verifier size.}}
\\label{{fig:reprobe-cost-params}}
\\end{{subfigure}}\\hfill
\\begin{{subfigure}}[b]{{0.39\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=\\linewidth, height=5.4cm,
  ybar=2pt, bar width=8pt, ymin=0, ymax=50,
  symbolic x coords={{PRM,ReProbe,InStep}}, xtick=data,
  xticklabels={{PRM-7B,ReProbe$_{{\\text{{HS}}}}$,\\textbf{{InStep}}}},
  x tick label style={{font=\\footnotesize, align=center}},
  enlarge x limits=0.25,
  ylabel={{Added latency (ms)}},
  tick label style={{font=\\footnotesize}}, label style={{font=\\small}},
  ymajorgrids, grid style={{gray!20}},
  nodes near coords, nodes near coords style={{font=\\scriptsize, /pgf/number format/fixed,
    /pgf/number format/fixed zerofill, /pgf/number format/precision=1}},
  legend style={{font=\\footnotesize, at={{(0.5,1.03)}}, anchor=south, draw=none, /tikz/every even column/.append style={{column sep=6pt}}}},
  legend columns=2, legend image code/.code={{\\draw[#1, draw=none] (0cm,-0.1cm) rectangle (0.25cm,0.1cm);}},
]
\\addplot[fill=gray!70, draw=none] coordinates {{{" ".join(f"({k},{v[0]})" for k, v in LATENCY.items())}}};
\\addlegendentry{{1 trajectory per call}}
\\addplot[fill=gray!30, draw=none] coordinates {{{" ".join(f"({k},{v[1]})" for k, v in LATENCY.items())}}};
\\addlegendentry{{10 per call}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Added latency, Qwen3-8B on MATH-500.}}
\\label{{fig:reprobe-cost-latency}}
\\end{{subfigure}}
"""


# ------------------------------------------------------------------------- test-time scaling figure
PRM7B = "Qwen/Qwen2.5-Math-PRM-7B"
# (bon_curve key, LATENCY key or None, label, pgfplots style), drawn in this order
BON_LINES = [
    ("majority", None, "Majority vote", "black!45, dashed, mark=none"),
    (PRM7B, "PRM", "PRM-7B", "black!75, mark=triangle*, mark size=2.2pt"),
    ("ReProbe HS self-anno", "ReProbe", "ReProbe", "black!75, densely dotted, mark=diamond*, mark size=2.4pt"),
    ("InStep-self", "InStep", "\\textbf{InStep}", "instepdark, thick, mark=*, mark size=2pt"),
]
# (beam arm, Table 2 row label, point label, mark style)
BEAM_ARMS = [
    ("random", "Random scores (same search)", "random scores", "mark=o, black!55, mark size=2.2pt"),
    ("prm7b", "Qwen2.5-Math-PRM-7B", "PRM-7B", "mark=triangle*, black!75, mark size=2.6pt"),
    ("rp_self", "ReProbe, Hidden States, Self-anno", "ReProbe", "mark=diamond*, black!75, mark size=2.8pt"),
    ("rt_self", "InStep, Self-anno", "\\textbf{InStep}", "mark=*, instepdark, mark size=3pt"),
]
SET_LABEL = {"math": "MATH", "gsm8k": "GSM8K", "proofnet": "ProofNet", "trips": "Trips", "meetings": "Meet.",
             "calendar": "Cal.", "strategyqa": "StrQA", "scienceqa": "SciQA"}


def beam_points(t2: dict) -> tuple[list[str], list[tuple]]:
    """Sets every available arm ran with timings, and per arm (label, style, mean seconds per
    question, mean accuracy) over those sets; arms missing on them are left out."""
    rows = {r["row"]: r for r in t2["rows"]}
    secs = t2.get("seconds", {})

    def ok(arm, row, s):
        return rows.get(row, {}).get(s) is not None and secs.get(arm, {}).get(s) is not None

    sets = [s for s in SET_LABEL if ok("random", BEAM_ARMS[0][1], s)]
    arms = [x for x in BEAM_ARMS if sets and all(ok(x[0], x[1], s) for s in sets)]
    sets = [s for s in sets if all(ok(x[0], x[1], s) for x in arms)]
    return sets, [(lab, sty, float(np.mean([secs[a][s] for s in sets])), float(np.mean([rows[r][s] for s in sets])))
                  for a, r, lab, sty in arms]


def _axis_pad(vals: list[float], pad: float = 1.0) -> tuple[float, float]:
    return float(np.floor(min(vals)) - pad), float(np.ceil(max(vals)) + pad)


def figure_tts(tts: dict) -> str:
    cur = tts["bon_curve"]
    ns, n_max = cur["n"], cur["n"][-1]
    # (a) best-of-N accuracy against N
    lines = []
    for key, _, label, style in BON_LINES:
        if key in cur["curves"]:
            co = " ".join(f"({n},{v:.2f})" for n, v in zip(ns, cur["curves"][key]))
            lines.append(f"\\addplot[{style}] coordinates {{{co}}};\n\\addlegendentry{{{label}}}")
    ys = [v for key, _, _, _ in BON_LINES if key in cur["curves"] for v in cur["curves"][key]]
    lo, hi = _axis_pad(ys)
    hi += 1  # headroom for the two-column legend
    # (b) best-of-N at the largest N against the verifier's time per question: N candidates at the
    # mean per-candidate latency of Figure 2b (10 per call)
    pts_b, ys_b = [], []
    for key, lat, label, style in BON_LINES:
        if lat is None or key not in cur["curves"]:
            continue
        x, y = n_max * LATENCY[lat][1], cur["curves"][key][-1]
        ys_b.append(y)
        mark = style.split(", mark=")[1] if ", mark=" in style else "mark=*"
        color = style.split(",")[0]
        where = {"InStep": "south east", "ReProbe": "south west"}.get(lat, "south")  # the two sit close
        pts_b.append(f"\\addplot[only marks, {color}, mark={mark}] coordinates {{({x:.1f},{y:.2f})}};\n"
                     f"\\node[font=\\footnotesize, text={color}, anchor={where}, inner sep=3pt] at (axis cs:{x:.1f},{y:.2f}) "
                     f"{{{label}}};")
    maj = cur["curves"].get("majority", [None])[-1]
    if maj is not None:
        ys_b.append(maj)
        pts_b.append(f"\\addplot[black!45, dashed, domain=10:1000, samples=2] {{{maj:.2f}}};\n"
                     f"\\node[font=\\scriptsize, black!60, anchor=north east, inner sep=2pt] at (axis cs:950,{maj:.2f}) "
                     f"{{majority vote (no verifier)}};")
    lo_b, hi_b = _axis_pad(ys_b, 0.5)
    # (c) beam search: accuracy against wall-clock seconds per question
    sets, bp = beam_points(tts["table2_beam"])
    pts_c = []
    for lab, sty, x, y in bp:
        color = sty.split(", ")[1]
        where = "east" if "InStep" in lab else "west"  # InStep and ReProbe sit close; labels on either side
        pts_c.append(f"\\addplot[only marks, {sty}] coordinates {{({x:.2f},{y:.2f})}};\n"
                     f"\\node[font=\\footnotesize, text={color}, anchor={where}, inner sep=4pt] at (axis cs:{x:.2f},{y:.2f}) "
                     f"{{{lab}}};")
    xs_c, ys_c = [p[2] for p in bp], [p[3] for p in bp]
    xlo, xhi = (np.floor(min(xs_c)) - 1, np.ceil(max(xs_c)) + 4) if bp else (0, 1)
    ylo, yhi = _axis_pad(ys_c, 0.5) if bp else (0, 1)
    common = ("width=\\linewidth, height=4.8cm, tick label style={font=\\footnotesize}, "
              "label style={font=\\small}, ymajorgrids, grid style={gray!20}")
    return f"""% generated by scripts/instep_v1/paper_tex.py
% (c) averages {", ".join(SET_LABEL[s] for s in sets)}
\\begin{{subfigure}}[b]{{0.325\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[{common},
  xmin=0.5, xmax={n_max + 0.5}, ymin={lo:g}, ymax={hi:g},
  xtick={{{",".join(str(n) for n in ns)}}},
  xlabel={{Candidates per question $N$}}, ylabel={{Accuracy (\\%)}},
  legend columns=2, legend style={{font=\\scriptsize, at={{(0.02,0.98)}}, anchor=north west, draw=none, fill=white,
                fill opacity=0.85, text opacity=1, cells={{anchor=west}}}},
]
{chr(10).join(lines)}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Best-of-$N$: accuracy.}}
\\label{{fig:tts-bon}}
\\end{{subfigure}}\\hfill
\\begin{{subfigure}}[b]{{0.325\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[{common},
  xmode=log, xmin=10, xmax=1000, ymin={lo_b:g}, ymax={hi_b:g}, clip=false,
  xtick={{30,100,300,1000}}, xticklabels={{30,100,300,1000}},
  xlabel={{Verifier time per question (ms)}}, ylabel={{Accuracy at $N={n_max}$ (\\%)}},
]
{chr(10).join(pts_b)}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Best-of-$N$: cost.}}
\\label{{fig:tts-bon-cost}}
\\end{{subfigure}}\\hfill
\\begin{{subfigure}}[b]{{0.325\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[{common},
  xmin={xlo:g}, xmax={xhi:g}, ymin={ylo:g}, ymax={yhi:g}, clip=false,
  xlabel={{Search time per question (s)}}, ylabel={{Accuracy (\\%)}},
]
{chr(10).join(pts_c)}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Beam search ($B = N = 5$).}}
\\label{{fig:tts-beam}}
\\end{{subfigure}}
"""


def add_step(ours: dict, backbone: str, summary: dict) -> None:
    """ours[backbone]["step"] = per-set seed-mean PR-AUC of the STEP baseline and their mean."""
    d = dict(summary["test_seed_mean"])
    d["overall"] = float(np.mean(list(d.values())))
    ours[backbone]["step"] = d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--final", type=Path, default=None)
    p.add_argument("--step", action="append", default=[],
                   help="BACKBONE=<scores>/<backbone>/step/summary.json (scripts/instep_v1/step_probe.py)")
    p.add_argument("--tts", type=Path, default=None, help="tts_tables.json (scripts/instep_v1/tts_tables.py)")
    p.add_argument("--out", type=Path, required=True, help="paper root; writes tables/body/ and figures/plots/ under it")
    a = p.parse_args()
    ours = json.loads(a.ours.read_text())
    for spec in a.step:
        bb, path = spec.split("=", 1)
        add_step(ours, bb, json.loads(Path(path).read_text()))
    finalj = json.loads(a.final.read_text()) if a.final else None
    tb, fp = a.out / "tables" / "body", a.out / "figures" / "plots"
    tb.mkdir(parents=True, exist_ok=True)
    fp.mkdir(parents=True, exist_ok=True)
    hdr = "% generated by scripts/instep_v1/paper_tex.py\n"
    t1 = tb / "reprobe_qwen3_8b.tex"
    t1.write_text(hdr + table_qwen3_8b(ours))
    t2 = tb / "reprobe_phi4.tex"
    t2.write_text(hdr + table_backbone(ours, "phi4"))
    t3 = tb / "reprobe_qwen3_32b.tex"
    t3.write_text(hdr + table_backbone(ours, "qwen3_32b", finalj))
    stored = (finalj or {}).get("backbones", {}).get("qwen3_32b", {}).get("metrics", {}).get(
        "prm7b_stored", {}).get("overall", {}).get("ap")
    f1 = fp / "reprobe_cost.tex"
    f1.write_text(figure_cost(ours))
    print(t1, t2, t3, f1, sep="\n")
    if a.tts:
        f2 = fp / "tts.tex"
        f2.write_text(figure_tts(json.loads(a.tts.read_text())))
        print(f2)
    if stored is not None:
        print(f"Qwen2.5-Math-PRM-7B on Qwen3-32B recomputed from the released scores: {stored:.3f}")


if __name__ == "__main__":
    main()
