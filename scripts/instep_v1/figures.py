"""Figures of the InStep paper V1 (Fig 2-6), drawn from final.json's structure.

Okabe-Ito hues in a fixed order per method family (checked for colour-vision-deficiency
separation), every bar or point also direct-labelled. Each figure is written as PNG and PDF;
a figure whose inputs are missing is skipped and reported, never raised.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"passive": "#E69F00", "maxprob": "#CC79A7", "reprobe": "#D55E00", "prm": "#009E73",
       "instep": "#0072B2", "ptrue": "#56B4E9", "ladder": "#0072B2", "ref": "#6B6B6B"}
MARK = {"passive": "s", "maxprob": "v", "reprobe": "D", "prm": "^", "instep": "o", "ptrue": "P"}
BBL = {"qwen3_8b": "Qwen3-8B", "phi4": "Phi-4", "qwen3_32b": "Qwen3-32B"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})


def _save(fig, out: Path, name: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None)
        paths.append(str(p))
    plt.close(fig)
    return paths


def _overall(R: dict, k: str):
    m = R["metrics"].get(k, {}).get("overall")
    return None if m is None else m


def _family(k: str) -> str:
    if k.startswith("instep"):
        return "instep"
    if k.startswith("rp"):
        return "reprobe"
    if k.startswith("prm"):
        return "prm"
    if k.startswith("maxprob"):
        return "maxprob"
    return {"passive": "passive", "ptrue": "ptrue"}.get(k, "ref")


def fig_pareto(final: dict, out: Path) -> list[str]:
    from scripts.instep_v1.paper_v1 import MS, PARAMS
    R = final["backbones"].get("qwen3_8b")
    if not R:
        raise LookupError("Fig 2: no Qwen3-8B results")
    pts = []
    for k in ("instep_ds", "rp_ds", "rp_rel_ds", "prm7b", "prm800k", "passive", "maxprob", "ptrue"):
        o = _overall(R, k)
        if o is not None:
            pts.append((k, R["arms"][k]["label"], o["ap"], R["arm_ci"].get(k, {}).get("overall", {}).get("ap")))
    if len(pts) < 2:
        raise LookupError("Fig 2: fewer than 2 arms with an overall PR-AUC")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, xs, xlabel in ((axes[0], PARAMS, "trainable parameters added (log scale; 0 drawn at 1)"),
                           (axes[1], MS, "added ms per trajectory (MATH-500, batch 1)")):
        for k, lab, y, c in pts:
            if k not in xs:
                continue
            x = xs[k]
            if ax is axes[0]:
                x = max(x, 1)
            f = _family(k)
            yerr = None if c is None else [[y - c[0]], [c[1] - y]]
            ax.errorbar(x, y, yerr=yerr, fmt=MARK.get(f, "o"), ms=8, color=COL.get(f, COL["ref"]),
                        mfc="white" if k == "rp_rel_ds" else COL.get(f, COL["ref"]), mew=1.6,
                        elinewidth=1.2, capsize=2)
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7, color="#333333")
        ax.set_xlabel(xlabel)
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.5, 3e10)
    axes[0].set_ylabel("ReProbe benchmark overall PR-AUC")
    axes[1].set_xlim(-1.5, 40)
    if "instep_ds" in [p[0] for p in pts]:
        y = next(p[2] for p in pts if p[0] == "instep_ds")
        axes[1].plot(35.3, y, "o", ms=8, color=COL["instep"], mfc="white", mew=1.6)
        axes[1].annotate("InStep offline\n(text pass + read)", (35.3, y), textcoords="offset points",
                         xytext=(-10, 8), fontsize=7, color="#333333", ha="right")
    fig.suptitle("Fig 2. Quality against added cost (Qwen3-8B; error bars: 95% trace-bootstrap CI)", fontsize=9)
    return _save(fig, out, "fig2_pareto")


def fig_backbones(final: dict, out: Path) -> list[str]:
    from scripts.instep_v1.paper_v1 import PRIMARY, best_of
    groups = [("passive", "passive", ("passive",)), ("maxprob", "MaxProb", ("maxprob", "maxprob_stored")),
              ("reprobe", "ReProbe retrained", ("rp_ds", "rp_q3", "rp_oss")),
              ("prm", "best PRM", None), ("instep", "InStep", None)]
    bbs = [bb for bb in ("qwen3_8b", "phi4", "qwen3_32b") if bb in final["backbones"]]
    if not bbs:
        raise LookupError("Fig 3: no backbone results")
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    w = 0.16
    drawn = 0
    for gi, (fam, lab, keys) in enumerate(groups):
        for bi, bb in enumerate(bbs):
            R = final["backbones"][bb]
            M = R["metrics"]
            if fam == "prm":
                k = best_of(M, ("prm7b", "prm800k")) or best_of(M, ("prm7b_stored", "prm800k_stored"))
            elif fam == "instep":
                k = PRIMARY[bb] if PRIMARY[bb] in M and "overall" in M[PRIMARY[bb]] else None
            else:
                k = next((x for x in keys if x in M and "overall" in M[x]), None)
            if not k:
                continue
            o = M[k]["overall"]
            x = bi + (gi - 2) * (w + 0.01)
            sd = o.get("ap_seed_sd") if len(o.get("ap_seeds", [])) > 1 else None
            ax.bar(x, o["ap"], w, color=COL[fam], edgecolor="white", linewidth=1.0)
            if sd:
                ax.errorbar(x, o["ap"], yerr=sd, color="#222222", capsize=2, elinewidth=1)
            ax.text(x, o["ap"] + (sd or 0) + 0.01, f"{o['ap']:.2f}", ha="center", va="bottom", fontsize=6.5,
                    color="#333333")
            drawn += 1
    if not drawn:
        raise LookupError("Fig 3: no bars")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[f]) for f, _, _ in groups]
    ax.legend(handles, [g[1] for g in groups], ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    ax.set_xticks(range(len(bbs)), [BBL[b] for b in bbs])
    ax.set_ylabel("overall PR-AUC (8 sets)")
    ax.set_ylim(0, 1)
    ax.set_title("Fig 3. Backbones (error bars: SD over seeds; best PRM = higher of our two PRM rows)", fontsize=9)
    return _save(fig, out, "fig3_backbones")


def fig_ladder(final: dict, out: Path) -> list[str]:
    from scripts.instep_v1.paper_v1 import LADDER, LADDER_SIDE
    R = final["backbones"].get("qwen3_8b")
    if not R:
        raise LookupError("Fig 4: no Qwen3-8B results")
    rungs = [(k, lab) for k, lab in LADDER + [LADDER_SIDE] if _overall(R, k)]
    if len(rungs) < 2:
        raise LookupError(f"Fig 4: only {len(rungs)} ladder rungs present")
    pb = R.get("pb", {})
    pbr = [(k, lab) for k, lab in (("emb_prm", "embedding only"), ("instep_prm", "InStep")) if k in pb]
    fig, axes = plt.subplots(1, 2 if pbr else 1, figsize=(9.6 if pbr else 6.4, 3.6),
                             gridspec_kw={"width_ratios": [3, 1]} if pbr else None, squeeze=False)
    ax = axes[0][0]
    for i, (k, lab) in enumerate(rungs):
        o = _overall(R, k)
        c = R["arm_ci"].get(k, {}).get("overall", {}).get("ap")
        side = k == LADDER_SIDE[0]
        col = COL["ptrue"] if k == "ptrue" else COL["passive"] if k == "passive" else COL["instep"]
        ax.errorbar(i, o["ap"], yerr=None if c is None else [[o["ap"] - c[0]], [c[1] - o["ap"]]],
                    fmt="o", ms=8, color=col, mfc="white" if side else col, mew=1.6, capsize=3)
        ax.text(i + 0.12, o["ap"], f"{o['ap']:.3f}", fontsize=7, va="center", color="#333333")
    main_idx = [i for i, (k, _) in enumerate(rungs) if k != LADDER_SIDE[0]]
    ax.plot(main_idx, [_overall(R, rungs[i][0])["ap"] for i in main_idx], color=COL["instep"], lw=1, alpha=0.4)
    ax.set_xticks(range(len(rungs)), [lab.replace(", ", ",\n").replace(" (= ", "\n(= ") for _, lab in rungs],
                  fontsize=7)
    ax.set_ylabel("ReProbe overall PR-AUC (DS tokens)")
    ax.set_title("ReProbe benchmark (95% trace-bootstrap CI; seeds averaged)", fontsize=8)
    if pbr:
        ax2 = axes[0][1]
        for i, (k, lab) in enumerate(pbr):
            v = pb[k]["auroc_mean"]
            c = R.get("pb_ci", {}).get(k, {}).get("auroc_mean")
            ax2.errorbar(i, v, yerr=None if c is None else [[v - c[0]], [c[1] - v]], fmt="o", ms=8,
                         color=COL["instep"], capsize=3)
            ax2.text(i + 0.1, v, f"{v:.3f}", fontsize=7, va="center")
        if "prm800k" in pb:
            ax2.axhline(pb["prm800k"]["auroc_mean"], color=COL["prm"], ls="--", lw=1)
            ax2.text(len(pbr) - 0.55, pb["prm800k"]["auroc_mean"], "Qwen2.5-Math-7B-PRM800K", color="#333333",
                     fontsize=6.5, va="bottom", ha="right")
        ax2.set_xticks(range(len(pbr)), [lab for _, lab in pbr], fontsize=7)
        ax2.set_xlim(-0.5, len(pbr) - 0.5)
        ax2.set_ylabel("ProcessBench first-error AUROC (mean of 4)")
        ax2.set_title("PRM800K-trained tokens", fontsize=8)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax2.margins(y=0.15)
    fig.suptitle("Fig 4. Ladder: each rung adds one ingredient (hollow: InStep without offsets)", fontsize=9)
    return _save(fig, out, "fig4_ladder")


def fig_per_layer(final: dict, out: Path) -> list[str]:
    paths = []
    for bb, R in final["backbones"].items():
        S = R.get("per_layer", {}).get("series")
        if not S:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        cols = [COL["passive"], COL["instep"], COL["reprobe"], COL["prm"], COL["maxprob"], COL["ptrue"]]
        for i, (name, pts) in enumerate(sorted(S.items())):
            x, y = zip(*pts)
            c = cols[i % len(cols)]
            low = name.lower()
            if "passive" in low or "text" in low:
                c = COL["passive"]
            elif "read" in low:
                c = COL["instep"]
            ax.plot(x, y, marker="o", ms=4, lw=2, color=c, label=name)
            ax.annotate(name.split("/")[-1], (x[-1], y[-1]), textcoords="offset points", xytext=(4, 0), fontsize=7)
        ax.set_xlabel("block")
        ax.set_ylabel("validation PR-AUC (linear probe)")
        ax.legend(fontsize=7)
        ax.set_title(f"Fig 5. Per-layer separability, {BBL.get(bb, bb)}", fontsize=9)
        paths += _save(fig, out, f"fig5_per_layer_{bb}")
    if not paths:
        raise LookupError("Fig 5: no per-layer series")
    return paths


def fig_abstention(final: dict, out: Path) -> list[str]:
    R = final["backbones"].get("qwen3_8b")
    A = (R or {}).get("abstention")
    if not A:
        raise LookupError("Fig 6: no abstention curves")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for k, v in A.items():
        f = _family(k)
        ax.plot(v["coverage"], v["risk"], lw=2, color=COL.get(f, COL["ref"]),
                ls="--" if k.endswith("_stored") or k == "rp_rel_ds" else "-",
                label=f"{R['arms'][k]['label']} (AURC {v['aurc']:.3f})")
    base = next(iter(A.values()))["base_rate"]
    ax.axhline(base, color=COL["ref"], lw=1, ls=":")
    ax.text(0.02, base, "no abstention", fontsize=7, va="bottom", color="#333333")
    ax.set_xlabel("coverage (fraction of traces kept)")
    ax.set_ylabel("fraction of kept traces with an incorrect step")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Fig 6. Abstention on the 8 test sets pooled (Qwen3-8B, trace level)", fontsize=9)
    return _save(fig, out, "fig6_abstention")


def make_all(final: dict, out: Path) -> tuple[list[str], list[str]]:
    out.mkdir(parents=True, exist_ok=True)
    paths, notes = [], []
    for fn in (fig_pareto, fig_backbones, fig_ladder, fig_per_layer, fig_abstention):
        try:
            paths += fn(final, out)
        except LookupError as e:
            notes.append(f"figure skipped: {e}")
        except Exception:  # noqa: BLE001
            notes.append(f"figure {fn.__name__} failed: {traceback.format_exc().strip().splitlines()[-1]}")
            plt.close("all")
    return paths, notes

