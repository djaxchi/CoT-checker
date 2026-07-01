#!/usr/bin/env python3
"""Plots for the per-token incorrectness trajectory (consumes s3_token_incorrectness_extract).

Reads {stem}_tokens.jsonl (+ {stem}_steps.jsonl) and answers, for the PRM800K 6k
held-out set, the question the last-token-only §15 study could not:

  1. TOKEN HEATMAP  For example incorrect steps, the step's tokens in reading order
     coloured by P(incorrect)=sigmoid(probe score), with a per-token certainty track.
     Shows *which* token fires the L28 detector.
  2. SPIKE          Is the incorrectness signal a localized spike or a plateau? Compares
     the per-step peakiness / prominence of incorrect vs correct steps, and where the
     peak sits within the step (mid-step vs the step-end the probe was trained on).
  3. COINCIDENCE    Does the firing token coincide with a certainty dip? Pooled
     probe-score vs certainty scatter, within-step argmax distance, and the
     probe<->certainty correlation.

Usage:
  python scripts/analysis/s3_token_incorrectness_plot.py \
    --tokens runs/s3_token_traj/qwen2_5_7b/prm800k_heldout_test_tokens.jsonl \
    --out runs/s3_token_traj/qwen2_5_7b/plots --probe_layer 28 --n_examples 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.token_trajectory import coincidence, spike_stats  # noqa: E402

GREEN, RED = "#2c7fb8", "#e6550d"
CMAP = "coolwarm"

# Per-token certainty metrics. "confidence" (p_top1 / logit_gap) is how peaked the
# model's OWN next-token distribution is (higher = more certain); "surprise" (nll /
# entropy) is about the realized token / distribution spread (higher = LESS certain).
# nll and p_top1 are different: nll = -log p(realized token); p_top1 = p(model's top
# token) regardless of what was written.
CERTAINTY = {
    "p_top1":    {"label": "confidence  p(top token)",             "more_certain": "high"},
    "logit_gap": {"label": "confidence margin  logit(top1-top2)",  "more_certain": "high"},
    "entropy":   {"label": "entropy (uncertainty)",                "more_certain": "low"},
    "nll":       {"label": "nll (surprise of realized token)",     "more_certain": "low"},
}


def _uncertainty(col_vals: np.ndarray, more_certain: str) -> np.ndarray:
    """Orient a certainty column so HIGHER always means LESS certain (for argmax=dip)."""
    return -col_vals if more_certain == "high" else col_vals


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def prettify(token: str) -> str:
    """Byte-level BPE token -> printable glyph (Qwen/GPT2 markers)."""
    t = token.replace("Ġ", "·").replace("▁", "·").replace("Ċ", "\\n").replace("ĉ", "\\t")
    return t if len(t) <= 12 else t[:11] + "…"


def group_by_step(rows: list[dict]) -> dict:
    steps: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        steps[r["uid"]].append(r)
    for uid in steps:
        steps[uid].sort(key=lambda r: r["tok_pos"])
    return steps


def _score_key(layer: int) -> str:
    return f"probe_score_l{layer}"


# --------------------------------------------------------------------------- #
# 1. token heatmap
# --------------------------------------------------------------------------- #

PROBE_C, CONF_C = "#d6301d", "#2166ac"   # red = probe (incorrect), blue = confidence


def plot_step_heatmap(step_rows, layer, out_path, cert="p_top1", cols=14):
    """One step, token by token: a colour strip of P(incorrect) + two aligned curves,
    P(incorrect) from the probe (red) and the model's confidence (blue), on one 0..1
    axis. Where both are high = the model is sure AND the probe says wrong."""
    pinc = sigmoid(np.array([r[_score_key(layer)] for r in step_rows]))  # 0..1
    conf = np.array([r[cert] for r in step_rows])
    cert_label = CERTAINTY[cert]["label"]
    is_prob = cert in ("p_top1", "p_realized")     # already a 0..1 probability
    toks = [prettify(r["token"]) for r in step_rows]
    T = len(step_rows)
    rows_n = int(np.ceil(T / cols))

    grid = np.full((rows_n, cols), np.nan)
    for k in range(T):
        grid[k // cols, k % cols] = pinc[k]

    fig, (axh, axc) = plt.subplots(
        2, 1, figsize=(1.05 * cols, 0.6 * rows_n + 3.2),
        gridspec_kw={"height_ratios": [rows_n + 0.5, 3.2]})

    # -- token strip, coloured by P(incorrect) --
    axh.imshow(grid, cmap=CMAP, norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
               aspect="auto")
    for k in range(T):
        r, c = divmod(k, cols)
        axh.text(c, r, toks[k], ha="center", va="center", fontsize=8, family="monospace",
                 color="white" if abs(pinc[k] - 0.5) > 0.32 else "black")
    axh.set_xticks([]); axh.set_yticks([])
    lab = "INCORRECT step" if step_rows[0]["label"] == 1 else "correct step"
    axh.set_title(f"{step_rows[0]['uid']}   [{lab}]\n"
                  f"token colour = P(incorrect) from the L{layer} probe   "
                  f"(red = probe says WRONG, blue = says right)", fontsize=9)

    # -- two aligned per-token curves on one probability axis --
    x = np.arange(T)
    axc.plot(x, pinc, "-o", ms=3, lw=1.6, color=PROBE_C,
             label="P(incorrect)  — the probe's score")
    if is_prob:
        axc.plot(x, conf, "-o", ms=3, lw=1.6, color=CONF_C,
                 label="confidence  — model's p(top token)")
        both = (pinc > 0.5) & (conf > 0.5)
        axc.fill_between(x, 0, 1, where=both, step="mid", color=PROBE_C, alpha=0.10,
                         label="confidently WRONG  (both above 0.5)")
        axc.set_ylim(0, 1); axc.set_ylabel("probability  (0 to 1)", fontsize=9)
    else:                                          # non-probability certainty -> twin axis
        axc.set_ylim(0, 1); axc.set_ylabel("P(incorrect)", fontsize=9, color=PROBE_C)
        axt = axc.twinx(); axt.plot(x, conf, "-", color=CONF_C, lw=1.4, alpha=0.85)
        axt.set_ylabel(cert_label, fontsize=8, color=CONF_C)
    axc.axhline(0.5, color="grey", lw=0.8, ls=":")
    axc.set_xlabel("token position   (0 = first token of the step  ->  last token)",
                   fontsize=9)
    axc.set_xlim(-0.5, T - 0.5)
    axc.legend(fontsize=8, loc="upper left", framealpha=0.92, ncol=1)
    axc.set_title("per token: how sure the model is (blue) vs how WRONG the probe thinks "
                  "it is (red)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def select_examples(steps, layer, n):
    """Teaching set for the confidently-wrong story, by mean P(incorrect) & confidence:
      confwrong        incorrect step, BOTH probe and confidence high (the headline case)
      uncertain_wrong  incorrect step, probe high but confidence LOW (contrast)
      confident_correct correct step, confidence high and probe low (control)
    """
    sk = _score_key(layer)
    def means(rows):
        pinc = sigmoid(np.array([r[sk] for r in rows]))
        conf = np.array([r["p_top1"] for r in rows])
        return float(pinc.mean()), float(conf.mean())

    inc, cor = [], []
    for u, rw in steps.items():
        if rw[0]["n_step_tokens"] < 6:
            continue
        mp, mc = means(rw)
        (inc if rw[0]["label"] == 1 else cor).append((u, mp, mc))
    confwrong = sorted(inc, key=lambda t: min(t[1], t[2]), reverse=True)   # both high
    uncwrong = sorted(inc, key=lambda t: t[1] - t[2], reverse=True)        # wrong, unsure
    confright = sorted(cor, key=lambda t: t[2] - t[1], reverse=True)       # sure, right
    k = max(1, n // 3)
    picks = []
    for grp, tag in ((confwrong, "confwrong"), (uncwrong, "uncertain_wrong"),
                     (confright, "confident_correct")):
        grp = [u for u, *_ in grp]
        for u in grp[:k]:
            if u not in [p[0] for p in picks]:
                picks.append((u, tag))
    return picks[:n]


# --------------------------------------------------------------------------- #
# 2. spike distributions
# --------------------------------------------------------------------------- #

def plot_spike(steps, layer, out_path):
    def feats(rows):
        return spike_stats([r[_score_key(layer)] for r in rows])
    inc = [feats(rw) for rw in steps.values() if rw[0]["label"] == 1 and rw[0]["n_step_tokens"] >= 4]
    cor = [feats(rw) for rw in steps.values() if rw[0]["label"] == 0 and rw[0]["n_step_tokens"] >= 4]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, title in (
        (axes[0], "peakiness", "Does one token stand out in the step?\n"
                               "top token's height above the step average (sigma)"),
        (axes[1], "prominence", "How far the top token sits above the typical token\n"
                                "(probe logits)"),
        (axes[2], "argmax_frac", "Where in the step is the top token?\n"
                                 "(0 = first token, 1 = last token)")):
        vi = np.array([f[key] for f in inc]); vc = np.array([f[key] for f in cor])
        lo, hi = np.percentile(np.concatenate([vi, vc]), [1, 99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(vc, bins=bins, color=GREEN, alpha=0.55, density=True, label="correct")
        ax.hist(vi, bins=bins, color=RED, alpha=0.55, density=True, label="incorrect")
        ax.axvline(np.median(vc), color=GREEN, lw=1.5, ls="--")
        ax.axvline(np.median(vi), color=RED, lw=1.5, ls="--")
        ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
    fig.suptitle(f"L{layer} within-step spike structure (incorrect vs correct)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {
        "peakiness_median_incorrect": float(np.median([f["peakiness"] for f in inc])),
        "peakiness_median_correct": float(np.median([f["peakiness"] for f in cor])),
        "argmax_at_last_token_frac_incorrect":
            float(np.mean([f["argmax_frac"] == 1.0 for f in inc])),
        "argmax_at_last_token_frac_correct":
            float(np.mean([f["argmax_frac"] == 1.0 for f in cor])),
        "n_incorrect": len(inc), "n_correct": len(cor),
    }


# --------------------------------------------------------------------------- #
# 3. coincidence
# --------------------------------------------------------------------------- #

def plot_coincidence(rows, steps, layer, out_path, cert="p_top1"):
    sk = _score_key(layer)
    score = np.array([r[sk] for r in rows])
    label = np.array([r["label"] for r in rows])
    cvals = np.array([r[cert] for r in rows])
    meta = CERTAINTY[cert]
    label_txt, more_certain = meta["label"], meta["more_certain"]
    dip_word = "least-certain" if more_certain == "high" else "most-surprising"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) pooled scatter probe score vs the certainty metric
    ax = axes[0]
    for lb, col, name in ((0, GREEN, "correct"), (1, RED, "incorrect")):
        m = label == lb
        ax.scatter(cvals[m], score[m], s=4, c=col, alpha=0.25, linewidths=0, label=name)
    r_all = float(np.corrcoef(cvals, score)[0, 1])
    ax.set_xlabel(f"per-token {label_txt}  ->")
    ax.set_ylabel(f"L{layer} probe score  (higher = more incorrect)  ->")
    trend = ("up-right = fires MORE on confident tokens" if more_certain == "high"
             else "down-right = fires LESS on surprising tokens")
    ax.set_title(f"1 dot = 1 token.  {trend}\ncorr = {r_all:+.2f}", fontsize=9)
    ax.set_xlim(np.percentile(cvals, [0.5, 99.5]))
    if more_certain == "high":
        ax.text(0.97, 0.03, "confidently wrong ->", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=PROBE_C, style="italic")
    ax.legend(fontsize=8)

    # (b) within-step distance: firing token vs the LEAST-certain token
    ax = axes[1]
    dist, corr_c = [], []
    for rw in steps.values():
        if rw[0]["label"] != 1 or rw[0]["n_step_tokens"] < 4:
            continue
        s = np.array([r[sk] for r in rw])
        cv = np.array([r[cert] for r in rw])
        c = coincidence(s, _uncertainty(cv, more_certain))  # argmax(unc) = least-certain tok
        dist.append(c["argmax_distance_frac"])
        wc = np.corrcoef(s, cv)[0, 1]  # signed corr with the certainty metric itself
        if not np.isnan(wc):
            corr_c.append(wc)
    ax.hist(dist, bins=np.linspace(0, 1, 21), color=RED, alpha=0.7)
    ax.axvline(np.median(dist), color="k", lw=1.5, ls="--",
               label=f"median={np.median(dist):.2f}")
    ax.set_xlabel(f"gap between them  (0 = same token,  1 = opposite ends of the step)")
    ax.set_title(f"Is the probe's top token the {dip_word} token?\n"
                 "(each incorrect step; left pile = yes, spread = no)", fontsize=9)
    ax.legend(fontsize=8)

    # (c) within-step corr(probe, certainty) distribution
    ax = axes[2]
    ax.hist(corr_c, bins=np.linspace(-1, 1, 31), color="#7a0177", alpha=0.7)
    ax.axvline(np.median(corr_c), color="k", lw=1.5, ls="--",
               label=f"median r={np.median(corr_c):.2f}")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"corr(probe, {cert}) inside one step   (<0 left | 0 | right >0)")
    ax.set_title("Inside a step, do probe score & confidence rise together?\n"
                 "(one bar = one incorrect step; right of 0 = yes)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    inc_m = label == 1
    # pooled corr of the probe score with EVERY certainty metric, so nothing is hidden.
    pooled_all = {k: float(np.corrcoef(np.array([r[k] for r in rows]), score)[0, 1])
                  for k in CERTAINTY}
    return {
        "certainty_metric": cert,
        f"pooled_corr_probe_{cert}_all": r_all,
        f"pooled_corr_probe_{cert}_incorrect":
            float(np.corrcoef(cvals[inc_m], score[inc_m])[0, 1]),
        "pooled_corr_probe_vs_all_certainty": pooled_all,
        "median_argmax_distance_incorrect": float(np.median(dist)),
        f"median_within_step_corr_probe_{cert}_incorrect": float(np.median(corr_c)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=Path, required=True, help="{stem}_tokens.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe_layer", type=int, default=28)
    ap.add_argument("--certainty", choices=list(CERTAINTY), default="p_top1",
                    help="per-token certainty axis. p_top1/logit_gap = the model's "
                         "CONFIDENCE (its own top-token prob/margin); nll/entropy = "
                         "surprise/uncertainty. Default p_top1 (confidence).")
    ap.add_argument("--n_examples", type=int, default=9)
    ap.add_argument("--cols", type=int, default=14, help="tokens per heatmap row")
    args = ap.parse_args()
    (args.out / "examples").mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.tokens)
    if _score_key(args.probe_layer) not in rows[0]:
        sys.exit(f"{_score_key(args.probe_layer)} not in tokens file; "
                 f"available: {[k for k in rows[0] if k.startswith('probe_score')]}")
    steps = group_by_step(rows)
    print(f"[plot] {len(rows)} tokens over {len(steps)} steps", flush=True)

    if args.certainty not in rows[0]:
        sys.exit(f"certainty '{args.certainty}' not in tokens file; "
                 f"available: {[k for k in CERTAINTY if k in rows[0]]}")

    made = []
    for uid, tag in select_examples(steps, args.probe_layer, args.n_examples):
        safe = uid.replace("::", "__").replace("/", "_")
        p = args.out / "examples" / f"{tag}__{safe}.png"
        plot_step_heatmap(steps[uid], args.probe_layer, p, cert=args.certainty, cols=args.cols)
        made.append(str(p))

    spike_metrics = plot_spike(steps, args.probe_layer, args.out / "spike_structure.png")
    coin_metrics = plot_coincidence(rows, steps, args.probe_layer,
                                    args.out / "coincidence.png", cert=args.certainty)
    made += [str(args.out / "spike_structure.png"), str(args.out / "coincidence.png")]

    metrics = {"probe_layer": args.probe_layer, "certainty_metric": args.certainty,
               "n_steps": len(steps), "n_tokens": len(rows),
               "spike": spike_metrics, "coincidence": coin_metrics}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Report the raw incorrect-vs-correct peakiness GAP, not a knife-edge binary label.
    peak_gap = (spike_metrics["peakiness_median_incorrect"]
                - spike_metrics["peakiness_median_correct"])
    verdict = (f"gap {peak_gap:+.2f} sigma -> "
               + ("essentially diffuse (no localized spike)" if abs(peak_gap) < 0.25
                  else "some localization"))
    summary = [
        "# Per-token incorrectness trajectory (PRM800K 6k held-out)\n",
        f"- probe layer L{args.probe_layer}; {len(steps)} steps, {len(rows)} tokens\n",
        "## Spike structure",
        f"- peakiness median: incorrect {spike_metrics['peakiness_median_incorrect']:.2f} "
        f"vs correct {spike_metrics['peakiness_median_correct']:.2f} -> **{verdict}**",
        f"- peak token is the LAST token in "
        f"{spike_metrics['argmax_at_last_token_frac_incorrect']*100:.0f}% of incorrect steps "
        f"(vs {spike_metrics['argmax_at_last_token_frac_correct']*100:.0f}% correct) "
        "-- how much of the firing is the step-end position the probe was trained on\n",
        f"## Coincidence with certainty (metric = {args.certainty})",
        f"- pooled corr(probe, {args.certainty}) = "
        f"{coin_metrics[f'pooled_corr_probe_{args.certainty}_all']:.2f} "
        f"(incorrect-only {coin_metrics[f'pooled_corr_probe_{args.certainty}_incorrect']:.2f})",
        "- pooled corr(probe, .) for every metric: "
        + ", ".join(f"{k} {v:+.2f}"
                    for k, v in coin_metrics["pooled_corr_probe_vs_all_certainty"].items()),
        f"- within incorrect steps: median |argmax(probe)-argmax(least-certain)| = "
        f"{coin_metrics['median_argmax_distance_incorrect']:.2f} of step length; "
        f"median within-step corr(probe,{args.certainty}) = "
        f"{coin_metrics[f'median_within_step_corr_probe_{args.certainty}_incorrect']:.2f}",
        "  (for p_top1: POSITIVE corr => probe fires on CONFIDENT tokens = 'confidently",
        "   wrong'; near-0 distance => firing token is the least-certain token)\n",
        "## Plots",
        "- `spike_structure.png`, `coincidence.png`, `examples/*.png`",
    ]
    (args.out / "token_incorrectness_report.md").write_text("\n".join(summary))

    print("[plot] wrote:")
    for p in made + [str(args.out / "metrics.json"),
                     str(args.out / "token_incorrectness_report.md")]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
