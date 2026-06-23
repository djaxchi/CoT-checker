#!/usr/bin/env python3
"""Stage 0 A/B battery: is the L28 correctness probe reducible to model surprise?

Joins the per-step confidence sidecar (``encode_fork_confidence.py``) to the existing
fork hidden states + probe and asks, on pooled positive/negative steps, three things:

  A/B-1  SUBSUMPTION   probe-score AUC raw vs after residualizing out the confidence
                       features {nll_mean, nll_max, entropy_mean, logit_gap_mean}.
                       Survives -> the probe is NOT just surprise.  Collapses -> it is.
  A/B-2  SUFFICIENCY   AUC of each confidence feature *alone* as a label predictor.
                       If a feature matches the probe AUC and the probe collapses under
                       residualization, the probe == that confidence meter.
  A/B-3  PARTIAL CORR  partial_corr(score, feat | label) and corr(feat, label), mirroring
                       the numeric analysis that label-neutralized "numeric".
  REVERSE              residualize each feature on the score; does it still predict the
                       label?  (does the probe subsume the confidence feature?)

The verdict band is driven by how much of the probe's AUC lift over chance is removed
when all confidence features are residualized out.

Inputs:
  --h          runs/s1_model_size_dense/<tag>/forks/forks_val_items_h.npy
  --confidence runs/.../forks_val_items_confidence.jsonl   (from encode_fork_confidence)
  --probe      runs/s1_model_size_dense/<tag>/linear_probe.pt
  --out        runs/fork_rep_audit/<tag>/confidence_battery
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the exact, already-tested estimators from the margin-driver analysis.
from scripts.inspect_margin_drivers import (  # noqa: E402
    auc, load_probe, partial_corr, read_jsonl, residualize,
)

CONF_FEATURES = ("nll_mean", "nll_max", "nll_last", "nll_first",
                 "entropy_mean", "logit_gap_mean")
# Features used for the headline subsumption test (the cleanest surprise/uncertainty set).
SUBSUME_FEATURES = ("nll_mean", "nll_max", "entropy_mean", "logit_gap_mean")


def verdict_band(frac_removed: float) -> str:
    if frac_removed < 0.25:
        return "probe is NOT reducible to confidence/surprise"
    if frac_removed < 0.60:
        return "confidence is a PARTIAL driver of the probe"
    return "probe IS largely a confidence/surprise meter"


def run_battery(score, label, conf):
    """conf: dict feature -> (n,) array. Returns the metrics dict."""
    raw = auc(score, label)
    lift = max(raw - 0.5, 1e-9)

    # A/B-1 subsumption: remove each feature, then all subsume features together.
    per_feat_resid = {}
    for k in CONF_FEATURES:
        if k in conf:
            per_feat_resid[k] = auc(residualize(score, conf[k][:, None]), label)
    sub_cols = np.column_stack([conf[k] for k in SUBSUME_FEATURES if k in conf])
    auc_resid_all = auc(residualize(score, sub_cols), label)
    frac_removed = (raw - auc_resid_all) / lift

    # A/B-2 sufficiency: each feature alone as a label predictor (orientation-free).
    sufficiency = {}
    for k in CONF_FEATURES:
        if k in conf:
            a = auc(conf[k], label)
            sufficiency[k] = {"auc": round(a, 4), "abs_lift": round(abs(a - 0.5), 4)}

    # A/B-3 partial corr + reverse subsumption.
    attribution, reverse = {}, {}
    lab = label.astype(float)
    for k in CONF_FEATURES:
        if k not in conf:
            continue
        attribution[k] = {
            "corr_with_score": round(partial_corr(score, conf[k], None), 4),
            "partial_corr_given_label": round(
                partial_corr(score, conf[k], lab[:, None]), 4),
            "corr_with_label": round(partial_corr(lab, conf[k], None), 4),
        }
        # does the confidence feature still predict the label after removing the score?
        reverse[k] = round(auc(residualize(conf[k], score[:, None]), label), 4)

    return {
        "n_steps": int(len(label)),
        "base_rate_incorrect": float(label.mean()),
        "auc_probe_raw": round(raw, 4),
        "auc_probe_after_removing_each_confidence_feature": {
            k: round(v, 4) for k, v in per_feat_resid.items()},
        "auc_probe_after_removing_all_confidence": round(auc_resid_all, 4),
        "frac_probe_lift_removed_by_confidence": round(frac_removed, 3),
        "verdict": verdict_band(frac_removed),
        "sufficiency_each_feature_alone": sufficiency,
        "attribution": attribution,
        "reverse_feature_auc_after_removing_score": reverse,
    }


def _load(args):
    H = np.load(args.h).astype(np.float64)
    conf_rows = [r for r in read_jsonl(args.confidence)
                 if r.get("role") in ("positive", "negative")
                 and r.get("nll_mean") is not None]
    w, b = load_probe(args.probe, H.shape[1])
    idx = np.array([r["row"] for r in conf_rows])
    label = np.array([1 if r["role"] == "negative" else 0 for r in conf_rows])
    score = H[idx] @ w + b
    conf = {k: np.array([float(r[k]) for r in conf_rows], dtype=np.float64)
            for k in CONF_FEATURES if conf_rows and k in conf_rows[0]}
    return score, label, conf, idx


def _plots(out, score, label, conf, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    GREEN, RED = "#2c7fb8", "#e6550d"
    for feat, fname in [("nll_mean", "score_vs_nll.png"),
                        ("entropy_mean", "score_vs_entropy.png"),
                        ("logit_gap_mean", "score_vs_logit_gap.png")]:
        if feat not in conf:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for lab, col, name in [(0, GREEN, "correct"), (1, RED, "incorrect")]:
            mk = label == lab
            ax.scatter(conf[feat][mk], score[mk], s=7, c=col, alpha=0.4,
                       label=name, linewidths=0)
        r = np.corrcoef(conf[feat], score)[0, 1]
        ax.set_xlabel(f"{feat} (nats)" if "nll" in feat or "entropy" in feat else feat)
        ax.set_ylabel("probe score (higher=incorrect)")
        ax.set_title(f"probe score vs {feat}  (r={r:.2f}; is the separating axis the "
                     f"surprise axis?)")
        ax.legend(); fig.tight_layout()
        fig.savefig(out / "plots" / fname, dpi=150); plt.close(fig)


def _report(m):
    L = ["# Stage 0 confidence battery: is the probe just model surprise?\n",
         f"- steps: {m['n_steps']}  (incorrect base rate {m['base_rate_incorrect']:.3f})\n",
         "## A/B-1 Subsumption (decisive)\n",
         f"- probe AUC raw = **{m['auc_probe_raw']:.3f}**",
         f"- after removing ALL confidence features = "
         f"**{m['auc_probe_after_removing_all_confidence']:.3f}**",
         f"- fraction of the probe's lift removed by confidence = "
         f"**{m['frac_probe_lift_removed_by_confidence']:.2f}** -> **{m['verdict']}**\n",
         "Per-feature (AUC after removing just that feature):"]
    for k, v in m["auc_probe_after_removing_each_confidence_feature"].items():
        L.append(f"- {k}: {v:.3f}")
    L += ["\n## A/B-2 Sufficiency (each confidence feature alone predicts label)\n",
          "| feature | AUC | |AUC-0.5| |", "|---|---|---|"]
    for k, v in m["sufficiency_each_feature_alone"].items():
        L.append(f"| {k} | {v['auc']:+.3f} | {v['abs_lift']:.3f} |")
    L += ["\n## A/B-3 Attribution (partial corr with score)\n",
          "| feature | corr w/ score | partial (given label) | corr w/ label | "
          "reverse AUC (feat after removing score) |",
          "|---|---|---|---|---|"]
    for k, a in m["attribution"].items():
        rev = m["reverse_feature_auc_after_removing_score"].get(k, float("nan"))
        L.append(f"| {k} | {a['corr_with_score']:+.3f} | "
                 f"{a['partial_corr_given_label']:+.3f} | {a['corr_with_label']:+.3f} | "
                 f"{rev:.3f} |")
    L += ["\n## Plots",
          "- `score_vs_{nll,entropy,logit_gap}.png` - if the probe were a surprise meter,",
          "  the correct/incorrect classes separate ALONG the confidence axis; if it is a",
          "  distinct correctness signal they separate ORTHOGONAL to it.\n",
          "## Reading",
          "- frac removed < 0.25: confidence does not explain the probe -> Stage 1 (on-policy).",
          "- frac removed > 0.60: the probe is largely surprise -> characterize/calibrate that.",
          "- A feature with high sufficiency AUC AND high `corr_with_label` is a real",
          "  confound; high partial-corr-with-score but LOW corr_with_label is label-neutral",
          "  modulation (cf the 'numeric' wrinkle), not the signal."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", required=True, type=Path)
    ap.add_argument("--confidence", required=True, type=Path)
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()
    (args.out / "plots").mkdir(parents=True, exist_ok=True)

    score, label, conf, idx = _load(args)
    if label.sum() == 0 or (1 - label).sum() == 0:
        sys.exit("[battery] need both positive and negative steps.")
    m = run_battery(score, label, conf)
    (args.out / "metrics.json").write_text(json.dumps(m, indent=2))
    (args.out / "confidence_battery_report.md").write_text(_report(m))

    import csv
    with open(args.out / "join.csv", "w", newline="") as f:
        wr = csv.writer(f)
        cols = ["row", "label_incorrect", "score", *conf.keys()]
        wr.writerow(cols)
        for i in range(len(label)):
            wr.writerow([int(idx[i]), int(label[i]), f"{score[i]:.5f}",
                         *[f"{conf[k][i]:.5f}" for k in conf]])

    if not args.no_plots:
        try:
            _plots(args.out, score, label, conf, m)
        except Exception as e:  # plotting is non-essential on login nodes
            print(f"[battery] plots skipped: {e}")

    print(f"[battery] probe AUC {m['auc_probe_raw']:.3f} -> "
          f"{m['auc_probe_after_removing_all_confidence']:.3f} after removing confidence "
          f"({m['frac_probe_lift_removed_by_confidence']:.2f} of lift) -> {m['verdict']}")
    print(f"[battery] wrote {args.out}")


if __name__ == "__main__":
    main()
