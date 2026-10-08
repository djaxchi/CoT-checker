#!/usr/bin/env python3
"""Post-hoc analysis for bidirectional_token_probe_v1 (plan 11-12, 16).

Inputs: the eval outputs (metrics.json, paired_differences.json), the final
fits' predictions and histories, the frozen manifest, the diagnostics JSON.
Writes into --out_dir:
  structural_baseline.json   logistic on position/continuation features only
  diagnostics_summary.json   paired score changes under future perturbations
  qualitative_audit.md       deterministic causal/full disagreement excerpts
  figures/*.png              paired conditions, horizon, score traces,
                             learning curves, diagnostics
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.eval.contextual_probe_metrics import (  # noqa: E402
    auroc, best_f1_threshold, best_pb_threshold, pb_trace_metrics, step_metrics,
)

PB = ("pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath")
TEST_NAME = "PRM800K test"
EXP = "bidirectional_token_probe_v1"
ARMS = ("local", "causal", "future1", "full")
EVAL_SPLITS = ("train", "calib", "test") + PB


def jl(p):
    return [json.loads(l) for l in open(p)]


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


# ------------------------------------------------------------- structural

def structural(manifest: Path) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    enc = {r["trace_id"]: r for r in jl(manifest / "encode_manifest.jsonl")}
    data = {}
    for sp in EVAL_SPLITS:
        X, y, keys = [], [], []
        for m in jl(manifest / "meta" / f"{sp}.jsonl"):
            e = enc.get(m["trace_id"])
            if e is None:
                continue
            ss, se, n, T = e["step_starts"], e["step_ends"], e["n_tokens"], e["n_steps"]
            for k in range(T):
                X.append([k, se[k] - ss[k], ss[k], T - 1 - k, n - se[k]])
                y.append(m["y"][k] if m["label_mask"][k] else -1)
                keys.append((m["trace_id"], k, m.get("pb_label")))
        data[sp] = (np.asarray(X, float), np.asarray(y), keys)
    Xtr, ytr, _ = data["train"]
    lab = ytr >= 0
    sc = StandardScaler().fit(Xtr[lab])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr[lab]), ytr[lab])
    p = {sp: clf.predict_proba(sc.transform(data[sp][0]))[:, 1] for sp in data}
    cm = data["calib"][1] >= 0
    thr, _ = best_f1_threshold(p["calib"][cm], data["calib"][1][cm])
    out = {"features": ["step_index", "target_tokens", "prefix_tokens", "later_steps", "later_tokens"],
           "coef_standardized": clf.coef_[0].tolist(), "calib_threshold": thr}
    for sp in ("test",) + PB:
        X, y, keys = data[sp]
        m = y >= 0
        r = step_metrics(p[sp][m], y[m], thr)
        r["oracle_f1"] = best_f1_threshold(p[sp][m], y[m])[1]
        if sp in PB:
            seqs = defaultdict(list); labels = {}
            for (tid, k, pl), s in zip(keys, p[sp]):
                seqs[tid].append(s); labels[tid] = pl
            ids = sorted(seqs)
            r["trace"] = pb_trace_metrics([seqs[t] for t in ids], [labels[t] for t in ids], thr)
            r["trace_oracle_F1_PB"] = best_pb_threshold([seqs[t] for t in ids], [labels[t] for t in ids])[1]
        out[sp] = r
    out["pb_mean_F1_PB"] = float(np.mean([out[sp]["trace"]["F1_PB"] for sp in PB]))
    return out


# ------------------------------------------------------------- diagnostics

def diagnostics(diag: dict, thresholds: dict) -> dict:
    rows = diag["rows"]
    out = {"n_selected": len(rows), "cell_sizes": diag["cell_sizes"], "variants": {}}
    runs = sorted(rows[0]["variants"]["unchanged"])
    for v in ("shuffled", "no_answer", "future_hidden"):
        el = [r for r in rows if r["variants"].get(v) is not None]
        res = {"eligible": len(el)}
        if v != "future_hidden":
            chk = [r["checks"][v] for r in el]
            res["prefix_target_ids_identical"] = int(sum(c["prefix_target_ids_identical"] for c in chk))
            res["prefix_target_state_max_dev_rel_max"] = float(max(c["prefix_target_state_max_dev_rel"] for c in chk)) if chk else None
        for run in runs:
            d = np.array([sig(r["variants"][v][run]) - sig(r["variants"]["unchanged"][run]) for r in el])
            y = np.array([r["y"] for r in el])
            base = np.array([sig(r["variants"]["unchanged"][run]) for r in el])
            new = np.array([sig(r["variants"][v][run]) for r in el])
            thr = thresholds.get(run)
            rr = {"mean_dP": float(d.mean()), "mean_abs_dP": float(np.abs(d).mean()),
                  "mean_dP_incorrect": float(d[y == 1].mean()) if (y == 1).any() else None,
                  "mean_dP_correct": float(d[y == 0].mean()) if (y == 0).any() else None,
                  "auroc_unchanged": auroc(base, y), "auroc_variant": auroc(new, y)}
            if thr is not None:
                rr["f1_unchanged"] = step_metrics(base, y, thr)["f1"]
                rr["f1_variant"] = step_metrics(new, y, thr)["f1"]
            res[run] = rr
        out["variants"][v] = res
    return out


# ------------------------------------------------------------- figures

def figures(res_dir: Path, fits: Path, out: Path, structural_res: dict | None, diag_sum: dict | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    met = json.loads((res_dir / "metrics.json").read_text())
    seeds = sorted({int(k.split("/")[1]) for k in met})
    paths = []
    colors = {"local": "#8c8c8c", "causal": "#1f77b4", "future1": "#ff7f0e", "full": "#d62728"}

    def per(metric_fn):
        return {a: [metric_fn(met[f"{a}/{s}"]) for s in seeds if f"{a}/{s}" in met] for a in ARMS}

    tname = TEST_NAME
    panels = [(f"{tname}: incorrect-step F1\n(calib threshold)", per(lambda m: m["prm_test"]["f1"]),
               per(lambda m: m["prm_test"]["oracle_f1"]), structural_res and structural_res["test"]["f1"]),
              (f"{tname}: AUROC", per(lambda m: m["prm_test"]["auroc"]), None,
               structural_res and structural_res["test"]["auroc"]),
              ("ProcessBench: mean F1_PB over 4 subsets\n(calib threshold)", per(lambda m: m["pb_mean_F1_PB"]),
               per(lambda m: np.mean([m[sp]["trace_oracle"]["F1_PB"] for sp in PB])),
               structural_res and structural_res["pb_mean_F1_PB"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (title, vals, oracle, struct) in zip(axes, panels):
        for i, a in enumerate(ARMS):
            v = vals[a]
            ax.scatter([i] * len(v), v, color=colors[a], s=30, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=colors[a], lw=2.5)
            if oracle:
                ax.scatter([i + 0.12] * len(oracle[a]), oracle[a], marker="x", color=colors[a], alpha=0.6, s=25)
        for s_i, s in enumerate(seeds):
            ax.plot(range(4), [vals[a][s_i] for a in ARMS], color="#bbbbbb", lw=0.8, zorder=1)
        allv = [x for a in ARMS for x in vals[a] + (oracle[a] if oracle else [])]
        lo, hi = min(allv), max(allv)
        pad = max(0.01, 0.15 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        if struct is not None:
            if lo - pad <= struct <= hi + pad:
                ax.axhline(struct, ls="--", color="#555555", lw=1, label="structural baseline")
                ax.legend(fontsize=8, loc="lower right")
            else:
                ax.text(0.99, 0.02, f"structural baseline: {struct:.3f} (off scale)", transform=ax.transAxes,
                        ha="right", fontsize=8, color="#555555")
        ax.set_xticks(range(4)); ax.set_xticklabels(["local", "causal", "future1\n(+1 step)", "full"])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].text(0.01, 0.96, "dots: seeds; bar: mean; x: oracle threshold (ceiling)", transform=axes[0].transAxes, fontsize=7)
    fig.suptitle(f"{EXP}: paired conditions (Qwen3-8B L35, {len(seeds)} seeds; grey lines pair seeds)")
    fig.tight_layout()
    p = out / "paired_conditions.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # horizon: per PB subset F1_PB and step AUROC vs condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for sp in PB:
        y1 = [np.mean([met[f"{a}/{s}"][sp]["trace"]["F1_PB"] for s in seeds]) for a in ARMS]
        y2 = [np.mean([met[f"{a}/{s}"][sp]["step_known_labels"]["auroc"] for s in seeds]) for a in ARMS]
        axes[0].plot(range(4), y1, marker="o", label=sp[3:])
        axes[1].plot(range(4), y2, marker="o", label=sp[3:])
    y2 = [np.mean([met[f"{a}/{s}"]["prm_test"]["auroc"] for s in seeds]) for a in ARMS]
    axes[1].plot(range(4), y2, marker="s", color="k", label=TEST_NAME)
    for ax, t in zip(axes, ["ProcessBench F1_PB (calib threshold), seed mean",
                            "Known-label step AUROC, seed mean"]):
        ax.set_xticks(range(4)); ax.set_xticklabels(["own step", "past (causal)", "+1 step", "all future"])
        ax.set_title(t, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Performance versus visible horizon")
    fig.tight_layout()
    p = out / "horizon.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # learning curves
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 3.6), squeeze=False)
    for ax, s in zip(axes[0], seeds):
        for a in ARMS:
            h = fits / f"final_s{s}_{a}" / a / "history.json"
            if h.exists():
                hh = json.loads(h.read_text())
                ax.plot([x["epoch"] for x in hh], [x["dev_f1"] for x in hh], color=colors[a], label=a)
        ax.set_title(f"seed {s}: dev F1 (best threshold)", fontsize=10); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    p = out / "learning_curves.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    if diag_sum:
        fig, ax = plt.subplots(figsize=(7, 4))
        vs = ["shuffled", "no_answer", "future_hidden"]
        for j, arm in enumerate(["full", "causal"]):
            for k, (yk, lab) in enumerate([("mean_dP_incorrect", "incorrect"), ("mean_dP_correct", "correct")]):
                vals = [np.mean([diag_sum["variants"][v][f"{arm}/{s}"][yk] for s in seeds]) for v in vs]
                ax.bar(np.arange(3) + (2 * j + k) * 0.2 - 0.3, vals, width=0.2, label=f"{arm} | {lab} targets")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{v}\n(n={diag_sum['variants'][v]['eligible']})" for v in vs])
        ax.set_ylabel("mean change in P(incorrect)"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        ax.set_title("Future perturbations (bounded diagnostic, seed mean)", fontsize=10)
        fig.tight_layout()
        p = out / "diagnostics.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def load_preds(path):
    out = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            out[r["split"]].setdefault(r["trace_id"], [None] * r["n_steps"])[r["step"]] = sig(r["logit"])
    return out


def score_traces_and_audit(res_dir, fits, manifest, out_fig, out_md, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    met = json.loads((res_dir / "metrics.json").read_text())
    P = {a: load_preds(fits / f"final_s{seed}_{a}" / a / "predictions.jsonl.gz") for a in ("causal", "full")}
    thr = {a: met[f"{a}/{seed}"]["calib"]["threshold"] for a in P}
    inputs, meta = {}, {}
    for sp in ("test",) + PB:
        for r in jl(manifest / "inputs" / f"{sp}.jsonl"):
            inputs[r["trace_id"]] = r
        for r in jl(manifest / "meta" / f"{sp}.jsonl"):
            meta[r["trace_id"]] = r
    cats = defaultdict(list)
    for sp in ("test",) + PB:
        for tid in sorted(P["full"][sp]):
            m = meta[tid]
            for k, (yk, mk) in enumerate(zip(m["y"], m["label_mask"])):
                fc = P["causal"][sp][tid][k] >= thr["causal"]
                ff = P["full"][sp][tid][k] >= thr["full"]
                if mk and fc != ff:
                    right_full = ff == bool(yk)
                    cats["full_gain" if right_full else "full_regression"].append((sp, tid, k))
            fe = m.get("first_error")
            if fe is not None:
                pf = next((i for i, s in enumerate(P["full"][sp][tid]) if s >= thr["full"]), -1)
                if 0 <= pf < fe:
                    cats["full_premature_alarm"].append((sp, tid, pf))
                if fe < len(m["y"]) - 2:
                    cats["post_error_pattern"].append((sp, tid, fe))
    lines = [f"# Qualitative audit: causal vs full disagreements (seed {seed})", "",
             "Deterministic sample: within each category, items sorted by sha1(trace_id:step), first 4.",
             f"Thresholds (calibration-selected): causal {thr['causal']:.4f}, full {thr['full']:.4f}.",
             "Scores are P(incorrect). Post-error validity judgements below are qualitative only; no labels exist there.", ""]
    for cat, items in cats.items():
        items = sorted(items, key=lambda x: hashlib.sha1(f"{x[1]}:{x[2]}".encode()).hexdigest())[:4]
        lines += [f"## {cat} ({len(cats[cat])} total)", ""]
        for sp, tid, k in items:
            m, tr = meta[tid], inputs[tid]
            lines.append(f"### {sp} {tid} step {k}  (first_error={m.get('first_error')}, label y={m['y'][k]})")
            lines.append(f"Problem: {tr['problem'][:300]}")
            lo, hi = max(0, k - 1), min(len(tr['steps']), k + 3)
            for j in range(lo, hi):
                lab = m['y'][j] if m['label_mask'][j] else 'unknown'
                lines.append(f"- step {j} [label {lab}] causal={P['causal'][sp][tid][j]:.3f} full={P['full'][sp][tid][j]:.3f}: "
                             f"{tr['steps'][j][:240]!r}")
            lines.append("")
    out_md.write_text("\n".join(lines))
    # score-trace figure: 4 PB error traces + 2 PRM traces, deterministic
    picks = []
    for sp in ("pb_math", "pb_olympiadbench", "pb_omnimath", "pb_gsm8k", "test"):
        cand = sorted((t for t in P["full"][sp] if meta[t].get("first_error") is not None
                       and len(meta[t]["y"]) >= 6), key=lambda t: hashlib.sha1(t.encode()).hexdigest())
        picks += [(sp, t) for t in cand[:(2 if sp == "test" else 1)]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
    for ax, (sp, t) in zip(axes.flat, picks):
        fe = meta[t]["first_error"]
        for a, c in (("causal", "#1f77b4"), ("full", "#d62728")):
            ax.plot(P[a][sp][t], marker="o", color=c, label=a)
            ax.axhline(thr[a], color=c, ls=":", lw=0.8)
        ax.axvline(fe, color="k", ls="--", lw=1, label="annotated first error")
        ax.axvspan(fe + 0.5, len(meta[t]["y"]) - 0.5, color="#eeeeee", label="no labels")
        ax.set_title(f"{sp} {t[:18]}", fontsize=8); ax.set_ylim(0, 1); ax.set_xlabel("step")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Per-step P(incorrect), seed {seed} (dotted: calibrated thresholds)")
    fig.tight_layout(); fig.savefig(out_fig, dpi=150); plt.close(fig)
    return {k: len(v) for k, v in cats.items()}


def label_structure(manifest: Path, split: str = "test") -> dict:
    """How predictable a label is from EARLIER labels alone (a property of the labels)."""
    from collections import Counter
    c = Counter()
    for m in jl(manifest / "meta" / f"{split}.jsonl"):
        seen_err = False
        for k, (y, mk) in enumerate(zip(m["y"], m["label_mask"])):
            if mk:
                key = "after_error" if seen_err else "no_earlier_error"
                c[f"{key}_n"] += 1
                c[f"{key}_incorrect"] += y == 1
                if y == 1:
                    seen_err = True
    return {"split": split,
            "P_incorrect_given_earlier_error": c["after_error_incorrect"] / max(c["after_error_n"], 1),
            "n_after_error": c["after_error_n"],
            "P_incorrect_given_no_earlier_error": c["no_earlier_error_incorrect"] / max(c["no_earlier_error_n"], 1),
            "n_no_earlier_error": c["no_earlier_error_n"]}


def slice_figure(res_dir: Path, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = json.loads((res_dir / "paired_differences.json").read_text())
    sets = [("rp_test_pre", "up to and incl.\nfirst error"), ("rp_test_post", "after the\nfirst error"),
            ("rp_test", "all labeled"), ("prm_human_test", "PRM800K human\n(first-error labels)")]
    sets = [x for x in sets if x[0] in p]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric in zip(axes, ("f1", "auroc")):
        for j, (c, col) in enumerate((("full-causal", "#d62728"), ("future1-causal", "#ff7f0e"), ("causal-local", "#1f77b4"))):
            xs = np.arange(len(sets)) + (j - 1) * 0.25
            for x, (k, _) in zip(xs, sets):
                v = p[k][c][metric]
                lo, hi = v["mean_ci95"]
                ax.errorbar([x], [v["mean"]], yerr=[[v["mean"] - lo], [hi - v["mean"]]], fmt="o", color=col,
                            label=c if x == xs[0] else None, capsize=3)
                ax.scatter([x] * len(v["per_seed"]), list(v["per_seed"].values()), color=col, s=10, alpha=0.4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(sets))); ax.set_xticklabels([s_[1] for s_ in sets], fontsize=8)
        ax.set_title(f"paired difference in step {metric.upper()} (seed mean, 95% CI; dots = seeds)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Does future access help once post-error steps are labeled? (DeepSeek-R1 labels)")
    fig.tight_layout()
    path = out / "slice_contrasts.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res_dir", type=Path, required=True)
    ap.add_argument("--fits_root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--diag", type=Path, default=None)
    ap.add_argument("--v2", action="store_true")
    a = ap.parse_args()
    global TEST_NAME, EXP
    if a.v2:
        TEST_NAME, EXP = "ReProbe held-out test (DeepSeek labels)", "bidirectional_token_probe_v2"
    st = structural(a.manifest)
    (a.res_dir / "structural_baseline.json").write_text(json.dumps(st, indent=2))
    print("[structural]", {k: (st[k]["f1"], st[k]["auroc"]) for k in ("test",) + PB}, st["pb_mean_F1_PB"])
    ds = None
    if a.diag and a.diag.exists():
        met = json.loads((a.res_dir / "metrics.json").read_text())
        thr = {k: v["calib"]["threshold"] for k, v in met.items()}
        thr_logit = {k: thr[k] for k in thr}
        ds = diagnostics(json.loads(a.diag.read_text()), thr_logit)
        (a.res_dir / "diagnostics_summary.json").write_text(json.dumps(ds, indent=2))
    figs = figures(a.res_dir, a.fits_root, a.res_dir / "figures", st, ds)
    p = a.res_dir / "figures" / "score_traces.png"
    counts = score_traces_and_audit(a.res_dir, a.fits_root, a.manifest, p, a.res_dir / "qualitative_audit.md")
    figs.append(p)
    if a.v2:
        ls = label_structure(a.manifest)
        (a.res_dir / "label_structure.json").write_text(json.dumps(ls, indent=2))
        print("[label_structure]", ls)
        figs.append(slice_figure(a.res_dir, a.res_dir / "figures"))
    print("[audit]", counts)
    print("[figures]", *map(str, figs), sep="\n  ")


if __name__ == "__main__":
    main()
