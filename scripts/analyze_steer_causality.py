#!/usr/bin/env python3
"""Aggregate the S3 Stage-5 steering sweep into the causal-validation verdict.

Reads the merged Tier-1 generation rows (s1ms_steer_generate.py output) and, optionally, the
Tier-0 fork battery JSONs (s1ms_steer_forks.py --directions_npz). Produces, per injection
layer:

  1. Dose-response   P(final-answer correct) vs alpha, one line per direction, vs the shared
                     alpha=0 baseline. A causal w is monotone; matched-norm controls are flat.
  2. Recovery (paired, by fork_id+sample_idx against the baseline):
       repair_rate  = P(steered correct | baseline incorrect), under toward-correct (alpha>0)
       corrupt_rate = P(steered incorrect | baseline correct), under toward-incorrect (alpha<0)
  3. Fluency guard   gradeable-rate and mean gen_len vs alpha (rules out "steering garbles text").
  4. Causal-vs-diagnostic  Tier-0 probe-logit shift (readout) vs Tier-1 dP(correct) (behaviour):
       behaviour moves only where the readout move is causal, not for every readout move.
  5. Dense vs sparse  probe vs sparse_restricted dose-response (is causation as compressible as
                      decoding?).

Outputs under --out_dir: steer_causality_report.md + dose_response_L{L}.png, recovery.png,
fluency_L{L}.png, causal_vs_diagnostic.png (if battery given).

Usage:
  python scripts/analyze_steer_causality.py \
      --gen runs/.../steering/steer_gen_merged.jsonl \
      --battery runs/.../steering/steer_forks_battery_L20.json \
                runs/.../steering/steer_forks_battery_L28.json \
      --out_dir runs/.../fork_rep_audit/qwen2_5_7b/steer_causality
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TREAT = ("probe", "sparse_restricted")


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, nargs="+", required=True,
                    help="merged Tier-1 generation jsonl(s) / shard files")
    ap.add_argument("--battery", type=Path, nargs="*", default=[],
                    help="optional Tier-0 steer_forks_battery_L{idx}.json files")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.gen)
    # baseline (direction == 'baseline') keyed by (fork_id, sample_idx)
    base = {}
    for r in rows:
        if r["direction"] == "baseline":
            base[(r["fork_id"], r["sample_idx"])] = r
    print(f"[analyze] {len(rows)} rows, {len(base)} baseline trajectories", flush=True)

    # group steered rows by (layer, direction, alpha)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["direction"] == "baseline":
            continue
        groups[(r["layer"], r["direction"], r["alpha"])].append(r)

    # baseline aggregate metrics
    b_grad = [r for r in base.values() if r["gradeable"]]
    base_pcorr = _mean([r["correct"] for r in b_grad])
    base_gradeable = _mean([r["gradeable"] for r in base.values()])
    base_len = _mean([r["gen_len"] for r in base.values()])

    cells = {}
    for (L, d, a), rs in groups.items():
        grad = [r for r in rs if r["gradeable"]]
        pcorr = _mean([r["correct"] for r in grad])
        # paired vs baseline
        rep_num = rep_den = cor_num = cor_den = 0
        for r in rs:
            b = base.get((r["fork_id"], r["sample_idx"]))
            if b is None or not b["gradeable"] or not r["gradeable"]:
                continue
            if not b["correct"]:                 # baseline wrong -> did we repair?
                rep_den += 1; rep_num += int(r["correct"])
            else:                                # baseline right -> did we corrupt?
                cor_den += 1; cor_num += int(not r["correct"])
        cells[(L, d, a)] = {
            "n": len(rs), "p_correct": pcorr,
            "gradeable_rate": _mean([r["gradeable"] for r in rs]),
            "mean_gen_len": _mean([r["gen_len"] for r in rs]),
            "repair_rate": (rep_num / rep_den) if rep_den else float("nan"),
            "repair_n": rep_den,
            "corrupt_rate": (cor_num / cor_den) if cor_den else float("nan"),
            "corrupt_n": cor_den,
            "d_p_correct": pcorr - base_pcorr,
        }

    layers = sorted({L for (L, _, _) in cells})
    dirs = sorted({d for (_, d, _) in cells}, key=lambda x: (x not in TREAT, x))

    # ---- 1. dose-response per layer -------------------------------------
    for L in layers:
        alphas = sorted({a for (LL, _, a) in cells if LL == L} | {0.0})
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for d in dirs:
            ys = []
            for a in alphas:
                ys.append(base_pcorr if a == 0 else cells.get((L, d, a), {}).get("p_correct"))
            style = "o-" if d in TREAT else "s--"
            ax.plot(alphas, ys, style, label=d, alpha=0.85)
        ax.axhline(base_pcorr, color="k", lw=0.6, ls=":")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("alpha  (+ = toward correct)"); ax.set_ylabel("P(final answer correct)")
        ax.set_title(f"Steering dose-response, hidden_states L{L}")
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.out_dir / f"dose_response_L{L}.png", dpi=150)
        plt.close(fig)

    # ---- 3. fluency guard per layer -------------------------------------
    for L in layers:
        alphas = sorted({a for (LL, _, a) in cells if LL == L} | {0.0})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for d in dirs:
            gr = [base_gradeable if a == 0 else cells.get((L, d, a), {}).get("gradeable_rate")
                  for a in alphas]
            ln = [base_len if a == 0 else cells.get((L, d, a), {}).get("mean_gen_len")
                  for a in alphas]
            style = "o-" if d in TREAT else "s--"
            ax[0].plot(alphas, gr, style, label=d, alpha=0.85)
            ax[1].plot(alphas, ln, style, label=d, alpha=0.85)
        ax[0].set_title("gradeable rate vs alpha"); ax[0].set_ylabel("gradeable rate")
        ax[1].set_title("mean gen length vs alpha"); ax[1].set_ylabel("tokens")
        for a_ in ax:
            a_.axvline(0, color="k", lw=0.5); a_.set_xlabel("alpha"); a_.grid(alpha=0.3)
        ax[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Fluency guard, L{L}"); fig.tight_layout()
        fig.savefig(args.out_dir / f"fluency_L{L}.png", dpi=150); plt.close(fig)

    # ---- 2. recovery bars (peak |alpha| per layer/direction) ------------
    fig, axes = plt.subplots(1, max(len(layers), 1), figsize=(6.2 * max(len(layers), 1), 4.4),
                             squeeze=False)
    for j, L in enumerate(layers):
        ax = axes[0][j]
        amax = max((a for (LL, _, a) in cells if LL == L), default=0.0)
        amin = min((a for (LL, _, a) in cells if LL == L), default=0.0)
        rep = [cells.get((L, d, amax), {}).get("repair_rate", float("nan")) for d in dirs]
        cor = [cells.get((L, d, amin), {}).get("corrupt_rate", float("nan")) for d in dirs]
        x = np.arange(len(dirs))
        ax.bar(x - 0.2, rep, 0.4, label=f"repair (a={amax:g})", color="#225522")
        ax.bar(x + 0.2, cor, 0.4, label=f"corrupt (a={amin:g})", color="#882222")
        ax.set_xticks(x); ax.set_xticklabels(dirs, rotation=40, ha="right", fontsize=7)
        ax.set_title(f"Recovery / corruption, L{L}"); ax.set_ylabel("rate")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(args.out_dir / "recovery.png", dpi=150); plt.close(fig)

    # ---- 4. causal-vs-diagnostic (needs Tier-0 battery) -----------------
    battery = {}
    for bp in args.battery:
        bj = json.loads(Path(bp).read_text())
        battery[int(bj["layer_index"])] = bj
    if battery:
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        for (L, d, a), c in cells.items():
            bj = battery.get(L)
            if not bj or d not in bj["by_dir"]:
                continue
            key = f"{a:g}"
            per = bj["by_dir"][d].get(key)
            if per is None:
                continue
            # readout shift on the correct (positive) candidate vs its own a=0
            base_logit = bj["by_dir"][d].get("0", {}).get("mean_probe_logit_pos")
            if base_logit is None:
                continue
            x = per["mean_probe_logit_pos"] - base_logit       # readout move
            y = c["d_p_correct"]                                # behaviour move
            col = "#1f77b4" if d in TREAT else "#aaaaaa"
            ax.scatter(x, y, c=col, s=30, alpha=0.8)
            ax.annotate(f"{d}\nL{L} a{a:g}", (x, y), fontsize=5, alpha=0.6)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("Tier-0 probe-logit shift on correct cand (readout)")
        ax.set_ylabel("Tier-1 dP(correct) vs baseline (behaviour)")
        ax.set_title("Causal vs diagnostic: does moving the readout move behaviour?")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.out_dir / "causal_vs_diagnostic.png", dpi=150)
        plt.close(fig)

    # ---- report ---------------------------------------------------------
    lines = ["# S3 Stage-5: steering causality report", "",
             f"Baseline (alpha=0, n={len(base)}): P(correct)={base_pcorr:.3f}, "
             f"gradeable={base_gradeable:.3f}, mean_len={base_len:.0f}", ""]
    for L in layers:
        lines += [f"## hidden_states L{L}", "",
                  "| direction | alpha | n | P(correct) | dP | repair | corrupt | "
                  "gradeable | len |", "|---|---|---|---|---|---|---|---|---|"]
        for d in dirs:
            for a in sorted({aa for (LL, dd, aa) in cells if LL == L and dd == d}):
                c = cells[(L, d, a)]
                lines.append(
                    f"| {d} | {a:+g} | {c['n']} | {c['p_correct']:.3f} | "
                    f"{c['d_p_correct']:+.3f} | {c['repair_rate']:.3f} (n{c['repair_n']}) | "
                    f"{c['corrupt_rate']:.3f} (n{c['corrupt_n']}) | "
                    f"{c['gradeable_rate']:.3f} | {c['mean_gen_len']:.0f} |")
        lines.append("")
    (args.out_dir / "steer_causality_report.md").write_text("\n".join(lines))
    print(f"[analyze] wrote {args.out_dir}/steer_causality_report.md + PNGs", flush=True)


if __name__ == "__main__":
    main()
