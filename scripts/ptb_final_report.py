#!/usr/bin/env python3
"""Generate FINAL_REPORT.md from PTB probe benchmark results.

Reads summary_results.json produced by eval_predictive_transition_probes.py
and writes a clean markdown report with the benchmark table and explicit
YES/NO answers to the four research questions.

Usage:
    python scripts/ptb_final_report.py \\
        --results  results/ptb_robust_probes/summary_results.json \\
        --out      results/ptb_robust_probes/FINAL_REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _nan_str(v, fmt=".3f") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return format(v, fmt)


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v * 100:.1f}%"


def _yn(cond: bool) -> str:
    return "**YES**" if cond else "**NO**"


def _margin(a, b) -> str:
    if a is None or b is None:
        return "n/a"
    if a != a or b != b:
        return "n/a"
    return f"{(a - b) * 100:+.1f} pp"


def _mf1(r) -> float:
    if r is None:
        return float("nan")
    return r.get("ms_macro_f1", float("nan"))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate FINAL_REPORT.md from PTB probe results")
    p.add_argument("--results", required=True, help="summary_results.json from eval script")
    p.add_argument("--out",     required=True, help="Output path for FINAL_REPORT.md")
    p.add_argument("--pb-threshold", type=float, default=0.25,
                   help="PB-F1 threshold for 'non-trivial localization' (default 0.25)")
    args = p.parse_args()

    data = json.loads(Path(args.results).read_text())
    rows = data["summary_rows"]

    # Look up key rows
    def find(label):
        return next((r for r in rows if r["label"] == label), None)

    baseline_random = find("random_bln")
    baseline_dense  = find("dense_h")
    ptb_no_l1       = find("ptb_no_l1")
    ptb_sparse_best = max(
        (r for r in rows if r["label"].startswith("ptb_") and r["label"] != "ptb_no_l1"),
        key=lambda r: _mf1(r) if _mf1(r) == _mf1(r) else -1,
        default=None,
    )

    # ---------- Build benchmark table (markdown) ----------
    lines: list[str] = []
    lines.append("# PTB Robustness Experiment: Final Report")
    lines.append("")
    lines.append("## Benchmark Table")
    lines.append("")
    lines.append(
        "| Representation | MS MacroF1 ± std | AUROC | AUPRC | MS thr | MS pos% "
        "| PB-F1 | PB MacroF1 | Transfer gap | MS | PB |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |")

    for r in rows:
        pb_f1_s  = _pct(r.get("pb_f1"))
        tgap     = _nan_str(r.get("transfer_gap"))
        ms_c     = "COLLAPSE" if r.get("ms_collapse") else "ok"
        pb_c     = "COLLAPSE" if r.get("pb_collapse") else "ok"
        lines.append(
            f"| {r['label']} "
            f"| {_nan_str(r['ms_macro_f1'])} ± {_nan_str(r.get('ms_mf1_std', 0.0))} "
            f"| {_nan_str(r.get('ms_auroc'))} "
            f"| {_nan_str(r.get('ms_auprc'))} "
            f"| {_nan_str(r.get('ms_best_threshold'), '.2f')} "
            f"| {_nan_str(r.get('ms_pos_rate'), '.2f')} "
            f"| {pb_f1_s} "
            f"| {_nan_str(r.get('pb_macro_f1'))} "
            f"| {tgap} "
            f"| {ms_c} "
            f"| {pb_c} |"
        )

    lines.append("")

    # ---------- 4 Research questions ----------
    lines.append("## Research Questions")
    lines.append("")

    # Q1: PTB vs random_bln
    q1_beat = (ptb_no_l1 is not None and baseline_random is not None
               and _mf1(ptb_no_l1) > _mf1(baseline_random))
    lines.append(f"### Q1. Does PTB beat random_bln?  {_yn(q1_beat)}")
    lines.append("")
    if ptb_no_l1 and baseline_random:
        lines.append(
            f"- `ptb_no_l1` macro-F1 = {_nan_str(_mf1(ptb_no_l1))}"
            f",  `random_bln` macro-F1 = {_nan_str(_mf1(baseline_random))}"
            f",  margin = {_margin(_mf1(ptb_no_l1), _mf1(baseline_random))}"
        )
    if not q1_beat and ptb_no_l1:
        lines.append(
            "- **Interpretation:** The transition prediction objective does not produce a "
            "representation better than an untrained random encoder. The PTB latent carries "
            "no more step-level information than random projections."
        )
    elif q1_beat:
        lines.append(
            "- **Interpretation:** Training on step-to-step activation deltas yields a "
            "representation that contains real correctness-relevant information beyond random chance."
        )
    lines.append("")

    # Q2: PTB vs dense_h
    q2_beat = (ptb_no_l1 is not None and baseline_dense is not None
               and _mf1(ptb_no_l1) > _mf1(baseline_dense))
    lines.append(f"### Q2. Does PTB beat dense_h?  {_yn(q2_beat)}")
    lines.append("")
    if ptb_no_l1 and baseline_dense:
        lines.append(
            f"- `ptb_no_l1` macro-F1 = {_nan_str(_mf1(ptb_no_l1))}"
            f",  `dense_h` macro-F1 = {_nan_str(_mf1(baseline_dense))}"
            f",  margin = {_margin(_mf1(ptb_no_l1), _mf1(baseline_dense))}"
        )
    if not q2_beat and ptb_no_l1:
        lines.append(
            "- **Interpretation:** The transition bottleneck compresses the raw hidden state "
            "without adding discriminative signal for step-level correctness. Raw hidden states "
            "remain the stronger representation."
        )
    elif q2_beat:
        lines.append(
            "- **Interpretation:** Projecting through the transition bottleneck extracts "
            "a more discriminative correctness signal than the raw hidden state."
        )
    lines.append("")

    # Q3: Sparsity effect
    q3_helps = (ptb_sparse_best is not None and ptb_no_l1 is not None
                and _mf1(ptb_sparse_best) > _mf1(ptb_no_l1))
    lines.append(f"### Q3. Does sparsity improve PTB?  {_yn(q3_helps)}")
    lines.append("")
    if ptb_sparse_best and ptb_no_l1:
        lines.append(
            f"- Best sparse variant: `{ptb_sparse_best['label']}`"
            f"  macro-F1 = {_nan_str(_mf1(ptb_sparse_best))}"
            f",  `ptb_no_l1` macro-F1 = {_nan_str(_mf1(ptb_no_l1))}"
            f",  margin = {_margin(_mf1(ptb_sparse_best), _mf1(ptb_no_l1))}"
        )
        # All sparse variants
        sparse_rows = [r for r in rows if r["label"].startswith("ptb_") and r["label"] != "ptb_no_l1"]
        if sparse_rows:
            lines.append("")
            lines.append("  All sparse variants vs. no_l1:")
            for r in sparse_rows:
                lines.append(
                    f"  - `{r['label']}`: macro-F1 = {_nan_str(_mf1(r))}"
                    f"  (margin {_margin(_mf1(r), _mf1(ptb_no_l1))})"
                )
    elif ptb_sparse_best is None:
        lines.append("- No sparse PTB variant found in results.")
    if not q3_helps and ptb_sparse_best:
        lines.append(
            "- **Interpretation:** Adding sparsity (via DWA or TopK) does not improve "
            "over the dense transition bottleneck. Sparsity either removes useful information "
            "or was not effectively calibrated."
        )
    elif q3_helps:
        lines.append(
            "- **Interpretation:** Sparse bottleneck variants outperform the dense baseline, "
            "suggesting that sparsity either acts as regularisation or enforces disentanglement "
            "that helps probe generalisation."
        )
    lines.append("")

    # Q4: ProcessBench transfer
    pb_candidates = [
        (r["label"], r.get("pb_f1", float("nan")))
        for r in rows if r.get("pb_f1") == r.get("pb_f1") and r.get("pb_f1") is not None
    ]
    pb_candidates.sort(key=lambda x: -(x[1] if x[1] == x[1] else -1))
    best_pb_label, best_pb_f1 = pb_candidates[0] if pb_candidates else (None, float("nan"))
    q4_transfers = isinstance(best_pb_f1, float) and best_pb_f1 == best_pb_f1 and best_pb_f1 > args.pb_threshold
    lines.append(f"### Q4. Does any representation transfer to ProcessBench?  {_yn(q4_transfers)}")
    lines.append("")
    if pb_candidates:
        lines.append("  Top representations by PB-F1:")
        for lbl, f1 in pb_candidates[:5]:
            pb_col = next((r.get("pb_collapse") for r in rows if r["label"] == lbl), None)
            col_s  = " (COLLAPSE)" if pb_col else ""
            lines.append(f"  - `{lbl}`: PB-F1 = {_pct(f1)}{col_s}")
        lines.append("")
        if not q4_transfers:
            lines.append(
                f"- **Interpretation:** No representation exceeded the {args.pb_threshold*100:.0f}% "
                "PB-F1 threshold. Math-Shepherd correctness (MC rollout viability) does not "
                "transfer to ProcessBench first-error localization. The learned signal may be "
                "specific to the MS training distribution."
            )
        else:
            lines.append(
                "- **Interpretation:** At least one representation achieves non-trivial "
                "first-error localization on ProcessBench, suggesting generalisation beyond "
                "the Math-Shepherd training distribution."
            )
    else:
        lines.append("- ProcessBench results not available (Stage 5 not run).")
    lines.append("")

    # ---------- Per-PTB validity ----------
    lines.append("## Per-PTB Variant Status")
    lines.append("")
    lines.append("| Variant | MS MacroF1 | Beats random? | MS collapse | PB collapse | Status |")
    lines.append("| :--- | ---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        if not r["label"].startswith("ptb_"):
            continue
        beats_random = (baseline_random is not None and _mf1(r) > _mf1(baseline_random))
        ms_c = "COLLAPSE" if r.get("ms_collapse") else "ok"
        pb_c = "COLLAPSE" if r.get("pb_collapse") else "ok"
        invalid = r.get("ms_collapse") or (
            baseline_random is not None and _mf1(r) <= _mf1(baseline_random)
        )
        status = "INVALID" if invalid else "VALID"
        lines.append(
            f"| {r['label']} "
            f"| {_nan_str(_mf1(r))} "
            f"| {_yn(beats_random)} "
            f"| {ms_c} "
            f"| {pb_c} "
            f"| {status} |"
        )
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Final report written to {out_path}")


if __name__ == "__main__":
    main()
