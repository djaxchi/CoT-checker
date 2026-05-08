#!/usr/bin/env python3
"""Print a compact validity table for all trained PTB variants.

Reads validity_report.json and calibration.json from each variant's
checkpoint directory and produces a one-row-per-variant summary table
followed by an explicit CONTINUE / STOP recommendation.

Usage:
    python scripts/ptb_validity_summary.py \\
        --ckpt-base /path/to/ptb_robust \\
        --variants  no_l1 dwa_calibrated active_fraction topk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(v, fmt=".4f") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "  n/a  "
    return format(v, fmt)


def _yn(cond: bool) -> str:
    return "YES" if cond else "NO"


def main() -> None:
    p = argparse.ArgumentParser(description="PTB validity summary table")
    p.add_argument("--ckpt-base", required=True, help="Base dir containing one subdir per variant")
    p.add_argument("--variants",  nargs="+",
                   default=["no_l1", "dwa_calibrated", "active_fraction", "topk"])
    p.add_argument("--cos-threshold",    type=float, default=0.05,
                   help="Minimum cosine_sim to count as 'transition learned'")
    p.add_argument("--r2-threshold",     type=float, default=0.0,
                   help="Minimum R2 to count as 'better than mean'")
    p.add_argument("--sat-threshold",    type=float, default=0.5,
                   help="Max clip_frac for DWA saturation before flagging INVALID")
    args = p.parse_args()

    base = Path(args.ckpt_base)

    # ---------- Collect rows ----------
    rows = []
    for name in args.variants:
        vr_path  = base / name / "validity_report.json"
        cal_path = base / name / "calibration.json"

        vr  = _load_json(vr_path)
        cal = _load_json(cal_path)

        if vr is None and cal is None:
            rows.append({"name": name, "missing": True})
            continue

        cos     = vr.get("final_val_cosine_sim") if vr else None
        r2      = vr.get("final_val_r2") if vr else None
        z_mean  = cal.get("z_mean") if cal else (vr.get("calibration", {}).get("z_mean") if vr else None)
        z_med   = cal.get("z_median") if cal else (vr.get("calibration", {}).get("z_median") if vr else None)
        act_dim = vr.get("final_val_mean_active_dims") if vr else None
        act_frc = vr.get("final_val_active_frac") if vr else None
        flags   = vr.get("flags", []) if vr else []

        # DWA saturation
        sat     = vr.get("dwa_saturation") if vr else None
        lam_final  = sat["final_weight"] if sat else None
        saturated  = False
        if sat:
            saturated = (sat.get("min_clip_frac", 0) > args.sat_threshold or
                         sat.get("max_clip_frac", 0) > args.sat_threshold)

        has_invalid = any("INVALID" in f for f in flags)
        has_warning = any("WARNING" in f for f in flags)

        rows.append({
            "name":       name,
            "missing":    False,
            "cos":        cos,
            "r2":         r2,
            "z_mean":     z_mean,
            "z_median":   z_med,
            "act_dims":   act_dim,
            "act_frac":   act_frc,
            "lam_final":  lam_final,
            "saturated":  saturated,
            "flags":      flags,
            "has_invalid": has_invalid,
            "has_warning": has_warning,
        })

    # ---------- Print table ----------
    hdr = (
        f"  {'variant':<20}  {'cos_sim':>7}  {'R2':>7}  {'z_mean':>7}  {'z_med':>7}"
        f"  {'act_dims':>8}  {'act@1e-3':>8}  {'lam_final':>9}  {'sat?':>5}  status"
    )
    sep = "-" * len(hdr)
    print(f"\nPTB Validity Summary  ({base})")
    print(f"\n{sep}\n{hdr}\n{sep}")

    for r in rows:
        if r["missing"]:
            print(f"  {r['name']:<20}  *** validity_report.json / calibration.json not found ***")
            continue
        status = "INVALID" if r["has_invalid"] else ("WARN" if r["has_warning"] else "OK")
        print(
            f"  {r['name']:<20}"
            f"  {_fmt(r['cos']):>7}"
            f"  {_fmt(r['r2']):>7}"
            f"  {_fmt(r['z_mean']):>7}"
            f"  {_fmt(r['z_median']):>7}"
            f"  {_fmt(r['act_dims'], '.1f'):>8}"
            f"  {_fmt(r['act_frac'], '.3f'):>8}"
            f"  {_fmt(r['lam_final'], '.2e'):>9}"
            f"  {_yn(r['saturated']):>5}"
            f"  {status}"
        )
    print(sep)

    # ---------- Detailed flags ----------
    any_flags = False
    for r in rows:
        if r.get("missing") or not r.get("flags"):
            continue
        ok_flags = [f for f in r["flags"] if f.startswith("OK")]
        bad_flags = [f for f in r["flags"] if not f.startswith("OK")]
        if bad_flags:
            any_flags = True
            print(f"\n  [{r['name']}] flags:")
            for f in bad_flags:
                print(f"    {f}")
    if not any_flags:
        print("\n  No non-OK flags across variants.")

    # ---------- CONTINUE / STOP recommendation ----------
    # Central baseline is no_l1. If it's invalid, the whole experiment is compromised.
    no_l1_row = next((r for r in rows if r["name"] == "no_l1" and not r.get("missing")), None)

    stop_reasons = []
    if no_l1_row is None:
        stop_reasons.append("ptb_no_l1 checkpoint missing -- cannot answer Q1/Q2")
    else:
        cos = no_l1_row.get("cos")
        r2  = no_l1_row.get("r2")
        z   = no_l1_row.get("z_mean")
        if cos is not None and cos < args.cos_threshold:
            stop_reasons.append(f"ptb_no_l1 cosine_sim={cos:.4f} < {args.cos_threshold} -- transition not learned")
        if r2 is not None and r2 < args.r2_threshold:
            stop_reasons.append(f"ptb_no_l1 R2={r2:.4f} < {args.r2_threshold} -- worse than mean delta")
        if z is not None and z <= 0:
            stop_reasons.append(f"ptb_no_l1 z_mean={z:.4f} -- encoder produces no activations")
        if no_l1_row.get("has_invalid"):
            stop_reasons.append("ptb_no_l1 has INVALID flags")

    print("\n" + "=" * 60)
    if stop_reasons:
        print("  RECOMMENDATION: STOP")
        print("  Reasons:")
        for r in stop_reasons:
            print(f"    - {r}")
        print("\n  Do not proceed to Stage 3 (extract latents).")
        print("  Debug the transition dataset or training pipeline first.")
        sys.exit(1)
    else:
        valid_variants = [r["name"] for r in rows
                          if not r.get("missing") and not r.get("has_invalid")]
        warn_variants  = [r["name"] for r in rows
                          if not r.get("missing") and r.get("has_warning") and not r.get("has_invalid")]
        print("  RECOMMENDATION: CONTINUE")
        if valid_variants:
            print(f"  Valid variants (no INVALID flags): {', '.join(valid_variants)}")
        if warn_variants:
            print(f"  Variants with warnings (proceed with caution): {', '.join(warn_variants)}")
        print("\n  Proceed to Stage 3 (extract latents).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
