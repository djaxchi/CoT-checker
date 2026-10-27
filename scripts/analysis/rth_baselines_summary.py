#!/usr/bin/env python3
"""read_token_v1 Stage G baselines against RTH: one table and the frozen readings.

Per baseline (3 seeds) and RTH: trained parameters (from the token checkpoint),
PRM800K test AUROC and ProcessBench calib-20 F1_PB of the head and of LDA (from
geometry_<run>/geometry.json), the paired trace bootstrap against RTH
(stage_f_bootstrap.py output, comparison "<group> - RTH"), PRM800K val AUROC and the
confidence-wrong pair accuracy (pair_complementarity.py output, pairs_<run>.json) of
every seed and of the seed picked on val AUROC. The readings are those of
docs/read_token_v1_plan.md, "Stage G".

    python scripts/analysis/rth_baselines_summary.py --root <OUT copy> --bootstrap <json> \
        --val <val_auroc json> --pairs_dir results/read_token_v1/baselines \
        --out results/read_token_v1/baselines/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

RTH = ["dq_correct", "rth_correct_seed1", "rth_correct_seed2"]
GROUPS = {"BL1": [f"bl1_correct_seed{s}" for s in range(3)],
          "BL1-SP": [f"bl1sp_seed{s}" for s in range(3)],
          "BL2": [f"bl2_offsets_seed{s}" for s in range(3)],
          "BL3": [f"bl3_loreft_seed{s}" for s in range(3)],
          "BL4": [f"bl4_lora_seed{s}" for s in range(3)]}
RTH_PAIRS = (0.530, 0.472, 0.583)  # rth_correct_seed1, results/read_token_v1/downstream/pairs_t10.json


def n_trained(ck: dict) -> int:
    """Parameters the run trained: head, the embedding unless frozen, offsets, intervention."""
    n = sum(v.numel() for v in ck["head"].values())
    if not ck.get("freeze_soft"):
        n += ck["soft"].numel()
    if ck.get("deep") is not None:
        n += ck["deep"].numel()
    return n + sum(v.numel() for v in (ck.get("interv_state") or {}).values())


def geometry(root: Path, run: str) -> dict:
    rows = json.loads((root / f"geometry_{run}" / "geometry.json").read_text())
    by = {r["method"]: r for r in rows}
    return {"head_test": by["prm_head"]["test_auroc"], "head_c20": by["prm_head"]["pb_calib20"],
            "lda_test": by["lda"]["test_auroc"], "lda_c20": by["lda"]["pb_calib20"]}


def ms(x: list[float]) -> str:
    return f"{np.mean(x):.3f} +- {np.std(x, ddof=1):.3f}" if len(x) > 1 else f"{x[0]:.3f}"


def readings(delta: dict[str, float], pairs: dict[str, tuple]) -> list[tuple[str, bool, str]]:
    """(reading, met, evidence); delta[g] = calib-20(g) - calib-20(RTH), 3-seed means."""
    out = []
    if "BL1" in delta:
        out.append(("BL1 at least 0.02 below RTH: the offsets are the ingredient",
                    delta["BL1"] <= -0.02, f"BL1 - RTH {delta['BL1']:+.3f}"))
    if "BL2" in delta:
        out.append(("BL2 within 0.01 of RTH: training the embedding adds nothing once offsets exist",
                    abs(delta["BL2"]) <= 0.01, f"BL2 - RTH {delta['BL2']:+.3f}"))
    if "BL3" in delta:
        d = delta["BL3"]
        out.append(("BL3 within 0.01 of RTH: a bias per layer is enough", abs(d) <= 0.01, f"BL3 - RTH {d:+.3f}"))
        out.append(("BL3 at least 0.02 above RTH: a richer intervention helps", d >= 0.02, f"BL3 - RTH {d:+.3f}"))
    hit = [g for g, (p, lo, _) in pairs.items() if p >= 0.60 and lo > 0.5]
    ev = ", ".join(f"{g} {p:.3f} [{lo:.3f}, {hi:.3f}]" for g, (p, lo, hi) in pairs.items())
    if "BL4" in delta:
        ev = f"BL4 - RTH {delta['BL4']:+.3f}; pairs: {ev}"
    out.append(("BL4 at least 0.02 above RTH, or any baseline >= 0.60 on the confidence-wrong pairs "
                "with the interval above 0.5: the confidence coupling is a matter of capacity "
                "(otherwise a property of reading a frozen policy)",
                delta.get("BL4", -1.0) >= 0.02 or bool(hit), ev))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="geometry_<run>/ and tokens/<run>/read_token.pt")
    p.add_argument("--bootstrap", type=Path, required=True)
    p.add_argument("--val", type=Path, nargs="+", required=True, help="val_auroc json(s) of the pairs job")
    p.add_argument("--pairs_dir", type=Path, required=True)
    p.add_argument("--group", nargs="*", default=[], help="name=run,run,run; overrides a default group")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    groups = dict(GROUPS)
    for spec in a.group:
        g, runs = spec.split("=", 1)
        groups[g] = runs.split(",")
    val = {k: v for f in a.val for k, v in json.loads(f.read_text())["val_auroc"].items()}
    boot = {r["comparison"]: r for r in json.loads(a.bootstrap.read_text())}

    rows, delta, pairs, res = [], {}, {}, {}
    for g, runs in [("RTH", RTH), *groups.items()]:
        geo = [geometry(a.root, r) for r in runs]
        ck = torch.load(a.root / "tokens" / runs[0] / "read_token.pt", map_location="cpu", weights_only=False)
        rec = {"runs": runs, "params": n_trained(ck), "geometry": geo,
               "val_auroc": {r: val.get(r) for r in runs}}
        b = boot.get(f"{g} - RTH")
        if b is not None:
            rec["bootstrap"] = {"delta": b["delta"], "ci95": b["ci95_normal"]}
            delta[g] = b["delta"]
        if g != "RTH":
            seen = {r: json.loads((a.pairs_dir / f"pairs_{r}.json").read_text())["math500"]
                    ["wrong_conf_question_weighted"]["probe"]
                    for r in runs if (a.pairs_dir / f"pairs_{r}.json").exists()}
            rec["pairs"] = seen
            cand = [r for r in runs if val.get(r) is not None and r in seen]
            if cand:
                pick = max(cand, key=lambda r: val[r])
                rec["pairs_pick"] = pick
                pairs[g] = tuple(seen[pick])
        res[g] = rec
        bcell = (f"{b['delta']:+.3f} [{b['ci95_normal'][0]:+.3f}, {b['ci95_normal'][1]:+.3f}]"
                 if b is not None else "-")
        pcell = "0.530 [0.472, 0.583] (seed1)" if g == "RTH" else (
            f"{pairs[g][0]:.3f} [{pairs[g][1]:.3f}, {pairs[g][2]:.3f}] ({rec['pairs_pick']})" if g in pairs else "-")
        rows.append(f"| {g} | {rec['params']:,} | {ms([x['head_test'] for x in geo])} | "
                    f"{ms([x['head_c20'] for x in geo])} | {ms([x['lda_c20'] for x in geo])} | {bcell} | {pcell} |")

    lines = ["# read_token_v1 Stage G baselines", "",
             "Frozen Qwen3-8B, verifier template, 100k PRM800K steps, 3 seeds. calib-20 = ProcessBench "
             "calib-20 F1_PB, four-subset mean (mean +- sd over seeds). Bootstrap: paired trace bootstrap "
             "of the 3-seed means against RTH's 3 seeds, 200 resamples. Pairs: accuracy on the 985 "
             "MATH-500 confidence-wrong pairs, question-weighted, 95% CI, seed picked on PRM800K val AUROC.",
             "",
             "| run | trained params | head test AUROC | head calib-20 | LDA calib-20 | calib-20 - RTH [95% CI] | confidence-wrong pairs |",
             "|---|---|---|---|---|---|---|", *rows, "", "## Readings (docs/read_token_v1_plan.md, Stage G)", ""]
    read = readings(delta, pairs)
    lines += [f"- {'MET' if met else 'NOT MET'}: {txt} ({ev})" for txt, met, ev in read]
    lines += ["", "## Per seed", ""]
    for g, rec in res.items():
        for r, x in zip(rec["runs"], rec["geometry"]):
            v = rec["val_auroc"].get(r)
            pr = rec.get("pairs", {}).get(r)
            lines.append(f"- {g} {r}: val {v:.4f}" if v is not None else f"- {g} {r}: val -")
            lines[-1] += (f", head test {x['head_test']:.4f}, head calib-20 {x['head_c20']:.3f}, "
                          f"LDA test {x['lda_test']:.4f}, LDA calib-20 {x['lda_c20']:.3f}")
            if pr:
                lines[-1] += f", pairs {pr[0]:.3f} [{pr[1]:.3f}, {pr[2]:.3f}]"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    a.out.with_suffix(".json").write_text(json.dumps(
        {"groups": res, "readings": [{"reading": t, "met": m, "evidence": e} for t, m, e in read]}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
