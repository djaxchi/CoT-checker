#!/usr/bin/env python3
"""Report for InStep (PRM800K) without ProcessBench problems (no-overlap protocol).

Reads (all written by slurm/instep_v1_nopb_tamia.sh):
  --selection   instep_lr_select.py's selection.json over the five seeds' sweep.json
  --sweeps      dir with seed<s>/sweep.json (final loss, constant-predictor loss, val AUROC)
  --filter      nopb_tools.py filter.json (labelled steps left, share incorrect)
  --logs        training logs instep_prm_nopb_seed<s>-*.log ("[data]" line: optimiser steps, cap)
  --final       paper_v1.py final.json with the arms instep_prm_nopb (rule-kept seeds),
                instep_prm_nopb_all (all five) and instep_prm_nopb_seed<s>
  --reference   the paper's final.json: every number of an arm or pair without instep_prm in it
                must be equal in --final (reported, never silently accepted)
  --pb_oracle   pb_oracle.py JSON for the same arms (F1 at P = 0.5, four-subset mean)
  --baselines   pb_oracle.py JSON of STEP / ReProbe's head on the filtered draws vs the no-overlap arm
ProcessBench per-subset F1 at P = 0.5 is computed here with pb_oracle.point on the arms'
pb_step_scores files that paper_v1.py wrote (<final dir>/pb_scores/qwen3_8b/<arm>/seed<i>).

Writes --out (markdown) and --json (the numbers of the report).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PB = ("gsm8k", "math", "olympiadbench", "omnimath")
TEST = ("math", "gsm8k", "proofnet", "trips", "meetings", "calendar", "strategyqa", "scienceqa")
DATA = re.compile(r"^\[data\] (\d+) items -> .*?; (\d+) optimizer steps at <= (\d+) items")
PB_VS = ("prm7b", "prm800k", "instep_ds", "instep_self", "rp_rel_ds")


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def fd(r: dict | None) -> str:
    if r is None:
        return "n/a"
    return f"{r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"


def stability(selection: dict, sweeps: Path, seeds: list[int]) -> list[dict]:
    t = next(iter(next(iter(selection["tables"].values())).values()))
    rows = []
    for s in seeds:
        sw = json.loads((sweeps / f"seed{s}" / "sweep.json").read_text())
        c = sw["configs"][0]
        r = t["seeds"][str(s)] if str(s) in t["seeds"] else t["seeds"][s]
        others = [t["seeds"][k]["metric"] for k in t["seeds"] if str(k) != str(s)]
        rows.append({"seed": s, "final_loss": c["final_loss"], "const_loss": sw["const_loss"],
                     "max_late_loss": r["max_late_loss"], "val_auroc": c["val_auroc"],
                     "median_others": float(np.median(others)), "flags": r["flags"], "kept": not r["flags"],
                     "reconstructed": bool(sw.get("reconstructed"))})
    return rows


def training(filt: dict, logs: Path, seeds: list[int]) -> list[dict]:
    out = []
    for s in seeds:
        f = sorted(logs.glob(f"instep_prm_nopb_seed{s}-*.log"))[-1]
        m = next(DATA.match(l) for l in f.read_text().splitlines() if DATA.match(l))
        c = filt["seeds"][str(s)]
        if int(m.group(1)) != c["kept"]:
            raise SystemExit(f"seed {s}: the log trained on {m.group(1)} items, the filter keeps {c['kept']}")
        out.append({"seed": s, "drawn": c["drawn"], "excluded": c["excluded"], "steps_labelled": c["kept"],
                    "share_incorrect": c["share_incorrect"], "optimiser_steps": int(m.group(2)),
                    "cap": int(m.group(3)), "log": f.name})
    return out


def p05_per_subset(final_dir: Path, arm: str) -> dict:
    from scripts.analysis.pb_threshold_calibration import load_traces
    from scripts.instep_v1.pb_oracle import point
    base = final_dir / "pb_scores" / "qwen3_8b" / arm
    seeds = sorted(base.glob("seed*"))
    per = {s: float(np.mean([point(load_traces(d / f"pb_step_scores_{s}.jsonl"), ("p05",))["p05"] for d in seeds]))
           for s in PB}
    per["mean"] = float(np.mean([per[s] for s in PB]))
    return per


def compare(new: dict, ref: dict, skip=lambda k: "instep_prm" in k) -> list[str]:
    """Every value of an arm or pair without instep_prm that differs between the two runs."""
    out = []
    a, b = new["backbones"]["qwen3_8b"], ref["backbones"]["qwen3_8b"]

    def walk(x, y, path):
        if isinstance(y, dict):
            for k in y:
                if k not in x:
                    out.append(f"{path}.{k}: missing in the new run")
                else:
                    walk(x[k], y[k], f"{path}.{k}")
        elif isinstance(y, list) and y and isinstance(y[0], (int, float)):
            if len(x) != len(y) or any(abs(p - q) > 1e-9 for p, q in zip(x, y)):
                out.append(f"{path}: {x} vs {y}")
        elif isinstance(y, float):
            if x is None or abs(x - y) > 1e-9:
                out.append(f"{path}: {x} vs {y}")
    for sec in ("metrics", "pb", "calib20"):
        for k, v in b.get(sec, {}).items():
            if not skip(k):
                walk(a.get(sec, {}).get(k, {}), v, f"{sec}.{k}")
    for sec in ("pairs", "ladder_pairs", "pb_pairs"):
        na = {(p["a"], p["b"]): p for p in a.get(sec, [])}
        for p in b.get(sec, []):
            if skip(p["a"]) or skip(p["b"]):
                continue
            q = na.get((p["a"], p["b"]))
            if q is None:
                out.append(f"{sec}.{p['a']}-{p['b']}: missing in the new run")
            else:
                walk(q, p, f"{sec}.{p['a']}-{p['b']}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("selection", "sweeps", "filter", "logs", "final", "reference", "pb_oracle", "baselines", "out", "json"):
        p.add_argument(f"--{k}", type=Path, required=k not in ("baselines",))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--main_label", default="", help="label of the instep_prm_nopb arm (default: rule-kept seeds)")
    a = p.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    sel = json.loads(a.selection.read_text())
    st = stability(sel, a.sweeps, seeds)
    tr = training(json.loads(a.filter.read_text()), a.logs, seeds)
    F = json.loads(a.final.read_text())
    R = F["backbones"]["qwen3_8b"]
    kept = [r["seed"] for r in st if r["kept"]]
    arms = [("all five", "instep_prm_nopb_all"), (f"rule-kept ({','.join(map(str, kept))})", "instep_prm_nopb")] + \
           [(f"seed {s}", f"instep_prm_nopb_seed{s}") for s in seeds]
    if set(kept) == set(seeds):
        arms = [x for x in arms if x[1] != "instep_prm_nopb_all"]  # the same seeds: one row
    if a.main_label:
        arms = [(a.main_label if k == "instep_prm_nopb" else lab, k) for lab, k in arms]
    arms = [x for x in arms if x[1] in R["pb"]]  # a run may carry only some of the arms
    orc = json.loads(a.pb_oracle.read_text())["arms"]
    pbp = {(x["a"], x["b"]): x for x in R["pb_pairs"]}
    tp = {(x["a"], x["b"]): x for x in R["pairs"]}
    diffs = compare(F, json.loads(a.reference.read_text()))
    res = {"stability": st, "kept": kept, "training": tr, "diffs_outside_instep_prm": diffs, "arms": {}}
    L = ["# InStep (PRM800K), no ProcessBench problem in training", "",
         f"Arm keys in {a.final}: " + ", ".join(f"`{k}` ({lab})" for lab, k in arms) + ".", "",
         "## Stability rule (scripts/analysis/instep_lr_select.py, validation AUROC on PRM800K val5k)", "",
         "| seed | final training loss | constant-predictor loss | max running loss after 25% | val AUROC "
         "| median of the other seeds | verdict |", "|---|---|---|---|---|---|---|"]
    for r in st:
        v = "kept" if r["kept"] else "UNSTABLE: " + "; ".join(r["flags"])
        L.append(f"| {r['seed']}{'*' if r['reconstructed'] else ''} | {r['final_loss']:.4f} | {r['const_loss']:.4f} | "
                 f"{r['max_late_loss']:.4f} | {r['val_auroc']:.4f} | {r['median_others']:.4f} | {v} |")
    L += ["", "\\* trained without --val (job 701011): sweep.json rebuilt from the training log (running loss "
          "every 50 steps) and an encode of val5k (nopb_tools.py sweep).", "",
          "## Training data", "",
          "| seed | drawn | removed (ProcessBench problems) | labelled steps left | share incorrect "
          "| optimiser steps | step cap |", "|---|---|---|---|---|---|---|"]
    for r in tr:
        L.append(f"| {r['seed']} | {r['drawn']:,} | {r['excluded']:,} | {r['steps_labelled']:,} | "
                 f"{r['share_incorrect']:.3f} | {r['optimiser_steps']:,} | {r['cap']} |")
    # ProcessBench
    L += ["", "## ProcessBench (pb: paper_v1.py; F1 at P = 0.5: pb_oracle.py)", "",
          "| arm | subset | first-error AUROC | PR-AUC | calib-20 F1 | F1 at P = 0.5 |", "|---|---|---|---|---|---|"]
    for lab, k in arms:
        pb, c20 = R["pb"][k], R["calib20"][k]
        p05 = p05_per_subset(a.final.parent, k)
        res["arms"][k] = {"label": lab, "pb": pb, "calib20": c20, "p05": p05, "pb_oracle": orc.get(k),
                          "metrics": R["metrics"][k]}
        L.append(f"| {lab} | mean | {f3(pb['auroc_mean'])} | {f3(pb['ap_mean'])} | {f3(c20['calib20_mean'])} | "
                 f"{f3(p05['mean'])} |")
        for s in PB:
            L.append(f"| | {s} | {f3(pb['per_subset'][s]['auroc'])} | {f3(pb['per_subset'][s]['ap'])} | "
                     f"{f3(c20['per_subset'][s])} | {f3(p05[s])} |")
    L += ["", "Old InStep (PRM800K), for reference: " + f"{f3(R['pb']['instep_prm']['auroc_mean'])} / "
          f"{f3(R['pb']['instep_prm']['ap_mean'])} / {f3(R['calib20']['instep_prm']['calib20_mean'])} / "
          f"{f3(orc.get('instep_prm', {}).get('p05'))} at 0.5.", "",
          "## ProcessBench paired intervals (no-overlap minus X, 2,000 paired resamples)", "",
          "| arm | " + " | ".join(f"− {x} AUROC | − {x} PR-AUC" for x in PB_VS) + " |",
          "|---" * (1 + 2 * len(PB_VS)) + "|"]
    for lab, k in arms:
        cells = []
        for x in PB_VS:
            q = pbp.get((k, x))
            cells += [fd(q and q["auroc_mean"]), fd(q and q["ap_mean"])]
            res["arms"][k].setdefault("pb_pairs", {})[x] = q
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    # ReProbe test sets
    L += ["", "## ReProbe's eight test sets (paper_v1.py; 2,000 paired trace resamples)", "",
          "| arm | set | PR-AUC | AUROC | F1 at val threshold | oracle F1 |", "|---|---|---|---|---|---|"]
    for lab, k in arms:
        m = R["metrics"][k]
        for s in ("overall",) + TEST:
            r = m[s]
            L.append(f"| {lab if s == 'overall' else ''} | {s} | {f3(r['ap'])} | {f3(r['auroc'])} | "
                     f"{f3(r.get('val_f1'))} | {f3(r.get('oracle_f1'))} |")
    L += ["", "| arm | InStep (DS) − arm, overall PR-AUC | arm − PRM-7B, overall PR-AUC |", "|---|---|---|"]
    for lab, k in arms:
        x, y = tp.get(("instep_ds", k)), tp.get((k, "prm7b"))
        res["arms"][k]["test_pairs"] = {"instep_ds_minus": x and x["overall"], "minus_prm7b": y and y["overall"]}
        L.append(f"| {lab} | {fd(x and x['overall']['ap'])} | {fd(y and y['overall']['ap'])} |")
    if a.baselines and a.baselines.exists():
        B = json.loads(a.baselines.read_text())
        res["baselines"] = B
        L += ["", "## Same-data baselines on the filtered draws (pb_oracle.py, 2,000 paired resamples)", "",
              "| arm | seeds | AUROC | PR-AUC | calib-20 F1 |", "|---|---|---|---|---|"]
        for k, v in B["arms"].items():
            L.append(f"| {k} | {v['seeds']} | {f3(v['auroc'])} | {f3(v['ap'])} | {f3(v['calib20'])} |")
        L += ["", "| pair | AUROC | PR-AUC |", "|---|---|---|"]
        for q in B["pairs"]:
            L.append(f"| {q['a']} − {q['b']} | {fd(q['auroc'])} | {fd(q['ap'])} |")
    L += ["", "## Numbers outside InStep (PRM800K) against the paper's final.json", ""]
    L += ["All equal." if not diffs else f"**{len(diffs)} values differ:**"] + [f"- {d}" for d in diffs[:200]]
    a.out.write_text("\n".join(L) + "\n")
    a.json.write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
