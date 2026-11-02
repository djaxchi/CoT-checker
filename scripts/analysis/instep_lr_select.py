#!/usr/bin/env python3
"""Pick InStep's training config from an lr sweep (docs/instep_lr_sweep_plan.md).

Reads the sweep.json files that `read_token_inline.py train --sweep ... --val ...` writes
(one per seed and training source) and applies the rule frozen in the plan:

* a run (config x seed) is UNSTABLE if its loss went non-finite, or if its running training
  loss, at any logged step from 25% of training on, is above the constant predictor's loss
  on the training labels, or if its validation metric is more than 0.05 below the median of
  the same config's other seeds;
* a config is STABLE when none of its seeds is unstable;
* per source, the pick is the stable config with the highest seed-mean validation metric;
  any stable config within 0.003 of it is a tie, broken by the lower peak lr, then the
  longer warm-up;
* one recipe for every source if a config is stable in all of them and within 0.003 of each
  source's pick (the same tie-break among such configs); otherwise one recipe per source.

Only validation splits are read. Usage:
    python scripts/analysis/instep_lr_select.py \
        --source "reprobe_ds:ap:$R/sweep/ds_seed*/sweep.json" \
        --source "prm800k:auroc:$R/sweep/prm_seed*/sweep.json" --out $R/sweep/selection
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

LATE = 0.25      # the loss check starts at this fraction of training
OUTLIER = 0.05   # a seed this far below the median of the config's other seeds is unstable
TIE = 0.003      # configs this close to the best mean are tied


def run_flags(cfg: dict, steps: int, const_loss: float) -> list[str]:
    """Reasons a single run is unstable, outlier check aside."""
    out = []
    if cfg["dead_at"] is not None:
        out.append(f"non-finite loss at step {cfg['dead_at']}")
    late = [l for s, l in cfg["trace"] if s >= LATE * steps]
    if late and max(late) > const_loss:
        out.append(f"running loss {max(late):.3f} above the constant predictor's {const_loss:.3f}")
    return out


def summarize(files: list[str], metric: str) -> dict[str, dict]:
    """Per config tag: lr, warm-up, per-seed metric, per-seed flags."""
    key = {"ap": "val_ap", "auroc": "val_auroc"}[metric]
    table: dict[str, dict] = {}
    for f in sorted(files):
        js = json.loads(Path(f).read_text())
        for c in js["configs"]:
            t = table.setdefault(c["tag"], {"lr": c["lr"], "warmup": c["warmup"], "seeds": {}})
            late = [l for s, l in c["trace"] if s >= LATE * js["steps"]]
            t["seeds"][js["seed"]] = {"metric": c[key], "flags": run_flags(c, js["steps"], js["const_loss"]),
                                      "max_late_loss": max(late) if late else None,
                                      "const_loss": js["const_loss"]}
    for t in table.values():
        for s, r in t["seeds"].items():
            others = [o["metric"] for k, o in t["seeds"].items() if k != s and o["metric"] is not None]
            if r["metric"] is None:
                r["flags"].append("no validation score")
            elif others and r["metric"] < np.median(others) - OUTLIER:
                r["flags"].append(f"val {r['metric']:.3f} more than {OUTLIER} below the other seeds' median")
        vals = [r["metric"] for r in t["seeds"].values() if r["metric"] is not None]
        t["mean"] = float(np.mean(vals)) if vals else None
        t["sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        t["min"] = float(min(vals)) if vals else None
        t["n_unstable"] = sum(bool(r["flags"]) for r in t["seeds"].values())
        t["stable"] = t["n_unstable"] == 0 and t["mean"] is not None
    return table


def tie_break(tags: list[str], table: dict[str, dict]) -> str:
    return min(tags, key=lambda k: (table[k]["lr"], -table[k]["warmup"]))


def pick(table: dict[str, dict]) -> str | None:
    stable = [k for k, t in table.items() if t["stable"]]
    if not stable:
        return None
    best = max(table[k]["mean"] for k in stable)
    return tie_break([k for k in stable if table[k]["mean"] >= best - TIE], table)


def joint(tables: dict[str, dict], picks: dict[str, str | None]) -> str | None:
    if any(p is None for p in picks.values()):
        return None
    common = set.intersection(*(set(t) for t in tables.values()))
    ok = [k for k in common
          if all(tables[s][k]["stable"] and tables[s][k]["mean"] >= tables[s][picks[s]]["mean"] - TIE
                 for s in tables)]
    return tie_break(ok, next(iter(tables.values()))) if ok else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True, help="name:ap|auroc:glob of sweep.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tables, picks, lines = {}, {}, []
    for spec in a.source:
        name, metric, pattern = spec.split(":", 2)
        files = glob.glob(pattern)
        if not files:
            raise SystemExit(f"{name}: no sweep.json matches {pattern}")
        tables[name] = summarize(files, metric)
        picks[name] = pick(tables[name])
        lines += [f"## {name} (validation {metric.upper()}, {len(files)} seeds)", "",
                  "| config | mean | sd | min | unstable seeds | max late loss / constant |",
                  "|---|---|---|---|---|---|"]
        for k, t in sorted(tables[name].items(), key=lambda kv: (kv[1]["lr"], kv[1]["warmup"])):
            ml = [r["max_late_loss"] / r["const_loss"] for r in t["seeds"].values() if r["max_late_loss"]]
            fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
            lines.append(f"| {k}{' **pick**' if k == picks[name] else ''} | {fmt(t['mean'])} | {fmt(t['sd'])} "
                         f"| {fmt(t['min'])} | {t['n_unstable']}/{len(t['seeds'])} | {max(ml):.2f} |"
                         if ml else f"| {k} | {fmt(t['mean'])} | | | {t['n_unstable']} | |")
        for k, t in tables[name].items():
            for s, r in t["seeds"].items():
                for fl in r["flags"]:
                    lines.append(f"- {k} seed {s}: {fl}")
        lines.append("")
    one = joint(tables, picks) if len(tables) > 1 else None
    lines.append(f"Picks: {picks}. One recipe for all sources: {one or 'none, use the per-source picks'}.")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "selection.json").write_text(json.dumps({"tables": tables, "picks": picks, "joint": one}, indent=1))
    (a.out / "selection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
