#!/usr/bin/env python3
"""Pick the shared LR/WD for bidirectional_token_probe_v1 from the 12 search fits.

Score of a setting = mean over {causal, full} of the best-epoch dev F1 (seed 0).
Ties: lower learning rate, then lower weight decay. Writes selection.json and the
final roster (4 arms x seeds 42 43 44, one process per fit)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final_fits", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--name_prefix", default="final_s")
    ap.add_argument("--arms", nargs="+", default=["future1", "full", "causal", "local"])
    ap.add_argument("--extra", default="", help="appended to every roster line (e.g. --score_splits ...)")
    a = ap.parse_args()
    scores = defaultdict(dict)
    for d in sorted(a.fits_root.glob("search_*/*/done.json")):
        r = json.loads(d.read_text())
        scores[(r["lr"], r["wd"])][r["arm"]] = r["dev_f1"]
    table = []
    for (lr, wd), arms in sorted(scores.items()):
        if set(arms) != {"causal", "full"}:
            raise SystemExit(f"[FATAL] incomplete search setting lr={lr} wd={wd}: {arms}")
        table.append({"lr": lr, "wd": wd, "causal": arms["causal"], "full": arms["full"],
                      "mean": (arms["causal"] + arms["full"]) / 2})
    if len(table) != 6:
        raise SystemExit(f"[FATAL] expected 6 settings, found {len(table)}")
    best = max(table, key=lambda r: (round(r["mean"], 12), -r["lr"], -r["wd"]))
    a.out.write_text(json.dumps({"rule": "max mean dev F1 of causal and full; ties -> lower lr, lower wd",
                                 "table": table, "selected": best}, indent=2))
    lines = [f"# final roster from {a.out}"]
    for s in a.seeds:
        for arm in a.arms:
            lines.append(f"{a.name_prefix}{s}_{arm} --arms {arm} --seed {s} --lr {best['lr']} --wd {best['wd']} {a.extra}".rstrip())
    a.final_fits.write_text("\n".join(lines) + "\n")
    print(json.dumps(best))


if __name__ == "__main__":
    main()
