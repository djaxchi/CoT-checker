#!/usr/bin/env python3
"""Record the rescaling setting on cells that predate the field.

`protocol.rescale` was added after two grids had already run, so their
results.json say nothing and the merge cannot tell them apart. The setting is
not in doubt: every cell writes a `[rescale]` line to its log when rescaling is
on, and none does when it is off, so the log is the evidence.

This writes the setting into results.json and stamps `rescale_backfilled` so it
is visible that the field was reconstructed rather than recorded at run time.
Only metadata is touched; no measurement is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_root", required=True, type=Path)
    p.add_argument("--rescale", required=True, choices=["none", "zscore"])
    p.add_argument("--require_log_evidence", action="store_true",
                   help="For zscore: only stamp cells whose log shows a [rescale] "
                        "line, so a mislabelled directory cannot be laundered.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    done = skipped = 0
    for f in sorted(args.run_root.rglob("results.json")):
        d = json.loads(f.read_text())
        if d.get("protocol", {}).get("rescale") is not None:
            skipped += 1
            continue
        if args.rescale == "zscore" and args.require_log_evidence:
            log = f.parent.parent / f"{f.parent.name}.log"
            if not (log.exists() and "[rescale]" in log.read_text()):
                print(f"  no evidence, skipping: {f.parent.name}")
                skipped += 1
                continue
        d.setdefault("protocol", {})["rescale"] = args.rescale
        d["protocol"]["rescale_backfilled"] = True
        if not args.dry_run:
            f.write_text(json.dumps(d, indent=2))
        done += 1
    print(f"[backfill] rescale={args.rescale}: stamped {done}, skipped {skipped}"
          + ("  (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
