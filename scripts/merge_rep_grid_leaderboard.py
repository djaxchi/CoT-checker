#!/usr/bin/env python3
"""Turn the grid's per-cell results.json files into the leaderboard.

The v1 leaderboard reported one number per row from one seed, which cannot rank
rows that differ by a point or two of F1_PB. Here a cell is summarised as mean +-
std over its seeds, and the table is laid out as representation x learner so the
two axes can be read separately: down a column, the learner is held fixed and
only the representation moves; across a row, the reverse.

Also emits the capacity view, F1_PB against learner parameter count, which is
what actually answers "representation or detector?": a representation that is
genuinely better dominates at every capacity, while one that only looked better
because its detector was larger converges as capacity grows.

Cells trained with a cap (`full_train: false`) are reported in a separate section
and never mixed into the main table, since v1's confound was exactly that kind of
silent mixing.

Before rendering anything it checks that every cell read the same inputs, by
comparing the store fingerprints each cell recorded. A benchmark whose rows were
trained on different activations is not a benchmark, so a disagreement is a hard
error rather than a footnote.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def load_cells(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.rglob("results.json")):
        try:
            out.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            print(f"[warn] unreadable: {f}")
    return out


def pb_avg(cell: dict, key: str) -> float | None:
    """Mean over the four subsets of a per-subset ProcessBench number."""
    vals = []
    for sub in PB_SUBSETS:
        entry = cell.get("processbench", {}).get(sub)
        if entry is None:
            return None
        vals.append(entry["val_selected"]["F1_PB"] if key == "val"
                    else entry["oracle_F1_PB"])
    return sum(vals) / len(vals)


def check_inputs(cells: list[dict]) -> dict[str, str]:
    """Every cell must have read the same splits. Raises if any disagree."""
    missing = [f"{c['rep']} x {c['learner']} seed {c['seed']}"
               for c in cells if not c.get("inputs")]
    if missing:
        raise SystemExit(
            "cells carry no input fingerprint, so it cannot be shown they read "
            "the same activations; rerun them with the current cell runner:\n  "
            + "\n  ".join(missing))

    reference = cells[0]["inputs"]
    problems: list[str] = []
    for c in cells[1:]:
        cell_id = f"{c['rep']} x {c['learner']} seed {c['seed']}"
        if set(c["inputs"]) != set(reference):
            problems.append(f"{cell_id}: read a different set of splits "
                            f"({sorted(set(c['inputs']) ^ set(reference))})")
            continue
        for split, digest in c["inputs"].items():
            if digest != reference[split]:
                problems.append(f"{cell_id}: {split} fingerprint {digest} != "
                                f"{reference[split]}")
    if problems:
        raise SystemExit("cells did not read the same inputs, the table would not "
                         "be a controlled comparison:\n  " + "\n  ".join(problems))
    return reference


def agg(values: list[float]) -> tuple[float, float]:
    return (st.mean(values), st.stdev(values) if len(values) > 1 else 0.0)


def fmt(mean: float, sd: float, n: int) -> str:
    return f"{mean:.3f} ± {sd:.3f}" if n > 1 else f"{mean:.3f}"


def summarise(cells: list[dict]) -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cells:
        groups[(c["rep"], c["learner"])].append(c)
    out = {}
    for key, members in groups.items():
        aurocs = [m["in_domain"]["auroc"] for m in members]
        f1s = [v for v in (pb_avg(m, "val") for m in members) if v is not None]
        oracles = [v for v in (pb_avg(m, "oracle") for m in members) if v is not None]
        out[key] = {
            "n_seeds": len(members),
            "seeds": sorted(m["seed"] for m in members),
            "n_params": members[0]["n_params"],
            "dim": members[0]["dim"],
            "n_train": members[0]["n_train"],
            "full_train": all(m["full_train"] for m in members),
            "auroc": agg(aurocs),
            "f1_pb_val": agg(f1s) if f1s else None,
            "f1_pb_oracle": agg(oracles) if oracles else None,
            "hp": [m["hp"]["selected"] for m in members],
        }
    return out


def render(summary: dict[tuple[str, str], dict], metric: str) -> list[str]:
    reps = sorted({r for r, _ in summary})
    learners = sorted({l for _, l in summary})
    lines = ["| representation | dim | " + " | ".join(learners) + " |",
             "|---|---|" + "---|" * len(learners)]
    for rep in reps:
        dim = next((v["dim"] for (r, _), v in summary.items() if r == rep), 0)
        row = [f"| `{rep}` | {dim} "]
        for learner in learners:
            v = summary.get((rep, learner))
            if v is None or v[metric] is None:
                row.append("| — ")
            else:
                row.append(f"| {fmt(*v[metric], v['n_seeds'])} ")
        lines.append("".join(row) + "|")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_root", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cells = load_cells(args.run_root)
    if not cells:
        raise SystemExit(f"no results.json under {args.run_root}")
    inputs = check_inputs(cells)
    full = [c for c in cells if c["full_train"]]
    capped = [c for c in cells if not c["full_train"]]

    lines = [f"# Representation x learner grid ({len(full)} full-split cells)", ""]
    summary = summarise(full)
    for metric, title in [("auroc", "In-domain PRM800K test AUROC"),
                          ("f1_pb_val", "ProcessBench F1_PB, val-selected threshold, 4-subset mean"),
                          ("f1_pb_oracle", "ProcessBench F1_PB, oracle threshold, 4-subset mean")]:
        lines += [f"## {title}", ""] + render(summary, metric) + [""]

    lines += ["## Capacity view (F1_PB against learner parameters)", "",
              "| representation | learner | params | AUROC | F1_PB (val) | seeds |",
              "|---|---|---|---|---|---|"]
    for (rep, learner), v in sorted(summary.items(), key=lambda kv: (kv[0][0], kv[1]["n_params"])):
        f1 = fmt(*v["f1_pb_val"], v["n_seeds"]) if v["f1_pb_val"] else "—"
        lines.append(f"| `{rep}` | `{learner}` | {v['n_params']:,} | "
                     f"{fmt(*v['auroc'], v['n_seeds'])} | {f1} | {v['n_seeds']} |")
    lines.append("")

    if capped:
        lines += ["## Excluded: cells trained with a cap", "",
                  "Not comparable to the table above; listed only so they are not lost.", "",
                  "| representation | learner | n_train | AUROC |", "|---|---|---|---|"]
        for c in capped:
            lines.append(f"| `{c['rep']}` | `{c['learner']}` | {c['n_train']:,} | "
                         f"{c['in_domain']['auroc']:.3f} |")
        lines.append("")

    n_train = next(iter(summary.values()))["n_train"] if summary else 0
    lines += ["## Protocol", "",
              f"Every cell above trains on the same {n_train:,} PRM800K steps, is "
              "tuned over the same lr x weight-decay grid selected on validation "
              "AUROC, and is trained by the same AdamW + BCE trainer with the same "
              "early-stopping rule. Only the representation and the learner vary.", "",
              "Every cell was verified to have read the same inputs "
              f"({len(cells)} cells, {len(inputs)} splits):", "",
              "| split | fingerprint |", "|---|---|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in sorted(inputs.items())]
    lines.append("")

    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
