#!/usr/bin/env python3
"""Turn the grid's per-cell results.json files into the leaderboard.

The v1 leaderboard reported one number per row from one seed, which cannot rank
rows that differ by a point or two of F1_PB. Here a cell is summarised as mean +-
std over its seeds, and the table is laid out as representation x learner so the
two axes can be read separately: down a column, the learner is held fixed and
only the representation moves; across a row, the reverse.

The headline metric is F1_PB at **calib-20**, the same one the v1 leaderboard
reported: hold out 20 ProcessBench traces per subset (stratified error/correct),
grid-max the first-error threshold there, apply it to the rest, average over 20
splits. It is recomputed here from each cell's saved per-trace scores using
`scripts/analysis/pb_threshold_calibration.py`, so the new table and the old one
are comparable on the number that matters rather than only on AUROC. The
val-selected threshold picks about 0.5 and under-tunes for ProcessBench's
correct-skew; oracle peeks at the whole test set. Both are kept for context.

Also emits the capacity view, F1_PB against learner parameter count, which is
what actually answers "representation or detector?": a representation that is
genuinely better dominates at every capacity, while one that only looked better
because its detector was larger converges as capacity grows. Learner columns are
ordered by parameter count, not alphabetically, so that story is readable
straight off the main table.

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
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analysis.pb_threshold_calibration import load_traces  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
CALIB_SIZE = 20
CALIB_SPLITS = 20
CALIB_GRID = np.arange(0.01, 1.00, 0.01)   # legacy uniform grid, kept for comparison
N_QUANTILES = 99


def load_cells(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.rglob("results.json")):
        try:
            cell = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"[warn] unreadable: {f}")
            continue
        cell["_dir"] = str(f.parent)
        out.append(cell)
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


def pred_matrix(traces: list, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(predictions (n_traces, n_thresholds), labels (n_traces,)).

    The first-error prediction is the first step whose score exceeds the
    threshold, else -1. Precomputing it for the whole grid once turns every
    later calibration split into counting, which is what makes 20 splits x 99
    thresholds x 57 cells finish at all: the straightforward nested loop is
    billions of trace scans.
    """
    n_t = len(grid)
    preds = np.empty((len(traces), n_t), dtype=np.int32)
    labels = np.empty(len(traces), dtype=np.int64)
    for i, (label, scores) in enumerate(traces):
        s = np.asarray(scores, dtype=np.float64)
        hit = s[None, :] > grid[:, None]              # (n_t, L)
        any_hit = hit.any(axis=1)
        preds[i] = np.where(any_hit, hit.argmax(axis=1), -1)
        labels[i] = label
    return preds, labels


def f1_pb_from_preds(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """F1_PB for every threshold column of `preds`, the harmonic mean of the
    error-trace and correct-trace accuracies (identical definition to
    evaluate_processbench and trace_f1_pb)."""
    is_cor = labels == -1
    n_err, n_cor = int((~is_cor).sum()), int(is_cor.sum())
    cor_hit = (preds[is_cor] == -1).sum(axis=0)
    err_hit = (preds[~is_cor] == labels[~is_cor, None]).sum(axis=0)
    acc_err = err_hit / n_err if n_err else np.zeros(preds.shape[1])
    acc_cor = cor_hit / n_cor if n_cor else np.zeros(preds.shape[1])
    denom = acc_err + acc_cor
    return np.where(denom > 0, 2 * acc_err * acc_cor / np.maximum(denom, 1e-12), 0.0)


def quantile_grid(traces: list, n: int = N_QUANTILES) -> np.ndarray:
    """Candidate thresholds at the score quantiles of `traces`.

    A uniform grid in probability space assumes scores are spread across [0,1].
    An overconfident probe breaks that assumption: on Qwen3 the wide
    representations pushed 54-66% of scores below 0.01 and 10-30% above 0.99, so
    a 0.01-step grid had almost no resolution where the decision boundary sat,
    and threshold selection on 20 traces became a coin flip (one seed of
    boundary_stats scored 0.248 against 0.498 and 0.494 for its siblings, while
    its in-domain AUROC was 0.8685, in line with theirs).

    Quantiles put the candidates where the scores actually are, whatever the
    calibration. `traces` should be the calibration split only, never the
    evaluation traces, or the grid has seen data the threshold is scored on.
    """
    s = np.concatenate([np.asarray(sc, dtype=np.float64) for _, sc in traces])
    q = np.unique(np.quantile(s, np.linspace(0.005, 0.995, n)))
    return q if len(q) >= 3 else CALIB_GRID


def calib20_subset(traces: list, grid_mode: str = "quantile") -> float:
    """F1_PB at calib-20 for one subset: hold out CALIB_SIZE traces (stratified),
    grid-max the threshold there, apply it to the remainder, mean over splits.

    With grid_mode='quantile' each split derives its candidate thresholds from
    its own calibration traces, so the grid never sees the evaluation traces.
    Predictions are still computed once, over the union of every split's grid,
    and each split reads back the columns belonging to its own.
    """
    idx = np.arange(len(traces))
    labels_all = np.array([t[0] for t in traces])
    splits, grids = [], []
    for sd in range(CALIB_SPLITS):
        rng = np.random.default_rng(sd)
        err, cor = idx[labels_all != -1], idx[labels_all == -1]
        frac = CALIB_SIZE / len(traces)
        n_err_c = min(len(err), round(len(err) * frac))
        n_cor_c = min(len(cor), CALIB_SIZE - n_err_c)
        err_p, cor_p = rng.permutation(len(err)), rng.permutation(len(cor))
        cal = np.concatenate([err[err_p[:n_err_c]], cor[cor_p[:n_cor_c]]])
        ev = np.concatenate([err[err_p[n_err_c:]], cor[cor_p[n_cor_c:]]])
        splits.append((cal, ev))
        grids.append(CALIB_GRID if grid_mode == "uniform"
                     else quantile_grid([traces[i] for i in cal]))

    union = np.unique(np.concatenate(grids))
    preds, labels = pred_matrix(traces, union)
    evals = []
    for (cal, ev), g in zip(splits, grids):
        cols = np.searchsorted(union, g)
        f1_cal = f1_pb_from_preds(preds[cal][:, cols], labels[cal])
        t = int(cols[int(np.argmax(f1_cal))])
        evals.append(float(f1_pb_from_preds(preds[ev], labels[ev])[t]))
    return float(np.mean(evals))


def calib20(cell: dict) -> float | None:
    """F1_PB at calib-20, averaged over the four subsets.

    Recomputed from the cell's saved per-trace scores under the same protocol the
    v1 rows used, so the two leaderboards report the same metric. Returns None if
    any subset's scores are missing, since a three-subset average would not be
    comparable to a four-subset one.
    """
    if cell.get("_calib20") is not None:
        return cell["_calib20"]
    out_dir = Path(cell.get("_dir", ""))
    per_subset = []
    for sub in PB_SUBSETS:
        path = out_dir / f"pb_step_scores_{sub}.jsonl"
        if not path.exists():
            return None
        traces = load_traces(path)
        if len(traces) <= CALIB_SIZE:
            return None
        per_subset.append(calib20_subset(traces))
    cell["_calib20"] = sum(per_subset) / len(per_subset)
    return cell["_calib20"]


def check_inputs(cells: list[dict]) -> dict[str, str]:
    """Every cell must have read the same splits. Raises if any disagree."""
    missing = [f"{c['rep']} x {c['learner']} seed {c['seed']}"
               for c in cells if not c.get("inputs")]
    if missing:
        raise SystemExit(
            "cells carry no input fingerprint, so it cannot be shown they read "
            "the same activations; rerun them with the current cell runner:\n  "
            + "\n  ".join(missing))

    # Rescaling changes the numbers entering every probe, so a table mixing
    # rescaled and un-rescaled cells is not one protocol. Same reasoning as the
    # capped/uncapped split, and the same refusal.
    modes = {c.get("protocol", {}).get("rescale", "unrecorded") for c in cells}
    if len(modes) > 1:
        by_mode = {}
        for c in cells:
            by_mode.setdefault(c.get("protocol", {}).get("rescale", "unrecorded"),
                               []).append(f"{c['rep']} x {c['learner']} seed {c['seed']}")
        raise SystemExit(
            "cells were trained under different rescaling settings, so the table "
            "would not be one protocol:\n  " + "\n  ".join(
                f"{m}: {len(v)} cells (e.g. {v[0]})" for m, v in sorted(by_mode.items())))

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
        calibs = [v for v in (calib20(m) for m in members) if v is not None]
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
            "f1_pb_calib20": agg(calibs) if calibs else None,
            "f1_pb_val": agg(f1s) if f1s else None,
            "f1_pb_oracle": agg(oracles) if oracles else None,
            "hp": [m["hp"]["selected"] for m in members],
        }
    return out


def learner_order(summary: dict[tuple[str, str], dict]) -> list[str]:
    """Learners smallest first, by parameter count. Alphabetical ordering hid the
    capacity story; this makes each row read left to right as a capacity curve."""
    params: dict[str, int] = {}
    for (_, learner), v in summary.items():
        params[learner] = min(params.get(learner, v["n_params"]), v["n_params"])
    return sorted(params, key=lambda k: (params[k], k))


def render(summary: dict[tuple[str, str], dict], metric: str) -> list[str]:
    reps = sorted({r for r, _ in summary})
    learners = learner_order(summary)
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
    rescale_mode = sorted({c.get("protocol", {}).get("rescale", "unrecorded")
                           for c in cells})[0]
    full = [c for c in cells if c["full_train"]]
    capped = [c for c in cells if not c["full_train"]]

    lines = [f"# Representation x learner grid ({len(full)} full-split cells)", ""]
    summary = summarise(full)
    for metric, title in [
            ("f1_pb_calib20",
             "ProcessBench F1_PB @ calib-20, 4-subset mean (HEADLINE)"),
            ("auroc", "In-domain PRM800K test AUROC"),
            ("f1_pb_val", "ProcessBench F1_PB, val-selected threshold, 4-subset mean"),
            ("f1_pb_oracle", "ProcessBench F1_PB, oracle threshold, 4-subset mean")]:
        lines += [f"## {title}", ""] + render(summary, metric) + [""]

    lines += ["## Capacity view (F1_PB @ calib-20 against learner parameters)", "",
              "| representation | learner | params | AUROC | F1_PB @ calib-20 | seeds |",
              "|---|---|---|---|---|---|"]
    for (rep, learner), v in sorted(summary.items(), key=lambda kv: (kv[0][0], kv[1]["n_params"])):
        f1 = fmt(*v["f1_pb_calib20"], v["n_seeds"]) if v["f1_pb_calib20"] else "—"
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
              f"Rescaling: {rescale_mode}. With `zscore`, each position's "
              "training average is subtracted and its swing divided out, so the "
              "numbers entering the probe sit near 0 and swing by about 1 instead "
              "of about 22; sparse codes are divided but not centred, to keep "
              "their zeros. Statistics are fitted on the training split only.", "",
              f"The headline is F1_PB at calib-20: {CALIB_SIZE} held-out "
              f"ProcessBench traces per subset (stratified) pick the first-error "
              f"threshold, which is applied to the rest, averaged over "
              f"{CALIB_SPLITS} splits. Candidate thresholds are the score "
              "quantiles of each split's own calibration traces, not a uniform "
              "grid in probability space: an overconfident probe piles its scores "
              "at 0 and 1, where a uniform grid has no resolution and threshold "
              "selection on 20 traces becomes near-random.", "",
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
