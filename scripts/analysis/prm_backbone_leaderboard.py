"""prm_backbone_v1: the Instruct leaderboard roster on Qwen2.5-Math-PRM-7B states,
side by side with the same cells on Qwen3-8B (Instruct).

Both grids read the same frozen PRM800K splits and ProcessBench files under the
same verifier template and protocol; only the backbone differs. Ranked by the
PRM arm's ProcessBench F1_PB at calib-20 (the Instruct leaderboard's ranking
metric). AUROC is the prevalence-invariant comparison; the in-domain F1 column
is f1_incorrect at the val-selected threshold on the balanced test split, where
the trivial always-incorrect predictor scores 0.667.

    python scripts/analysis/prm_backbone_leaderboard.py \
        --prm_root .../prm_backbone_v1/cells --instruct_root .../instruct_leaderboard_v1/cells \
        --out results/prm_backbone_v1/leaderboard.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.merge_rep_grid_leaderboard import load_cells, summarise  # noqa: E402

TRIVIAL_F1_IN_DOMAIN = 2 / 3   # always "incorrect" on the balanced test split


def in_domain_f1(cells: list[dict]) -> dict[tuple[str, str], tuple[float, float]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for c in cells:
        groups.setdefault((c["rep"], c["learner"]), []).append(
            c["in_domain"]["val_selected"]["f1_incorrect"])
    return {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
            for k, v in groups.items()}


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x: list[float]) -> list[float]:
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prm_root", type=Path, required=True)
    p.add_argument("--instruct_root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    prm_cells, ins_cells = load_cells(a.prm_root), load_cells(a.instruct_root)
    prm, ins = summarise(prm_cells), summarise(ins_cells)
    prm_f1, ins_f1 = in_domain_f1(prm_cells), in_domain_f1(ins_cells)

    def m(row: dict | None, key: str) -> float | None:
        return None if row is None or row.get(key) is None else row[key][0]

    def fmt(x: float | None, nd: int = 4) -> str:
        return "-" if x is None else f"{x:.{nd}f}"

    def dfmt(x: float | None, y: float | None) -> str:
        return "-" if x is None or y is None else f"{x - y:+.4f}"

    ranked = sorted(prm.items(), key=lambda kv: -(m(kv[1], "f1_pb_calib20") or -1))
    lines = [
        "**prm_backbone_v1: leaderboard on Qwen2.5-Math-PRM-7B states (P) against Qwen3-8B Instruct (I)**", "",
        "Same splits, template, cells and protocol; only the backbone differs. "
        "Means over seeds (count in the seeds column). F1_PB is the ProcessBench "
        "first-error F1, four-subset mean; ID F1 is f1_incorrect on the balanced "
        f"PRM800K test at the val-selected threshold (trivial baseline {TRIVIAL_F1_IN_DOMAIN:.3f}).", "",
        "| rank | representation | learner | seeds P/I | P AUROC | I AUROC | dAUROC "
        "| P ID F1 | I ID F1 | P F1_PB val | I F1_PB val | P F1_PB oracle | I F1_PB oracle "
        "| P F1_PB calib20 | I F1_PB calib20 | dcalib20 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows_json = []
    for rank, (key, pr) in enumerate(ranked, 1):
        ir = ins.get(key)
        row = {
            "rep": key[0], "learner": key[1],
            "seeds_prm": pr["seeds"], "seeds_instruct": ir["seeds"] if ir else [],
            "prm": {"auroc": pr["auroc"], "id_f1": prm_f1[key], "f1_pb_val": pr["f1_pb_val"],
                    "f1_pb_oracle": pr["f1_pb_oracle"], "f1_pb_calib20": pr["f1_pb_calib20"]},
            "instruct": None if ir is None else {
                "auroc": ir["auroc"], "id_f1": ins_f1[key], "f1_pb_val": ir["f1_pb_val"],
                "f1_pb_oracle": ir["f1_pb_oracle"], "f1_pb_calib20": ir["f1_pb_calib20"]},
        }
        rows_json.append(row)
        lines.append(
            f"| {rank} | {key[0]} | {key[1]} | {pr['n_seeds']}/{ir['n_seeds'] if ir else 0} "
            f"| {fmt(m(pr, 'auroc'))} | {fmt(m(ir, 'auroc'))} | {dfmt(m(pr, 'auroc'), m(ir, 'auroc'))} "
            f"| {fmt(prm_f1[key][0])} | {fmt(ins_f1[key][0] if key in ins_f1 else None)} "
            f"| {fmt(m(pr, 'f1_pb_val'))} | {fmt(m(ir, 'f1_pb_val'))} "
            f"| {fmt(m(pr, 'f1_pb_oracle'))} | {fmt(m(ir, 'f1_pb_oracle'))} "
            f"| {fmt(m(pr, 'f1_pb_calib20'))} | {fmt(m(ir, 'f1_pb_calib20'))} "
            f"| {dfmt(m(pr, 'f1_pb_calib20'), m(ir, 'f1_pb_calib20'))} |")

    shared = [k for k in prm if k in ins]
    lines += ["", f"Cells on both backbones: {len(shared)}."]
    for metric in ("auroc", "f1_pb_val", "f1_pb_oracle", "f1_pb_calib20"):
        pairs = [(m(prm[k], metric), m(ins[k], metric)) for k in shared]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 3:
            continue
        wins = sum(x > y for x, y in pairs)
        mean_d = statistics.mean(x - y for x, y in pairs)
        rho = spearman([x for x, _ in pairs], [y for _, y in pairs])
        lines.append(f"- {metric}: PRM higher on {wins}/{len(pairs)} cells, mean delta "
                     f"{mean_d:+.4f}, rank Spearman {rho:+.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    a.out.with_suffix(".json").write_text(json.dumps(rows_json, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
