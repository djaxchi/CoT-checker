#!/usr/bin/env python3
"""judge_prompt_v1: Qwen3-8B under a verification prompt against the same cells on
Qwen3-8B under the verifier template (Instruct leaderboard) and on
Qwen2.5-Math-PRM-7B (prm_backbone_v1). docs/judge_prompt_v1_plan.md.

Rows:
  judge-span    step-span readouts, only the backbone context changed
  judge-verdict the state at the end of the verdict question (rep `last_token`
                on the judge-token store; compared with each baseline's best
                vector cell, since the baselines have no verdict token)
  zero-shot     P(No) at the verdict token, no probe

Per row: test_2k AUROC, in-domain f1_incorrect at the val threshold (trivial
0.667), ProcessBench F1_PB oracle and calib-20 (4-subset mean), seed mean +- sd.
Paired trace bootstrap of the calib-20 difference (seed-averaged scores of both
sides recomputed on the same resampled traces) for the comparisons named in
the plan.

    python scripts/analysis/judge_prompt_report.py \
        --span_root .../judge_prompt_v1/cells_span \
        --verdict_root .../judge_prompt_v1/cells_verdict \
        --zeroshot_root .../judge_prompt_v1/cells_zeroshot \
        --instruct_root cot-checker-results/prm_backbone_v1/instruct_cells \
        --prm_root cot-checker-results/prm_backbone_v1/prm_cells \
        --out results/judge_prompt_v1/leaderboard.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.pb_threshold_calibration import load_traces  # noqa: E402
from scripts.merge_rep_grid_leaderboard import (  # noqa: E402
    PB_SUBSETS, calib20_subset, load_cells, summarise,
)

TRIVIAL_F1 = 2 / 3


def id_f1(cells: list[dict]) -> dict[tuple[str, str], tuple[float, float]]:
    g: dict[tuple[str, str], list[float]] = {}
    for c in cells:
        g.setdefault((c["rep"], c["learner"]), []).append(
            c["in_domain"]["val_selected"]["f1_incorrect"])
    return {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
            for k, v in g.items()}


def table(cells: list[dict]) -> dict[tuple[str, str], dict]:
    s, f = summarise(cells), id_f1(cells)
    for k in s:
        s[k]["id_f1"] = f[k]
        s[k]["dirs"] = sorted(c["_dir"] for c in cells if (c["rep"], c["learner"]) == k)
    return s


def boot_delta(dirs_a: list[str], dirs_b: list[str], n_boot: int, seed: int = 0) -> dict:
    """Paired trace bootstrap of calib-20(a) - calib-20(b), each side the mean
    over its seeds, both recomputed on the same resampled traces per subset."""
    def load(dirs):
        return [{s: load_traces(Path(d) / f"pb_step_scores_{s}.jsonl") for s in PB_SUBSETS}
                for d in dirs]
    A, B = load(dirs_a), load(dirs_b)
    for s in PB_SUBSETS:
        ids = {len(x[s]) for x in A + B}
        if len(ids) != 1:
            raise ValueError(f"{s}: trace counts differ across cells {ids}")
    n = {s: len(A[0][s]) for s in PB_SUBSETS}

    def score(side, idx):
        return float(np.mean([np.mean([calib20_subset([c[s][i] for i in idx[s]])
                                       for s in PB_SUBSETS]) for c in side]))
    full = {s: np.arange(n[s]) for s in PB_SUBSETS}
    point = score(A, full) - score(B, full)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = {s: rng.integers(0, n[s], n[s]) for s in PB_SUBSETS}
        d.append(score(A, idx) - score(B, idx))
    d = np.asarray(d)
    return {"delta": point, "se": float(d.std(ddof=1)),
            "ci95_normal": [point - 1.96 * float(d.std(ddof=1)),
                            point + 1.96 * float(d.std(ddof=1))],
            "frac_boot_le0": float((d <= 0).mean()), "n_boot": n_boot}


def fmt(row: dict | None, key: str) -> str:
    if row is None or row.get(key) is None:
        return "-"
    m, sd = row[key][0], row[key][1]
    return f"{m:.4f} +- {sd:.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("span_root", "verdict_root", "zeroshot_root", "instruct_root", "prm_root"):
        p.add_argument(f"--{k}", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_boot", type=int, default=200)
    a = p.parse_args()

    span, verdict = table(load_cells(a.span_root)), table(load_cells(a.verdict_root))
    zs, ins, prm = (table(load_cells(a.zeroshot_root)), table(load_cells(a.instruct_root)),
                    table(load_cells(a.prm_root)))
    keys = ("auroc", "id_f1", "f1_pb_oracle", "f1_pb_calib20")
    head = ("| arm | rep x learner | AUROC | in-domain F1 | F1_PB oracle | F1_PB calib-20 |\n"
            "|---|---|---:|---:|---:|---:|")
    lines = ["# judge_prompt_v1 leaderboard", "",
             f"In-domain F1 is f1_incorrect at the val threshold; trivial = {TRIVIAL_F1:.3f}.",
             "Seed mean +- sd. Baselines: Qwen3-8B under the verifier template "
             "(instruct) and Qwen2.5-Math-PRM-7B (prm), same cells, same protocol.", "",
             head]
    out_json: dict = {"rows": [], "bootstrap": []}

    def add(arm, key, row):
        lines.append(f"| {arm} | {key[0]} x {key[1]} | "
                     + " | ".join(fmt(row, k) for k in keys) + " |")
        out_json["rows"].append({"arm": arm, "rep": key[0], "learner": key[1],
                                 **{k: row.get(k) for k in keys}})

    for key in sorted(span, key=lambda k: -span[k]["f1_pb_calib20"][0]):
        add("judge-span", key, span[key])
        for name, base in (("instruct", ins), ("prm", prm)):
            if key in base:
                add(name, key, base[key])
    lines.append("|  |  |  |  |  |  |")
    for key in verdict:
        add("judge-verdict", ("verdict_token", key[1]), verdict[key])
    for key in zs:
        add("zero-shot", ("P(No)", "none"), zs[key])

    def best(t):
        return max(t, key=lambda k: t[k]["f1_pb_calib20"][0])
    pairs = []
    for jname, jt in (("judge-span", span), ("judge-verdict", verdict)):
        jb = best(jt)
        pairs.append((f"{jname} best vs instruct best", jt[jb], ins[best(ins)], jb, best(ins)))
        pairs.append((f"{jname} best vs prm best", jt[jb], prm[best(prm)], jb, best(prm)))
    for key in span:
        if key in ins:
            pairs.append(("judge-span vs instruct, same cell", span[key], ins[key], key, key))
    lines += ["", "## Paired trace bootstrap, calib-20 F1_PB difference", "",
              "| comparison | A | B | delta | SE | 95% CI (normal) |", "|---|---|---|---:|---:|---|"]
    for name, ra, rb, ka, kb in pairs:
        r = boot_delta(ra["dirs"], rb["dirs"], a.n_boot)
        lines.append(f"| {name} | {ka[0]} x {ka[1]} | {kb[0]} x {kb[1]} | {r['delta']:+.4f} "
                     f"| {r['se']:.4f} | [{r['ci95_normal'][0]:+.4f}, {r['ci95_normal'][1]:+.4f}] |")
        out_json["bootstrap"].append({"comparison": name, "a": list(ka), "b": list(kb), **r})
        print(lines[-1], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    a.out.with_suffix(".json").write_text(json.dumps(out_json, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
