"""Validate the frozen Instruct roster before reuse or leaderboard publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.merge_rep_grid_leaderboard import check_inputs, summarise  # noqa: E402

SEEDS = (42, 43, 44)
PROTOCOL = {"epochs": 30, "patience": 3, "batch_size": 256, "rescale": "none",
            "t_max": 512, "dropout": 0.1, "bucketed": True,
            "threshold_grid": "0.01", "test_is_val": False,
            "stems": {"train": "probe_train_full", "val": "val_5k", "test": "test_2k"}}
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def roster(path: Path) -> list[tuple[str, str]]:
    pairs = [tuple(line.split("#", 1)[0].split()) for line in path.read_text().splitlines()
             if line.split("#", 1)[0].strip()]
    if any(len(pair) != 2 for pair in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("Malformed or duplicate roster entry")
    return pairs


def cell_tag(rep: str, learner: str, seed: int) -> str:
    return f"{rep}__{learner.replace(':', '_').replace(',', '_')}__seed{seed}"


def validate_result(result: dict, rep: str, learner: str, seed: int) -> None:
    if (result["rep"], result["learner"], result["seed"]) != (rep, learner, seed):
        raise ValueError("Cell identity does not match the frozen roster")
    if result.get("n_train") != 513810 or not result.get("full_train"):
        raise ValueError("Every cell must fit all 513810 training steps")
    # Length bucketing applies to sequence learners only; vector cells record
    # bucketed=False on both backbones, so the field is checked per kind.
    expected = dict(PROTOCOL, bucketed=(rep == "step_tokens"))
    if any(result.get("protocol", {}).get(key) != val for key, val in expected.items()):
        raise ValueError("Training protocol does not match the Instruct reference")
    if result["hp"].get("search_rows") != 100000:
        raise ValueError("Hyperparameter search must use 100000 rows")
    if seed == 42:
        trials = {(t["lr"], t["weight_decay"]) for t in result["hp"]["trials"]}
        if trials != {(lr, wd) for lr in (1e-3, 3e-4, 1e-4) for wd in (0.0, 0.01)}:
            raise ValueError("Hyperparameter search differs from the frozen six-point grid")
    elif not result["hp"].get("reused_from"):
        raise ValueError("Later seeds must reuse the seed-42 hyperparameters")
    if learner == "transformer:d512,l2,f2048,h8" and result["n_params"] != 8665089:
        raise ValueError("Truncated transformer specification")
    if set(result.get("processbench", {})) != set(SUBSETS):
        raise ValueError("Need all four ProcessBench subsets")


def validate_grid(root: Path, pairs: list[tuple[str, str]], reference: dict,
                  seeds_by_pair: dict[tuple[str, str], tuple[int, ...]] | None = None) -> list[dict]:
    if seeds_by_pair is not None and set(seeds_by_pair) != set(pairs):
        raise ValueError("Explicit seed roster must cover exactly the requested pairs")
    cells = []
    for rep, learner in pairs:
        seeds = SEEDS if seeds_by_pair is None else seeds_by_pair[rep, learner]
        if not seeds or seeds[0] != 42 or tuple(sorted(set(seeds))) != seeds or not set(seeds) <= set(SEEDS):
            raise ValueError("Invalid explicit seed roster")
        members = []
        for seed in seeds:
            path = root / cell_tag(rep, learner, seed)
            for filename in ("model.pt", "results.json", *[f"pb_step_scores_{s}.jsonl" for s in SUBSETS]):
                if not (path / filename).is_file():
                    raise ValueError(f"Incomplete cell: {path / filename}")
            result = json.loads((path / "results.json").read_text())
            result["_dir"] = str(path)
            validate_result(result, rep, learner, seed)
            if result["inputs"] != reference["inputs"]:
                raise ValueError(f"Input fingerprint mismatch: {path}")
            members.append(result)
        if any(m["hp"]["selected"] != members[0]["hp"]["selected"] for m in members[1:]):
            raise ValueError("Seeds used different selected hyperparameters")
        cells.extend(members)
    check_inputs(cells)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cells-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ranked-out", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    cells = validate_grid(args.root, roster(args.cells_file), reference)
    print(f"Validated {len(cells)} complete Instruct cells under one fixed protocol")
    if args.ranked_out:
        summary = summarise(cells)
        if any(row["f1_pb_calib20"] is None for row in summary.values()):
            raise ValueError("Missing calibration result")
        ranked = sorted(summary.items(), key=lambda item: -item[1]["f1_pb_calib20"][0])
        lines = ["**Qwen3-8B Instruct leaderboard**", "",
                 "ProcessBench first-error F1 at calib-20, four-subset mean; "
                 "mean and sample standard deviation over seeds 42, 43 and 44.", "",
                 "| Rank | Representation | Learner | F1_PB | Parameters | Roster |",
                 "|---:|---|---|---:|---:|---|"]
        for rank, ((rep, learner), row) in enumerate(ranked, 1):
            mean, sd = row["f1_pb_calib20"]
            cohort = "extension" if rep == "lengthfree_geom" else "core"
            lines.append(f"| {rank} | {rep} | {learner} | {mean:.4f} +/- {sd:.4f} "
                         f"| {row['n_params']:,} | {cohort} |")
        args.ranked_out.write_text("\n".join(lines) + "\n")
        args.ranked_out.with_suffix(".json").write_text(json.dumps([
            {"rep": key[0], "learner": key[1], **value} for key, value in ranked
        ], indent=2) + "\n")


if __name__ == "__main__":
    main()
