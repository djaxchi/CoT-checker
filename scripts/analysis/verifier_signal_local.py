#!/usr/bin/env python3
"""Replay verified cluster activations through small frozen probes on a laptop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.verifier_signal_experiment import (  # noqa: E402
    analyze,
    git_revision,
    load_bundle,
    seed_roster,
    write_json,
)
from scripts.validate_instruct_leaderboard import cell_tag, validate_result  # noqa: E402
from src.analysis.verifier_signal import family_metrics, sha256  # noqa: E402


def validate_checkpoint(cell: Path, reference: dict, identity: tuple[str, str, int],
                        dataset_hash: str, store_hash: str) -> dict:
    """Require the exact checkpoint and inputs used by cluster reference scoring."""
    import math

    result = json.loads((cell / "results.json").read_text())
    validate_result(result, *identity)
    if (reference["rep"], reference["learner"], reference["seed"]) != identity:
        raise ValueError("reference checkpoint identity mismatch")
    expected = {"model_sha256": sha256(cell / "model.pt"),
                "results_sha256": sha256(cell / "results.json"),
                "dataset_sha256": dataset_hash, "store_fingerprint": store_hash,
                "score_orientation": "P(incorrect)", "rescale": "none"}
    if any(reference.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint or reference provenance mismatch")
    recorded = reference["source_test_auroc_recorded"]
    reproduced = reference["source_test_auroc_reproduced"]
    if (not math.isfinite(recorded) or not math.isfinite(reproduced)
            or recorded != result["in_domain"]["auroc"] or abs(recorded - reproduced) > 0.001):
        raise ValueError("cluster source-test AUROC validation failed")
    return result


def compare_scores(actual: dict[str, float], reference: dict[str, float]) -> dict:
    """Check platform agreement at a fixed, prespecified absolute tolerance."""
    import numpy as np

    if not actual or set(actual) != set(reference):
        raise ValueError("local/reference score coverage mismatch")
    values = np.array([[actual[uid], reference[uid]] for uid in sorted(actual)])
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("invalid probability scores")
    differences = np.abs(values[:, 0] - values[:, 1])
    maximum = float(differences.max())
    if maximum > 1e-4:
        raise ValueError(f"local/cluster score disagreement: max absolute error {maximum}")
    return {"n_examples": len(actual), "max_absolute_error": maximum,
            "mean_absolute_error": float(differences.mean()), "absolute_tolerance": 1e-4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "experiments/verifier_signal_v1_available64")
    parser.add_argument("--reference-run", type=Path, default=ROOT / "results/verifier_signal_v1/cluster_reference")
    parser.add_argument("--cells-root", type=Path, default=ROOT / "results/verifier_signal_v1/local_assets/cells")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch

    from scripts.onpolicy.score_cells_on_split import score_cell
    from src.repstore import split_fingerprint
    from src.repstore.store import ShardedRepSplit

    torch.set_num_threads(4)
    config, rows = load_bundle(args.bundle)
    data_hash = sha256(args.bundle / "examples.jsonl")
    if sha256(args.reference_run / "config.json") != sha256(args.bundle / "config.json"):
        raise ValueError("reference configuration mismatch")
    split = args.reference_run / "store/diagnostic"
    store_hash = split_fingerprint(split)
    view = ShardedRepSplit(split)
    expected = {r["uid"]: r["label"] for r in rows}
    meta = view.meta()
    if len(meta) != len(rows) or {m["uid"]: m["label"] for m in meta} != expected:
        raise ValueError("diagnostic store row/label mismatch")
    manifests = [json.loads(p.read_text()) for p in sorted(split.glob("shard_*/extraction.json"))]
    if len(manifests) != 4 or {m["shard_idx"] for m in manifests} != set(range(4)):
        raise ValueError("incomplete extraction manifests")
    revision = (args.reference_run / "backbone_revision.txt").read_text().strip()
    for manifest in manifests:
        if (manifest["dataset_sha256"] != data_hash
                or manifest["config_sha256"] != sha256(args.bundle / "config.json")
                or manifest["model_revision"] != revision or manifest["num_shards"] != 4):
            raise ValueError("extraction provenance mismatch")
    roster = [(rep, learner, seed) for (rep, learner), seeds in seed_roster(args.bundle, config).items()
              for seed in seeds]
    validated = []
    for identity in roster:
        tag = cell_tag(*identity)
        reference_path = args.reference_run / "scores" / f"{tag}.json"
        reference = json.loads(reference_path.read_text())
        result = validate_checkpoint(args.cells_root / tag, reference, identity, data_hash, store_hash)
        family_metrics(rows, reference["scores"])
        validated.append((tag, result, reference, reference_path))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "scores").mkdir()
    comparisons = []
    for tag, result, reference, reference_path in validated:
        scores, _, meta = score_cell(args.cells_root / tag, result, split, None,
                                    torch.device(args.device), args.batch_size, config["t_max"])
        mapped = {m["uid"]: float(s) for m, s in zip(meta, scores)}
        family_metrics(rows, mapped)
        comparison = {"cell": tag, **compare_scores(mapped, reference["scores"])}
        comparisons.append(comparison)
        payload = dict(reference, scores=mapped, execution_device=args.device,
                       torch_version=torch.__version__, git_revision=git_revision(),
                       source_test_validation="inherited from hash-matched cluster reference; not rerun locally",
                       reference_scores_sha256=sha256(reference_path), local_comparison=comparison)
        write_json(args.out / "scores" / f"{tag}.json", payload)
        print(f"{tag}: max score difference {comparison['max_absolute_error']:.3g}", flush=True)
    write_json(args.out / "local_validation.json", {
        "device": args.device, "torch_version": torch.__version__, "cells": comparisons,
        "dataset_sha256": data_hash, "store_fingerprint": store_hash,
        "backbone_revision": revision,
        "source_test_validation": "cluster reference; training store not downloaded"})
    analyze(args, config, rows)


if __name__ == "__main__":
    main()
