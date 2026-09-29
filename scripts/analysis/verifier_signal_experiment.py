#!/usr/bin/env python3
"""Build, validate, extract, score, and analyze verifier_signal_v1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.verifier_signal import (  # noqa: E402
    build_dataset,
    family_metrics,
    read_rows,
    sha256,
    summarize,
    validate_dataset,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else os.environ.get("SOURCE_REVISION", "unavailable")


def load_bundle(bundle: Path) -> tuple[dict, list[dict]]:
    config = json.loads((bundle / "config.json").read_text())
    rows = read_rows(bundle / "examples.jsonl")
    manifest = json.loads((bundle / "manifest.json").read_text())
    for filename, digest in manifest["sha256"].items():
        if sha256(bundle / filename) != digest:
            raise ValueError(f"frozen bundle hash mismatch: {filename}")
    validate_dataset(rows, config)
    if rows != build_dataset(config):
        raise ValueError("data does not reproduce from frozen configuration")
    return config, rows


def build(bundle: Path) -> None:
    config = json.loads((bundle / "config.json").read_text())
    rows = build_dataset(config)
    for name in ("examples.jsonl", "manifest.json", "review_samples.md"):
        if (bundle / name).exists():
            raise ValueError(f"refusing to replace frozen {name}; use a new bundle directory")
    (bundle / "examples.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    families = {r["family_id"] for r in rows}
    preview = ["# Verifier signal diagnostic samples", "",
               "Generated arithmetic witnesses pass programmatic checks. Human review is pending.",
               "Read the experiment plan before interpreting local and inherited labels.", ""]
    selected = {next(r["family_id"] for r in rows if r["arm"] == arm and r["domain"] == domain)
                for arm, domain in {(r["arm"], r["domain"]) for r in rows}}
    for r in rows:
        if r["family_id"] in selected and r["style"] == "plain":
            preview += [f"**{r['uid']}**", "", f"Problem: {r['problem']}", "",
                        f"Prefix: {r['prefix'] or '(empty)'}", "",
                        f"Candidate: {r['candidate_step']}", "",
                        f"Local invalid={r['local_invalid']}; prefix invalid={r['prefix_invalid']}; "
                        f"conclusion invalid={r['conclusion_invalid']}; trace invalid={r['trace_invalid']}.", ""]
    (bundle / "review_samples.md").write_text("\n".join(preview))
    write_json(bundle / "manifest.json", {
        "version": config["version"], "n_rows": len(rows), "n_families": len(families),
        "n_target_rows": sum(r["role"] == "target" for r in rows),
        "n_before_rows": sum(r["role"] == "before" for r in rows),
        "human_review": "pending; generated and arithmetic-validated only",
        "sha256": {name: sha256(bundle / name) for name in
                   ("config.json", "probes.cells", "examples.jsonl")},
        "generator_sha256": sha256(ROOT / "src/analysis/verifier_signal.py"),
    })
    print(f"Prepared {len(rows)} rows in {len(families)} families")


def seed_roster(bundle: Path, config: dict) -> dict[tuple[str, str], tuple[int, ...]]:
    from scripts.validate_instruct_leaderboard import roster

    result = {pair: tuple(config["seeds"]) for pair in roster(bundle / "probes.cells")}
    seen = set()
    for item in config.get("seed_overrides", []):
        pair = (item["rep"], item["learner"])
        if pair not in result or pair in seen:
            raise ValueError("Unknown or duplicate seed override")
        seen.add(pair)
        result[pair] = tuple(item["seeds"])
    for seeds in result.values():
        if not seeds or seeds[0] != 42 or tuple(sorted(set(seeds))) != seeds or not set(seeds) <= {42, 43, 44}:
            raise ValueError("Invalid explicit seed roster")
    return result


def preflight(args, config: dict) -> list[dict]:
    from scripts.validate_instruct_leaderboard import roster, validate_grid
    from src.repstore import split_fingerprint
    from src.repstore.store import ShardedRepSplit

    if not args.cells_root or not args.reference or not args.prm_store:
        raise ValueError("need --cells-root, --reference, and --prm-store")
    reference = json.loads(args.reference.read_text())
    results = validate_grid(args.cells_root, roster(args.bundle / "probes.cells"), reference,
                            seeds_by_pair=seed_roster(args.bundle, config))
    view = ShardedRepSplit(args.prm_store / "probe_train_full")
    for shard in view.shards:
        spec = shard.spec
        if (spec.layer != config["hidden_state_index"] or spec.dim != config["hidden_dim"]
                or spec.prompt_style != "verifier" or "Qwen3-8B" not in spec.backbone
                or "Base" in spec.backbone):
            raise ValueError("training store is not the specified Instruct verifier representation")
    if split_fingerprint(args.prm_store / "probe_train_full") != reference["inputs"]["prm/probe_train_full"]:
        raise ValueError("training store differs from frozen reference")
    return results


def extract(args, config: dict, rows: list[dict]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.encode_prm800k_token_store import encode_split, tokenize_with_offsets

    if args.out is None or not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("need --out and valid shard index")
    destination = args.out / "store" / "diagnostic" / f"shard_{args.shard_idx:02d}"
    if destination.exists():
        raise ValueError(f"refusing to overwrite {destination}")
    revision = os.environ.get("SIGNAL_MODEL_REVISION")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=revision, local_files_only=True,
        torch_dtype=torch.bfloat16).to("cuda").eval()
    if revision and getattr(model.config, "_commit_hash", None) != revision:
        raise ValueError("loaded backbone differs from requested revision")
    if model.config.hidden_size != config["hidden_dim"] or model.config.num_hidden_layers < config["hidden_state_index"]:
        raise ValueError("backbone architecture mismatch")
    lengths = []
    for row in rows:
        ids, start = tokenize_with_offsets(tokenizer, row, config["max_seq_len"])
        if len(ids) - start > config["t_max"]:
            raise ValueError(f"candidate would be truncated: {row['uid']}")
        lengths.append({"uid": row["uid"], "input_tokens": len(ids), "step_tokens": len(ids)-start})
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encode_split(args.bundle / "examples.jsonl", args.out / "store", "diagnostic",
                 tokenizer, model, torch.device("cuda"), config["hidden_state_index"],
                 config["max_seq_len"], args.batch_size, pad, args.shard_idx,
                 args.num_shards, config["model"], None, span_only=True)
    write_json(destination / "extraction.json", {
        "dataset_sha256": sha256(args.bundle / "examples.jsonl"),
        "config_sha256": sha256(args.bundle / "config.json"),
        "model": config["model"], "model_revision": getattr(model.config, "_commit_hash", None),
        "layer": config["hidden_state_index"], "dtype": "bfloat16", "storage_dtype": "float16",
        "prompt_style": "verifier", "chat_template": False, "git_revision": git_revision(),
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
        "torch_version": torch.__version__,
        "lengths": lengths[args.shard_idx::args.num_shards],
    })


def score(args, config: dict, rows: list[dict]) -> None:
    import numpy as np
    import torch

    from scripts.onpolicy.score_cells_on_split import score_cell
    from scripts.train_easy_probe_method import auroc_numpy
    from src.repstore import split_fingerprint
    from src.repstore.store import ShardedRepSplit

    results = preflight(args, config)
    split = args.out / "store" / "diagnostic"
    view = ShardedRepSplit(split)
    actual = view.meta()
    expected = {r["uid"]: r for r in rows}
    if len(actual) != len(rows) or {m["uid"] for m in actual} != set(expected):
        raise ValueError("store is missing rows or contains duplicates")
    extraction = [json.loads(p.read_text()) for p in sorted(split.glob("shard_*/extraction.json"))]
    if len(extraction) != args.num_shards or {e["shard_idx"] for e in extraction} != set(range(args.num_shards)):
        raise ValueError("incomplete extraction manifests")
    for e in extraction:
        if (e["dataset_sha256"] != sha256(args.bundle / "examples.jsonl")
                or e["config_sha256"] != sha256(args.bundle / "config.json")
                or e["num_shards"] != args.num_shards):
            raise ValueError("extraction provenance mismatch")
    if len({e["model_revision"] for e in extraction}) != 1:
        raise ValueError("mixed backbone revisions")
    for m in actual:
        if m["label"] != expected[m["uid"]]["label"]:
            raise ValueError("store labels do not match dataset")
    out = args.out / "scores"
    out.mkdir(exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for result in results:
        cell = Path(result["_dir"])
        test_split = args.prm_store / "test_2k"
        if split_fingerprint(test_split) != result["inputs"]["prm/test_2k"]:
            raise ValueError("source test split fingerprint mismatch")
        source_scores, source_y, _ = score_cell(cell, result, test_split, None, device,
                                               args.batch_size, config["t_max"])
        source_auc = float(auroc_numpy(source_y, source_scores))
        if abs(source_auc - result["in_domain"]["auroc"]) > 0.001:
            raise ValueError(f"source AUROC reproduction failed for {cell.name}: {source_auc}")
        scores, _, meta = score_cell(cell, result, split, None, device,
                                     args.batch_size, config["t_max"])
        mapped = {m["uid"]: float(s) for m, s in zip(meta, scores)}
        family_metrics(rows, mapped)  # Validate full coverage and score orientation/range contract.
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite model output")
        payload = {"cell": cell.name, "rep": result["rep"], "learner": result["learner"],
                   "seed": result["seed"], "scores": mapped,
                   "model_sha256": sha256(cell / "model.pt"),
                   "results_sha256": sha256(cell / "results.json"),
                   "dataset_sha256": sha256(args.bundle / "examples.jsonl"),
                   "store_fingerprint": split_fingerprint(split),
                   "source_test_auroc_reproduced": source_auc,
                   "source_test_auroc_recorded": result["in_domain"]["auroc"],
                   "score_orientation": "P(incorrect)", "rescale": "none",
                   "git_revision": git_revision()}
        write_json(out / f"{cell.name}.json", payload)
        print(f"Scored {cell.name}: {len(mapped)} examples", flush=True)


def analyze(args, config: dict, rows: list[dict]) -> None:
    from collections import defaultdict

    by_rep = defaultdict(list)
    per_cell = []
    seen = set()
    for path in sorted((args.out / "scores").glob("*.json")):
        result = json.loads(path.read_text())
        identity = (result["rep"], result["learner"], result["seed"])
        if identity in seen:
            raise ValueError("duplicate checkpoint scores")
        seen.add(identity)
        if result["dataset_sha256"] != sha256(args.bundle / "examples.jsonl"):
            raise ValueError("score file has different data")
        metrics = family_metrics(rows, result["scores"])
        by_rep[identity[:2]].append(metrics)
        per_cell.append({"cell": result["cell"], "family_metrics": metrics,
                         "summary": summarize(metrics, config["bootstrap_replicates"], config["bootstrap_seed"])})
    frozen_roster = seed_roster(args.bundle, config)
    wanted = {(rep, learner, seed) for (rep, learner), seeds in frozen_roster.items()
              for seed in seeds}
    if seen != wanted:
        raise ValueError(f"score roster incomplete or unexpected: {seen ^ wanted}")
    pooled = []
    for (rep, learner), seed_metrics in sorted(by_rep.items()):
        averaged = []
        for index, item in enumerate(seed_metrics[0]):
            row = dict(item)
            row["metrics"] = {key: sum(m[index]["metrics"][key] for m in seed_metrics)/len(seed_metrics)
                              for key in item["metrics"]}
            averaged.append(row)
        pooled.append({"rep": rep, "learner": learner,
                       "seeds": list(frozen_roster[rep, learner]),
                       "family_metrics": averaged,
                       "summary": summarize(averaged, config["bootstrap_replicates"], config["bootstrap_seed"])})
    write_json(args.out / "analysis.json", {
        "dataset_sha256": sha256(args.bundle / "examples.jsonl"),
        "bootstrap_unit": "family within domain; average per-seed family metrics before resampling",
        "note": "Descriptive intervals, no multiple-testing correction or natural-data generalization claim.",
        "per_cell": per_cell, "seed_averaged": pooled})
    lines = ["# Verifier signal v1: descriptive results", "",
             "Test families, plain wording, metrics averaged across training seeds before family bootstrap.",
             "Intervals describe these numeric families within templates, not natural-data generalization.", "",
             "| Representation | Learner | Seeds | Arm | Domain | Metric | Families | Mean | 95% interval |",
             "|---|---|---|---|---|---|---:|---:|---|" ]
    headline = {"local_contrast_mean", "both_preferences_correct", "prefix_error_contrast",
                "global_conclusion_contrast"}
    for cell in pooled:
        for row in cell["summary"]:
            if row["partition"] == "test" and row["style"] == "plain" and row["metric"] in headline:
                lo, hi = row["ci95"]
                lines.append(f"| {cell['rep']} | {cell['learner']} | {cell['seeds']} | {row['arm']} | {row['domain']} | "
                             f"{row['metric']} | {row['n_families']} | {row['mean']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ["", "Consult analysis.json for per-seed, dev, wording-control, and prefix-before results.",
              "Probability-scale contrasts are within-checkpoint effects; their size is calibration-dependent.", ""]
    (args.out / "summary.md").write_text("\n".join(lines))
    print(f"Analyzed {len(per_cell)} checkpoints; wrote {args.out / 'analysis.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "validate", "preflight", "extract", "score", "analyze"))
    parser.add_argument("--bundle", type=Path, default=ROOT / "experiments/verifier_signal_v1")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cells-root", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prm-store", type=Path)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "build":
        build(args.bundle)
        return
    config, rows = load_bundle(args.bundle)
    if args.phase == "validate":
        print(f"Validated frozen bundle: {len(rows)} rows")
    elif args.phase == "preflight":
        print(f"Validated {len(preflight(args, config))} trained checkpoints")
    else:
        if args.out is None:
            parser.error("--out is required")
        {"extract": extract, "score": score, "analyze": analyze}[args.phase](args, config, rows)


if __name__ == "__main__":
    main()
