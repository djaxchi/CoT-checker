#!/usr/bin/env python3
"""Pre-encode token-length audit for the model-size DenseLinear ablation.

Hard invariant: NO context truncation. Every step is encoded conditioned on
``question + all previous reasoning steps + current step``. Before spending any
GPU time, this script tokenizes the exact sequences the encoders will build
(reusing build_prompt_prefix from the encoders) and verifies that every example
fits inside the model context window.

For each dataset (PRM800K split files + each ProcessBench subset) it reports
max / mean / p95 / p99 observed token lengths under the model tokenizer, and
asserts:

    max_observed_tokens <= model.config.max_position_embeddings
    num_truncated_examples == 0

If any example exceeds the model context window it FAILS LOUDLY, printing the
offending dataset/subset, example id, step index, and token length, and exits
non-zero so the launcher stops THIS model size (and only this one).

Usage:
    python scripts/s1ms_audit_token_lengths.py \\
        --model_name_or_path Qwen/Qwen2.5-7B \\
        --prm_split_dir   $PRM_SPLIT_DIR \\
        --prm_splits prm800k_probe_train_40k.jsonl prm800k_val_1k.jsonl \\
        --pb_subset gsm8k:$PB_DIR/gsm8k.json math:$PB_DIR/math.json ... \\
        --out_json <model_dir>/length_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from encode_prm800k_hidden_states import build_prompt_prefix, read_jsonl  # type: ignore  # noqa: E402
from encode_processbench_hidden_states import load_trace_file  # type: ignore  # noqa: E402


def _stats(lengths: list[int]) -> dict:
    lt = np.asarray(lengths, dtype=np.int64) if lengths else np.zeros(0, dtype=np.int64)
    return {
        "num_examples": int(lt.size),
        "max_observed_tokens": int(lt.max()) if lt.size else 0,
        "mean_observed_tokens": float(lt.mean()) if lt.size else 0.0,
        "p95_observed_tokens": int(np.percentile(lt, 95)) if lt.size else 0,
        "p99_observed_tokens": int(np.percentile(lt, 99)) if lt.size else 0,
    }


def _seq_len(tokenizer, problem: str, prefix: str, step: str) -> int:
    prefix_ids = tokenizer(build_prompt_prefix(problem, prefix),
                           add_special_tokens=True, truncation=False)["input_ids"]
    step_ids = tokenizer(step, add_special_tokens=False, truncation=False)["input_ids"]
    return len(prefix_ids) + len(step_ids)


def audit_prm(tokenizer, jsonl_path: Path, model_max: int, tag: str) -> tuple[dict, int]:
    lengths: list[int] = []
    n_over = 0
    for ex in read_jsonl(jsonl_path):
        n = _seq_len(tokenizer, ex["problem"], ex["prefix"], ex["candidate_step"])
        lengths.append(n)
        if n > model_max:
            n_over += 1
            print(
                f"[audit] OVERLENGTH prm:{tag} uid={ex.get('uid')} "
                f"tokens={n} > model_max={model_max}",
                file=sys.stderr, flush=True,
            )
    return _stats(lengths), n_over


def audit_pb(tokenizer, raw_path: Path, model_max: int, subset: str) -> tuple[dict, int]:
    lengths: list[int] = []
    n_over = 0
    for trace in load_trace_file(raw_path):
        problem = trace["problem"]
        steps = trace["steps"]
        for k, step_text in enumerate(steps):
            prefix = "\n\n".join(steps[:k])
            n = _seq_len(tokenizer, problem, prefix, step_text)
            lengths.append(n)
            if n > model_max:
                n_over += 1
                print(
                    f"[audit] OVERLENGTH pb:{subset} id={trace.get('id')} "
                    f"step_idx={k} tokens={n} > model_max={model_max}",
                    file=sys.stderr, flush=True,
                )
    return _stats(lengths), n_over


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--prm_split_dir", type=Path, required=True)
    p.add_argument("--prm_splits", nargs="+", required=True,
                   help="PRM800K split JSONL basenames under --prm_split_dir.")
    p.add_argument("--pb_subset", nargs="+", required=True,
                   help="<subset>:<raw_path> pairs, one per ProcessBench subset.")
    p.add_argument("--out_json", type=Path, required=True)
    args = p.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    model_max = int(getattr(cfg, "max_position_embeddings", 0) or 0)
    if model_max <= 0:
        sys.exit(f"[audit] {args.model_name_or_path} has no max_position_embeddings; cannot audit.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)

    print(f"[audit] model={args.model_name_or_path} max_position_embeddings={model_max}", flush=True)

    report: dict = {
        "model_name": args.model_name_or_path,
        "tokenizer_name": args.model_name_or_path,
        "model_max_position_embeddings": model_max,
        "datasets": {},
    }
    total_over = 0

    for basename in args.prm_splits:
        path = args.prm_split_dir / basename
        if not path.exists():
            sys.exit(f"[audit] PRM800K split not found: {path}")
        stats, n_over = audit_prm(tokenizer, path, model_max, basename)
        stats["num_truncated_examples"] = n_over
        report["datasets"][f"prm800k:{basename}"] = stats
        total_over += n_over
        print(f"[audit] prm800k:{basename} {stats}", flush=True)

    for spec in args.pb_subset:
        if ":" not in spec:
            sys.exit(f"[audit] --pb_subset entry must be <subset>:<path>, got {spec!r}")
        subset, raw = spec.split(":", 1)
        raw_path = Path(raw)
        if not raw_path.exists():
            sys.exit(f"[audit] ProcessBench subset raw file not found: {raw_path}")
        stats, n_over = audit_pb(tokenizer, raw_path, model_max, subset)
        stats["num_truncated_examples"] = n_over
        report["datasets"][f"processbench:{subset}"] = stats
        total_over += n_over
        print(f"[audit] processbench:{subset} {stats}", flush=True)

    report["num_truncated_examples"] = total_over
    report["max_observed_tokens"] = max(
        (d["max_observed_tokens"] for d in report["datasets"].values()), default=0
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))

    if total_over > 0:
        sys.exit(
            f"[audit] FATAL: {total_over} example(s) exceed the model context "
            f"window ({model_max}) for {args.model_name_or_path}. No truncation "
            "is permitted; stopping THIS model size. See messages above."
        )
    assert report["num_truncated_examples"] == 0
    print(
        f"[audit] OK: 0 truncated; global max_observed_tokens="
        f"{report['max_observed_tokens']} <= {model_max}. Wrote {args.out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
