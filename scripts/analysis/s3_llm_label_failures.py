"""LLM judge: label sampled first-error steps by failure mode (Claude Opus 4.8).

Reads the stratified sample, asks Claude to classify the marked first-error step
into the S3 taxonomy (enum-constrained structured output), and writes per-step
labels plus a human-review CSV and a detected-vs-missed contingency table.

Detection and probe score are deliberately NOT shown to the judge, so the
failure-mode label is unbiased. They are re-joined afterwards for the analysis.

Requires ANTHROPIC_API_KEY in the environment (or an `ant auth login` profile).

Outputs (results/s3_first_error/):
  - failure_labels.jsonl        one label per sampled step (resumable)
  - failure_labels_review.csv   human-verification table
  - failure_mode_by_detection.csv  failure-mode x detected/missed counts

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/analysis/s3_llm_label_failures.py
    python scripts/analysis/s3_llm_label_failures.py --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.eval.failure_taxonomy import FAILURE_MODES, taxonomy_prompt_block

MODEL = "claude-opus-4-8"

SYSTEM = (
    "You are an expert annotator of mathematical chain-of-thought reasoning. You "
    "are given a problem, the reasoning steps so far, and one step that has been "
    "independently identified as the FIRST incorrect step. Classify ONLY that step "
    "into exactly one failure mode from this taxonomy:\n\n"
    f"{taxonomy_prompt_block()}\n\n"
    "Pick the single best-fitting mode for the primary error in that step. Judge "
    "the step on its own merits; do not assume it is wrong in a particular way. "
    "Give a one-sentence rationale naming the specific error, and a confidence."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "failure_mode": {"type": "string", "enum": FAILURE_MODES},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "rationale": {"type": "string"},
    },
    "required": ["failure_mode", "confidence", "rationale"],
    "additionalProperties": False,
}


def build_user_message(rec: dict) -> str:
    prior = rec.get("prior_steps", [])
    prior_block = (
        "\n".join(f"  Step {i}: {s}" for i, s in enumerate(prior))
        if prior else "  (none — this is the first step)"
    )
    return (
        f"Problem:\n{rec['problem']}\n\n"
        f"Reasoning steps so far:\n{prior_block}\n\n"
        f"First incorrect step (Step {rec['step_idx']}):\n{rec['error_step']}\n\n"
        "Classify this step's primary failure mode."
    )


def label_one(client, rec: dict, effort: str) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": SCHEMA},
                       "effort": effort},
        system=SYSTEM,
        messages=[{"role": "user", "content": build_user_message(rec)}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    out = json.loads(text)
    return {
        "sample_id": rec["sample_id"],
        "subset": rec["subset"],
        "trace_id": rec["trace_id"],
        "step_idx": rec["step_idx"],
        "detected": rec["detected"],
        "probe_score": rec["probe_score"],
        "gold_first_error": rec["gold_first_error"],
        "failure_mode": out["failure_mode"],
        "confidence": out["confidence"],
        "rationale": out["rationale"],
        "model": MODEL,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path,
                    default=Path("results/s3_first_error/first_error_sample.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/s3_first_error/failure_labels.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all (smoke-test with e.g. 20)")
    args = ap.parse_args()

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("[label] set ANTHROPIC_API_KEY (or run `ant auth login`) first.")
    import anthropic

    recs = [json.loads(l) for l in args.sample.read_text().splitlines() if l]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["sample_id"] for l in args.out.read_text().splitlines() if l}
    todo = [r for r in recs if r["sample_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[label] {len(recs)} sampled | {len(done)} already labelled | {len(todo)} to do")

    client = anthropic.Anthropic()
    labels: list[dict] = []
    if done:
        labels = [json.loads(l) for l in args.out.read_text().splitlines() if l]

    with args.out.open("a") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(label_one, client, r, args.effort): r for r in todo}
        n_ok = n_err = 0
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001 - log and continue
                n_err += 1
                print(f"  [err] {r['sample_id']}: {type(e).__name__}: {e}")
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            labels.append(row)
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"  labelled {n_ok}/{len(todo)} ...")
    print(f"[label] done: {n_ok} new, {n_err} errors -> {args.out}")

    # ---- review CSV ----
    review = args.out.with_name("failure_labels_review.csv")
    with review.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "subset", "detected", "probe_score",
                    "failure_mode", "confidence", "rationale"])
        for r in sorted(labels, key=lambda x: (x["subset"], x["failure_mode"])):
            w.writerow([r["sample_id"], r["subset"], r["detected"], r["probe_score"],
                        r["failure_mode"], r["confidence"], r["rationale"]])

    # ---- failure-mode x detection contingency ----
    by_mode: dict[str, Counter] = defaultdict(Counter)
    for r in labels:
        by_mode[r["failure_mode"]]["detected" if r["detected"] else "missed"] += 1
    cont = args.out.with_name("failure_mode_by_detection.csv")
    with cont.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["failure_mode", "detected", "missed", "n", "detection_rate"])
        print("\n[contingency] failure mode x probe detection:")
        for mode in FAILURE_MODES:
            det = by_mode[mode]["detected"]
            mis = by_mode[mode]["missed"]
            n = det + mis
            if n == 0:
                continue
            rate = det / n
            w.writerow([mode, det, mis, n, round(rate, 3)])
            print(f"  {mode:34s} n={n:3d}  detected={det:3d}  missed={mis:3d}  rate={rate:.2f}")
    print(f"\n[label] review -> {review}\n[label] contingency -> {cont}")


if __name__ == "__main__":
    main()
