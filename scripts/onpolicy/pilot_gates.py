#!/usr/bin/env python3
"""Stage 1 gates: is this generation run worth encoding and judging?

Four questions, in the order in which a bad answer kills the arm.

1. **Are both classes present?** A run where the model is always right has no
   incorrect steps to detect, and one where it is always wrong has no correct
   ones. The 48-problem pilot sat at 0.435 at temperature 0.8.

2. **Do the solutions segment into steps?** The whole arm assumes a base model
   keeps the blank-line convention without instruction tuning. The pilot said it
   does (4.2% single-step solutions).

3. **Are the steps comparable in length to the off-policy ones?** Measured in
   *tokens*, not words: the earlier pilot reported ~24 words against PRM800K's
   38.8 tokens, which are not the same unit and cannot be compared. Pass
   `--reference_items` to compute the same statistic on the off-policy split and
   `--tokenizer` to count with the backbone's own tokenizer. Step length was
   flagged as a minor confound in the probe-anatomy work, so this gate only asks
   that the distributions overlap; the rank comparison is reported inside
   length-matched strata regardless.

4. **Is there room for a reranker to show anything?** This is the gate that
   decides whether T2 is answerable at all. If any-of-N is right almost exactly
   as often as one sample is, then every verifier scores the same best-of-N
   accuracy no matter how good it is, and the correlation T2 wants to measure is
   noise. Two headrooms are reported: oracle@N over a single sample, and the
   sharper one, oracle@N over self-consistency, which is the baseline a reranker
   actually has to beat.

Reads the generator's *_trajectories.jsonl (all shards). No GPU unless a
tokenizer is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import read_jsonl  # noqa: E402
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from src.eval.math_grade import is_equiv, normalize_answer  # noqa: E402


def majority_answer(preds: list[str | None]) -> str | None:
    """Self-consistency: the most common normalised answer, ties broken by order."""
    counts: Counter = Counter()
    first: dict = {}
    for p in preds:
        n = normalize_answer(p)
        if n is None:
            continue
        counts[n] += 1
        first.setdefault(n, p)
    if not counts:
        return None
    top = max(counts.values())
    for n, c in counts.items():
        if c == top:
            return first[n]
    return None


def per_problem(trajs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for t in trajs:
        out[str(t["fork_id"])].append(t)
    return out


def bon_headroom(trajs: list[dict]) -> dict:
    """pass@1, oracle@N, self-consistency, and the room between them."""
    groups = per_problem([t for t in trajs if t.get("gradeable")])
    pass1, oracle, sc, sizes = [], [], [], []
    for _pid, ts in groups.items():
        correct = [bool(t["correct"]) for t in ts]
        sizes.append(len(ts))
        pass1.append(float(np.mean(correct)))
        oracle.append(float(any(correct)))
        maj = majority_answer([t.get("pred") for t in ts])
        gold = ts[0].get("gold")
        sc.append(float(is_equiv(maj, gold)) if maj is not None else 0.0)
    p1, orc, scv = float(np.mean(pass1)), float(np.mean(oracle)), float(np.mean(sc))
    return {"n_problems": len(groups), "samples_per_problem": float(np.mean(sizes)),
            "pass_at_1": p1, "oracle_at_n": orc, "self_consistency": scv,
            "headroom_over_sample": orc - p1, "headroom_over_sc": orc - scv}


def step_lengths(texts: list[str], tokenizer) -> np.ndarray:
    if tokenizer is None:
        return np.array([len(t.split()) for t in texts], dtype=np.float64)
    return np.array([len(tokenizer(t, add_special_tokens=False)["input_ids"])
                     for t in texts], dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", nargs="+", required=True, type=Path)
    p.add_argument("--reference_items", type=Path, default=None,
                   help="Off-policy per-step jsonl (candidate_step field) to "
                        "measure step length against, in the same unit.")
    p.add_argument("--tokenizer", default=None,
                   help="Model path for token counts. Without it lengths are "
                        "reported in words and the gate says so.")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--min_accuracy", type=float, default=0.20)
    p.add_argument("--max_accuracy", type=float, default=0.80)
    p.add_argument("--min_gradeable", type=float, default=0.95)
    p.add_argument("--max_single_step", type=float, default=0.10)
    p.add_argument("--min_headroom", type=float, default=0.10)
    p.add_argument("--min_headroom_over_sc", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    trajs: list[dict] = []
    for f in args.trajectories:
        trajs.extend(read_jsonl(f))
    if not trajs:
        sys.exit("[gates] no trajectories")

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                            local_files_only=args.local_files_only)
    unit = "tokens" if tok is not None else "words"

    graded = [t for t in trajs if t.get("gradeable")]
    gradeable_rate = len(graded) / len(trajs)
    accuracy = float(np.mean([bool(t["correct"]) for t in graded])) if graded else 0.0

    steps_per = np.array([len(split_into_steps(t["solution"])) for t in graded],
                         dtype=np.float64)
    all_steps = [s for t in graded for s in split_into_steps(t["solution"])]
    lens = step_lengths(all_steps, tok)
    ref = None
    if args.reference_items:
        ref_steps = [r["candidate_step"] for r in read_jsonl(args.reference_items)
                     if r.get("candidate_step")]
        ref = step_lengths(ref_steps, tok)

    bon = bon_headroom(trajs)
    single = float((steps_per <= 1).mean()) if len(steps_per) else 1.0

    report = {
        "n_trajectories": len(trajs), "n_gradeable": len(graded),
        "gradeable_rate": gradeable_rate, "trajectory_accuracy": accuracy,
        "steps_per_solution_median": float(np.median(steps_per)) if len(steps_per) else 0.0,
        "steps_per_solution_mean": float(steps_per.mean()) if len(steps_per) else 0.0,
        "single_step_fraction": single,
        "length_unit": unit,
        "step_length_median": float(np.median(lens)) if len(lens) else 0.0,
        "step_length_mean": float(lens.mean()) if len(lens) else 0.0,
        "reference_step_length_median": float(np.median(ref)) if ref is not None else None,
        "reference_step_length_mean": float(ref.mean()) if ref is not None else None,
        **bon,
    }

    checks = [
        ("both classes present",
         args.min_accuracy <= accuracy <= args.max_accuracy,
         f"accuracy {accuracy:.3f} in [{args.min_accuracy}, {args.max_accuracy}]"),
        ("solutions are gradeable", gradeable_rate >= args.min_gradeable,
         f"{gradeable_rate:.3f} >= {args.min_gradeable} (raise max_new_tokens if not)"),
        ("segmentation holds", single <= args.max_single_step,
         f"single-step fraction {single:.3f} <= {args.max_single_step}"),
        ("reranking has room", bon["headroom_over_sample"] >= args.min_headroom,
         f"oracle@N - pass@1 = {bon['headroom_over_sample']:.3f} >= {args.min_headroom}"),
        ("room over self-consistency", bon["headroom_over_sc"] >= args.min_headroom_over_sc,
         f"oracle@N - SC = {bon['headroom_over_sc']:.3f} >= {args.min_headroom_over_sc}"),
    ]

    print(f"trajectories {len(trajs)}  gradeable {len(graded)}  "
          f"problems {bon['n_problems']} x {bon['samples_per_problem']:.1f} samples")
    print(f"trajectory accuracy   {accuracy:.3f}")
    print(f"steps per solution    median {report['steps_per_solution_median']:.0f}  "
          f"mean {report['steps_per_solution_mean']:.1f}  "
          f"single-step {single:.3f}")
    print(f"step length ({unit})  median {report['step_length_median']:.0f}  "
          f"mean {report['step_length_mean']:.1f}" +
          (f"   off-policy reference median {report['reference_step_length_median']:.0f}"
           f" mean {report['reference_step_length_mean']:.1f}" if ref is not None else
           "   (no --reference_items: nothing to compare against)"))
    if tok is None:
        print("  NOTE lengths are in words. PRM800K's 38.8 is tokens; pass "
              "--tokenizer before drawing any conclusion from this line.")
    print(f"pass@1 {bon['pass_at_1']:.3f}   self-consistency {bon['self_consistency']:.3f}"
          f"   oracle@N {bon['oracle_at_n']:.3f}")
    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<28} {detail}")
    verdict = all(ok for _, ok, _ in checks)
    report["checks"] = {name: bool(ok) for name, ok, _ in checks}
    report["verdict"] = "GO" if verdict else "NO-GO"
    print(f"\nVERDICT: {report['verdict']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[gates] wrote {args.out}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
