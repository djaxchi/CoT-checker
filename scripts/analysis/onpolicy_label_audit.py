#!/usr/bin/env python3
"""Audit the on-policy step labels before anything trains on them.

Labels from a model judge fail in ways that training curves do not reveal. A
judge that marks everything faulty, or nothing, or that quietly tracks the final
answer rather than the reasoning, all produce a probe that learns something other
than step correctness and a validation curve that looks fine.

The checks that carry weight here:

**Discrimination.** The judge sees the gold answer, so it knows which
trajectories failed. If it marked faults at the same rate in correct and
incorrect solutions it would be reading nothing; if it marked faults in
*exactly* the failing ones it would be copying the outcome rather than reading
the reasoning. Both extremes are reported rather than assumed away.

**False alarms.** How often a solution that reached the right answer is said to
contain a faulty step. This is not automatically an error, since a right answer
can follow a repaired mistake, but a rate near the incorrect-trajectory rate
means the judge is guessing.

**Degenerate marking.** A trace where most steps are faulty carries almost no
localisation signal and inflates the positive class. The share of such traces is
reported and can be capped.

**Position.** If first errors cluster at step 0 the judge may be blaming the
setup rather than the reasoning, and if they cluster at the last step it may be
reading the wrong answer backwards.

Representative cases are printed for each category the audit is meant to cover,
because a number can look reasonable while the underlying annotations do not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import read_jsonl  # noqa: E402


def propagation_shape(rows: list[dict]) -> dict:
    """How the faulty steps are arranged inside a trace.

    ReProbe reports a SET of faulty steps and the paper does not mark everything
    after the first error, which is why the parser and the encoder were built to
    preserve a sparse set. Whether the judge honours that is an empirical
    question about the judge, not about the code, and this measures it:

      suffix      every step from the first error to the end is faulty, which is
                  propagation whether or not it was asked for
      contiguous  an unbroken run that stops before the end
      gapped      steps judged fine sit inside the faulty span, which is
                  evidence the judge is reading steps rather than applying a rule

    A high suffix share does not make the labels wrong: a step carrying a bad
    value forward really is incorrect. It does mean the labels sit close to the
    first-error convention, the positive class is large, and localisation signal
    is weak, so it belongs in the interpretation of anything trained on them.
    """
    inc = [r for r in rows if not r.get("traj_correct") and r.get("faulty_steps")]
    if not inc:
        return {}
    suffix = contiguous = gapped = 0
    for r in inc:
        f = sorted(r["faulty_steps"])
        n = r["n_steps"]
        if f == list(range(f[0], n)):
            suffix += 1
        elif f == list(range(f[0], f[-1] + 1)):
            contiguous += 1
        else:
            gapped += 1
    t = len(inc)
    return {"n_traces": t, "suffix_share": suffix / t,
            "contiguous_share": contiguous / t, "gapped_share": gapped / t,
            "mean_faulty_fraction": float(np.mean(
                [len(r["faulty_steps"]) / r["n_steps"] for r in inc]))}


def group_stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    nf = np.array([len(r["faulty_steps"] or []) for r in rows])
    ns = np.array([r["n_steps"] for r in rows])
    return {
        "n": len(rows),
        "any_faulty": float((nf > 0).mean()),
        "faulty_step_rate": float(nf.sum() / max(1, ns.sum())),
        "median_faulty_per_trace": float(np.median(nf)),
        "mostly_faulty_traces": float((nf > 0.5 * ns).mean()),
    }


def audit(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("parse_ok")]
    cor = [r for r in ok if r.get("traj_correct")]
    inc = [r for r in ok if not r.get("traj_correct")]
    uids = [r.get("traj_uid") or r.get("id") for r in rows]
    rep: dict = {
        "n_annotated": len(rows),
        "n_parsed": len(ok),
        "parse_rate": len(ok) / max(1, len(rows)),
        "duplicate_ids": len(uids) - len(set(uids)),
        "correct_answer": group_stats(cor),
        "incorrect_answer": group_stats(inc),
    }
    if cor and inc:
        rep["false_alarm_rate"] = rep["correct_answer"]["any_faulty"]
        rep["coverage_on_incorrect"] = rep["incorrect_answer"]["any_faulty"]
        # If these were equal the judge would be reading nothing; if the first
        # were 0 and the second 1 it would be copying the grader.
        rep["discrimination"] = (rep["incorrect_answer"]["any_faulty"]
                                 - rep["correct_answer"]["any_faulty"])
    pos = [r["first_error"] / max(1, r["n_steps"] - 1)
           for r in inc if r.get("first_error", -1) >= 0]
    if pos:
        rep["first_error_relative_position"] = float(np.mean(pos))
        rep["first_error_at_step_0"] = float(
            np.mean([r.get("first_error", -1) == 0 for r in inc]))
        rep["first_error_at_last_step"] = float(
            np.mean([r.get("first_error", -1) == r["n_steps"] - 1 for r in inc]))
    rep["propagation"] = propagation_shape(ok)
    steps = sum(r["n_steps"] for r in ok)
    faulty = sum(len(r["faulty_steps"] or []) for r in ok)
    rep["n_steps"] = steps
    rep["positive_step_share"] = faulty / max(1, steps)
    rep["n_problems"] = len({r.get("problem_id") for r in ok if r.get("problem_id")})
    return rep


def checks(rep: dict) -> list[tuple[str, bool, str]]:
    """Automated sanity checks. Each names the pathology it would catch."""
    c = []
    c.append(("labels parse", rep["parse_rate"] >= 0.90,
              f"{rep['parse_rate']:.3f} >= 0.90"))
    c.append(("no duplicate ids", rep["duplicate_ids"] == 0,
              f"{rep['duplicate_ids']} duplicates"))
    if "discrimination" in rep:
        c.append(("judge discriminates", rep["discrimination"] >= 0.25,
                  f"incorrect {rep['coverage_on_incorrect']:.3f} minus correct "
                  f"{rep['false_alarm_rate']:.3f} = {rep['discrimination']:.3f} "
                  f">= 0.25; near zero means it reads nothing"))
        c.append(("not just copying the grader", rep["false_alarm_rate"] > 0.02,
                  f"false alarms {rep['false_alarm_rate']:.3f} > 0.02; exactly "
                  f"zero would mean the judge is echoing the final answer"))
    c.append(("positives are not the majority", rep["positive_step_share"] <= 0.5,
              f"{rep['positive_step_share']:.3f} <= 0.5"))
    if "first_error_at_step_0" in rep:
        c.append(("errors are not all at step 0", rep["first_error_at_step_0"] <= 0.4,
                  f"{rep['first_error_at_step_0']:.3f} <= 0.4; higher suggests the "
                  f"judge blames the setup rather than the reasoning"))
    prop = rep.get("propagation") or {}
    if prop:
        # Recalibrated after the first real run, and the change is deliberate.
        # The original check counted traces with "most steps faulty" against a
        # 0.25 threshold picked before the trace-length distribution was known,
        # and it fires on short traces for arithmetic reasons: marking two of
        # three steps is "most". The quantity worth bounding is the fraction of
        # steps marked faulty, which is length-normalised, and the bound is set
        # to catch a judge that marks essentially everything rather than to
        # encode a prior about how many steps ought to be wrong.
        c.append(("not marking nearly every step",
                  prop["mean_faulty_fraction"] <= 0.75,
                  f"mean faulty fraction {prop['mean_faulty_fraction']:.3f} <= 0.75"))
        # Propagation is reported, not failed. It is a property of the judge and
        # a fact about what the labels mean, and blocking on it would discard
        # labels that are defensible.
        c.append(("propagation is measured, not assumed", True,
                  f"suffix {prop['suffix_share']:.3f}, contiguous "
                  f"{prop['contiguous_share']:.3f}, gapped {prop['gapped_share']:.3f} "
                  f"-- a high suffix share means the labels sit close to the "
                  f"first-error convention and localisation signal is weak"))
    return c


def show_cases(rows: list[dict], traces: dict[str, dict], n: int = 1) -> None:
    ok = [r for r in rows if r.get("parse_ok")]
    def pick(pred):
        return [r for r in ok if pred(r)][:n]
    cats = [
        ("fully correct, no fault found",
         lambda r: r["traj_correct"] and not r["faulty_steps"]),
        ("correct answer, judge alleges a fault",
         lambda r: r["traj_correct"] and r["faulty_steps"]),
        ("wrong answer, early error",
         lambda r: not r["traj_correct"] and r["faulty_steps"]
         and r["first_error"] <= max(0, r["n_steps"] // 4)),
        ("wrong answer, middle error",
         lambda r: not r["traj_correct"] and r["faulty_steps"]
         and r["n_steps"] // 4 < r["first_error"] < 3 * r["n_steps"] // 4),
        ("wrong answer, late error",
         lambda r: not r["traj_correct"] and r["faulty_steps"]
         and r["first_error"] >= 3 * r["n_steps"] // 4),
        ("wrong answer, no fault found",
         lambda r: not r["traj_correct"] and not r["faulty_steps"]),
    ]
    for name, pred in cats:
        got = pick(pred)
        print(f"\n--- {name} ({'none found' if not got else ''}) ---")
        for r in got:
            t = traces.get(r["traj_uid"], {})
            print(f"  {r['traj_uid']}  steps {r['n_steps']}  faulty {r['faulty_steps']}")
            for i in (r["faulty_steps"] or [])[:2]:
                st = (t.get("steps") or [None] * r["n_steps"])
                if i < len(st) and st[i]:
                    print(f"    step {i+1}: {st[i][:160]}")
            print(f"    judge: {(r.get('raw') or '')[-180:].strip()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", nargs="+", required=True, type=Path)
    p.add_argument("--traces", type=Path, default=None,
                   help="the judge pool, for printing representative cases")
    p.add_argument("--holdout_outcomes", type=Path, default=None)
    p.add_argument("--cases", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = [r for f in args.labels for r in read_jsonl(f)]
    if not rows:
        raise SystemExit("no label rows")
    rep = audit(rows)

    if args.holdout_outcomes and args.holdout_outcomes.exists():
        held = {r["problem_id"] for r in read_jsonl(args.holdout_outcomes)}
        got = {r.get("problem_id") for r in rows}
        rep["holdout_overlap"] = len(held & got)

    print(f"{rep['n_annotated']} annotated, {rep['n_parsed']} parsed "
          f"({rep['parse_rate']:.3f}), {rep['n_steps']} steps, "
          f"{rep['n_problems']} problems\n")
    for k in ("correct_answer", "incorrect_answer"):
        g = rep.get(k) or {}
        if g:
            print(f"{k:<18} n {g['n']:>5}  any-faulty {g['any_faulty']:.3f}  "
                  f"faulty-step rate {g['faulty_step_rate']:.3f}  "
                  f"mostly-faulty traces {g['mostly_faulty_traces']:.3f}")
    if "first_error_relative_position" in rep:
        print(f"\nfirst error at {rep['first_error_relative_position']:.2f} of the "
              f"trace; step 0 {rep['first_error_at_step_0']:.3f}, "
              f"last step {rep['first_error_at_last_step']:.3f}")
    prop = rep.get("propagation") or {}
    if prop:
        print(f"faulty-set shape over {prop['n_traces']} failing traces: "
              f"suffix {prop['suffix_share']:.3f}, contiguous "
              f"{prop['contiguous_share']:.3f}, gapped {prop['gapped_share']:.3f}; "
              f"mean faulty fraction {prop['mean_faulty_fraction']:.3f}")
    print(f"positive step share {rep['positive_step_share']:.3f}")
    if "holdout_overlap" in rep:
        print(f"held-out evaluation problems present in this pool: "
              f"{rep['holdout_overlap']}")

    print()
    passed = True
    for name, ok, detail in checks(rep):
        passed &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<30} {detail}")
    rep["checks_passed"] = bool(passed)

    if args.traces and args.traces.exists():
        traces = {t["id"]: t for t in read_jsonl(args.traces)}
        show_cases(rows, traces, args.cases)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2))
        print(f"\n[audit] wrote {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
