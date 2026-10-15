#!/usr/bin/env python3
"""How much of the step_value_labels_v1 risk is grading noise.

A step changes the answer's spelling as well as its value. When the project grader
accepts one spelling of the gold answer and rejects another ("(3/4, 5)" against
"\\left(\\frac34,5\\right)"), err jumps at that step with no change in the math,
and risk inherits the jump.

This re-grades every distinct predicted answer of every trace with a second,
lenient grader (math_verify, plus mixed numbers such as "3 11/15") and calls a
rollout correct when either grader accepts it. Per-step risk is recomputed from the
re-graded rollouts and compared with the shipped labels: the share of rollouts and
of steps whose label changes, per class.

Needs the math-verify package (run with `uv run --with math-verify`). Writes
<out_dir>/regrade.json and regraded.shard00.jsonl (labeler rows with the lenient
grades, for analyze_labels.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.step_value_labels import step_risks  # noqa: E402

MIXED = re.compile(r"^\s*(-?\d+)\s+(\d+)\s*/\s*(\d+)\s*$")
_GOLD_NUM = re.compile(r"^\s*(-?\d+)\s*/\s*(\d+)\s*$|^\s*-?\\frac\{(\d+)\}\{(\d+)\}\s*$")


def mixed_value(s: str) -> Fraction | None:
    m = MIXED.match(s)
    if not m:
        return None
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if c == 0:
        return None
    f = abs(a) + Fraction(b, c)
    return -f if a < 0 else f


def gold_fraction(s: str) -> Fraction | None:
    m = _GOLD_NUM.match(s)
    if not m:
        return None
    try:
        if m.group(1) is not None:
            return Fraction(int(m.group(1)), int(m.group(2)))
        f = Fraction(int(m.group(3)), int(m.group(4)))
        return -f if s.strip().startswith("-") else f
    except (ZeroDivisionError, ValueError):
        return None


def make_lenient():
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    cfg = [LatexExtractionConfig(), ExprExtractionConfig()]

    def ok(pred: str, gold: str) -> bool:
        mv, gf = mixed_value(pred), gold_fraction(gold)
        if mv is not None and gf is not None and mv == gf:
            return True
        try:
            G = parse(f"${gold}$", extraction_config=cfg)
            return any(verify(G, parse(p, extraction_config=cfg)) for p in (f"${pred}$", pred))
        except Exception:
            return False
    return ok


def regrade_row(r: dict, lenient) -> tuple[dict, int]:
    """The row with lenient grades and recomputed steps, and the number of rollouts flipped to correct."""
    cache, flips = {}, 0
    cuts = []
    for c in r["cuts"]:
        ro = []
        for x in c["rollouts"]:
            ok = bool(x["correct"])
            if not ok and x.get("pred") is not None:
                if x["pred"] not in cache:
                    cache[x["pred"]] = lenient(x["pred"], r["gold"])
                ok = cache[x["pred"]]
                flips += ok
            ro.append({**x, "correct": ok})
        cuts.append({**c, "rollouts": ro, "err": sum(not x["correct"] for x in ro) / len(ro)})
    err = {c["cut"]: [not x["correct"] for x in c["rollouts"]] for c in cuts}
    return {**r, "cuts": cuts, "steps": step_risks(err, r["n_steps"])}, flips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--values_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(str(a.values_dir / "values.shard*.jsonl"))):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    lenient = make_lenient()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    n_roll = n_flip = 0
    pairs = []                                   # (old step, new step, y_ds, phase)
    with open(a.out_dir / "regraded.shard00.jsonl", "w") as f:
        for r in rows:
            nr, fl = regrade_row(r, lenient)
            n_flip += fl; n_roll += sum(len(c["rollouts"]) for c in r["cuts"])
            fe = r.get("first_error_ds")
            for o, n in zip(r["steps"], nr["steps"]):
                i = o["step"]
                ph = "pre_error" if fe is None or i < fe else ("first_error" if i == fe else "post_error")
                pairs.append((o, n, r["y_ds"][i] if r["mask_ds"][i] else None, ph))
            f.write(json.dumps(nr) + "\n")
    out = {"n_traces": len(rows), "n_rollouts": n_roll, "share_rollouts_flipped_to_correct": n_flip / n_roll}

    def cmp(sel):
        x = [(o, n) for o, n, y, ph in pairs if sel(y, ph)]
        if not x:
            return {"n": 0}
        ho, hn = np.array([o["harmful"] for o, _ in x]), np.array([n["harmful"] for _, n in x])
        ro, rn = np.array([o["risk"] for o, _ in x]), np.array([n["risk"] for _, n in x])
        return {"n": len(x), "harmful_old": float(ho.mean()), "harmful_new": float(hn.mean()),
                "harmful_lost": float(np.mean(ho & ~hn)), "harmful_gained": float(np.mean(~ho & hn)),
                "share_risk_changed": float(np.mean(ro != rn)), "risk_mean_old": float(ro.mean()),
                "risk_mean_new": float(rn.mean()), "mean_abs_risk_change": float(np.abs(ro - rn).mean())}
    out["by_class"] = {"all": cmp(lambda y, ph: True), "ds_correct": cmp(lambda y, ph: y == 0),
                       "ds_incorrect": cmp(lambda y, ph: y == 1),
                       **{ph: cmp(lambda y, p, ph=ph: p == ph) for ph in ("pre_error", "first_error", "post_error")}}
    (a.out_dir / "regrade.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
