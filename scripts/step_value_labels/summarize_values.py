#!/usr/bin/env python3
"""Summarize step_value_labels_v1 shards (docs/step_value_labels_v1_plan.md, section 4).

Label set (default): writes <out_dir>/step_values.jsonl, one row per labeled step
(view id = <trace_id>#<claim index>, the context_ablation_v3 view id), and
summary.{json,md}: cut and rollout counts, err(0) (problem error rate), risk by
DeepSeek validity, harmful shares, first-error against post-error steps.

Validation set (--track1_dir): joins each fork's err(s) and err(s + 1) to
factorized_step_labels_v1 track 1's err(C) and err(A) and applies the gate of
plan section 4: the fork-bootstrap 95% CI of each mean difference includes 0 and
its absolute value is below --gate_tol.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.factorized_step_labels import boot_ci, spearman  # noqa: E402


def load_shards(d: Path) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(d / "values.shard*.jsonl"))):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    ids = [r["trace_id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("[sum] duplicate traces across shards")
    return rows


def validate(rows: list[dict], track1_dir: Path, tol: float) -> dict:
    t1 = {}
    for p in sorted(glob.glob(str(track1_dir / "track1.shard*.jsonl"))):
        for l in open(p):
            r = json.loads(l); t1[r["fork_id"]] = r["risk"]
    pairs = []
    for r in rows:
        s = r["fork_step"]
        e = {c["cut"]: c["err"] for c in r["cuts"]}
        old = t1.get(r["trace_id"])
        if old is None:
            continue
        pairs.append((e[s], old["err_c_mean"], e[s + 1], old["err_a"], e[s + 1] - e[s], old["risk_diff"]))
    P = np.array(pairs, float)
    d_before, d_after = P[:, 0] - P[:, 1], P[:, 2] - P[:, 3]
    out = {"n_forks": len(P)}
    for name, d in (("err_before_minus_err_c", d_before), ("err_after_minus_err_a", d_after)):
        ci = boot_ci(lambda k: float(d[k].mean()), len(d), 2000, 1)
        out[name] = {"mean": float(d.mean()), "ci95": ci,
                     "pass": bool(ci[0] <= 0 <= ci[1] and abs(d.mean()) < tol)}
    out["spearman_err_before_err_c"] = spearman(P[:, 0], P[:, 1])
    out["spearman_err_after_err_a"] = spearman(P[:, 2], P[:, 3])
    out["spearman_risk_new_old"] = spearman(P[:, 4], P[:, 5])
    out["pass"] = out["err_before_minus_err_c"]["pass"] and out["err_after_minus_err_a"]["pass"]
    return out


def step_table(rows: list[dict], finished: dict[str, bool] | None = None) -> list[dict]:
    """One row per labeled step; `finished` (trace id -> flag, from the frozen set) marks capped traces."""
    out = []
    for r in rows:
        e0 = r["cuts"][0]["err"] if r["cuts"][0]["cut"] == 0 else None
        for s in r["steps"]:
            i = s["step"]
            row = {"trace_id": r["trace_id"], "problem_key": r["problem_key"], "group": r["group"],
                   "n_steps": r["n_steps"], "err0": e0, **s}
            if finished is not None:
                row["finished"] = finished[r["trace_id"]]
            if "step_claim_idx" in r:
                ci = r["step_claim_idx"][i]
                fe = r.get("first_error_ds")
                row.update({"view_id": f"{r['trace_id']}#{ci}", "claim_idx": ci,
                            "y_ds": r["y_ds"][i], "mask_ds": r["mask_ds"][i],
                            "y_self": r["y_self"][i], "mask_self": r["mask_self"][i],
                            "relpos": i / max(1, r["n_steps"] - 1),
                            "phase": "pre_error" if fe is None or i < fe else ("first_error" if i == fe else "post_error")})
            out.append(row)
    return out


def describe(T: list[dict]) -> dict:
    def stats(sel):
        x = [t for t in T if sel(t)]
        if not x:
            return {"n": 0}
        rk = np.array([t["risk"] for t in x])
        return {"n": len(x), "risk_mean": float(rk.mean()), "risk_median": float(np.median(rk)),
                "share_harmful": float(np.mean([t["harmful"] for t in x])),
                "share_risk_zero": float(np.mean(rk == 0)),
                "err_before_mean": float(np.mean([t["err_before"] for t in x]))}
    lab = [t for t in T if t.get("mask_ds")]
    return {"n_steps": len(T), "n_labeled_ds": len(lab),
            "by_validity": {"incorrect": stats(lambda t: t.get("mask_ds") and t["y_ds"] == 1),
                            "correct": stats(lambda t: t.get("mask_ds") and t["y_ds"] == 0)},
            "by_phase": {ph: stats(lambda t, ph=ph: t.get("phase") == ph)
                         for ph in ("pre_error", "first_error", "post_error")},
            "by_group": {g: stats(lambda t, g=g: t["group"] == g) for g in sorted({t["group"] for t in T})},
            "first_error_by_finished": {str(f): stats(lambda t, f=f: t.get("phase") == "first_error"
                                                      and t.get("finished") == f) for f in (True, False)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--values_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--track1_dir", type=Path, default=None, help="validation set: track-1 shards to check against")
    ap.add_argument("--gate_tol", type=float, default=0.03)
    ap.add_argument("--trace_set", type=Path, default=None,
                    help="frozen label set, to carry each trace's `finished` flag into the step rows")
    a = ap.parse_args()
    rows = load_shards(a.values_dir)
    fin = ({t["trace_id"]: bool(t["finished"]) for t in json.loads(a.trace_set.read_text())["traces"]}
           if a.trace_set else None)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    roll = [x for r in rows for c in r["cuts"] for x in c["rollouts"]]
    S = {"n_traces": len(rows), "n_cuts": sum(len(r["cuts"]) for r in rows), "n_rollouts": len(roll),
         "share_truncated": float(np.mean([x["truncated"] for x in roll])) if roll else None,
         "share_ungradeable": float(np.mean([not x["gradeable"] for x in roll])) if roll else None,
         "n_new_median": float(np.median([x["n_new"] for x in roll])) if roll else None}
    T = step_table(rows, fin)
    S["steps"] = describe(T)
    if a.track1_dir:
        S["validation"] = validate(rows, a.track1_dir, a.gate_tol)
    else:
        e0 = np.array([r["cuts"][0]["err"] for r in rows if r["cuts"][0]["cut"] == 0])
        S["err0"] = {"mean": float(e0.mean()), "share_zero": float(np.mean(e0 == 0)),
                     "share_one": float(np.mean(e0 == 1))} if len(e0) else None
    with open(a.out_dir / "step_values.jsonl", "w") as f:
        for t in T:
            f.write(json.dumps(t) + "\n")
    (a.out_dir / "summary.json").write_text(json.dumps(S, indent=2))
    md = ["# step_value_labels_v1 summary", "", "```", json.dumps(S, indent=1), "```"]
    (a.out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
