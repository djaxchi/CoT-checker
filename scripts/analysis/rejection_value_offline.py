#!/usr/bin/env python3
"""Offline value of a step-rejection score: mean downstream risk of the steps it flags.

Plan and prespecified reading: docs/rejection_value_offline_v0_plan.md.
risk(s) = err(s+1) - err(s) is the expected drop in final-answer error from one
redraw of step s from its prefix, so a score that flags high-risk steps spends a
resampling budget where it can pay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

QS = (0.05, 0.10, 0.20, 0.30)


def flagged_mean(score: np.ndarray, risk: np.ndarray, q: float) -> tuple[float, float]:
    """Mean risk of the top-q steps by score, and their share of all positive risk."""
    k = max(1, int(round(q * len(score))))
    top = np.argsort(-score, kind="stable")[:k]
    pos = np.clip(risk, 0, None)
    return float(risk[top].mean()), float(pos[top].sum() / max(pos.sum(), 1e-12))


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--values", type=Path, default=Path("results/step_value_labels_v1/label_summary/step_values.jsonl"))
    p.add_argument("--scores", type=Path, default=Path("results/risk_probe_v1/analysis_prod/step_scores.jsonl"))
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("results/rejection_value_offline_v0/summary.json"))
    a = p.parse_args()
    vals = {r["view_id"]: r for r in map(json.loads, open(a.values))}
    rows = [(vals[s["view_id"]], s) for s in map(json.loads, open(a.scores)) if s["view_id"] in vals]
    risk = np.array([v["risk"] for v, _ in rows])
    groups = np.array([v["problem_key"] for v, _ in rows])
    S = {"validity": np.array([s["xfit5"] for _, s in rows]),
         "A_last": np.array([s["A_last"] for _, s in rows]),
         "A_last_h": np.array([s["A_last_h"] for _, s in rows])}
    S["combo"] = zscore(S["validity"]) + zscore(S["A_last_h"])
    rng = np.random.default_rng(0)
    S["random"] = rng.random(len(risk))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = {"n_steps": int(len(risk)), "n_problems": int(len(uniq)), "mean_risk_all": float(risk.mean()), "q": {}}
    boots = []
    for _ in range(a.n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        boots.append(np.concatenate([members[g] for g in pick]))
    for q in QS:
        res = {name: dict(zip(("mean_risk", "share_pos_risk"), flagged_mean(sc, risk, q))) for name, sc in S.items()}
        diffs = {}
        for name in ("A_last", "A_last_h", "combo", "random"):
            d = [flagged_mean(S[name][b], risk[b], q)[0] - flagged_mean(S["validity"][b], risk[b], q)[0] for b in boots]
            diffs[f"{name}_minus_validity"] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        out["q"][str(q)] = {"scorers": res, "diff_vs_validity": diffs}
        print(f"q={q}: " + "  ".join(f"{n}={r['mean_risk']:+.4f}" for n, r in res.items()))
        print("      " + "  ".join(f"{n} {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]" for n, v in diffs.items()))
    r10 = out["q"]["0.1"]
    rel = r10["scorers"]["combo"]["mean_risk"] / max(r10["scorers"]["validity"]["mean_risk"], 1e-12) - 1
    out["go"] = bool(rel >= 0.25 and r10["diff_vs_validity"]["combo_minus_validity"][1] > 0)
    out["combo_rel_gain_q10"] = float(rel)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"combo relative gain at q=0.10: {rel:+.3f}  GO={out['go']}")


if __name__ == "__main__":
    main()
