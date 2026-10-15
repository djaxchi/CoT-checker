"""Gate G1 of step_bert_probe_v0: does reconstruction error flag incorrect steps?

For each visibility mode the pretrained model was scored in (e_left, e_right,
e_both), reports on the on-policy TEST split:

* AUROC of the raw error (higher error = predicted incorrect),
* AUROC after residualizing the error on structural features (step index, step
  length, prefix length, later-step count, later-token count; OLS fit on train),
* both AUROCs on two slices: steps up to and including the first error, and
  steps after it,
* paired differences between modes,

with 95% intervals from a bootstrap that resamples problems. A structural-only
logistic baseline is reported for scale. On ProcessBench, AUROC over known-label
steps (up to and including the first error), per subset.

Pass rule from the plan: residualized e_both AUROC CI above 0.5, and
e_both - e_left CI above 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score


def structural(z) -> np.ndarray:
    tid, step, ntok, nst = z["trace_id"], z["step_idx"], z["n_tokens"].astype(float), z["n_steps"]
    prefix = np.zeros(len(tid))
    total = {}
    for i, t in enumerate(tid.tolist()):
        prefix[i] = total.get(t, 0.0)
        total[t] = prefix[i] + ntok[i]
    later_tok = np.array([total[t] for t in tid.tolist()]) - prefix - ntok
    f = np.stack([step, ntok, prefix, nst - 1 - step, later_tok], 1).astype(float)
    return np.hstack([f, np.log1p(f)])


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else float("nan")


def cluster_boot(y, scores: dict[str, np.ndarray], groups, n_boot=2000, seed=0):
    """Point AUROC and 95% CI per score, plus the problem-resampling index sets."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    draws = {k: [] for k in scores}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(len(uniq), size=len(uniq))])
        for k, s in scores.items():
            draws[k].append(auc(y[idx], s[idx]))
    out = {}
    for k, s in scores.items():
        d = np.array(draws[k])
        out[k] = {"auroc": round(auc(y, s), 4), "lo": round(float(np.nanpercentile(d, 2.5)), 4),
                  "hi": round(float(np.nanpercentile(d, 97.5)), 4), "_draws": d}
    return out


def diff_ci(a, b):
    d = a["_draws"] - b["_draws"]
    return {"diff": round(a["auroc"] - b["auroc"], 4),
            "lo": round(float(np.nanpercentile(d, 2.5)), 4), "hi": round(float(np.nanpercentile(d, 97.5)), 4)}


def strip(res):
    return {k: {kk: vv for kk, vv in v.items() if kk != "_draws"} for k, v in res.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pooled", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--scores_dir", type=Path, required=True)
    p.add_argument("--pb", type=Path, nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()

    z = np.load(args.pooled)
    e = dict(np.load(args.scores_dir / "scores_onpolicy.npz"))
    modes = sorted(k[2:] for k in e)
    split = json.loads(args.split.read_text())
    tid = z["trace_id"]
    is_tr, is_te = np.isin(tid, split["train"]), np.isin(tid, split["test"])
    y, fe, step, grp = z["y"].astype(int), z["first_error_idx"], z["step_idx"], z["problem_id"]
    X = structural(z)

    resid_models = {m: LinearRegression().fit(X[is_tr], e[f"e_{m}"][is_tr]) for m in modes}
    scores = {}
    for m in modes:
        scores[f"{m}_raw"] = e[f"e_{m}"]
        scores[f"{m}_resid"] = e[f"e_{m}"] - resid_models[m].predict(X)
    mu, sd = X[is_tr].mean(0), X[is_tr].std(0) + 1e-9
    base = LogisticRegression(max_iter=2000).fit((X[is_tr] - mu) / sd, y[is_tr])
    scores["structural_baseline"] = base.predict_proba((X - mu) / sd)[:, 1]

    post = (fe >= 0) & (step > fe)
    report = {"n_test_steps": int(is_te.sum()), "test_prevalence": round(float(y[is_te].mean()), 4),
              "modes": modes, "slices": {}}
    for name, mask in (("all", is_te), ("upto_first_error", is_te & ~post), ("after_first_error", is_te & post)):
        res = cluster_boot(y[mask], {k: v[mask] for k, v in scores.items()}, grp[mask], args.n_boot)
        diffs = {}
        for kind in ("raw", "resid"):
            for a, b in (("both", "left"), ("both", "right"), ("right", "left")):
                if a in modes and b in modes:
                    diffs[f"{a}-{b}_{kind}"] = diff_ci(res[f"{a}_{kind}"], res[f"{b}_{kind}"])
        report["slices"][name] = {"n": int(mask.sum()), "prevalence": round(float(y[mask].mean()), 4),
                                  "auroc": strip(res), "paired_diffs": diffs}

    report["processbench"] = {}
    for pb_path in args.pb:
        name = pb_path.stem.replace("pooled_", "")
        zp = np.load(pb_path)
        ep = dict(np.load(args.scores_dir / f"scores_{name}.npz"))
        known = (zp["first_error_idx"] < 0) | (zp["step_idx"] <= zp["first_error_idx"])
        Xp = structural(zp)
        yp = zp["y"].astype(int)[known]
        row = {"n_known": int(known.sum()), "prevalence": round(float(yp.mean()), 4)}
        for m in modes:
            row[f"{m}_raw"] = round(auc(yp, ep[f"e_{m}"][known]), 4)
            row[f"{m}_resid"] = round(auc(yp, (ep[f"e_{m}"] - resid_models[m].predict(Xp))[known]), 4)
        report["processbench"][name] = row

    pass_resid = report["slices"]["all"]["auroc"].get("both_resid", {}).get("lo", 0) > 0.5
    d = report["slices"]["all"]["paired_diffs"].get("both-left_resid", {"lo": -1})
    report["G1_pass"] = bool(pass_resid and d["lo"] > 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    for name, sl in report["slices"].items():
        print(f"\n[{name}] n={sl['n']} prevalence={sl['prevalence']}")
        for k, v in sl["auroc"].items():
            print(f"  {k:22s} {v['auroc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
        for k, v in sl["paired_diffs"].items():
            print(f"  diff {k:17s} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    for name, row in report["processbench"].items():
        print(name, row)
    print("G1_pass", report["G1_pass"])


if __name__ == "__main__":
    main()
