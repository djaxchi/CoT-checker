#!/usr/bin/env python3
"""risk_probe_v0 analysis (docs/risk_probe_v0_plan.md, section 4). CPU only.

Inputs (pulled from the TamIA run):
  --run_dir     prod/: fits/xfit_v2ds_s4*/local/predictions.jsonl.gz,
                existing_scores/<name>.jsonl, features/fsl_forks_L35.npz
  --saved_preds dir with <name>.predictions.jsonl.gz, the §21.18 fits' own
                predictions, for the reproduction gate
  --views_meta  the frozen manifest's meta/fsl_forks.jsonl (view id -> fork id)
  --fork_factors  factorized_step_labels_v1 fork_factors.jsonl (full, pilot)
  --fork_splits   the two frozen fork splits (training-overlap flags)
Writes <out_dir>/summary.json and fork_scores.jsonl.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.factorized_step_labels import boot_ci, partial_spearman, spearman  # noqa: E402

GATE_MEDIAN, GATE_MAX = 0.002, 0.02
EXISTING = [f"full_{src}_s{s}" for src in ("v2ds", "v1human") for s in (42, 43, 44)]


def jl(p: Path) -> list[dict]:
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def auroc(score, y) -> float | None:
    s, y = np.asarray(score, float), np.asarray(y, int)
    if y.min() == y.max():
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def gate(run_dir: Path, saved: Path) -> dict:
    out = {}
    for name in EXISTING:
        new = {r["trace_id"]: r["logit"] for r in jl(run_dir / "existing_scores" / f"{name}.jsonl")
               if r["split"] == "gate_rp_test"}
        old = {r["trace_id"]: r["logit"] for r in jl(saved / f"{name}.predictions.jsonl.gz")
               if r["split"] == "rp_test" and r["trace_id"] in new}
        # on the probability scale, as the project's earlier gate: the saved logits are
        # bf16 (spacing 0.03 near |logit| 4), so a one-ulp flip alone exceeds 0.02 logits
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
        d = np.abs([sig(new[t]) - sig(old[t]) for t in old])
        out[name] = {"n": len(d), "median_abs": float(np.median(d)), "max_abs": float(d.max()),
                     "pass": bool(len(d) == len(new) and np.median(d) <= GATE_MEDIAN and d.max() <= GATE_MAX)}
    out["pass_all"] = all(v["pass"] for v in out.values())
    return out


def q1(score: np.ndarray, F: dict, mask: np.ndarray) -> dict:
    """Does a validity score rank risk within validity classes?"""
    inc, rd = F["incorrect"][mask], F["risk_diff"][mask]
    s, harm, byp = score[mask], F["harmful"][mask], F["bypassed"][mask]
    covs = [F["incorrect"][mask], F["log_n_cont"][mask], F["relpos"][mask]]
    i, c = inc == 1, inc == 0
    si, rdi = s[i], rd[i]
    sc, rdc = s[c], rd[c]
    out = {"n": int(mask.sum()), "validity_auroc": auroc(s, inc),
           "incorrect": {"n": int(i.sum()),
                         "spearman_score_risk": spearman(si, rdi),
                         "spearman_ci95": boot_ci(lambda k: spearman(si[k], rdi[k]), len(si), 1000, 3),
                         "auroc_harmful": auroc(si, harm[i]),
                         "auroc_harmful_ci95": boot_ci(lambda k: auroc(si[k], harm[i][k]), len(si), 1000, 4),
                         "n_harmful": int(harm[i].sum()),
                         "auroc_not_bypassed": auroc(si, 1 - byp[i]), "n_bypassed": int(byp[i].sum())},
           "correct": {"n": int(c.sum()), "spearman_score_risk": spearman(sc, rdc),
                       "spearman_ci95": boot_ci(lambda k: spearman(sc[k], rdc[k]), len(sc), 1000, 5),
                       "harmful_score_percentiles": [float((sc < v).mean()) for v in sc[harm[c] == 1]]},
           "partial_score_risk": partial_spearman(s, rd, covs),
           "partial_ci95": boot_ci(lambda k: partial_spearman(s[k], rd[k], [x[k] for x in covs]),
                                   len(s), 1000, 6)}
    return out


def oof_ridge(X, y, folds=10, repeats=5, seed=0):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = np.zeros((repeats, len(y)))
    for r in range(repeats):
        for tr, te in KFold(folds, shuffle=True, random_state=seed + r).split(X):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 13)))
            m.fit(X[tr], y[tr]); pred[r, te] = m.predict(X[te])
    return pred.mean(0)


def oof_logit(X, y, folds=10, repeats=5, seed=0):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = np.zeros((repeats, len(y)))
    for r in range(repeats):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=5,
                                                                     scoring="roc_auc", max_iter=5000))
            m.fit(X[tr], y[tr]); pred[r, te] = m.predict_proba(X[te])[:, 1]
    return pred.mean(0)


def q2(X: np.ndarray, F: dict, n_perm: int, seed: int = 0) -> dict:
    """Linear risk probes on pooled states, out of fold, with a permutation null."""
    # no PCA: the correctness signal on these states sits in a low-variance subspace
    # (§15.4), which a top-k variance projection would discard; ridge copes with p > n
    Z = X
    C = np.column_stack([np.ones(len(X)), F["incorrect"], F["log_n_cont"], F["relpos"]])
    resid = F["risk_diff"] - C @ np.linalg.lstsq(C, F["risk_diff"], rcond=None)[0]
    pr = oof_ridge(Z, F["risk_diff"]); pres = oof_ridge(Z, resid)
    ph = oof_logit(Z, F["harmful"])
    rng = np.random.default_rng(seed)
    null_r, null_res, null_h = [], [], []
    for _ in range(n_perm):
        p = rng.permutation(len(X))
        null_r.append(spearman(oof_ridge(Z, F["risk_diff"][p], repeats=1, seed=1), F["risk_diff"][p]))
        null_res.append(spearman(oof_ridge(Z, resid[p], repeats=1, seed=1), resid[p]))
        null_h.append(auroc(oof_logit(Z, F["harmful"][p], repeats=1, seed=1), F["harmful"][p]))
    q = lambda v: float(np.percentile([x for x in v if x is not None], 95))  # noqa: E731
    return {"dims": Z.shape[1],
            "spearman_risk": spearman(pr, F["risk_diff"]), "null95_spearman_risk": q(null_r),
            "spearman_resid": spearman(pres, resid), "null95_spearman_resid": q(null_res),
            "auroc_harmful": auroc(ph, F["harmful"]), "null95_auroc_harmful": q(null_h),
            "_pred_risk": pr, "_pred_harm": ph}


def q3(val_score: np.ndarray, risk_pred: np.ndarray, F: dict) -> dict:
    """Does a risk readout add to the validity score? OOF logistic, 2 targets."""
    base = np.column_stack([val_score, F["log_n_cont"], F["relpos"]])
    both = np.column_stack([base, risk_pred])
    out = {}
    for tgt in ("harmful", "fail"):
        y = F[tgt]
        pa, pb = oof_logit(base, y, seed=10), oof_logit(both, y, seed=10)
        aa, ab = auroc(pa, y), auroc(pb, y)
        out[tgt] = {"n_pos": int(y.sum()), "auroc_validity": aa, "auroc_validity_plus_risk": ab,
                    "delta": ab - aa,
                    "delta_ci95": boot_ci(lambda k: (auroc(pb[k], y[k]) or np.nan) - (auroc(pa[k], y[k]) or np.nan),
                                          len(y), 2000, 11)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--saved_preds", type=Path, required=True)
    ap.add_argument("--views_meta", type=Path, required=True)
    ap.add_argument("--fork_factors", type=Path, nargs="+", required=True)
    ap.add_argument("--fork_splits", type=Path, nargs="+", required=True)
    ap.add_argument("--v2_train_meta", type=Path, required=True,
                    help="context_ablation_v3 views/full/meta/v2_train.jsonl (in-sample flags)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_perm", type=int, default=200)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    vid2fork = {m["trace_id"]: m["fork_id"] for m in jl(a.views_meta)}
    fac = {r["fork_id"]: r for p in a.fork_factors for r in jl(p)}
    forks = {f["fork_id"]: f for p in a.fork_splits for f in json.loads(p.read_text())["forks"]}
    trained = {m["orig_trace_id"] for m in jl(a.v2_train_meta)}
    feats = np.load(a.run_dir / "features" / "fsl_forks_L35.npz")
    vids = [str(v) for v in feats["ids"]]
    fids = [vid2fork[v] for v in vids]
    assert set(fids) == set(fac), "fork set mismatch between features and factor labels"

    F = {k: np.array([fac[f][k] for f in fids], float) for k in ("incorrect", "risk_diff", "risk_se",
                                                                  "err_a", "log_n_cont", "relpos")}
    F["harmful"] = ((F["risk_diff"] > 2 * F["risk_se"]) & (F["risk_se"] > 0)).astype(int)
    F["bypassed"] = (F["err_a"] == 0).astype(int)
    F["fail"] = (F["err_a"] >= 0.5).astype(int)
    in_sample = np.array([forks[f]["trace_id"] in trained for f in fids])

    def seed_mean(names, loader):
        return np.mean([loader(n) for n in names], axis=0)

    def xfit(n):
        d = {r["trace_id"]: r["logit"] for r in jl(a.run_dir / "fits" / n / "local" / "predictions.jsonl.gz")
             if r["split"] == "fsl_forks"}
        return np.array([d[v] for v in vids])

    def existing(n):
        d = {r["trace_id"]: r["logit"] for r in jl(a.run_dir / "existing_scores" / f"{n}.jsonl")
             if r["split"] == "fsl_forks"}
        return np.array([d[v] for v in vids])

    S = {"n_forks": len(fids), "n_in_sample_v2ds": int(in_sample.sum()),
         "n_harmful": int(F["harmful"].sum()), "n_bypassed_incorrect": int((F["bypassed"] * F["incorrect"]).sum())}
    S["gate"] = gate(a.run_dir, a.saved_preds)
    scores = {"xfit_v2ds": seed_mean([f"xfit_v2ds_s{s}" for s in (42, 43, 44)], xfit),
              "v2ds": seed_mean([f"full_v2ds_s{s}" for s in (42, 43, 44)], existing),
              "v1human": seed_mean([f"full_v1human_s{s}" for s in (42, 43, 44)], existing)}
    allm = np.ones(len(fids), bool)
    S["q1"] = {"xfit_v2ds": q1(scores["xfit_v2ds"], F, allm),
               "v2ds_out_of_sample": q1(scores["v2ds"], F, ~in_sample),
               "v1human": q1(scores["v1human"], F, allm)}
    S["q2"] = {}
    preds = {}
    for pool in ("mean", "last"):
        r = q2(feats[pool].astype(np.float64), F, a.n_perm)
        preds[pool] = r.pop("_pred_risk"); r.pop("_pred_harm")
        S["q2"][pool] = r
    S["q3"] = {pool: q3(scores["xfit_v2ds"], preds[pool], F) for pool in ("mean", "last")}
    (a.out_dir / "summary.json").write_text(json.dumps(S, indent=2))
    with open(a.out_dir / "fork_scores.jsonl", "w") as f:
        for j, fid in enumerate(fids):
            f.write(json.dumps({"fork_id": fid, "view_id": vids[j], "in_sample_v2ds": bool(in_sample[j]),
                                **{f"score_{k}": float(v[j]) for k, v in scores.items()},
                                **{f"risk_pred_{k}": float(v[j]) for k, v in preds.items()},
                                **{k: float(F[k][j]) for k in F}}) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
