#!/usr/bin/env python3
"""risk_probe_v1 analysis (docs/risk_probe_v1_plan.md, sections 5 to 7). CPU only.

Inputs:
  --fit_dir      fit_readouts.py outputs: preds_e2.npz, preds_e1.npz, hparams.json,
                 null_*.npz
  --run_dir      the encode/fit job's OUT_ROOT: fits/xfit5_f<k>/ (cross-fitted
                 validity), existing_scores/xfit_v2ds_s4*.jsonl (v0's validity
                 probe rescored), features/rows_L35.npz
  --v0_run       risk_probe_v0's run dir (features/fsl_forks_L35.npz and
                 fits/xfit_v2ds_s4*/local/predictions.jsonl.gz, for the gate)
  --views        this experiment's frozen view manifest (meta/fsl_forks.jsonl,
                 lab_views.jsonl, build_audit.json)
  --step_values, --trace_set   the frozen label set
  --fork_factors, --track1_dirs  factorized_step_labels_v1 fork labels
Writes <out_dir>/summary.json, summary.md and step_scores.jsonl.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "factorized_step_labels"))
from src.analysis.risk_probe_v1 import N_FOLDS, auroc, fold_of, residualize, spearman  # noqa: E402

GATE_MEDIAN, GATE_MAX, GATE_COS = 0.002, 0.02, 0.9999
N_TRACES = 5874
PRIMARY = ("A_last", "B")
N_BOOT = 2000


def jl(p) -> list[dict]:
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #

class Boot:
    """Cluster bootstrap over groups (problems; one fork per problem on E1)."""

    def __init__(self, groups, n_boot: int = N_BOOT, seed: int = 0):
        by = defaultdict(list)
        for i, g in enumerate(groups):
            by[g].append(i)
        self.idx = [np.array(v) for v in by.values()]
        rng = np.random.default_rng(seed)
        self.draws = [np.concatenate([self.idx[j] for j in rng.integers(0, len(self.idx), len(self.idx))])
                      for _ in range(n_boot)]

    def ci(self, stat, level: float = 0.95) -> list[float] | None:
        v = [x for x in (stat(k) for k in self.draws) if x is not None and np.isfinite(x)]
        if len(v) < len(self.draws) // 2:
            return None
        lo = (1 - level) / 2 * 100
        return [float(np.percentile(v, lo)), float(np.percentile(v, 100 - lo))]


def null95(vals) -> float | None:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.percentile(v, 95)) if v else None


# --------------------------------------------------------------------------- #
# out-of-fold combination (Q3)
# --------------------------------------------------------------------------- #

def logit_cv():
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=5,
                                                                scoring="roc_auc", max_iter=5000))


def oof_by_fold(X: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """E2: the combination fit on four problem folds, applied to the fifth."""
    p = np.zeros(len(y))
    for o in range(N_FOLDS):
        tr, te = fold != o, fold == o
        if te.any():
            p[te] = logit_cv().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def q3_contrasts(base: np.ndarray, extra: dict[str, np.ndarray], y: np.ndarray, oof, boot: Boot,
                 levels=(0.95, 0.975)) -> dict:
    pa = oof(base, y)
    out = {"n_pos": int(y.sum()), "n": int(len(y)), "auroc_validity": auroc(pa, y)}
    for name, Z in extra.items():
        pb = oof(np.column_stack([base, Z]), y)
        d = {"auroc": auroc(pb, y)}
        d["delta"] = d["auroc"] - out["auroc_validity"]
        for lv in levels:
            d[f"delta_ci{int(lv * 1000) / 10:g}"] = boot.ci(
                lambda k: (auroc(pb[k], y[k]) or np.nan) - (auroc(pa[k], y[k]) or np.nan), lv)
        out[name] = d
    return out


def above0(ci) -> bool:
    return ci is not None and ci[0] > 0


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def label_frame(step_values: list[dict], trace_set: dict, view_ids: np.ndarray) -> dict:
    """Per label step (in preds_e2 order): labels and covariates."""
    traces = {t["trace_id"]: t for t in trace_set["traces"]}
    sv = {}
    for r in step_values:
        t = traces[r["trace_id"]]
        sv[f"{r['trace_id']}#{t['step_claim_idx'][r['step']]}"] = r
    F = defaultdict(list)
    for v in view_ids:
        r = sv[v]
        t = traces[r["trace_id"]]
        n_cont = len(t["reply_ids"]) - t["cut_offsets"][r["step"] + 1]
        F["trace_id"].append(r["trace_id"]); F["problem_key"].append(r["problem_key"])
        F["risk"].append(r["risk"]); F["harmful"].append(int(r["harmful"]))
        F["err_before"].append(r["err_before"]); F["mask_ds"].append(bool(r["mask_ds"]))
        F["y_ds"].append(int(r["y_ds"]) if r["mask_ds"] else -1)
        F["phase"].append(r.get("phase")); F["finished"].append(bool(t["finished"]))
        F["relpos"].append(r["relpos"]); F["log_n_cont"].append(math.log1p(max(n_cont, 0)))
    F = {k: np.array(v) for k, v in F.items()}
    m = F["mask_ds"]
    F["resid"] = np.full(len(m), np.nan)
    F["resid"][m] = residualize(F["risk"][m].astype(float), [F["y_ds"][m], F["log_n_cont"][m], F["relpos"][m]])
    return F


def fork_frame(views_meta: list[dict], fork_factors: list[dict], track1: list[dict], view_ids: np.ndarray) -> dict:
    vid2fork = {m["trace_id"]: m["fork_id"] for m in views_meta}
    fac = {r["fork_id"]: r for r in fork_factors}
    errc = {r["fork_id"]: r["risk"]["err_c_mean"] for r in track1}
    fids = [vid2fork[str(v)] for v in view_ids]
    F = {k: np.array([fac[f][k] for f in fids], float) for k in ("incorrect", "risk_diff", "risk_se", "err_a",
                                                                  "log_n_cont", "relpos")}
    F["harmful"] = ((F["risk_diff"] > 2 * F["risk_se"]) & (F["risk_se"] > 0)).astype(int)
    F["fail"] = (F["err_a"] >= 0.5).astype(int)
    F["err_c"] = np.array([errc.get(f, np.nan) for f in fids], float)
    F["fork_id"] = np.array(fids)
    C = np.column_stack([np.ones(len(fids)), F["incorrect"], F["log_n_cont"], F["relpos"]])
    F["resid"] = F["risk_diff"] - C @ np.linalg.lstsq(C, F["risk_diff"], rcond=None)[0]
    return F


def load_nulls(fit_dir: Path):
    fs = sorted(glob.glob(str(fit_dir / "null_*.npz")))
    if not fs:
        return None
    zs = [np.load(f) for f in fs]
    arms = [str(a) for a in zs[0]["arms"]]
    return arms, np.concatenate([z["e2"] for z in zs]).astype(np.float64), \
        np.concatenate([z["e1"] for z in zs]).astype(np.float64)


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def gate_features(rows: np.lib.npyio.NpzFile, v0_feats: np.lib.npyio.NpzFile) -> dict:
    ids = [str(v) for v in rows["ids"]]
    pos = {v: i for i, v in enumerate(ids)}
    v0_ids = [str(x) for x in v0_feats["ids"]]
    take = np.array([pos[v] for v in v0_ids])
    out = {}
    for new, old in (("after", "last"), ("mean", "mean")):
        A = rows[new][take].astype(np.float64)          # NpzFile arrays: read each once
        Bm = v0_feats[old].astype(np.float64)
        cos = (A * Bm).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(Bm, axis=1))
        out[new] = {"n": int(len(cos)), "median_cos": float(np.median(cos)), "min_cos": float(np.min(cos))}
    out["pass"] = all(out[k]["median_cos"] >= GATE_COS for k in ("after", "mean"))
    return out


def gate_scores(run_dir: Path, v0_run: Path) -> dict:
    out = {}
    for s in (42, 43, 44):
        new = {(r["split"], r["trace_id"]): r["logit"] for r in jl(run_dir / "existing_scores" / f"xfit_v2ds_s{s}.jsonl")}
        old = {(r["split"], r["trace_id"]): r["logit"]
               for r in jl(v0_run / "fits" / f"xfit_v2ds_s{s}" / "local" / "predictions.jsonl.gz")}
        keys = [k for k in new if k in old]
        d = np.abs(sig([new[k] for k in keys]) - sig([old[k] for k in keys]))
        out[f"s{s}"] = {"n": len(d), "n_new": len(new), "median_abs": float(np.median(d)), "max_abs": float(d.max()),
                        "pass": bool(len(d) == len(new) and np.median(d) <= GATE_MEDIAN and d.max() <= GATE_MAX)}
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def gate_leakage(run_dir: Path, lab_pk: set, fork_pk: set) -> dict:
    out = {"label_fork_problem_overlap": len(lab_pk & fork_pk), "folds": {}}
    ok = out["label_fork_problem_overlap"] == 0
    for k in range(N_FOLDS):
        cfg = json.loads((run_dir / "fits" / f"xfit5_f{k}" / "config.json").read_text())
        tr = cfg["train_split"].split(",")
        good = f"xf{k}" not in tr and f"xf{k}" in cfg["score_splits"] and len(tr) == N_FOLDS - 1
        out["folds"][k] = {"train_split": cfg["train_split"], "ok": good}
        ok &= good
    out["pass"] = bool(ok)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def q2_block(pred: dict, y: np.ndarray, resid: np.ndarray, harm: np.ndarray, boot: Boot | None,
             null: dict | None) -> dict:
    out = {}
    for arm, p in pred.items():
        if arm.endswith("_h"):
            d = {"auroc_harmful": auroc(p, harm)}
            if null is not None:
                d["null95_auroc_harmful"] = null95([auroc(q, harm) for q in null[arm]])
        else:
            m = np.isfinite(resid)
            d = {"spearman_risk": spearman(p, y), "spearman_resid": spearman(p[m], resid[m]),
                 "auroc_harmful": auroc(p, harm)}
            if boot is not None and arm in PRIMARY:
                d["spearman_risk_ci95"] = boot.ci(lambda k: spearman(p[k], y[k]))
            if null is not None:
                d["null95_spearman_risk"] = null95([spearman(q, y) for q in null[arm]])
                d["null95_spearman_resid"] = null95([spearman(q[m], resid[m]) for q in null[arm]])
                d["null95_auroc_harmful"] = null95([auroc(q, harm) for q in null[arm]])
        out[arm] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit_dir", type=Path, required=True)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--v0_run", type=Path, required=True)
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--step_values", type=Path, required=True)
    ap.add_argument("--trace_set", type=Path, required=True)
    ap.add_argument("--fork_factors", type=Path, nargs="+", required=True)
    ap.add_argument("--track1_dirs", type=Path, nargs="*", default=[])
    ap.add_argument("--out_dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    S: dict = {}

    e2z, e1z = np.load(a.fit_dir / "preds_e2.npz"), np.load(a.fit_dir / "preds_e1.npz")
    arms = [str(x) for x in e2z["arms"]]
    P2 = {arm: e2z["pred"][:, j] for j, arm in enumerate(arms)}
    P1 = {arm: e1z["pred"][:, j] for j, arm in enumerate(arms)}
    step_values = jl(a.step_values)
    trace_set = json.loads(a.trace_set.read_text())
    L = label_frame(step_values, trace_set, e2z["view_id"])
    track1 = [r for d in a.track1_dirs for p in sorted(d.glob("track1.shard*.jsonl")) for r in jl(p)]
    Fk = fork_frame(jl(a.views / "meta" / "fsl_forks.jsonl"), [r for p in a.fork_factors for r in jl(p)], track1,
                    e1z["view_id"])

    # gates
    rows = np.load(a.run_dir / "features" / "rows_L35.npz")
    S["gates"] = {
        "labels": {"n_traces": len({r["trace_id"] for r in step_values}), "n_steps": len(step_values),
                   "n_steps_with_predictions": int(len(L["risk"])),
                   "pass": len({r["trace_id"] for r in step_values}) == N_TRACES},
        "reproduction_features": gate_features(rows, np.load(a.v0_run / "features" / "fsl_forks_L35.npz")),
        "reproduction_scores": gate_scores(a.run_dir, a.v0_run),
        "alignment": {k: json.loads((a.views / "build_audit.json").read_text())[k] for k in ("counts", "aligned_share")},
        "leakage": gate_leakage(a.run_dir, set(L["problem_key"]),
                                {m["problem_key"] for m in jl(a.views / "meta" / "fsl_forks.jsonl")}),
    }
    S["gates"]["alignment"]["pass"] = S["gates"]["alignment"]["aligned_share"] >= 0.99
    e1_ok = S["gates"]["reproduction_features"]["pass"] and S["gates"]["reproduction_scores"]["pass"]
    S["gates"]["e1_interpretable"] = bool(e1_ok and S["gates"]["leakage"]["pass"])
    S["gates"]["e2_interpretable"] = bool(S["gates"]["labels"]["pass"] and S["gates"]["alignment"]["pass"]
                                          and S["gates"]["leakage"]["pass"])

    # validity scores
    xf5 = {}
    for k in range(N_FOLDS):
        for r in jl(a.run_dir / "fits" / f"xfit5_f{k}" / "local" / "predictions.jsonl.gz"):
            if r["split"] == f"xf{k}":
                xf5[r["trace_id"]] = r["logit"]
    L["xfit5"] = np.array([xf5[str(v)] for v in e2z["view_id"]])
    vx = defaultdict(list)
    for s in (42, 43, 44):
        for r in jl(a.run_dir / "existing_scores" / f"xfit_v2ds_s{s}.jsonl"):
            if r["split"] == "fsl_forks":
                vx[r["trace_id"]].append(r["logit"])
    Fk["xfit_v2ds"] = np.array([np.mean(vx[str(v)]) for v in e1z["view_id"]])

    fold = np.array([fold_of(p) for p in L["problem_key"]])
    boot2 = Boot(L["problem_key"])
    boot1 = Boot(Fk["fork_id"])
    nulls = load_nulls(a.fit_dir)
    null2 = null1 = None
    if nulls is not None:
        narms, N2, N1 = nulls
        null2 = {arm: N2[:, :, j] for j, arm in enumerate(narms)}
        null1 = {arm: N1[:, :, j] for j, arm in enumerate(narms)}
        S["n_null_perms"] = int(N2.shape[0])

    # Q1 (E2): the cross-fitted validity probe within DeepSeek-incorrect steps
    lab = L["mask_ds"]
    inc = lab & (L["y_ds"] == 1)
    ii = np.where(inc)[0]
    bi = Boot(L["problem_key"][ii])
    xs, rk, hm = L["xfit5"][ii], L["risk"][ii], L["harmful"][ii]
    S["q1_e2"] = {"validity_auroc": auroc(L["xfit5"][lab], L["y_ds"][lab]), "n_incorrect": int(len(ii)),
                  "spearman_score_risk": spearman(xs, rk), "spearman_ci95": bi.ci(lambda k: spearman(xs[k], rk[k])),
                  "auroc_harmful": auroc(xs, hm), "auroc_harmful_ci95": bi.ci(lambda k: auroc(xs[k], hm[k])),
                  "n_harmful": int(hm.sum())}

    # Q2
    risk_arms = ["A_last", "A_mean", "B", "A_last_res", "A_mean_res", "A_last_h", "A_mean_h"]
    S["q2_e1"] = q2_block({k: P1[k] for k in risk_arms}, Fk["risk_diff"], Fk["resid"], Fk["harmful"], boot1,
                          None if null1 is None else {k: null1[k] for k in risk_arms})
    ok = np.isfinite(Fk["err_c"])
    S["q2_e1"]["B_calibration"] = {"spearman_before_err_c": spearman(P1["B_before"][ok], Fk["err_c"][ok]),
                                   "spearman_after_err_a": spearman(P1["B_after"], Fk["err_a"]), "n_err_c": int(ok.sum())}
    S["q2_e2"] = {}
    subsets = {"all": np.ones(len(L["risk"]), bool), "first_error": L["phase"] == "first_error",
               "headroom": (L["err_before"] > 0) & (L["err_before"] < 1), "finished": L["finished"]}
    for name, m in subsets.items():
        idx = np.where(m)[0]
        S["q2_e2"][name] = q2_block({k: P2[k][idx] for k in risk_arms}, L["risk"][idx], L["resid"][idx],
                                    L["harmful"][idx], Boot(L["problem_key"][idx], 500) if name == "all" else None,
                                    None if (null2 is None or name != "all") else {k: null2[k][:, idx] for k in risk_arms})
        S["q2_e2"][name]["n"] = int(len(idx))
    S["q2_e2"]["B_calibration"] = {"spearman_before_err": spearman(P2["B_before"], L["err_before"])}

    # Q3
    from analyze_risk_probe import oof_logit
    base1 = np.column_stack([Fk["xfit_v2ds"], Fk["log_n_cont"], Fk["relpos"]])
    ex1 = {"A_last": P1["A_last"], "B": P1["B"], "A_last+B": np.column_stack([P1["A_last"], P1["B"]]),
           "A_mean": P1["A_mean"]}
    S["q3_e1"] = {tgt: q3_contrasts(base1, ex1, Fk[tgt], lambda X, y: oof_logit(X, y, seed=10), boot1)
                  for tgt in ("harmful", "fail")}
    base2 = np.column_stack([L["xfit5"], L["log_n_cont"], L["relpos"]])
    ex2 = {"A_last": P2["A_last"], "B": P2["B"], "A_last+B": np.column_stack([P2["A_last"], P2["B"]]),
           "A_mean": P2["A_mean"]}
    S["q3_e2"] = {"harmful": q3_contrasts(base2, ex2, L["harmful"], lambda X, y: oof_by_fold(X, y, fold), boot2,
                                          levels=(0.95,))}

    # A against B
    def ab(P, y, boot):
        d = (spearman(P["B"], y) or np.nan) - (spearman(P["A_last"], y) or np.nan)
        return {"delta_spearman_B_minus_A": d,
                "ci95": boot.ci(lambda k: (spearman(P["B"][k], y[k]) or np.nan) - (spearman(P["A_last"][k], y[k]) or np.nan))}
    S["a_vs_b"] = {"e1": ab(P1, Fk["risk_diff"], boot1), "e2": ab(P2, L["risk"], Boot(L["problem_key"], 500))}

    # prespecified reading (section 6)
    R = {}
    for arm in PRIMARY:
        e1c = S["q3_e1"]["harmful"][arm]["delta_ci97.5"]
        e2c = S["q3_e2"]["harmful"][arm]["delta_ci95"]
        R[arm] = {"e1_delta_ci97.5_above0": above0(e1c), "e2_delta_ci95_above0": above0(e2c)}
    adds = any(v["e1_delta_ci97.5_above0"] and v["e2_delta_ci95_above0"] for v in R.values())
    e2_only = any(v["e2_delta_ci95_above0"] and not v["e1_delta_ci97.5_above0"] for v in R.values())
    abe = S["a_vs_b"]
    b_better = above0(abe["e2"]["ci95"]) and (abe["e1"]["delta_spearman_B_minus_A"] or 0) > 0
    a_better = (abe["e2"]["ci95"] is not None and abe["e2"]["ci95"][1] < 0
                and (abe["e1"]["delta_spearman_B_minus_A"] or 0) < 0)
    R["verdict"] = ("risk adds to validity at this scale" if adds else
                    "E2 only: helps on the label distribution, not shown on the forks" if e2_only else
                    "null at this scale")
    R["target"] = "B better" if b_better else "A better" if a_better else "no difference shown"
    R["interpretable"] = {"e1": S["gates"]["e1_interpretable"], "e2": S["gates"]["e2_interpretable"]}
    S["reading"] = R

    (a.out_dir / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    (a.out_dir / "summary.md").write_text(render(S))
    with open(a.out_dir / "step_scores.jsonl", "w") as f:
        for j, v in enumerate(e2z["view_id"]):
            f.write(json.dumps({"view_id": str(v), "xfit5": float(L["xfit5"][j]),
                                **{arm: float(P2[arm][j]) for arm in arms}}) + "\n")
    print(render(S))


def render(S: dict) -> str:
    f = lambda x: "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))  # noqa: E731
    ci = lambda c: "n/a" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"  # noqa: E731
    g = S["gates"]
    L = ["# risk_probe_v1 summary", "",
         f"**Reading:** {S['reading']['verdict']}; target: {S['reading']['target']}. "
         f"Interpretable: E1 {g['e1_interpretable']}, E2 {g['e2_interpretable']}.", "",
         "## Gates", "",
         f"* labels: {g['labels']}",
         f"* features: {g['reproduction_features']}",
         f"* validity rescoring: pass {g['reproduction_scores']['pass']}",
         f"* alignment: {g['alignment']}",
         f"* leakage: pass {g['leakage']['pass']}", "",
         "## Q3: harmful detection, validity score plus a risk readout", "",
         "| set | readout | AUROC validity | AUROC + readout | delta | CI |", "|---|---|---|---|---|---|"]
    for setn, key, lv in (("E1 forks", "q3_e1", "delta_ci97.5"), ("E2 label set", "q3_e2", "delta_ci95")):
        q = S[key]["harmful"]
        for arm in ("A_last", "B", "A_last+B", "A_mean"):
            L.append(f"| {setn} | {arm} | {f(q['auroc_validity'])} | {f(q[arm]['auroc'])} | {f(q[arm]['delta'])} | "
                     f"{ci(q[arm].get(lv))} ({lv.replace('delta_ci', '')}%) |")
    L += ["", "## Q2: out-of-sample risk readouts", "",
          "| set | arm | Spearman risk (null95) | Spearman resid (null95) | AUROC harmful (null95) |", "|---|---|---|---|---|"]
    for setn, q in (("E1", S["q2_e1"]), ("E2 all", S["q2_e2"]["all"])):
        for arm in ("A_last", "A_mean", "B", "A_last_res", "A_mean_res", "A_last_h", "A_mean_h"):
            d = q[arm]
            L.append(f"| {setn} | {arm} | {f(d.get('spearman_risk'))} ({f(d.get('null95_spearman_risk'))}) | "
                     f"{f(d.get('spearman_resid'))} ({f(d.get('null95_spearman_resid'))}) | "
                     f"{f(d.get('auroc_harmful'))} ({f(d.get('null95_auroc_harmful'))}) |")
    L += ["", "E2 subsets (Spearman risk, A_last / B): " + ", ".join(
        f"{k} {f(v['A_last']['spearman_risk'])} / {f(v['B']['spearman_risk'])} (n={v['n']})"
        for k, v in S["q2_e2"].items() if isinstance(v, dict) and "A_last" in v),
          f"B calibration: E1 {S['q2_e1']['B_calibration']}, E2 {S['q2_e2']['B_calibration']}", "",
          "## Q1 (E2): cross-fitted validity probe within DeepSeek-incorrect steps", "",
          f"```\n{json.dumps(S['q1_e2'], indent=1, default=float)}\n```", "",
          "## A against B (Spearman with risk, B minus A_last)", "",
          f"```\n{json.dumps(S['a_vs_b'], indent=1, default=float)}\n```"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
