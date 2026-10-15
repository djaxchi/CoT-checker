#!/usr/bin/env python3
"""Fit risk_probe_v1's linear readouts (docs/risk_probe_v1_plan.md, sections 2 and 4).

Arms, all on L35 rows of the label-set step views:

  A_last, A_mean          ridge on the step's risk (its last row / its mean row)
  A_last_res, A_mean_res  ridge on residual risk (validity, log continuation
                          length and relative position regressed out), v0's Q2
  A_last_h, A_mean_h      L2 logistic on `harmful`
  B                       binomial value probe V on (wrong rollouts, k) at each
                          cut: the row ahead of the step for cut s, the step's
                          last row for cut s + 1; risk = V(after) - V(before),
                          also kept as B_before and B_after

Hyperparameters come from 3 problem-grouped inner folds of each training set,
over v0's grids. Two evaluation sets:

  E2  5 outer folds by problem (risk_probe_v1.fold_of): out-of-fold predictions
      for every label step
  E1  one fit on all label steps, applied to the 531 fork views

--stage real writes preds_e2.npz, preds_e1.npz and hparams.json. --stage null
refits every arm with the real fit's hyperparameters on labels permuted across
traces of the same (fold, n_steps), for permutations [perm_start, perm_end), and
writes null_<start>_<end>.npz (predictions only; analyze.py scores them).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.risk_probe_v1 import (N_FOLDS, RidgeSolver, choose_C, choose_alpha,  # noqa: E402
                                        group_permutation, logistic_fit, residualize, standardize)

RIDGE_ARMS = {"A_last": ("after", "risk"), "A_mean": ("mean", "risk"),
              "A_last_res": ("after", "resid"), "A_mean_res": ("mean", "resid")}
LOGIT_ARMS = {"A_last_h": ("after", "harmful"), "A_mean_h": ("mean", "harmful")}
ARMS = list(RIDGE_ARMS) + list(LOGIT_ARMS) + ["B", "B_before", "B_after"]


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def assemble(rows: dict, lab_views: list[dict], step_values: list[dict], traces: dict[str, dict]) -> dict:
    """Join label steps to their rows and labels. Returns arrays in label-step order."""
    idx = {v: i for i, v in enumerate(rows["ids"].tolist())}
    sv = {(r["trace_id"], r["step"]): r for r in step_values}
    keep, recs = [], []
    for lv in lab_views:
        r = sv.get((lv["trace_id"], lv["step"]))
        if r is None or lv["view_id"] not in idx:
            continue
        t = traces[lv["trace_id"]]
        n_cont = len(t["reply_ids"]) - t["cut_offsets"][lv["step"] + 1]
        keep.append(idx[lv["view_id"]])
        recs.append({**r, "view_id": lv["view_id"], "fold": lv["fold"], "problem_key": lv["problem_key"],
                     "log_n_cont": math.log1p(max(n_cont, 0)), "n_steps": t["n_steps"]})
    D = {"view_id": np.array([r["view_id"] for r in recs]), "trace_id": np.array([r["trace_id"] for r in recs]),
         "step": np.array([r["step"] for r in recs]), "problem_key": np.array([r["problem_key"] for r in recs]),
         "fold": np.array([r["fold"] for r in recs]), "n_steps": np.array([r["n_steps"] for r in recs])}
    for k in ("risk", "se", "err_before", "err_after", "log_n_cont", "relpos"):
        D[k] = np.array([float(r[k]) for r in recs])
    D["harmful"] = np.array([int(r["harmful"]) for r in recs])
    D["k_before"] = np.array([float(r["k_before"]) for r in recs])
    D["k_after"] = np.array([float(r["k_after"]) for r in recs])
    D["wrong_before"] = np.rint(D["err_before"] * D["k_before"])
    D["wrong_after"] = np.rint(D["err_after"] * D["k_after"])
    D["mask_ds"] = np.array([bool(r["mask_ds"]) for r in recs])
    D["y_ds"] = np.array([int(r["y_ds"]) if r["mask_ds"] else -1 for r in recs])
    # residual risk for every step (a finite target for every row): the few steps
    # without a DeepSeek label take their self label; metrics use labeled steps only
    y_res = np.array([int(r["y_ds"]) if r["mask_ds"] else int(r.get("y_self") or 0) for r in recs])
    D["resid"] = residualize(D["risk"], [y_res, D["log_n_cont"], D["relpos"]])
    for pool in ("before", "after", "mean"):
        D[pool] = rows[pool][np.array(keep)] if keep else rows[pool][:0]
    return D


def permuted_targets(D: dict, rng: np.random.Generator) -> dict:
    """Labels moved between traces of the same (fold, n_steps), step position kept."""
    tids = list(dict.fromkeys(D["trace_id"].tolist()))
    pos = {}
    for i, (t, s) in enumerate(zip(D["trace_id"], D["step"])):
        pos[(t, int(s))] = i
    first = {t: i for i, t in reversed(list(enumerate(D["trace_id"].tolist())))}
    keys = [(int(D["fold"][first[t]]), int(D["n_steps"][first[t]])) for t in tids]
    perm = group_permutation(keys, rng)
    rows_of: dict = {}
    for i, t in enumerate(D["trace_id"].tolist()):
        rows_of.setdefault(t, []).append(i)
    src = np.arange(len(D["step"]))
    for ti, t in enumerate(tids):
        u = tids[perm[ti]]
        for i in rows_of[t]:
            src[i] = pos.get((u, int(D["step"][i])), i)
    out = dict(D)
    for k in ("risk", "resid", "harmful", "wrong_before", "wrong_after", "k_before", "k_after"):
        out[k] = D[k][src]
    return out


class Fitter:
    """Fits every arm on a training index set, predicts on given rows.

    Training tensors, their standardization and the ridge eigendecompositions
    depend only on (training set, pooling), so they are cached across arms and
    across null permutations.
    """

    def __init__(self, device: str):
        import torch
        self.torch, self.device = torch, device
        self.cache: dict = {}

    def data(self, key: str, X: np.ndarray):
        if key not in self.cache:
            Xt = self.torch.as_tensor(np.asarray(X, np.float32), device=self.device)
            self.cache[key] = {"X": Xt, "std": standardize(Xt)}
        return self.cache[key]

    def solver(self, key: str, X: np.ndarray) -> RidgeSolver:
        c = self.data(key, X)
        if "ridge" not in c:
            c["ridge"] = RidgeSolver(c["X"], std=c["std"])
        return c["ridge"]

    def fit_all(self, D: dict, tr: np.ndarray, key: str, hp: dict | None = None) -> tuple[dict, dict]:
        """Models for every arm on training rows tr; hp fixes the hyperparameters (null)."""
        hp_out, models = {}, {}
        g = D["problem_key"][tr]
        for arm, (pool, tgt) in RIDGE_ARMS.items():
            y = D[tgt][tr]
            a = choose_alpha(self.data(f"{key}/{pool}", D[pool][tr])["X"], y, g) if hp is None else hp[arm]
            models[arm], hp_out[arm] = self.solver(f"{key}/{pool}", D[pool][tr]).fit(y, a), a
        for arm, (pool, tgt) in LOGIT_ARMS.items():
            c, y = self.data(f"{key}/{pool}", D[pool][tr]), D[tgt][tr].astype(float)
            ones = np.ones(len(y))
            C = choose_C(c["X"], y, ones, g, "auroc") if hp is None else hp[arm]
            models[arm], hp_out[arm] = logistic_fit(c["X"], y, ones, C, std=c["std"]), C
        c = self.data(f"{key}/cuts", np.concatenate([D["before"][tr], D["after"][tr]]))
        k = np.concatenate([D["wrong_before"][tr], D["wrong_after"][tr]])
        n = np.concatenate([D["k_before"][tr], D["k_after"][tr]])
        C = choose_C(c["X"], k, n, np.concatenate([g, g]), "deviance") if hp is None else hp["B"]
        models["B"], hp_out["B"] = logistic_fit(c["X"], k, n, C, std=c["std"]), C
        return models, hp_out

    @staticmethod
    def predict(models: dict, rows: dict) -> dict:
        out = {}
        for arm, (pool, _) in {**RIDGE_ARMS, **LOGIT_ARMS}.items():
            out[arm] = models[arm].predict(rows[pool])
        out["B_before"] = models["B"].predict(rows["before"])
        out["B_after"] = models["B"].predict(rows["after"])
        out["B"] = out["B_after"] - out["B_before"]
        return out


def run(D: dict, forks: dict, fitter: Fitter, hp: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """E2 out-of-fold predictions [n_lab, arms], E1 fork predictions [n_fork, arms], hyperparameters."""
    e2 = np.full((len(D["step"]), len(ARMS)), np.nan)
    hps = {}
    for o in range(N_FOLDS):
        tr, te = np.where(D["fold"] != o)[0], np.where(D["fold"] == o)[0]
        if not len(te):
            continue
        models, hps[f"fold{o}"] = fitter.fit_all(D, tr, f"fold{o}", None if hp is None else hp[f"fold{o}"])
        p = fitter.predict(models, {k: D[k][te] for k in ("before", "after", "mean")})
        for j, arm in enumerate(ARMS):
            e2[te, j] = p[arm]
    models, hps["full"] = fitter.fit_all(D, np.arange(len(D["step"])), "full", None if hp is None else hp["full"])
    p = fitter.predict(models, forks)
    e1 = np.column_stack([p[arm] for arm in ARMS])
    return e2, e1, hps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True, help="dump_rows.py output")
    ap.add_argument("--lab_views", type=Path, required=True)
    ap.add_argument("--step_values", type=Path, required=True)
    ap.add_argument("--trace_set", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--stage", choices=("real", "null"), default="real")
    ap.add_argument("--perm_start", type=int, default=0)
    ap.add_argument("--perm_end", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    z = np.load(a.rows)
    rows = {k: z[k] for k in ("ids", "split", "before", "after", "mean")}
    traces = {t["trace_id"]: t for t in json.loads(a.trace_set.read_text())["traces"]}
    D = assemble(rows, jl(a.lab_views), jl(a.step_values), traces)
    fk = np.where(rows["split"] == "fsl_forks")[0]
    forks = {k: rows[k][fk] for k in ("before", "after", "mean")}
    fork_ids = rows["ids"][fk]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fitter = Fitter(device)
    print(f"[fit] {len(D['step'])} label steps, {len(fk)} forks, device {device}", flush=True)
    if a.stage == "real":
        e2, e1, hps = run(D, forks, fitter)
        np.savez(a.out_dir / "preds_e2.npz", view_id=D["view_id"], arms=np.array(ARMS), pred=e2)
        np.savez(a.out_dir / "preds_e1.npz", view_id=fork_ids, arms=np.array(ARMS), pred=e1)
        (a.out_dir / "hparams.json").write_text(json.dumps({"hparams": hps, "n_label_steps": int(len(D["step"])),
                                                            "n_forks": int(len(fk)), "device": device,
                                                            "seconds": time.time() - t0}, indent=2))
        print(f"[fit] real done in {time.time() - t0:.0f}s", flush=True)
        return
    hp = json.loads((a.out_dir / "hparams.json").read_text())["hparams"]
    E2, E1 = [], []
    for p in range(a.perm_start, a.perm_end):
        Dp = permuted_targets(D, np.random.default_rng(10_000 + p))
        e2, e1, _ = run(Dp, forks, fitter, hp)
        E2.append(e2.astype(np.float16)); E1.append(e1.astype(np.float16))
        print(f"[null] perm {p} done ({time.time() - t0:.0f}s)", flush=True)
    np.savez(a.out_dir / f"null_{a.perm_start:03d}_{a.perm_end:03d}.npz", perms=np.arange(a.perm_start, a.perm_end),
             arms=np.array(ARMS), e2=np.stack(E2), e1=np.stack(E1))


if __name__ == "__main__":
    main()
