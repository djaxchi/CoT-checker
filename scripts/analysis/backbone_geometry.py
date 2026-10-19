#!/usr/bin/env python3
"""Closed-form geometry of pooled step vectors across backbones, layers and formats.

Reads the shards written by scripts/encode_pooled_multilayer.py for one or more
configs and, for every (config, layer, readout), fits on the PRM800K training
subsample only:

    lda       class means + shrunk pooled covariance, shrinkage by val AUROC
    mean_axis the raw class-mean difference, no whitening
    best_pc   the single top-50 principal component with the best val AUROC

and reports PRM800K test AUROC, ProcessBench known-step AUROC and calib-20 F1_PB
(four-subset means). For a prm_native config it also scores the PRM's own head
(suspicion = 1 - P(correct) at the step's separator), the reproduction gate for
§21.15 (calib-20 0.701). Plan and prespecified readings:
docs/backbone_geometry_v0_plan.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.one_class_correct import Gauss  # noqa: E402
from scripts.merge_rep_grid_leaderboard import calib20_subset  # noqa: E402

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
SHRINK = (1e-4, 1e-3, 1e-2, 0.1, 0.5)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_split(d: Path, name: str) -> tuple[dict, list[dict]]:
    shards = sorted(glob.glob(str(d / f"{name}.shard*.npz")),
                    key=lambda p: int(p.split(".shard")[1].split(".")[0]))
    if not shards:
        raise FileNotFoundError(f"{d}/{name}")
    parts = [dict(np.load(s)) for s in shards]
    metas = [[json.loads(l) for l in open(s.replace(".npz", ".meta.jsonl"))] for s in shards]
    out = {"layers": parts[0]["layers"]}
    for k in parts[0]:
        if k != "layers":
            out[k] = np.concatenate([p[k] for p in parts])
    meta = [m for ms in metas for m in ms]
    kept = out["kept"].astype(bool)
    for k in list(out):
        if k != "layers":
            out[k] = out[k][kept]
    return out, [m for m, k in zip(meta, kept) if k]


def pb_eval(score: np.ndarray, y: np.ndarray, meta: list[dict]) -> tuple[float, float]:
    known = np.array([m["first_error"] == -1 or m["step_idx"] <= m["first_error"] for m in meta])
    au = roc_auc_score(y[known], score[known])
    by: dict[str, list] = {}
    fe: dict[str, int] = {}
    rk = (np.argsort(np.argsort(score)) + 0.5) / len(score)
    for m, s in zip(meta, rk):
        by.setdefault(m["id"], []).append((m["step_idx"], float(s)))
        fe[m["id"]] = int(m["first_error"])
    traces = [(fe[k], [s for _, s in sorted(v)]) for k, v in by.items()]
    return float(au), float(calib20_subset(traces))


def evaluate(scorer, data: dict) -> dict:
    r = {"test_auroc": float(roc_auc_score(data["test"][1], scorer(data["test"][0])))}
    aus, cs = [], []
    for sub in SUBSETS:
        x, y, meta = data[sub]
        au, c = pb_eval(scorer(x), y, meta)
        r[f"pb_{sub}_auroc"], r[f"pb_{sub}_calib20"] = au, c
        aus.append(au); cs.append(c)
    r["pb_auroc"], r["pb_calib20"] = float(np.mean(aus)), float(np.mean(cs))
    return r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", required=True, help="name:dir")
    p.add_argument("--reps", nargs="+", default=["mean", "last", "boundary", "sep"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in a.configs:
        cname, d = spec.split(":", 1)
        d = Path(d)
        S = {"train": load_split(d, "train"), "val": load_split(d, "val"), "test": load_split(d, "test")}
        for sub in SUBSETS:
            S[sub] = load_split(d, f"pb_{sub}")
        layers = [int(l) for l in S["train"][0]["layers"]]
        if not np.isnan(S["test"][0]["head_p_correct"]).all():
            sc = lambda key: 1 - S[key][0]["head_p_correct"]  # noqa: E731
            r = {"test_auroc": float(roc_auc_score(S["test"][0]["y"], sc("test")))}
            cs, aus = [], []
            for sub in SUBSETS:
                au, c = pb_eval(sc(sub), S[sub][0]["y"], S[sub][1])
                r[f"pb_{sub}_calib20"] = c; cs.append(c); aus.append(au)
            r.update(pb_auroc=float(np.mean(aus)), pb_calib20=float(np.mean(cs)),
                     config=cname, layer="head", rep="head", method="prm_head")
            rows.append(r)
            log(f"{cname} PRM head: test={r['test_auroc']:.4f} pb_calib20={r['pb_calib20']:.4f}")
        for rep in a.reps:
            if rep not in S["train"][0]:
                continue
            for li, layer in enumerate(layers):
                X = {k: (S[k][0][rep][:, li].astype(np.float64), S[k][0]["y"], S[k][1]) for k in S}
                xtr, ytr = X["train"][0], X["train"][1]
                mu1, mu0 = xtr[ytr == 1].mean(0), xtr[ytr == 0].mean(0)
                pooled = Gauss(np.concatenate([xtr[ytr == 1] - mu1, xtr[ytr == 0] - mu0]))
                pooled.mu = xtr.mean(0)
                dmu = mu1 - mu0
                dmu_e = dmu @ pooled.evecs

                def lda(x, lam):
                    w = (1 - lam) * pooled.evals + lam * pooled.tr
                    return ((x - pooled.mu) @ pooled.evecs) @ (dmu_e / w)
                va = {lam: roc_auc_score(X["val"][1], lda(X["val"][0], lam)) for lam in SHRINK}
                lam = max(va, key=va.get)
                data = {"test": X["test"][:2], **{s: X[s] for s in SUBSETS}}
                res = {"lda": evaluate(lambda x: lda(x, lam), data) | {"shrink": lam}}
                res["mean_axis"] = evaluate(lambda x: (x - pooled.mu) @ dmu, data)
                top = pooled.evecs[:, ::-1][:, :50]
                pc_va = [roc_auc_score(X["val"][1], (X["val"][0] - pooled.mu) @ top[:, j]) for j in range(50)]
                j = int(np.argmax(np.abs(np.array(pc_va) - 0.5)))
                sign = 1.0 if pc_va[j] >= 0.5 else -1.0
                res["best_pc"] = evaluate(lambda x: sign * ((x - pooled.mu) @ top[:, j]), data) | {"pc": j}
                for m, r in res.items():
                    rows.append(r | {"config": cname, "layer": layer, "rep": rep, "method": m})
                log(f"{cname} L{layer} {rep}: lda test={res['lda']['test_auroc']:.4f} "
                    f"pb_calib20={res['lda']['pb_calib20']:.4f} | best_pc test={res['best_pc']['test_auroc']:.4f}")
                (a.out / "geometry.json").write_text(json.dumps(rows, indent=1))
    log("done")


if __name__ == "__main__":
    main()
