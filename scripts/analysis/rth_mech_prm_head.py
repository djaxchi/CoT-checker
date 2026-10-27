#!/usr/bin/env python3
"""M5: how much of the PRM's own verdict the read state carries, per layer (read_token_v1, Stage G).

Target: Qwen2.5-Math-PRM-7B's own head on each ProcessBench step (log-odds of
"incorrect", prm_backbone_v1/prm_head_scores/pb_step_scores_<subset>.jsonl, read
at its own <extra_0> positions in its own format). Inputs, from the mechanistic
bundle's ProcessBench splits: every row of the read token, of the Qwen3-8B last step
token and of the PRM's last step token under the verifier template (a reference),
and RTH's head logit alone (a 1-D baseline).

Per row: ridge, 5-fold cross-validation grouped by trace (all four subsets pooled,
all steps), out-of-fold R2. The ridge penalty is picked per row on the same folds'
mean R2 out of six values, a small optimism stated in the summary.

Then on the steps where RTH and the PRM disagree and ProcessBench labels the step
(the first error, or a step before it): which of the two is right, and which side
the out-of-fold prediction from the read state (and from the last step token)
takes. Verdicts use a threshold per scorer that flags as many labelled steps as
there are first errors ("rate", the headline: RTH's P = 0.5 transfers badly to
ProcessBench), and P = 0.5 ("p05"). If the read state predicts the PRM's verdict on
those steps, the PRM's information is present at the read position and RTH's head does not use it.

    python scripts/analysis/rth_mech_prm_head.py --bundle <mech>/bundle \
        --prm_scores /scratch/d/dchikhi/cot_mech/prm_backbone_v1/prm_head_scores --out m5_prm_head.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.prm_geometry import pb_step_class  # noqa: E402
from scripts.read_token_mech_bundle import load_bundle  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
RELS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def prm_targets(path: Path, meta: list[dict], subset: str) -> np.ndarray:
    """The PRM head's log-odds for each bundle step, in bundle order."""
    by = {}
    for r in map(json.loads, (path / f"pb_step_scores_{subset}.jsonl").read_text().splitlines()):
        for j, v in enumerate(r["log_odds_incorrect"]):
            by[(r["id"], j)] = float(v)
    missing = [(m["id"], int(m["step_idx"])) for m in meta if (m["id"], int(m["step_idx"])) not in by]
    if missing:
        raise SystemExit(f"{subset}: {len(missing)} steps without a PRM head score, e.g. {missing[:3]}")
    return np.array([by[(m["id"], int(m["step_idx"]))] for m in meta])


def group_folds(groups: list[str], k: int, seed: int = 0) -> np.ndarray:
    u = sorted(set(groups))
    perm = np.random.default_rng(seed).permutation(len(u))
    f = {g: int(perm[i]) % k for i, g in enumerate(u)}
    return np.array([f[g] for g in groups])


def cv_ridge(X: np.ndarray, y: np.ndarray, folds: np.ndarray, dev, rels=RELS) -> dict:
    """Out-of-fold ridge predictions for each penalty; returns the best by pooled R2."""
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev)
    yt = torch.as_tensor(np.asarray(y, np.float64), device=dev)
    oof = torch.zeros((len(rels), len(y)), dtype=torch.float64, device=dev)
    for f in np.unique(folds):
        tr = torch.as_tensor(folds != f, device=dev)
        Xa, ya = Xt[tr].double(), yt[tr]
        mu, ym = Xa.mean(0), ya.mean()
        Xc = Xa - mu
        e, V = torch.linalg.eigh(Xc.T @ Xc)
        proj = V.T @ (Xc.T @ (ya - ym))
        Xb = Xt[~tr].double() - mu
        XbV = Xb @ V
        for i, r in enumerate(rels):
            oof[i, ~tr] = XbV @ (proj / (e + r * e.mean())) + ym
    ss = ((yt - yt.mean()) ** 2).sum()
    r2 = (1 - ((oof - yt[None]) ** 2).sum(1) / ss).cpu().numpy()
    i = int(np.argmax(r2))
    return {"r2": float(r2[i]), "rel": rels[i], "oof": oof[i].cpu().numpy()}


def verdicts(s: np.ndarray, cls: np.ndarray, mode: str) -> np.ndarray:
    """"incorrect" verdicts: s > 0 (P = 0.5) or, with mode "rate", s above the quantile
    that flags as many labelled steps as ProcessBench labels as first errors (removes
    each scorer's calibration, which differs on ProcessBench)."""
    if mode == "p05":
        return s > 0
    lab = cls >= 0
    return s > np.quantile(s[lab], 1 - (cls[lab] == 1).mean())


def disagreement(rth_logit: np.ndarray, prm_lo: np.ndarray, cls: np.ndarray, preds: dict,
                 mode: str = "rate") -> dict:
    """On labelled steps where RTH and the PRM disagree: who is right, and which side each
    out-of-fold prediction of the PRM's log-odds takes (each thresholded the same way)."""
    rth_v, prm_v = verdicts(rth_logit, cls, mode), verdicts(prm_lo, cls, mode)
    sel = (rth_v != prm_v) & (cls >= 0)
    truth = cls[sel] == 1
    out = {"threshold": mode, "n_labelled": int((cls >= 0).sum()), "n_disagree": int(sel.sum()),
           "agreement_rate": float((rth_v == prm_v)[cls >= 0].mean()),
           "rth_right": float((rth_v[sel] == truth).mean()), "prm_right": float((prm_v[sel] == truth).mean()),
           "rth_flags_prm_not": int((rth_v & ~prm_v & sel).sum()),
           "prm_flags_rth_not": int((prm_v & ~rth_v & sel).sum()), "pred_sides_with_prm": {}}
    for name, p in preds.items():
        out["pred_sides_with_prm"][name] = float((verdicts(p, cls, mode)[sel] == prm_v[sel]).mean())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--prm_scores", type=Path, required=True)
    p.add_argument("--reps", default="read,last_step,prm")
    p.add_argument("--rows", default="all", help="comma list of rows, or all")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    reps = a.reps.split(",")
    X = {r: [] for r in reps}
    y, cls, rth, groups = [], [], [], []
    for sub in PB_SUBSETS:
        b = load_bundle(a.bundle / sub, arrays=tuple(reps) + ("head_logit",), mmap=False)
        y.append(prm_targets(a.prm_scores, b["meta"], sub))
        cls.append(pb_step_class(b["meta"]))
        rth.append(np.asarray(b["head_logit"], np.float64))
        groups += [f"{sub}/{m['id']}" for m in b["meta"]]
        for r in reps:
            X[r].append(b[r])
        print(f"[m5] loaded {sub}: {len(b['meta'])} steps ({time.time() - t0:.0f}s)", flush=True)
    X = {r: np.concatenate(v) for r, v in X.items()}
    y, cls, rth = np.concatenate(y), np.concatenate(cls), np.concatenate(rth)
    folds = group_folds(groups, a.folds)
    out = {"n_steps": int(len(y)), "n_traces": len(set(groups)), "folds": a.folds, "rels": list(RELS),
           "pearson_rth_prm": float(np.corrcoef(rth, y)[0, 1]), "reps": {}}
    base = cv_ridge(rth[:, None], y, folds, dev, rels=(1e-8,))
    out["rth_logit_only_r2"] = base["r2"]
    print(f"[m5] {len(y)} steps, {len(set(groups))} traces; RTH logit alone R2 {base['r2']:.3f}, "
          f"Pearson(RTH, PRM) {out['pearson_rth_prm']:.3f}", flush=True)
    keep = {"rth_logit_only": base["oof"]}
    for r in reps:
        rows = range(X[r].shape[1]) if a.rows == "all" else [int(x) for x in a.rows.split(",")]
        res = {}
        for l in rows:
            c = cv_ridge(X[r][:, l], y, folds, dev)
            res[str(l)] = {"r2": c["r2"], "rel": c["rel"]}
            if l == X[r].shape[1] - 1:
                keep[f"{r}_final"] = c["oof"]
            print(f"[m5] {r} row {l}: R2 {c['r2']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        best = max(res, key=lambda k: res[k]["r2"])
        out["reps"][r] = {"rows": res, "best_row": int(best), "best_r2": res[best]["r2"]}
        if best != str(X[r].shape[1] - 1):
            keep[f"{r}_best_row{best}"] = cv_ridge(X[r][:, int(best)], y, folds, dev)["oof"]
    out["disagreement"] = {m: disagreement(rth, y, cls, keep, m) for m in ("rate", "p05")}
    np.savez(a.out.with_suffix(".oof.npz"), y=y, cls=cls, rth=rth, groups=np.array(groups), **keep)
    out["oof_r2_check"] = {k: float(1 - ((v - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()) for k, v in keep.items()}
    print(f"[m5] disagreement: {json.dumps(out['disagreement'])}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
