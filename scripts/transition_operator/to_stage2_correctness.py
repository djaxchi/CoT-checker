"""transition_operator_v0 Stage 2 correctness probes on frozen z (+ baselines).

Two evaluations on the problem-disjoint splits (correct=1 / wrong=0, balanced by
fork construction, so chance = 50%):

1. GLOBAL probe: logistic regression on train, evaluated on test.
   ROC-AUC, balanced accuracy, macro-F1, Brier (calibration), train-vs-test AUC.

2. WITHIN-FORK ranking: with the SAME trained w, score both siblings of every test
   fork, s+ = w.z_correct, s- = w.z_wrong. Report pair accuracy P(s+ > s-), mean
   margin, and a paired bootstrap 95% CI over forks. Cleanest test: question+prefix
   identical across siblings.

Both are repeated after residualizing surface features (step length, equation
presence, numeric-token count, final-character type) out of the representation
(linear fit on train only).

Representations: z_A/z_B/z_AB (per seed, aggregated), max/mean-pool(H_t), S_t,
S_t - S_{t-1}, Target-A effect (residualized dL PCA-64), Target-B effect (d_belief).

  python scripts/transition_operator/to_stage2_correctness.py --run_dir runs/transition_operator
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

_NUM = re.compile(r"\d")
_FINAL = ("digit", "letter", "punct", "other")


def surface_matrix(texts, n_steps) -> np.ndarray:
    rows = []
    for t, n in zip(texts, n_steps):
        s = t.rstrip()
        last = s[-1] if s else " "
        cls = ("digit" if last.isdigit() else "letter" if last.isalpha()
               else "punct" if not last.isalnum() and not last.isspace() else "other")
        rows.append([float(n), 1.0 if "=" in t else 0.0,
                     float(len(_NUM.findall(t))),
                     *[1.0 if cls == c else 0.0 for c in _FINAL]])
    return np.asarray(rows, np.float32)


def residualize(X, surf, train_mask):
    Sb = np.concatenate([surf, np.ones((len(surf), 1), np.float32)], 1)
    beta, *_ = np.linalg.lstsq(Sb[train_mask], X[train_mask], rcond=None)
    return X - Sb @ beta


def mmap_member(npz_path, member, tmp):
    out = tmp / f"{member}.npy"
    if not out.exists():
        subprocess.run(["unzip", "-o", "-q", str(npz_path), f"{member}.npy",
                        "-d", str(tmp)], check=True)
    return np.load(out, mmap_mode="r")


def global_probe(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    pte = clf.predict_proba(sc.transform(X[te]))[:, 1]
    ptr = clf.predict_proba(sc.transform(X[tr]))[:, 1]
    return {
        "auc_test": float(roc_auc_score(y[te], pte)),
        "auc_train": float(roc_auc_score(y[tr], ptr)),
        "bal_acc_test": float(balanced_accuracy_score(y[te], (pte > 0.5))),
        "macro_f1_test": float(f1_score(y[te], (pte > 0.5), average="macro")),
        "brier_test": float(brier_score_loss(y[te], pte)),
    }, sc, clf


def within_fork(X, sc, clf, corr_idx, wrong_idx, seed=0):
    w_dot = lambda idx: clf.decision_function(sc.transform(X[idx]))  # noqa: E731
    sp, sm = w_dot(corr_idx), w_dot(wrong_idx)
    diff = sp - sm
    pair_acc = float((diff > 0).mean())
    margin = float(diff.mean())
    rng = np.random.default_rng(seed)
    accs, margins = [], []
    for _ in range(2000):
        b = rng.integers(0, len(diff), len(diff))
        accs.append((diff[b] > 0).mean())
        margins.append(diff[b].mean())
    return {
        "pair_acc": pair_acc,
        "pair_acc_ci": [float(np.percentile(accs, 2.5)),
                        float(np.percentile(accs, 97.5))],
        "margin": margin,
        "margin_ci": [float(np.percentile(margins, 2.5)),
                      float(np.percentile(margins, 97.5))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    a = args.run_dir / "stage2" / "arrays"
    trans_rows = json.loads((a / "trans_rows.json").read_text())
    fork_rows = json.loads((a / "fork_rows.json").read_text())
    fpos = {r["fork_id"]: i for i, r in enumerate(fork_rows)}
    splits = json.loads((args.run_dir / "splits.json").read_text())
    train_f, test_f = set(splits["train"]) & set(fpos), set(splits["test"]) & set(fpos)

    N = len(trans_rows)
    y = np.array([1 if r["branch"] == "correct" else 0 for r in trans_rows])
    in_train = np.array([r["fork_id"] in train_f for r in trans_rows])
    in_test = np.array([r["fork_id"] in test_f for r in trans_rows])
    tr, te = np.where(in_train)[0], np.where(in_test)[0]
    f_of = np.array([fpos[r["fork_id"]] for r in trans_rows])
    # sibling indices per fork for within-fork ranking (test only)
    by_fork = {}
    for i, r in enumerate(trans_rows):
        by_fork.setdefault(r["fork_id"], {})[r["branch"]] = i
    test_forks = [fk for fk in splits["test"]
                  if fk in by_fork and "correct" in by_fork[fk]
                  and "wrong" in by_fork[fk]]
    corr_idx = np.array([by_fork[fk]["correct"] for fk in test_forks])
    wrong_idx = np.array([by_fork[fk]["wrong"] for fk in test_forks])
    print(f"N={N} train={len(tr)} test={len(te)} test_forks={len(test_forks)}",
          flush=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        S_prev_all = mmap_member(a / "fork_arrays.npz", "S_prev", tmp)
        S_post_all = mmap_member(a / "trans_arrays.npz", "S_post", tmp)
        H_all = mmap_member(a / "trans_arrays.npz", "H_steps", tmp)
        n_all = np.asarray(mmap_member(a / "trans_arrays.npz", "n_steps", tmp))
        surf = surface_matrix([r["text"] for r in trans_rows], n_all)
        S_post = np.asarray(S_post_all[:], np.float32)
        S_prev = np.asarray(S_prev_all[f_of], np.float32)
        hidden = S_post.shape[1]
        meanpool = np.zeros((N, hidden), np.float32)
        maxpool = np.zeros((N, hidden), np.float32)
        for lo in range(0, N, 128):
            Hc = np.asarray(H_all[lo:lo + 128], np.float32)
            nc = n_all[lo:lo + 128]
            m = (np.arange(Hc.shape[1])[None] < nc[:, None])[..., None]
            meanpool[lo:lo + len(Hc)] = np.where(m, Hc, 0.0).sum(1) / np.maximum(nc[:, None], 1)
            maxpool[lo:lo + len(Hc)] = np.where(m, Hc, -np.inf).max(1)

    effects = np.load(args.run_dir / "stage2" / "effects.npy")
    reps = {
        "maxpool": maxpool, "meanpool": meanpool, "S_t": S_post,
        "delta": S_post - S_prev,
        "targetA_effect": effects[:, :64], "targetB_effect": effects[:, 64:],
    }
    z_arms = {}
    for arm in ("A", "B", "AB"):
        zs = [np.load(p) for p in sorted(
            (args.run_dir / "stage2").glob(f"{arm}_seed*/z_all.npy"))]
        if zs:
            z_arms[arm] = zs

    def run_rep(X, tag):
        g, sc, clf = global_probe(X, y, tr, te)
        wf = within_fork(X, sc, clf, corr_idx, wrong_idx)
        Xr = residualize(X, surf, in_train)
        gr, scr, clfr = global_probe(Xr, y, tr, te)
        wfr = within_fork(Xr, scr, clfr, corr_idx, wrong_idx)
        return {"raw": {**g, **wf}, "residualized": {**gr, **wfr}}

    out = {}
    for name, X in reps.items():
        out[name] = run_rep(X, name)
        print(f"[{name}] done", flush=True)
    for arm, zs in z_arms.items():
        per = [run_rep(z, f"z_{arm}") for z in zs]
        agg = {}
        for cond in ("raw", "residualized"):
            keys = per[0][cond]
            agg[cond] = {}
            for k in keys:
                vals = [p[cond][k] for p in per]
                if isinstance(vals[0], list):
                    agg[cond][k] = [float(np.mean([v[0] for v in vals])),
                                    float(np.mean([v[1] for v in vals]))]
                else:
                    agg[cond][k + "_mean"] = float(np.mean(vals))
                    agg[cond][k + "_std"] = float(np.std(vals))
        out[f"z_{arm}"] = agg
        print(f"[z_{arm}] done ({len(zs)} seeds)", flush=True)

    dst = args.run_dir / "stage2" / "correctness_probes.json"
    dst.write_text(json.dumps(out, indent=2))

    def gv(d, cond, k):
        return d[cond].get(k, d[cond].get(k + "_mean", float("nan")))
    print(f"\n{'repr':16} | RAW auc  balAcc  pairAcc margin | RESID auc  pairAcc")
    for name, d in out.items():
        print(f"{name:16} | "
              f"{gv(d,'raw','auc_test'):.3f} {gv(d,'raw','bal_acc_test'):.3f} "
              f"{gv(d,'raw','pair_acc'):.3f} {gv(d,'raw','margin'):+.3f} | "
              f"{gv(d,'residualized','auc_test'):.3f} "
              f"{gv(d,'residualized','pair_acc'):.3f}")
    print(f"\n[correctness] wrote {dst}")


if __name__ == "__main__":
    main()
