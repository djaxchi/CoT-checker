"""parametric_retrieval_geometry_v0 exp 2 analysis: within-instance CoT
success geometry (local, after rollouts are merged + synced).

Two predictors of per-rollout success, at each trajectory checkpoint, with
fact-grouped 5-fold CV AUROC:

  raw          checkpoint hidden state (carries fact info + path info)
  within       instance-demeaned state: subtract, per instance, the mean over
               that instance's rollouts. This ERASES all static-prompt / fact
               information (identical across an instance's rollouts) and keeps
               only how THIS reasoning path differs from the instance's own
               average. If `within` predicts success, the reasoning path itself
               carries recoverable recall signal -- exactly the
               reasoning_unlocked signal the static prompt geometry could not
               see.

Reported per checkpoint (final_prompt_token = shared across an instance's
rollouts, so `within` there is ~0 by construction and serves as a sanity
floor; gen0 / sentence-end quartiles / pre-answer show whether and WHEN the
paths diverge). A rising `within` AUROC along the trajectory = genuine
mid-reasoning divergence; already-high at gen0 = fate set by the first tokens.

  python scripts/parametric_retrieval/prg_rollout_analysis.py \
      --out_dir runs/parametric_retrieval_geometry_v0 --hs 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STEP_ORDER = ["final_prompt_token", "first_generated_token", "se_q1", "se_q2",
              "se_q3", "se_q4", "token_before_final_answer",
              "first_final_answer_token"]


def canon_step(row, n_se):
    p = row.position_name
    if p == "sentence_end":
        frac = (row.position_rank + 1) / (n_se.get(
            (row.question_id, row.rollout_idx), 1) + 1)
        return f"se_q{min(3, int(frac * 4)) + 1}"
    return p


def cv_auc(X, yb, groups):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(yb)) < 2 or len(np.unique(groups)) < 5:
        return float("nan")
    k = min(128, X.shape[1], X.shape[0] - 5)
    aucs = []
    for tr, te in GroupKFold(5).split(X, yb, groups):
        if len(np.unique(yb[tr])) < 2 or len(np.unique(yb[te])) < 2:
            continue
        pipe = Pipeline([("sc", StandardScaler()),
                         ("pca", PCA(n_components=k, random_state=0)),
                         ("lr", LogisticRegression(max_iter=2000))])
        pipe.fit(X[tr], yb[tr])
        aucs.append(roc_auc_score(yb[te], pipe.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--hs", type=int, nargs="+", default=[20, 24])
    args = ap.parse_args()

    from safetensors.numpy import load_file

    rd = args.out_dir / "rollouts"
    meta = pd.read_parquet(rd / "ckpt_meta.parquet")
    n_se = (meta[meta.position_name == "sentence_end"]
            .groupby(["question_id", "rollout_idx"]).size().to_dict())
    meta = meta.reset_index(drop=True)
    meta["step"] = meta.apply(lambda r: canon_step(r, n_se), axis=1)

    # keep only instances that actually have BOTH a success and a failure
    inst_ok = (meta.groupby("question_id").success.nunique() == 2)
    meta = meta[meta.question_id.map(inst_ok)].copy()
    roll = pd.DataFrame([json.loads(ln) for ln in
                         (rd / "rollouts.jsonl").read_text().splitlines()
                         if ln.strip()])
    print(f"[roll-an] {meta.question_id.nunique()} mixed instances; "
          f"rollout success rate {roll.success.mean():.2f}; "
          f"{len(meta)} checkpoints")

    rows = []
    for K in args.hs:
        H = load_file(rd / f"layer_{K:02d}.safetensors")["h"].astype(np.float32)
        for step in STEP_ORDER:
            sub = meta[meta.step == step]
            if len(sub) < 60:
                continue
            X = H[sub.index.to_numpy()]
            yb = sub.success.to_numpy().astype(int)
            groups = sub.fact_id.to_numpy()
            # instance-demean (erase fact/prompt info)
            Xw = X.copy()
            for qid, idx in sub.groupby("question_id").groups.items():
                pos = sub.index.get_indexer(idx)
                Xw[pos] -= Xw[pos].mean(0, keepdims=True)
            rows.append({
                "hs_idx": K, "step": step, "n": len(sub),
                "n_instances": sub.question_id.nunique(),
                "success_rate": round(float(yb.mean()), 3),
                "auc_raw": round(cv_auc(X, yb, groups), 3),
                "auc_within": round(cv_auc(Xw, yb, groups), 3),
            })
            print(f"[roll-an] hs{K} {step:26s} raw={rows[-1]['auc_raw']} "
                  f"within={rows[-1]['auc_within']} (n={len(sub)})", flush=True)

    res = pd.DataFrame(rows)
    if res.empty:
        print("[roll-an] no checkpoints with >=2 outcomes; nothing to score "
              "(need mixed-outcome rollouts from the real model)")
        return
    res["step_order"] = res.step.map({s: i for i, s in enumerate(STEP_ORDER)})
    res = res.sort_values(["hs_idx", "step_order"])
    out = args.out_dir / "rollouts" / "rollout_success_geometry.csv"
    res.to_csv(out, index=False)
    print("\n==== within-instance CoT success AUROC ====")
    print(res.drop(columns="step_order").to_string(index=False))
    print(f"\n[roll-an] wrote {out}")


if __name__ == "__main__":
    main()
