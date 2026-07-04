"""parametric_retrieval_geometry_v0: separation search ("twist and turn").

Systematically searches representations built from the extracted hidden
states for ones that separate the four retrieval classes, or at least
direct_retrieval vs non_retrieved. Two families:

  state reps        prompt-token states per layer, multi-layer stacks,
                    cot-minus-direct prompt deltas, answer-token states
  trajectory reps   CoT position stacks [prompt, gen0, answer], displacement
                    vectors, sentence-end means, engineered path-shape
                    features (segment norms, path/net meander, curvature)

Transforms: raw, L2, family-centered, popularity-residualized (log gbc +
family regressed out of every dim; label-free so no class leakage).

Scores per representation:
  ch4 / ch2         Calinski-Harabasz for 4 classes / dr-vs-nr (unsupervised)
  auc2              GroupKFold(5, by fact_id) CV AUROC, direct_retrieval vs
                    non_retrieved, scaler->PCA(<=256)->logistic
  bacc4 / auc4      4-class balanced accuracy / macro OVR AUROC (same CV)

A surface-feature baseline row (log gbc, family, generation lengths, marker
flag, sentence count) anchors what "cheap confounds" already achieve; a rep
only matters if it clears that. AUROC (not F1) because both label sets here
are explicit head-to-head class comparisons, not deployment thresholds.

Also prints the Test 5 answer: within-gbc-bin CH + AUROC for the best rep.

Outputs: geometry/separation_search.csv + printed leaderboard.

  python scripts/parametric_retrieval/prg_separation_search.py \
      --out_dir runs/parametric_retrieval_geometry_v0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLASSES = ["direct_retrieval", "reasoning_unlocked", "unstable_retrieval",
           "non_retrieved"]
LAYERS = [4, 8, 12, 16, 20, 24, 28]
STACK = [12, 16, 20, 24, 28]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def ch_score(X: np.ndarray, y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    total = float((Xc ** 2).sum())
    between = 0.0
    classes = np.unique(y)
    for c in classes:
        m = y == c
        s = Xc[m].sum(axis=0)
        between += float(s @ s) / m.sum()
    within = total - between
    c_, n = len(classes), len(y)
    if within <= 0 or c_ < 2:
        return float("nan")
    return (between / (c_ - 1)) / (within / (n - c_))


def reduce_dims(X: np.ndarray, k: int = 256) -> np.ndarray:
    """Global (label-free) scale + PCA so the CV loop only refits the
    classifier; PCA fit uses no labels, so no class leakage."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    if X.shape[1] <= k:
        return X
    Xs = StandardScaler().fit_transform(X)
    return PCA(n_components=min(k, X.shape[0] - 5),
               random_state=0).fit_transform(Xs)


def make_pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=3000, C=1.0))])


def cv_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              binary_pair=("direct_retrieval", "non_retrieved")) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    out = {}
    # ---- binary dr vs nr ---------------------------------------------------
    mb = np.isin(y, binary_pair)
    Xb, yb, gb = X[mb], (y[mb] == binary_pair[0]).astype(int), groups[mb]
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(Xb, yb, gb):
        pipe = make_pipe()
        pipe.fit(Xb[tr], yb[tr])
        p = pipe.predict_proba(Xb[te])[:, 1]
        aucs.append(roc_auc_score(yb[te], p))
    out["auc2"] = float(np.mean(aucs))
    out["auc2_std"] = float(np.std(aucs))

    # ---- 4-class -----------------------------------------------------------
    preds = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(CLASSES)))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        pipe = make_pipe()
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])
        pr = pipe.predict_proba(X[te])
        cols = {c: i for i, c in enumerate(pipe.classes_)}
        for j, c in enumerate(CLASSES):
            if c in cols:
                proba[te, j] = pr[:, cols[c]]
    out["bacc4"] = float(balanced_accuracy_score(y, preds.astype(str)))
    y_onehot = np.stack([(y == c).astype(int) for c in CLASSES], axis=1)
    out["auc4"] = float(roc_auc_score(y_onehot, proba, average="macro"))
    return out


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-8)


def residualize(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(Z, X, rcond=None)
    return X - Z @ beta


def family_center(X: np.ndarray, fam: np.ndarray) -> np.ndarray:
    X = X.copy()
    for f in np.unique(fam):
        m = fam == f
        X[m] -= X[m].mean(axis=0, keepdims=True)
    return X


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--skip_slow", action="store_true",
                    help="skip transform variants, keep raw only")
    ap.add_argument("--within_bin_only", action="store_true",
                    help="skip the battery; only run the within-gbc-bin check")
    ap.add_argument("--within_bin_reps", nargs="+",
                    default=["direct_answer_hs20:l2",
                             "delta_cot-direct_prompt_hs20:raw",
                             "direct_prompt_hs20:raw"],
                    help="rep:transform pairs for the within-bin check")
    args = ap.parse_args()
    t0 = time.perf_counter()

    from safetensors.numpy import load_file

    hs_dir = args.out_dir / "hidden_states"
    meta = pd.read_parquet(hs_dir / "hs_meta.parquet")
    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    md = pd.read_parquet(args.out_dir / "metadata.parquet")

    qa = grading[~grading.is_control
                 & grading.retrieval_class.isin(CLASSES)].copy()
    qa = qa.merge(md[["question_id", "gbc"]], on="question_id", how="left")
    qa = qa.sort_values("question_id").reset_index(drop=True)
    qid_i = {q: i for i, q in enumerate(qa.question_id)}
    n = len(qa)
    y = qa.retrieval_class.to_numpy()
    groups = qa.fact_id.to_numpy()
    fam = qa.family.to_numpy()
    print(f"[sep] {n} QA instances; classes: "
          f"{dict(qa.retrieval_class.value_counts())}", flush=True)

    # ---- position row lookup ----------------------------------------------
    meta = meta.reset_index().rename(columns={"index": "row_pos"})
    meta_qa = meta[meta.question_id.isin(qid_i)]
    fixed = {}
    for (mode, pos), g in meta_qa[meta_qa.position_name != "sentence_end"] \
            .groupby(["prompt_mode", "position_name"]):
        rows = np.full(n, -1, dtype=int)
        rows[[qid_i[q] for q in g.question_id]] = g.row_pos.to_numpy()
        fixed[(mode, pos)] = rows
    se = meta_qa[meta_qa.position_name == "sentence_end"]
    se_rows = {}
    for q, g in se.groupby("question_id"):
        se_rows[qid_i[q]] = g.sort_values("position_rank").row_pos.to_numpy()

    H = {k: load_file(hs_dir / f"layer_{k:02d}.safetensors")["h"]
         for k in LAYERS}
    print(f"[sep] loaded {len(H)} layers in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    def gather(mode, pos, k):
        rows = fixed[(mode, pos)]
        ok = rows >= 0
        X = np.zeros((n, H[k].shape[1]), dtype=np.float32)
        X[ok] = H[k][rows[ok]].astype(np.float32)
        return X, ok

    # ---- representations ---------------------------------------------------
    reps: list[tuple[str, np.ndarray, np.ndarray]] = []  # (name, X, valid)

    for k in LAYERS:
        X, ok = gather("direct", "final_prompt_token", k)
        reps.append((f"direct_prompt_hs{k}", X, ok))
    for k in [16, 20, 24, 28]:
        X, ok = gather("cot", "final_prompt_token", k)
        reps.append((f"cot_prompt_hs{k}", X, ok))
    for k in [20, 24]:
        Xd, okd = gather("direct", "final_prompt_token", k)
        Xc, okc = gather("cot", "final_prompt_token", k)
        reps.append((f"delta_cot-direct_prompt_hs{k}", Xc - Xd, okd & okc))
        reps.append((f"concat_direct+cot_prompt_hs{k}",
                     np.concatenate([Xd, Xc], axis=1), okd & okc))
    for mode in ["direct", "cot"]:
        Xs, oks = [], np.ones(n, bool)
        for k in STACK:
            X, ok = gather(mode, "final_prompt_token", k)
            Xs.append(X)
            oks &= ok
        reps.append((f"stack_{mode}_prompt_hs{'-'.join(map(str, STACK))}",
                     np.concatenate(Xs, axis=1), oks))
    for k in [20, 24]:
        X, ok = gather("direct", "final_answer_token", k)
        reps.append((f"direct_answer_hs{k}", X, ok))

    # trajectory reps (cot)
    for k in [20, 24]:
        Xp, okp = gather("cot", "final_prompt_token", k)
        Xg, okg = gather("cot", "first_generated_token", k)
        Xa, oka = gather("cot", "final_answer_token", k)
        ok3 = okp & okg & oka
        reps.append((f"traj_stack_p-g0-ans_hs{k}",
                     np.concatenate([Xp, Xg, Xa], axis=1), ok3))
        reps.append((f"traj_disp_g0-p_hs{k}", Xg - Xp, ok3))
        reps.append((f"traj_disp_ans-p_hs{k}", Xa - Xp, ok3))
        # sentence-end mean state and mean displacement from prompt
        Xm = np.zeros_like(Xp)
        okm = np.zeros(n, bool)
        for i, rows in se_rows.items():
            Xm[i] = H[k][rows].astype(np.float32).mean(axis=0)
            okm[i] = True
        reps.append((f"traj_se_mean_hs{k}", Xm, okm & okp))
        reps.append((f"traj_se_mean-disp_hs{k}", Xm - Xp, okm & okp))

    # engineered path-shape features at hs20
    k = 20
    Xp, okp = gather("cot", "final_prompt_token", k)
    Xa, oka = gather("cot", "final_answer_token", k)
    feats = np.zeros((n, 8), dtype=np.float32)
    okf = okp & oka
    for i in range(n):
        if not okf[i]:
            continue
        pts = [Xp[i]]
        if i in se_rows:
            pts += [H[k][r].astype(np.float32) for r in se_rows[i]]
        pts.append(Xa[i])
        P = np.stack(pts)
        segs = np.diff(P, axis=0)
        seg_n = np.linalg.norm(segs, axis=1)
        net = np.linalg.norm(P[-1] - P[0])
        path = float(seg_n.sum())
        cosns = []
        for a, b in zip(segs[:-1], segs[1:]):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-6 and nb > 1e-6:
                cosns.append(float(a @ b / (na * nb)))
        feats[i] = [len(P), path, net, path / max(net, 1e-6),
                    float(seg_n.mean()), float(seg_n.std()),
                    float(np.mean(cosns)) if cosns else 0.0,
                    float(np.linalg.norm(P.std(axis=0)))]
    reps.append(("traj_shape_feats_hs20", feats, okf))

    # surface-confound baseline
    fam_dum = pd.get_dummies(qa.family).to_numpy(dtype=np.float32)
    g_meta = meta_qa[(meta_qa.prompt_mode == "cot")
                     & (meta_qa.position_name == "final_prompt_token")]
    ng = np.zeros(n, np.float32)
    ng[[qid_i[q] for q in g_meta.question_id]] = \
        g_meta.n_gen_tokens.to_numpy()
    nse = np.zeros(n, np.float32)
    for i, rows in se_rows.items():
        nse[i] = len(rows)
    marker = np.zeros(n, np.float32)
    marker[[qid_i[q] for q in g_meta.question_id]] = \
        g_meta.has_final_marker.to_numpy().astype(np.float32)
    dg_meta = meta_qa[(meta_qa.prompt_mode == "direct")
                      & (meta_qa.position_name == "final_prompt_token")]
    ndg = np.zeros(n, np.float32)
    ndg[[qid_i[q] for q in dg_meta.question_id]] = \
        dg_meta.n_gen_tokens.to_numpy()
    surface = np.concatenate(
        [np.log1p(qa.gbc.to_numpy(np.float32))[:, None], fam_dum,
         ng[:, None], ndg[:, None], nse[:, None], marker[:, None]], axis=1)
    reps.append(("BASELINE_surface_feats", surface, np.ones(n, bool)))

    # residualization design (label-free)
    Z = np.concatenate([np.ones((n, 1), np.float32),
                        np.log1p(qa.gbc.to_numpy(np.float32))[:, None],
                        fam_dum], axis=1)

    # ---- evaluate ----------------------------------------------------------
    rows_out = []
    for name, X, ok in (reps if not args.within_bin_only else []):
        variants = [("raw", X)]
        if not args.skip_slow and X.shape[1] > 20:
            variants += [("l2", l2(X)),
                         ("fam_center", family_center(X, fam)),
                         ("gbc_resid", residualize(X, Z))]
        for tag, Xv in variants:
            Xs, ys, gs = Xv[ok], y[ok], groups[ok]
            mb = np.isin(ys, ["direct_retrieval", "non_retrieved"])
            row = {"rep": name, "transform": tag, "n": int(ok.sum()),
                   "dims": Xv.shape[1],
                   "ch4": round(ch_score(Xs, ys), 2),
                   "ch2": round(ch_score(Xs[mb], ys[mb]), 2)}
            row.update({k2: round(v, 4) for k2, v in
                        cv_scores(reduce_dims(Xs), ys, gs).items()})
            rows_out.append(row)
            print(f"[sep] {name:42s} {tag:10s} auc2={row['auc2']:.3f} "
                  f"bacc4={row['bacc4']:.3f} auc4={row['auc4']:.3f} "
                  f"ch4={row['ch4']}", flush=True)

    if rows_out:
        lb = pd.DataFrame(rows_out).sort_values("auc2", ascending=False)
        out_csv = args.out_dir / "geometry" / "separation_search.csv"
        lb.to_csv(out_csv, index=False)
        print("\n==== leaderboard (top 15 by binary dr-vs-nr AUROC) ====")
        print(lb.head(15).to_string(index=False))
        base = lb[lb.rep == "BASELINE_surface_feats"].iloc[0]
        print(f"\nsurface baseline: auc2={base.auc2:.3f} "
              f"bacc4={base.bacc4:.3f} auc4={base.auc4:.3f}")

    # ---- Test 5: within-gbc-bin check on selected reps ---------------------
    rep_map = {nm: (x, okm) for nm, x, okm in reps}
    gbc_bin = md.set_index("question_id").gbc_bin \
        .reindex(qa.question_id).to_numpy()
    for spec in args.within_bin_reps:
        name, _, tag = spec.partition(":")
        if name not in rep_map:
            print(f"[sep] unknown rep {name}; skipping")
            continue
        X, ok = rep_map[name]
        if tag == "l2":
            X = l2(X)
        elif tag == "fam_center":
            X = family_center(X, fam)
        elif tag == "gbc_resid":
            X = residualize(X, Z)
        print(f"\n==== within-gbc-bin (Test 5) on {name}/{tag or 'raw'} ====")
        for b in ["low", "mid", "high", "very_high"]:
            m = ok & (gbc_bin == b)
            Xs, ys, gs = X[m], y[m], groups[m]
            try:
                sc = cv_scores(reduce_dims(Xs), ys, gs)
                print(f"  {b:10s} n={int(m.sum()):4d} "
                      f"ch4={ch_score(Xs, ys):6.2f} "
                      f"auc2={sc['auc2']:.3f} bacc4={sc['bacc4']:.3f}",
                      flush=True)
            except Exception as e:  # tiny classes in a bin
                print(f"  {b:10s} n={int(m.sum()):4d} skipped ({e})")

    if rows_out:
        print(f"\n[sep] wrote {out_csv}", flush=True)
    print(f"[sep] done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
