"""parametric_retrieval_geometry_v0: public-SAE arm.

Encodes the extracted Qwen2.5-7B-Instruct hidden states through the matching
public SAEs (andyrdt/saes-qwen2.5-7b-instruct, BatchTopK, dict 131072,
trainer_1 = k64) and asks where the retrieval-class signal lives:

  z        sparse SAE latents            (does sparsification keep it?)
  h_hat    SAE reconstruction            (signal inside the SAE span)
  resid    h - h_hat                     (signal the SAE fails to model)
  z_l1     L1-logistic on latents        (how FEW features carry it?)
  zdelta   z(cot prompt) - z(direct prompt)   sparse instruction-sensitivity

Layer mapping (verified in scripts/public_sae/download_public_sae.md):
hidden_states[K] = output of block K-1 = SAE folder resid_post_layer_{K-1};
so hs20 -> resid_post_layer_19, hs24 -> resid_post_layer_23. Our states ARE
Instruct-model states, so the Instruct-matched hard rule is satisfied.

FVU gate: last-token deep-layer readouts are mildly OOD for this SAE (audit
saw FVU 0.58 at L20 on PRM800K steps); FVU is reported per rep and stored.

Scoring matches prg_separation_search.py: GroupKFold(5) by fact_id, logistic,
binary direct_retrieval-vs-non_retrieved AUROC + 4-class balanced accuracy.

Outputs under <out_dir>/sae/:
  sae_arm_results.csv        leaderboard rows (rep x layer x view)
  top_features_hs{K}_{rep}.csv   per-feature dr-vs-nr AUROC, top 40
  encode manifests with FVU / active-count stats

  python scripts/parametric_retrieval/prg_sae_arm.py \
      --out_dir runs/parametric_retrieval_geometry_v0 \
      --sae_root data/public_sae/andyrdt-qwen2.5-7b-instruct --hs 20 24
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
sys.path.insert(0, str(ROOT / "scripts" / "public_sae"))

CLASSES = ["direct_retrieval", "reasoning_unlocked", "unstable_retrieval",
           "non_retrieved"]
REPS = [("direct", "final_prompt_token", "direct_prompt"),
        ("cot", "final_prompt_token", "cot_prompt"),
        ("direct", "final_answer_token", "direct_answer")]


def cv_scores(X, y, groups) -> dict:
    """Fact-grouped CV; X may be scipy sparse or dense."""
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler

    def new_pipe():
        from sklearn.pipeline import Pipeline
        sc = MaxAbsScaler() if sparse.issparse(X) else StandardScaler()
        return Pipeline([("sc", sc),
                         ("lr", LogisticRegression(max_iter=2000, C=1.0))])

    out = {}
    mb = np.isin(y, ["direct_retrieval", "non_retrieved"])
    Xb, yb, gb = X[mb], (y[mb] == "direct_retrieval").astype(int), groups[mb]
    aucs = []
    for tr, te in GroupKFold(5).split(Xb, yb, gb):
        p = new_pipe().fit(Xb[tr], yb[tr])
        aucs.append(roc_auc_score(yb[te], p.predict_proba(Xb[te])[:, 1]))
    out["auc2"] = float(np.mean(aucs))
    preds = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(X, y, groups):
        p = new_pipe().fit(X[tr], y[tr])
        preds[te] = p.predict(X[te])
    out["bacc4"] = float(balanced_accuracy_score(y, preds.astype(str)))
    return out


def l1_probe(X, y, groups, C=0.05) -> dict:
    """Sparse-feature readout: L1 logistic, report AUROC + active features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import MaxAbsScaler

    mb = np.isin(y, ["direct_retrieval", "non_retrieved"])
    Xb, yb, gb = X[mb], (y[mb] == "direct_retrieval").astype(int), groups[mb]
    aucs, nnzs = [], []
    for tr, te in GroupKFold(5).split(Xb, yb, gb):
        sc = MaxAbsScaler().fit(Xb[tr])
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                max_iter=2000)
        lr.fit(sc.transform(Xb[tr]), yb[tr])
        aucs.append(roc_auc_score(
            yb[te], lr.decision_function(sc.transform(Xb[te]))))
        nnzs.append(int((lr.coef_ != 0).sum()))
    return {"auc2": float(np.mean(aucs)), "nnz_mean": float(np.mean(nnzs))}


def per_feature_auc(Xcsr, y) -> np.ndarray:
    """dr-vs-nr AUROC per latent, rank-based, chunked over features."""
    from scipy.stats import rankdata
    mb = np.isin(y, ["direct_retrieval", "non_retrieved"])
    Xb = Xcsr[mb].tocsc()
    yb = (y[mb] == "direct_retrieval").astype(int)
    n1, n0 = int(yb.sum()), int((1 - yb).sum())
    d = Xb.shape[1]
    auc = np.full(d, 0.5, dtype=np.float32)
    for a in range(0, d, 8192):
        b = min(a + 8192, d)
        block = np.asarray(Xb[:, a:b].todense(), dtype=np.float32)
        ranks = np.apply_along_axis(rankdata, 0, block)
        r1 = ranks[yb == 1].sum(axis=0)
        auc[a:b] = (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--sae_root", type=Path,
                    default=Path("data/public_sae/andyrdt-qwen2.5-7b-instruct"))
    ap.add_argument("--hs", type=int, nargs="+", default=[20, 24])
    ap.add_argument("--trainer", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--skip_feature_table", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    import torch
    from safetensors.numpy import load_file
    from scipy import sparse

    from encode_public_sae import BatchTopKSAE  # noqa: E402

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")

    hs_dir = args.out_dir / "hidden_states"
    sae_out = args.out_dir / "sae"
    sae_out.mkdir(exist_ok=True)
    meta = pd.read_parquet(hs_dir / "hs_meta.parquet")
    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    qa = grading[~grading.is_control
                 & grading.retrieval_class.isin(CLASSES)].copy()
    qa = qa.sort_values("question_id").reset_index(drop=True)
    qid_i = {q: i for i, q in enumerate(qa.question_id)}
    n = len(qa)
    y = qa.retrieval_class.to_numpy()
    groups = qa.fact_id.to_numpy()
    meta = meta.reset_index().rename(columns={"index": "row_pos"})
    meta_qa = meta[meta.question_id.isin(qid_i)]

    def gather(mode, pos, H):
        g = meta_qa[(meta_qa.prompt_mode == mode)
                    & (meta_qa.position_name == pos)]
        rows = np.full(n, -1, dtype=int)
        rows[[qid_i[q] for q in g.question_id]] = g.row_pos.to_numpy()
        ok = rows >= 0
        X = np.zeros((n, H.shape[1]), dtype=np.float32)
        X[ok] = H[rows[ok]].astype(np.float32)
        return X, ok

    results = []
    for K in args.hs:
        folder = args.sae_root / f"resid_post_layer_{K - 1}" \
            / f"trainer_{args.trainer}"
        ae_pt = folder / "ae.pt"
        if not ae_pt.exists():
            print(f"[sae] missing {ae_pt}; skipping hs{K}")
            continue
        print(f"[sae] hs{K} (block {K - 1}): loading SAE {ae_pt}", flush=True)
        ae = BatchTopKSAE.from_ae_pt(ae_pt, device)
        H = load_file(hs_dir / f"layer_{K:02d}.safetensors")["h"]

        z_by_rep = {}
        for mode, pos, rep in REPS:
            X, ok = gather(mode, pos, H)
            # encode
            zs, hhat = [], np.zeros_like(X, dtype=np.float16)
            sse = 0.0
            hbar = torch.tensor(X[ok].mean(0), device=device)
            sst = float(((torch.tensor(X[ok], device=device) - hbar) ** 2)
                        .sum().item())
            active = []
            for i in range(0, n, args.batch_size):
                xb = torch.tensor(X[i:i + args.batch_size], device=device)
                z = ae.encode(xb)
                xh = ae.decode(z)
                hhat[i:i + xb.shape[0]] = xh.to(torch.float16).cpu().numpy()
                okb = torch.tensor(ok[i:i + xb.shape[0]], device=device)
                sse += float((((xb - xh) ** 2).sum(1) * okb).sum().item())
                active.append((z > 0).sum(1).cpu().numpy())
                zs.append(sparse.csr_matrix(z.cpu().numpy()))
            Z = sparse.vstack(zs).tocsr()
            fvu = sse / max(sst, 1e-9)
            act = np.concatenate(active)[ok]
            print(f"[sae] hs{K} {rep:14s} FVU={fvu:.3f} "
                  f"active/row={act.mean():.0f} (k=64 nominal)", flush=True)
            resid = X - hhat.astype(np.float32)
            z_by_rep[rep] = (Z, ok)

            views = [("latents_z", Z), ("recon_hhat", hhat.astype(np.float32)),
                     ("residual", resid)]
            for view, Xv in views:
                sc = cv_scores(Xv[ok] if not sparse.issparse(Xv)
                               else Xv[np.flatnonzero(ok)], y[ok], groups[ok])
                results.append({"hs_idx": K, "rep": rep, "view": view,
                                "fvu": round(fvu, 3),
                                "auc2": round(sc["auc2"], 4),
                                "bacc4": round(sc["bacc4"], 4)})
                print(f"[sae]   {view:12s} auc2={sc['auc2']:.3f} "
                      f"bacc4={sc['bacc4']:.3f}", flush=True)
            lp = l1_probe(Z[np.flatnonzero(ok)], y[ok], groups[ok])
            results.append({"hs_idx": K, "rep": rep, "view": "latents_L1",
                            "fvu": round(fvu, 3),
                            "auc2": round(lp["auc2"], 4),
                            "bacc4": np.nan, "nnz": lp["nnz_mean"]})
            print(f"[sae]   latents_L1   auc2={lp['auc2']:.3f} "
                  f"nnz~{lp['nnz_mean']:.0f} features", flush=True)

            if not args.skip_feature_table and rep in ("direct_prompt",
                                                       "direct_answer"):
                auc_f = per_feature_auc(Z[np.flatnonzero(ok)], y[ok])
                order = np.argsort(-np.abs(auc_f - 0.5))[:40]
                freq = np.asarray(
                    (Z[np.flatnonzero(ok)] > 0).mean(axis=0)).ravel()
                pd.DataFrame({
                    "feature": order, "auc_dr_vs_nr": auc_f[order],
                    "activation_freq": freq[order],
                }).to_csv(sae_out / f"top_features_hs{K}_{rep}.csv",
                          index=False)
                strong = int((np.abs(auc_f - 0.5) > 0.10).sum())
                print(f"[sae]   per-feature: {strong} latents with "
                      f"|AUROC-0.5|>0.10; best "
                      f"{auc_f[order[0]]:.3f} (feat {order[0]}, "
                      f"freq {freq[order[0]]:.3f})", flush=True)

        # sparse instruction-sensitivity delta in latent space
        (Zd, okd), (Zc, okc) = z_by_rep["direct_prompt"], z_by_rep["cot_prompt"]
        okb = okd & okc
        Zdelta = (Zc - Zd).tocsr()[np.flatnonzero(okb)]
        sc = cv_scores(Zdelta, y[okb], groups[okb])
        results.append({"hs_idx": K, "rep": "zdelta_cot-direct", "view":
                        "latents_z", "fvu": np.nan,
                        "auc2": round(sc["auc2"], 4),
                        "bacc4": round(sc["bacc4"], 4)})
        print(f"[sae] hs{K} zdelta_cot-direct auc2={sc['auc2']:.3f} "
              f"bacc4={sc['bacc4']:.3f}", flush=True)
        del ae, H
        if device.type == "mps":
            torch.mps.empty_cache()

    lb = pd.DataFrame(results)
    lb.to_csv(sae_out / "sae_arm_results.csv", index=False)
    print("\n==== SAE arm results ====")
    print(lb.to_string(index=False))
    print(f"\n[sae] wrote {sae_out}/sae_arm_results.csv in "
          f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
