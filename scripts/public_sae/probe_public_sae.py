"""Linear-probe representation audit: dense h vs SAE z vs h_hat vs residual.

Trains a logistic-regression probe on each representation of the Instruct
held-out PRM800K steps, using ONE fixed stratified split shared across all
representations, and reports the audit metrics + localization + direction
comparisons used by plot_public_sae_audit.py and the report.

Representations per layer L:
  dense       heldout_{L}_h.npy                 (3584-d, StandardScaler + LR)
  sae_z       {L}_t{t}_z.npz (CSR, 131072-d)    (raw sparse + LR)
  hhat        {L}_t{t}_hhat.npy                  (3584-d, StandardScaler + LR)
  resid       {L}_t{t}_resid.npy                (3584-d, StandardScaler + LR)
  norm_only   [||h||]                            (1-d control)
  active_count[n active SAE feats]               (1-d control)
  dense_shuf  dense h with labels shuffled       (null)

Label convention: y=1 INCORRECT, y=0 correct; probe score = P(incorrect), so
roc_auc_score(y, score) is oriented (higher score = more error-like).

Outputs (--out_dir):
  metrics.csv                 the key table (layer|model|representation|dim|AUROC|bal-acc|d'|notes)
  metrics.json                same rows, structured
  scores_{L}.npz              per-rep test decision scores + y_test (for plots B/C)
  localization_{L}.json       cumulative AUROC vs top-k SAE feats, n for 95/99%
  direction_{L}.json          dense vs sae probe-score scatter, decoder-mapped cosines,
                              per-feature (w_i, dz_i) for plot E
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


def d_prime(score: np.ndarray, y: np.ndarray) -> float:
    a, b = score[y == 1], score[y == 0]
    v = 0.5 * (a.var() + b.var())
    return float((a.mean() - b.mean()) / np.sqrt(v)) if v > 0 else float("nan")


def overlap_coef(score: np.ndarray, y: np.ndarray, bins: int = 60) -> float:
    """Histogram-intersection overlap of the two class score distributions in [0,1]."""
    lo, hi = float(score.min()), float(score.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(score[y == 1], bins=edges, density=True)
    q, _ = np.histogram(score[y == 0], bins=edges, density=True)
    w = edges[1] - edges[0]
    return float(np.minimum(p, q).sum() * w)


def oracle_balacc(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_v, best_t = -1.0, 0.5
    for t in np.linspace(0.01, 0.99, 99):
        v = balanced_accuracy_score(y, (score >= t).astype(int))
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return best_t, best_v


def fit_eval(Xtr, Xte, ytr, yte, *, sparse_in=False):
    """Fit LR, return (proba_test, logit_test, metrics dict, coef raw-space, scaler)."""
    scaler = None
    if sparse_in:
        Xtr_f, Xte_f = Xtr, Xte
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
    else:
        scaler = StandardScaler().fit(Xtr)
        Xtr_f, Xte_f = scaler.transform(Xtr), scaler.transform(Xte)
        clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(Xtr_f, ytr)
    proba = clf.predict_proba(Xte_f)[:, 1]
    logit = clf.decision_function(Xte_f)
    o_t, o_bal = oracle_balacc(proba, yte)
    m = {
        "auroc": float(roc_auc_score(yte, proba)),
        "balacc_oracle": o_bal, "oracle_threshold": o_t,
        "balacc_0p5": float(balanced_accuracy_score(yte, (proba >= 0.5).astype(int))),
        "acc_0p5": float((( proba >= 0.5).astype(int) == yte).mean()),
        "f1_incorrect_oracle": float(f1_score(yte, (proba >= o_t).astype(int), pos_label=1, zero_division=0)),
        "dprime": d_prime(logit, yte),
        "overlap": overlap_coef(proba, yte),
        "pred_pos_rate_0p5": float((proba >= 0.5).mean()),
    }
    return proba, logit, m, clf.coef_.ravel(), scaler


def _trunc(s, n=240):
    s = " ".join(str(s or "").split())
    return s[:n] + (" ..." if len(s) > n else "")


def write_feature_table(out_path, Z, order, coef_z, dz, y, gidx, jsonl_rows, n_top, n_ex):
    """Table G: top SAE features ranked by |w*dz| with max-activating step text."""
    lines = [f"# Top SAE features (layer table G)", "",
             "Ranked by |probe_weight * class_delta|. freq = fraction of class with feature active.", ""]
    Zc = Z.tocsc()
    for f in order[:n_top]:
        col = np.asarray(Zc[:, f].todense()).ravel()
        act = col > 0
        fc = float(act[y == 0].mean()) if (y == 0).any() else 0.0
        fi = float(act[y == 1].mean()) if (y == 1).any() else 0.0
        lines += [f"## feature {int(f)}  (w={coef_z[f]:+.3f}, dz={dz[f]:+.4f}, "
                  f"freq_correct={fc:.3f}, freq_incorrect={fi:.3f})", ""]
        top = np.argsort(col)[::-1][:n_ex]
        for r in top:
            if col[r] <= 0:
                continue
            ex = jsonl_rows[int(gidx[r])] if jsonl_rows and int(gidx[r]) < len(jsonl_rows) else {}
            lab = "INCORRECT" if y[r] == 1 else "correct"
            lines += [f"- act={col[r]:.3f} [{lab}]  Q: {_trunc(ex.get('problem'), 160)}",
                      f"    prev: {_trunc(ex.get('prefix'), 160)}",
                      f"    step: {_trunc(ex.get('candidate_step'), 200)}"]
        lines.append("")
    Path(out_path).write_text("\n".join(lines))


def load_z_csr(npz_path: Path):
    d = np.load(npz_path)
    n, dim = int(d["shape"][0]), int(d["shape"][1])
    return sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=(n, dim))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enc_dir", type=Path, required=True, help="merged dense residuals dir")
    ap.add_argument("--sae_dir", type=Path, required=True, help="encode_public_sae outputs")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--layers", nargs="+", default=["L20", "L28"])
    ap.add_argument("--trainer", type=int, default=1)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--sae_root", type=Path, default=None,
                    help="optional: load decoder.weight for the SAE->h direction map (fig F)")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="optional: held-out steps jsonl -> dump top-feature examples (table G)")
    ap.add_argument("--n_top_feats", type=int, default=12)
    ap.add_argument("--n_examples_per_feat", type=int, default=4)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    y = np.load(args.enc_dir / "heldout_y.npy").astype(int)
    n = len(y)
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=args.test_size,
                                      random_state=SEED, stratify=y)
    ytr, yte = y[idx_tr], y[idx_te]
    rows: list[dict] = []

    jsonl_rows, gidx = None, None
    if args.jsonl is not None:
        jsonl_rows = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
        meta = [json.loads(l) for l in
                (args.enc_dir / "heldout_meta.jsonl").read_text().splitlines() if l.strip()]
        gidx = np.array([m["global_index"] for m in meta], dtype=np.int64)

    for L in args.layers:
        t = args.trainer
        h = np.load(args.enc_dir / f"heldout_{L}_h.npy").astype(np.float32)
        hhat = np.load(args.sae_dir / f"{L}_t{t}_hhat.npy").astype(np.float32)
        resid = np.load(args.sae_dir / f"{L}_t{t}_resid.npy").astype(np.float32)
        Z = load_z_csr(args.sae_dir / f"{L}_t{t}_z.npz")
        active = np.load(args.sae_dir / f"{L}_t{t}_z.npz")["active_per_row"].astype(np.float32)
        man = json.loads((args.sae_dir / f"{L}_t{t}_encode_manifest.json").read_text())
        scores = {"y_test": yte}

        # ---- dense, hhat, resid (3584-d) -----------------------------------
        p_h, lg_h, m_h, coef_h, sc_h = fit_eval(h[idx_tr], h[idx_te], ytr, yte)
        scores["dense"] = p_h
        rows.append(_row(L, "dense h", h.shape[1], m_h, f"FVU={man['recon_fvu']}"))

        p_hh, _, m_hh, _, _ = fit_eval(hhat[idx_tr], hhat[idx_te], ytr, yte)
        scores["hhat"] = p_hh
        rows.append(_row(L, "SAE h_hat", hhat.shape[1], m_hh, "reconstruction"))

        p_r, _, m_r, _, _ = fit_eval(resid[idx_tr], resid[idx_te], ytr, yte)
        scores["resid"] = p_r
        rows.append(_row(L, "SAE residual h-h_hat", resid.shape[1], m_r, "what SAE discards"))

        # ---- SAE z (sparse 131072-d) ---------------------------------------
        p_z, _, m_z, coef_z, _ = fit_eval(Z[idx_tr], Z[idx_te], ytr, yte, sparse_in=True)
        scores["sae_z"] = p_z
        eff = f"{man['n_features_ever_active']} feats active; k={man['k']}"
        rows.append(_row(L, "SAE z", Z.shape[1], m_z, eff, used_dim=man["n_features_ever_active"]))

        # ---- controls + null ----------------------------------------------
        hn = np.linalg.norm(h, axis=1, keepdims=True)
        p_n, _, m_n, _, _ = fit_eval(hn[idx_tr], hn[idx_te], ytr, yte)
        rows.append(_row(L, "norm-only ||h||", 1, m_n, "control"))

        ac = active.reshape(-1, 1)
        p_ac, _, m_ac, _, _ = fit_eval(ac[idx_tr], ac[idx_te], ytr, yte)
        rows.append(_row(L, "active-count", 1, m_ac, "control"))

        rng = np.random.default_rng(SEED)
        y_sh = ytr.copy(); rng.shuffle(y_sh)
        _, _, m_sh, _, _ = fit_eval(h[idx_tr], h[idx_te], y_sh, yte)
        rows.append(_row(L, "dense h (shuffled-label null)", h.shape[1], m_sh, "null"))

        np.savez(args.out_dir / f"scores_{L}.npz", **scores)

        # ---- feature localization (z): cumulative AUROC vs top-k ----------
        dz = (np.asarray(Z[idx_tr][ytr == 1].mean(0)).ravel()
              - np.asarray(Z[idx_tr][ytr == 0].mean(0)).ravel())
        contrib = np.abs(coef_z * dz)
        order = np.argsort(contrib)[::-1]
        Zte = Z[idx_te].tocsc()
        full = m_z["auroc"]
        ks = [k for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
              if k <= (man["n_features_ever_active"] or Z.shape[1])]
        cum = []
        for k in ks:
            top = order[:k]
            s = np.asarray(Zte[:, top] @ coef_z[top]).ravel()
            cum.append(float(roc_auc_score(yte, s)) if len(np.unique(yte)) > 1 else float("nan"))
        def n_for(frac):
            tgt = frac * full
            for k, a in zip(ks, cum):
                if a >= tgt:
                    return k
            return None
        (args.out_dir / f"localization_{L}.json").write_text(json.dumps({
            "layer": L, "full_z_auroc": full, "top_k_grid": ks, "cumulative_auroc": cum,
            "n_feats_for_95pct": n_for(0.95), "n_feats_for_99pct": n_for(0.99),
            "top20_feature_ids": order[:20].tolist(),
            "top20_probe_weight": coef_z[order[:20]].tolist(),
            "top20_class_delta": dz[order[:20]].tolist(),
        }, indent=2))

        if jsonl_rows is not None:
            write_feature_table(args.out_dir / f"feature_examples_{L}.md", Z, order,
                                coef_z, dz, y, gidx, jsonl_rows, args.n_top_feats,
                                args.n_examples_per_feat)

        # ---- direction comparison (fig E/F) -------------------------------
        w_dense_raw = coef_h / sc_h.scale_           # dense probe dir in raw h-space
        mean_diff = h[idx_tr][ytr == 1].mean(0) - h[idx_tr][ytr == 0].mean(0)
        direction = {
            "layer": L,
            "cos_dense_vs_meandiff": _cos(w_dense_raw, mean_diff),
            "pearson_dense_vs_z_score": float(np.corrcoef(p_h, p_z)[0, 1]),
            "top_feat_contrib": contrib[order[:200]].tolist(),
        }
        # SAE->h direction map (fig F). Prefer a saved decoder.npy (format-agnostic,
        # written by encode_gemma_sae.py); fall back to the Qwen BatchTopK ae.pt.
        Wdec = None
        dec_npy = args.sae_dir / f"{L}_t{t}_decoder.npy"
        if dec_npy.exists():
            Wdec = np.load(dec_npy)                          # (act_dim, dict_size)
        elif args.sae_root is not None:
            import torch
            sd = torch.load(args.sae_root / man["sae_folder"] / f"trainer_{t}" / "ae.pt",
                            map_location="cpu")
            Wdec = sd["decoder.weight"].float().numpy()     # (act_dim, dict_size)
        if Wdec is not None:
            v_sae_to_h = Wdec @ coef_z                       # (act_dim,)
            direction.update({
                "cos_saemap_vs_dense": _cos(v_sae_to_h, w_dense_raw),
                "cos_saemap_vs_meandiff": _cos(v_sae_to_h, mean_diff),
            })
        (args.out_dir / f"direction_{L}.json").write_text(json.dumps(direction, indent=2))
        print(f"[probe] {L}: dense AUROC={m_h['auroc']:.3f}  z={m_z['auroc']:.3f}  "
              f"hhat={m_hh['auroc']:.3f}  resid={m_r['auroc']:.3f}  "
              f"null={m_sh['auroc']:.3f}", flush=True)

    # ---- key table ----------------------------------------------------------
    cols = ["layer", "representation", "dim", "used_dim", "auroc", "balacc_oracle",
            "dprime", "overlap", "f1_incorrect_oracle", "notes"]
    with open(args.out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    (args.out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"[probe] wrote {args.out_dir}/metrics.csv ({len(rows)} rows)", flush=True)


def _row(layer, rep, dim, m, notes, used_dim=None):
    return {"layer": layer, "model": "Qwen2.5-7B-Instruct", "representation": rep,
            "dim": dim, "used_dim": used_dim if used_dim is not None else dim,
            "auroc": round(m["auroc"], 4), "balacc_oracle": round(m["balacc_oracle"], 4),
            "dprime": round(m["dprime"], 4), "overlap": round(m["overlap"], 4),
            "f1_incorrect_oracle": round(m["f1_incorrect_oracle"], 4),
            "balacc_0p5": round(m["balacc_0p5"], 4), "acc_0p5": round(m["acc_0p5"], 4),
            "pred_pos_rate_0p5": round(m["pred_pos_rate_0p5"], 4), "notes": notes}


def _cos(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


if __name__ == "__main__":
    main()
