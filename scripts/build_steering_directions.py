#!/usr/bin/env python3
"""Build the set of steering directions for the S3 Stage-5 causal-validation test.

For one layer hidden_states index L, this materialises every direction we inject into
the residual stream, all as **unit vectors in raw hidden space**, oriented so that
``+alpha = toward correct`` for the correctness treatments:

  probe              the dense correctness direction w (the direction under test).
                     L == deployed_layer -> loaded from --probe_path (the deployed
                     linear_probe.pt). Otherwise a fresh logistic probe is trained in
                     this layer's own space from the cache and saved next to --out_path.
  sparse_restricted  w with weights zeroed outside the L1 minimal subset (~the Stage-2
                     235/3584 dims) and renormalised. Tests whether causal power is as
                     compressible as decoding power.
  meandiff           mean(h|incorrect) - mean(h|correct); cross-check (cos~0.99 at L28).
  top_pc             PC1 of the layer's hidden states (high variance, Stage-2-orthogonal
                     to correctness).
  random_{k}         Gaussian directions (matched norm via s_layer at inject time).
  surface            (optional) direction along which token-length+step position vary.
  perplexity         (optional) direction predicting per-step NLL (Gate-A surprise).

The probe vector + bias (in raw space, incorrect = positive logit) are also stored so the
steering scripts can read off the probe-score shift w.h_steered and separate "moved the
readout" from "changed behaviour".

Norm calibration: s_layer = median ||h|| over the cache rows. At inject time the vector is
alpha * s_layer * unit_dir, so alpha is a dimensionless fraction of the typical residual norm.

Inputs (per-layer 2D cache, the one s1ms_steer_forks.py uses):
  <cache_dir>/<cache_prefix>_L{L}_h.npy   (n, hidden) float
  <cache_dir>/<cache_prefix>_y.npy        (n,) int, 1 = incorrect

Output: <out_path> .npz with names / vectors (toward-correct oriented for treatments) /
s_layer / probe score weight+bias / per-direction metadata.

Usage:
  python scripts/build_steering_directions.py \
      --cache_dir runs/.../qwen2_5_7b/multilayer --layer_index 28 \
      --probe_path runs/.../qwen2_5_7b/linear_probe.pt --deployed_layer 28 \
      --out_path runs/.../qwen2_5_7b/steering/directions_L28.npz
  # L20: no deployed probe at this layer -> trains + saves linear_probe_L20.pt
  python scripts/build_steering_directions.py \
      --cache_dir runs/.../qwen2_5_7b/multilayer --layer_index 20 \
      --probe_path runs/.../qwen2_5_7b/linear_probe.pt --deployed_layer 28 \
      --out_path runs/.../qwen2_5_7b/steering/directions_L20.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _load_probe(path: Path, hidden_dim: int) -> tuple[np.ndarray, float]:
    """Load (w, b) from a linear_probe.pt; w is (hidden,), positive logit = incorrect."""
    import torch
    obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    wkey = next(k for k in state if k.endswith("fc.weight") or k.endswith("weight"))
    bkey = next((k for k in state if k.endswith("fc.bias") or k.endswith("bias")), None)
    w = np.asarray(state[wkey], dtype=np.float64).reshape(-1)
    if w.shape[0] != hidden_dim:
        raise ValueError(f"probe dim {w.shape[0]} != hidden {hidden_dim}")
    b = float(np.asarray(state[bkey]).reshape(-1)[0]) if bkey else 0.0
    return w, b


def _save_probe(path: Path, w: np.ndarray, b: float) -> None:
    import torch
    state = {"weight": torch.tensor(w.reshape(1, -1), dtype=torch.float32),
             "bias": torch.tensor([b], dtype=torch.float32)}
    torch.save(state, path)


def _l1_minimal_mask(h: np.ndarray, y: np.ndarray, tol: float, seed: int) -> np.ndarray:
    """Smallest L1-selected dim set whose CV acc is within tol of the full-dim probe.

    Mirrors scripts/analysis/s3_prm800k_minimal_subspace.py: standardise, sweep C on an
    L1 logistic, take the fewest-nonzero solution that stays within tol of full-dim acc.
    """
    Xs = StandardScaler().fit_transform(h).astype(np.float32)
    full = float(cross_val_score(
        LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced"),
        Xs, y, cv=5, scoring="accuracy").mean())
    curve = []
    for C in (0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0):
        est = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                 max_iter=3000, class_weight="balanced")
        acc = float(cross_val_score(est, Xs, y, cv=5, scoring="accuracy").mean())
        coef = est.fit(Xs, y).coef_.ravel()
        curve.append((C, int((coef != 0).sum()), acc, coef))
        print(f"[sparse] C={C:<6} nnz={int((coef != 0).sum()):4d} acc={acc:.3f}", flush=True)
    ok = [r for r in curve if r[2] >= full - tol and r[1] > 0]
    pick = min(ok, key=lambda r: r[1]) if ok else max(curve, key=lambda r: r[2])
    mask = pick[3] != 0
    print(f"[sparse] full={full:.3f} pick C={pick[0]} nnz={int(mask.sum())} "
          f"acc={pick[2]:.3f} kept {100*mask.sum()/h.shape[1]:.1f}%", flush=True)
    return mask


def _nuisance_dir(h: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """Direction in h-space along which standardised features `feats` (n,k) vary.

    Regress every hidden dim on [1, z(feats)]; the steering direction is the unit sum of
    the per-feature response columns (a direction the nuisance moves along).
    """
    z = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    X = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(X, h, rcond=None)   # (1+k, hidden)
    return _unit(beta[1:].sum(0))                  # sum the feature-response rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--cache_prefix", type=str, default="probe_train_40k")
    ap.add_argument("--layer_index", type=int, required=True)
    ap.add_argument("--probe_path", type=Path, required=True,
                    help="deployed linear_probe.pt (used iff layer_index == deployed_layer)")
    ap.add_argument("--deployed_layer", type=int, default=28)
    ap.add_argument("--out_path", type=Path, required=True)
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--sparse_tol", type=float, default=0.01)
    ap.add_argument("--row_features_jsonl", type=Path, default=None,
                    help="optional per-cache-row {length, step_idx, nll} for surface/"
                         "perplexity controls (aligned by row order)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    L = args.layer_index
    rng = np.random.default_rng(args.seed)

    h = np.load(args.cache_dir / f"{args.cache_prefix}_L{L}_h.npy").astype(np.float32)
    y = np.load(args.cache_dir / f"{args.cache_prefix}_y.npy").astype(int)
    n, hidden = h.shape
    assert len(y) == n, f"y/h mismatch: {len(y)} vs {n}"
    s_layer = float(np.median(np.linalg.norm(h, axis=1)))
    print(f"[build] L{L} n={n} hidden={hidden} s_layer(median||h||)={s_layer:.3f} "
          f"correct={int((y == 0).sum())} incorrect={int((y == 1).sum())}", flush=True)

    # ---- probe w (raw space, incorrect = positive logit) -------------------
    if L == args.deployed_layer:
        w_raw, b_raw = _load_probe(args.probe_path, hidden)
        print(f"[build] probe: loaded deployed {args.probe_path.name}", flush=True)
    else:
        clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
        clf.fit(h, y)                              # raw space, label 1 = incorrect
        w_raw, b_raw = clf.coef_.ravel().astype(np.float64), float(clf.intercept_[0])
        saved = args.out_path.parent / f"linear_probe_L{L}.pt"
        _save_probe(saved, w_raw, b_raw)
        acc = float((clf.predict(h) == y).mean())
        print(f"[build] probe: trained L{L} in-space (train acc {acc:.3f}) -> {saved.name}",
              flush=True)
    w_unit = _unit(w_raw)                           # points toward incorrect

    # ---- sparse-restricted: dense w masked to the L1 minimal subset --------
    mask = _l1_minimal_mask(h, y, args.sparse_tol, args.seed)
    w_sparse = _unit(np.where(mask, w_raw, 0.0))    # toward incorrect, on minimal dims

    # ---- mean-diff and top PC ---------------------------------------------
    d_unit = _unit(h[y == 1].mean(0) - h[y == 0].mean(0))   # toward incorrect
    hc = h - h.mean(0)
    # top right singular vector of centred h == PC1
    _, _, vt = np.linalg.svd(hc[rng.choice(n, size=min(n, 8000), replace=False)],
                             full_matrices=False)
    pc1 = _unit(vt[0])

    # treatments stored oriented TOWARD CORRECT (so +alpha = toward correct)
    names: list[str] = []
    vecs: list[np.ndarray] = []
    meta: dict[str, dict] = {}

    def add(name: str, unit_vec: np.ndarray, toward: str, n_active: int) -> None:
        names.append(name)
        vecs.append(unit_vec.astype(np.float32))
        meta[name] = {"toward": toward, "n_active_dims": int(n_active),
                      "cos_to_probe_incorrect": float(np.dot(_unit(unit_vec), w_unit))}

    add("probe", -w_unit, "correct", hidden)                     # +alpha -> correct
    add("sparse_restricted", -w_sparse, "correct", int(mask.sum()))
    add("meandiff", -d_unit, "correct", hidden)
    add("top_pc", pc1, "none", hidden)
    for k in range(args.n_random):
        add(f"random_{k}", _unit(rng.standard_normal(hidden)), "none", hidden)

    # ---- optional nuisance controls (need aligned per-row features) --------
    if args.row_features_jsonl and args.row_features_jsonl.exists():
        rows = [json.loads(l) for l in args.row_features_jsonl.read_text().splitlines()
                if l.strip()]
        if len(rows) != n:
            print(f"[build] WARN row_features n={len(rows)} != cache n={n}; "
                  "skipping surface/perplexity", flush=True)
        else:
            length = np.array([r.get("length", np.nan) for r in rows], float)
            step = np.array([r.get("step_idx", np.nan) for r in rows], float)
            nll = np.array([r.get("nll", np.nan) for r in rows], float)
            if np.isfinite(length).all() and np.isfinite(step).all():
                add("surface", _nuisance_dir(h, np.column_stack([length, step])),
                    "none", hidden)
            if np.isfinite(nll).all():
                add("perplexity", _nuisance_dir(h, nll[:, None]), "none", hidden)
    if "surface" not in meta:
        print("[build] note: surface/perplexity controls omitted (no aligned features)",
              flush=True)

    V = np.stack(vecs, 0)
    np.savez(
        args.out_path,
        names=np.array(names),
        vectors=V,                                  # (k, hidden) unit, toward-correct/none
        s_layer=np.float32(s_layer),
        layer_index=np.int64(L),
        w_score=w_raw.astype(np.float32),           # probe direction toward incorrect
        b_score=np.float32(b_raw),                  # so logit = h@w_score + b_score
        sparse_mask=mask,
        meta_json=json.dumps(meta),
    )
    print(f"[build] wrote {args.out_path}  directions={names}", flush=True)
    for nm in names:
        print(f"        {nm:18s} toward={meta[nm]['toward']:8s} "
              f"active={meta[nm]['n_active_dims']:5d} "
              f"cos(probe_inc)={meta[nm]['cos_to_probe_incorrect']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
