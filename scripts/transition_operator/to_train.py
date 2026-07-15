"""transition_operator_v0 Stage 2b: train the latent transition operator.

Arms (v0.3 three-arm ablation, InfoNCE always on):
  A   L = 1.0*L_A + 0.5*L_NCE            (frozen-decoder KL through D(z))
  B   L = 1.0*L_B + 0.5*L_NCE            (trained head h_B(z) -> d_belief)
  AB  L = 1.0*L_A + 1.0*L_B + 0.5*L_NCE

Reads stage2/arrays (to_extract_train), splits.json (to_splits), forks.jsonl.
--prep_only fits and caches the measurement transforms (train-split PCA-64 of dL,
whitened + format-residualized; d_belief whitening) plus per-transition effect
vectors; run it ONCE before parallel arm/seed runs.

Outputs per run in stage2/<arm>_seed<seed>/: weights.pt, log.jsonl, metrics.json
(final losses, D(z) naturalness audits A1-A4 for A arms, test-set z decodability
and cross-problem retrieval vs the Stage-1 baseline numbers).

  python scripts/transition_operator/to_train.py --arm AB --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "transition_operator"))

from src.analysis.transition_operator import sep_join_ids  # noqa: E402
from src.analysis.transition_operator_train import (  # noqa: E402
    BeliefHead,
    ContrastiveProjections,
    TransitionEncoder,
    UpperDecoder,
    effect_close_mask,
    format_features,
    info_nce,
    kl_to_actual,
    percentile_of,
    rms,
)


# ---------------------------------------------------------------------------
# data plumbing
# ---------------------------------------------------------------------------

class Data:
    def __init__(self, run_dir: Path, layer: int):
        a = run_dir / "stage2" / "arrays"
        # npz member access re-reads the WHOLE member every time (mmap_mode is
        # ignored for npz), so load each array into RAM once; the training node
        # has plenty (~17 GB total for 10k transitions)
        with np.load(a / "fork_arrays.npz") as z:
            self.fork = {k: np.asarray(z[k]) for k in z.files}
        with np.load(a / "trans_arrays.npz") as z:
            self.trans = {k: np.asarray(z[k]) for k in z.files}
        self.fork_rows = json.loads((a / "fork_rows.json").read_text())
        self.trans_rows = json.loads((a / "trans_rows.json").read_text())
        self.fork_idx = {r["fork_id"]: i for i, r in enumerate(self.fork_rows)}
        self.trans_idx: dict[str, dict[str, int]] = {}
        for i, r in enumerate(self.trans_rows):
            self.trans_idx.setdefault(r["fork_id"], {})[r["branch"]] = i
        forks_meta = {json.loads(l)["fork_id"]: json.loads(l)
                      for l in open(run_dir / "forks.jsonl") if l.strip()}
        self.forks_meta = forks_meta
        splits = json.loads((run_dir / "splits.json").read_text())
        # keep only forks that survived extraction
        self.splits = {k: [f for f in splits[k] if f in self.fork_idx]
                       for k in ("train", "val", "test")}

    def trans_of(self, fork_ids: list[str]) -> tuple[list[int], list[int]]:
        c = [self.trans_idx[f]["correct"] for f in fork_ids]
        w = [self.trans_idx[f]["wrong"] for f in fork_ids]
        return c, w

    def d_belief(self, t_idx: np.ndarray, f_idx: np.ndarray) -> np.ndarray:
        return (self.trans["belief_post"][t_idx]
                - self.fork["belief_pre"][f_idx]).astype(np.float32)


# ---------------------------------------------------------------------------
# transforms (fit on train only; cached)
# ---------------------------------------------------------------------------

def fit_transforms(data: Data, run_dir: Path, seed: int) -> Path:
    out = run_dir / "stage2" / "transforms.npz"
    if out.exists():
        return out
    from sklearn.decomposition import PCA
    rng_fids = data.splits["train"]
    tc, tw = data.trans_of(rng_fids)
    t_all = np.array(tc + tw)
    f_all = np.array([data.fork_idx[f] for f in rng_fids] * 2)
    print(f"[prep] fitting transforms on {len(t_all)} train transitions", flush=True)
    dl = (data.trans["post_logits"][t_all].astype(np.float32)
          - data.fork["pre_logits"][f_all].astype(np.float32))
    pca = PCA(n_components=64, svd_solver="randomized", whiten=True,
              random_state=seed)
    dl64 = pca.fit_transform(dl)
    X = np.array([format_features(data.trans_rows[i]["text"],
                                  int(data.trans["n_steps"][i])) for i in t_all],
                 np.float32)
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    beta, *_ = np.linalg.lstsq(Xb, dl64, rcond=None)
    resid = dl64 - Xb @ beta
    fmt_r2 = 1.0 - resid.var(0) / np.maximum(dl64.var(0), 1e-12)
    db = data.d_belief(t_all, f_all)
    db_std = db.std(0) + 1e-8
    np.savez(out, components=pca.components_, mean=pca.mean_,
             explained_std=np.sqrt(pca.explained_variance_), beta=beta,
             db_std=db_std, format_r2=fmt_r2)
    print(f"[prep] format R^2 median {np.median(fmt_r2):.3f} "
          f"max {fmt_r2.max():.3f}", flush=True)
    # precompute effect vectors for ALL transitions
    return out


def effect_vectors(data: Data, run_dir: Path) -> np.ndarray:
    out = run_dir / "stage2" / "effects.npy"
    if out.exists():
        return np.load(out)
    t = np.load(run_dir / "stage2" / "transforms.npz")
    N = len(data.trans_rows)
    f_all = np.array([data.fork_idx[r["fork_id"]] for r in data.trans_rows])
    effects = np.zeros((N, 72), np.float32)
    chunk = 1024
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        dl = (data.trans["post_logits"][lo:hi].astype(np.float32)
              - data.fork["pre_logits"][f_all[lo:hi]].astype(np.float32))
        dl64 = (dl - t["mean"]) @ t["components"].T / t["explained_std"]
        X = np.array([format_features(data.trans_rows[i]["text"],
                                      int(data.trans["n_steps"][i]))
                      for i in range(lo, hi)], np.float32)
        Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
        effects[lo:hi, :64] = dl64 - Xb @ t["beta"]
        effects[lo:hi, 64:] = data.d_belief(np.arange(lo, hi),
                                            f_all[lo:hi]) / t["db_std"]
    np.save(out, effects)
    return effects


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def batch_tensors(data: Data, fork_ids: list[str], device):
    f_idx = np.array([data.fork_idx[f] for f in fork_ids])
    tc, tw = data.trans_of(fork_ids)
    t_idx = np.array(tc + tw)
    s_prev = torch.tensor(np.asarray(data.fork["S_prev"][f_idx], np.float32),
                          device=device)
    s_prev2 = torch.cat([s_prev, s_prev])
    H = torch.tensor(np.asarray(data.trans["H_steps"][t_idx], np.float32),
                     device=device)
    n = torch.tensor(np.asarray(data.trans["n_steps"][t_idx]), device=device)
    mask = (torch.arange(H.shape[1], device=device)[None] < n[:, None])
    return f_idx, t_idx, s_prev, s_prev2, H, mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--arm", choices=["A", "B", "AB"], default="AB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--d_z", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_forks", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--prep_only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = Data(args.run_dir, args.layer)
    fit_transforms(data, args.run_dir, seed=42)
    effects_all = effect_vectors(data, args.run_dir)
    if args.prep_only:
        print("[prep] done")
        return

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    use_A = "A" in args.arm
    use_B = "B" in args.arm

    hidden = data.fork["S_prev"].shape[1]
    enc = TransitionEncoder(hidden=hidden, d_z=args.d_z).to(device)
    proj = ContrastiveProjections(args.d_z).to(device)
    params = list(enc.parameters()) + list(proj.parameters())
    D = h_B = model = ud = tok = None
    if use_B:
        h_B = BeliefHead(args.d_z).to(device)
        params += list(h_B.parameters())
    if use_A:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                            local_files_only=args.local_files_only)
        dtype = torch.bfloat16 if device == "cuda" else (
            torch.float16 if device == "mps" else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=dtype,
            local_files_only=args.local_files_only).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        ud = UpperDecoder(model, args.layer)
        D = torch.nn.Linear(args.d_z, hidden).to(device)
        torch.nn.init.zeros_(D.bias)
        with torch.no_grad():
            D.weight.mul_(0.01)  # start near D(z)=0: decoder emits pre-context dist
        params += list(D.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(data.splits["train"]) // args.batch_forks)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch)

    pre_ids_cache: dict[str, list[int]] = {}

    def prefill_ids(fork_ids):
        rows = []
        for f in fork_ids:
            if f not in pre_ids_cache:
                m = data.forks_meta[f]
                pre_ids_cache[f] = sep_join_ids(tok, [m["question"],
                                                      *m["prefix_steps"]])
            rows.append(pre_ids_cache[f][:-1])  # cache excludes the boundary token
        lens = torch.tensor([len(r) for r in rows])
        width = int(lens.max())
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        am = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)], am[i, :len(r)] = torch.tensor(r), 1
        return ids.to(device), am.to(device), lens

    def losses(fork_ids, train: bool):
        f_idx, t_idx, s_prev, s_prev2, H, mask = batch_tensors(data, fork_ids, device)
        z = enc(s_prev2, H, mask)
        out = {}
        total = 0.0
        if use_A:
            ids, am, lens = prefill_ids(fork_ids)
            cache = ud.prefill(ids, am)
            base_len = ud.cache_len(cache)
            F = len(fork_ids)
            preds = []
            for half in (slice(0, F), slice(F, 2 * F)):
                h_hat = s_prev + D(z[half])
                preds.append(ud.decode_boundary(cache, h_hat, lens))
                ud.crop(cache, base_len)
            pred = torch.cat(preds)
            actual = torch.tensor(np.asarray(
                data.trans["post_logits"][t_idx], np.float32), device=device)
            out["L_A"] = kl_to_actual(actual, pred)
            total = total + out["L_A"]
        if use_B:
            db = torch.tensor(data.d_belief(t_idx, np.concatenate([f_idx, f_idx])),
                              device=device)
            out["L_B"] = torch.nn.functional.mse_loss(h_B(z), db)
            total = total + out["L_B"]
        eff = torch.tensor(effects_all[t_idx], device=device)
        za, ea = proj(z, eff)
        close = effect_close_mask(eff)
        out["L_NCE"] = info_nce(za, ea, close, tau=args.tau)
        out["mask_rate"] = close.float().mean().item()
        total = total + 0.5 * out["L_NCE"]
        out["total"] = total
        return out

    run_dir = args.run_dir / "stage2" / f"{args.arm}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "log.jsonl", "w")
    best_val, best_state, bad = float("inf"), None, 0
    rng = np.random.default_rng(args.seed)
    train_fids = list(data.splits["train"])
    modules = [m for m in (enc, proj, D, h_B) if isinstance(m, torch.nn.Module)]

    for epoch in range(args.epochs):
        rng.shuffle(train_fids)
        for m in modules:
            m.train()
        tr = []
        for lo in range(0, steps_per_epoch * args.batch_forks, args.batch_forks):
            out = losses(train_fids[lo:lo + args.batch_forks], train=True)
            opt.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tr.append({k: float(v.detach() if torch.is_tensor(v) else v)
                       for k, v in out.items()})
        for m in modules:
            m.eval()
        with torch.no_grad():
            va = []
            vf = data.splits["val"]
            for lo in range(0, len(vf), args.batch_forks):
                out = losses(vf[lo:lo + args.batch_forks], train=False)
                va.append({k: float(v) for k, v in out.items()})
        mean = lambda rows, k: float(np.mean([r[k] for r in rows if k in r]))  # noqa: E731
        rec = {"epoch": epoch,
               **{f"train_{k}": mean(tr, k) for k in tr[0]},
               **{f"val_{k}": mean(va, k) for k in va[0]}}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[{args.arm} s{args.seed}] ep{epoch} "
              f"train {rec['train_total']:.4f} val {rec['val_total']:.4f}", flush=True)
        if rec["val_total"] < best_val - 1e-5:
            best_val, bad = rec["val_total"], 0
            best_state = {f"m{i}": {k: v.detach().cpu().clone() for k, v in
                                    m.state_dict().items()}
                          for i, m in enumerate(modules)}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{args.arm} s{args.seed}] early stop at epoch {epoch}",
                      flush=True)
                break
    for i, m in enumerate(modules):
        m.load_state_dict(best_state[f"m{i}"])
    torch.save({"args": vars(args) | {"run_dir": str(args.run_dir)},
                **{f"m{i}": m.state_dict() for i, m in enumerate(modules)}},
               run_dir / "weights.pt")

    # ---- z for all transitions (best weights) --------------------------------
    all_fids = [f for s in ("train", "val", "test") for f in data.splits[s]]
    zs = np.zeros((len(data.trans_rows), args.d_z), np.float32)
    with torch.no_grad():
        for lo in range(0, len(all_fids), args.batch_forks):
            fids = all_fids[lo:lo + args.batch_forks]
            _, t_idx, _, s_prev2, H, mask = batch_tensors(data, fids, device)
            zs[t_idx] = enc(s_prev2, H, mask).cpu().numpy()
    np.save(run_dir / "z_all.npy", zs)

    metrics: dict = {"best_val_total": best_val, "arm": args.arm,
                     "seed": args.seed,
                     "created": datetime.now(timezone.utc).isoformat()}

    # ---- D(z) naturalness audits (A arms) -------------------------------------
    if use_A:
        with torch.no_grad():
            vf = data.splits["val"]
            f_idx, t_idx, s_prev, s_prev2, H, mask = batch_tensors(data, vf, device)
            z = enc(s_prev2, H, mask)
            dz = D(z)
            h_hat = torch.cat([s_prev, s_prev])[:len(dz)] + dz
            s_post = torch.tensor(np.asarray(data.trans["S_post"][t_idx],
                                             np.float32), device=device)
            delta_true = s_post - torch.cat([s_prev, s_prev])
            natural = torch.cat([s_prev, s_post])
            metrics["audit_A1_rms_percentile"] = {
                "p1": float(np.percentile(percentile_of(rms(h_hat),
                                                        rms(natural)).cpu(), 1)),
                "p50": float(np.percentile(percentile_of(rms(h_hat),
                                                         rms(natural)).cpu(), 50)),
                "p99": float(np.percentile(percentile_of(rms(h_hat),
                                                         rms(natural)).cpu(), 99))}
            # A2: Mahalanobis-style distance to the natural boundary-state
            # manifold, via whitened PCA-256 fit on the natural states
            from sklearn.decomposition import PCA as _PCA
            nat_np = natural.cpu().numpy()
            n_comp = min(256, nat_np.shape[0] - 1, nat_np.shape[1])
            p256 = _PCA(n_components=n_comp, whiten=True,
                        random_state=0).fit(nat_np)
            d_nat = np.linalg.norm(p256.transform(nat_np), axis=1)
            d_hat = np.linalg.norm(p256.transform(h_hat.cpu().numpy()), axis=1)
            metrics["audit_A2_manifold_dist"] = {
                "hat_median_percentile_in_natural": float(
                    (d_nat[None, :] < d_hat[:, None]).mean(1).mean() * 100),
                "d_hat_median": float(np.median(d_hat)),
                "d_natural_median": float(np.median(d_nat))}
            metrics["audit_A3_edit_norms"] = {
                "Dz_median": float(dz.norm(dim=-1).median()),
                "delta_true_median": float(delta_true.norm(dim=-1).median()),
                "sibling_diff_median": float(
                    (s_post[:len(vf)] - s_post[len(vf):]).norm(dim=-1).median())}
            # A4: norm-matched clipping stress on val L_A
            ids, am, lens = prefill_ids(vf)
            cache = ud.prefill(ids, am)
            base_len = ud.cache_len(cache)
            actual = torch.tensor(np.asarray(data.trans["post_logits"][t_idx],
                                             np.float32), device=device)

            def val_LA(scale_to: float | None):
                preds = []
                for half in (slice(0, len(vf)), slice(len(vf), 2 * len(vf))):
                    d = D(z[half])
                    if scale_to is not None:
                        d = d * (scale_to / (d.norm(dim=-1, keepdim=True) + 1e-8))
                    preds.append(ud.decode_boundary(cache, s_prev + d, lens))
                    ud.crop(cache, base_len)
                return float(kl_to_actual(actual, torch.cat(preds)))

            la = val_LA(None)
            med = float(delta_true.norm(dim=-1).median())
            p95 = float(np.percentile(delta_true.norm(dim=-1).cpu(), 95))
            metrics["audit_A4_clip_stress"] = {
                "val_L_A": la,
                "val_L_A_scaled_to_median": val_LA(med),
                "val_L_A_capped_p95": val_LA(p95)}

    # ---- z test-set eval vs Stage-1 protocol ----------------------------------
    import pandas as pd
    from to_stage1 import cross_problem_retrieval, decodability  # noqa: E402
    labels = pd.read_parquet(args.run_dir / "stage1" / "step_labels.parquet")
    lab_idx = {(r.fork_id, r.branch): (r.op_symbolic, r.tag_top)
               for r in labels.itertuples()}
    test_fids = set(data.splits["test"])
    rows = [(i, *lab_idx.get((r["fork_id"], r["branch"]), (None, None)),
             r["question"])
            for i, r in enumerate(data.trans_rows) if r["fork_id"] in test_fids]
    idx = np.array([r[0] for r in rows])
    groups = pd.factorize(np.asarray([r[3] for r in rows], dtype=object))[0]
    X = zs[idx]
    for name, col in (("op_symbolic", 1), ("tag_top", 2)):
        y = pd.Series([r[col] for r in rows])
        y = y.where(y != "NONE")
        m = y.notna().to_numpy()
        if m.sum() >= 30:
            metrics[f"z_test_{name}"] = {
                "decodability": decodability(X[m], y[m].to_numpy().astype(str),
                                             groups[m], seed=args.seed),
                "retrieval": cross_problem_retrieval(
                    X[m], y[m].to_numpy().astype(str), groups[m])}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
