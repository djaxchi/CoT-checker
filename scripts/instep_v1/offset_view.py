#!/usr/bin/env python3
"""InStep paper, "what the offsets do": the step token's own state with and without offsets,
seen without any projection fitted on the plotted labels.

For each token (--tok name=states_dir=read_token.pt), from read_token_inline.py encode outputs
(`sep` = the state the token's linear head reads, `head_p_correct`, `y`, `kept`) on val and the
8 ReProbe test sets:
  score   the head's own logit of P(incorrect), on the test steps (held out)
  pca     principal directions of the val states (no labels), raw-centred and per-dimension
          standardized; test steps projected on the top --n_pcs; AUROC of each PC on the pooled
          test steps (sign free: max(a, 1 - a), reported with the sign)
  orth    first principal direction of the val states orthogonal to the head weight (the
          y axis of a "head logit x orthogonal PC" view: fitted without test labels)
Writes <out>/<name>.csv (set, y, logit, pc1.., spc1.., orth) for every incorrect test step and
--n_correct sampled correct ones per set, and <out>/summary.json (per token: test PR-AUC and
AUROC of the logit, pooled and 8-set mean; per-PC AUROC, raw and standardized; variance shares).

    python scripts/instep_v1/offset_view.py --data $D --out $OUT/offset_view \\
        --tok emb=$ST/view_emb_ds_seed0=$TOK_EMB off=$ST/view_off_ds_seed0=$TOK_OFF instep=$ST8=$TOK_IN
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.instep_v1.passive_fit import val_labels  # noqa: E402
from scripts.reprobe_cmp.analyze import TEST, auroc, fast_ap  # noqa: E402


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def load(d: Path, s: str, vl: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(states (n, d) float32, y, logit of P(incorrect)) of the kept, labelled steps."""
    with np.load(d / f"{s}.shard0.npz") as z:
        X, y, kept, pc = z["sep"], z["y"].astype(int), z["kept"].astype(bool), z["head_p_correct"].astype(np.float64)
    if vl is not None:
        uids = [r["uid"] for r in jl(d / f"{s}.shard0.meta.jsonl")]
        y = np.array([vl.get(u, -1) for u in uids])
    m = kept & (y >= 0)
    p = np.clip(1 - pc[m], 1e-12, 1 - 1e-12)
    return X.reshape(len(X), -1)[m].astype(np.float32), y[m], np.log(p / (1 - p))


def head_weight(ckpt: Path) -> np.ndarray:
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["head"]
    w = [v for v in st.values() if v.ndim == 2 and v.shape[0] == 1]
    if len(w) != 1:
        raise SystemExit(f"{ckpt}: expected one (1, d) head weight, found {[tuple(v.shape) for v in st.values()]}")
    return w[0][0].float().numpy()


def signed_auroc(y: np.ndarray, x: np.ndarray) -> float:
    a = auroc(y, x)
    return a if a >= 0.5 else -(1 - a)  # negative = the incorrect steps sit on the low side


def pca(X: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(top-k directions (d, k), variance share of each) of already centred X."""
    _, S, V = torch.pca_lowrank(X, q=k, center=False, niter=6)
    return V, (S ** 2) / (X ** 2).sum()


def view(name: str, d: Path, ckpt: Path, data: Path, n_pcs: int, n_correct: int, out: Path, dev) -> dict:
    vl = val_labels(data, "ds")
    Xv, _, _ = load(d, "val", vl)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    Xvc = torch.from_numpy(Xv - mu).to(dev)
    V, share = pca(Xvc, n_pcs)
    Vs, share_s = pca(torch.from_numpy((Xv - mu) / sd).to(dev), n_pcs)
    w = torch.from_numpy(head_weight(ckpt)).to(dev)
    wu = w / w.norm()
    Vo, _ = pca(Xvc - (Xvc @ wu)[:, None] * wu[None], 1)
    del Xvc
    rows, ys, logits, P, Ps, O, per = [], [], [], [], [], [], {}
    rng = np.random.default_rng(0)
    for s in TEST:
        X, y, lg = load(d, s)
        Xc = torch.from_numpy(X - mu).to(dev)
        p = (Xc @ V).cpu().numpy()
        ps = (torch.from_numpy((X - mu) / sd).to(dev) @ Vs).cpu().numpy()
        o = (Xc @ Vo[:, 0]).cpu().numpy()
        per[s] = {"ap": fast_ap(y, lg), "auroc": auroc(y, lg), "n": int(len(y)), "pos": float(y.mean())}
        ys.append(y), logits.append(lg), P.append(p), Ps.append(ps), O.append(o)
        keep = np.flatnonzero(y == 1).tolist()
        cor = np.flatnonzero(y == 0)
        keep += sorted(rng.choice(cor, min(n_correct, len(cor)), replace=False).tolist())
        for i in sorted(keep):
            rows.append([s, int(y[i]), float(lg[i])] + p[i].tolist() + ps[i].tolist() + [float(o[i])])
    y, lg, P, Ps, O = map(np.concatenate, (ys, logits, P, Ps, O))
    with open(out / f"{name}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["set", "y", "logit"] + [f"pc{j + 1}" for j in range(n_pcs)]
                    + [f"spc{j + 1}" for j in range(n_pcs)] + ["orth"])
        wr.writerows(rows)
    res = {"states": str(d), "ckpt": str(ckpt),
           "logit": {"pooled_ap": fast_ap(y, lg), "pooled_auroc": auroc(y, lg),
                     "mean_ap": float(np.mean([v["ap"] for v in per.values()])),
                     "mean_auroc": float(np.mean([v["auroc"] for v in per.values()])), "per_set": per},
           "pc_auroc": [signed_auroc(y, P[:, j]) for j in range(n_pcs)],
           "spc_auroc": [signed_auroc(y, Ps[:, j]) for j in range(n_pcs)],
           "pc_share": share.cpu().tolist(), "spc_share": share_s.cpu().tolist(),
           "orth_auroc": signed_auroc(y, O), "cos_w_pc1": float((wu @ V[:, 0]).abs()),
           "n_test": int(len(y)), "n_val": int(len(Xv))}
    print(f"[view] {name}: logit 8-set PR-AUC {res['logit']['mean_ap']:.4f} AUROC {res['logit']['mean_auroc']:.4f} | "
          f"PC AUROC raw {[round(v, 3) for v in res['pc_auroc'][:6]]} std {[round(v, 3) for v in res['spc_auroc'][:6]]} | "
          f"orth {res['orth_auroc']:.3f} | |cos(w, PC1)| {res['cos_w_pc1']:.3f}", flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tok", nargs="+", required=True, help="name=states_dir=read_token.pt")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_pcs", type=int, default=10)
    p.add_argument("--n_correct", type=int, default=600)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for spec in a.tok:
        name, d, ck = spec.split("=")
        res[name] = view(name, Path(d), Path(ck), a.data, a.n_pcs, a.n_correct, a.out, dev)
        (a.out / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
