#!/usr/bin/env python3
"""Interchange only the correctness-readout component of a step (das_probe_direction_v0).

§15.8 tested the probe direction `w` by ADDING it during generation and found it
no better than a random push. It left one intervention open: activation
patching of the `w` component between matched siblings, which swaps the value a
readout sees instead of pushing it. The das_branch caches (runs/das_train,
`cache_L{12,20}.pt`) hold, per PRM800K fork, the last-8 step-token states of the
correct and the wrong sibling and the teacher-forced gold-answer margin, so this
is an evaluation of fixed subspaces with no training of the backbone:

    H_wrong <- H_wrong + Q Q^T (H_correct - H_wrong)    on the step's last 8 tokens

for Q built from the TRAIN forks only:

    mean_diff     the mean sibling difference, the matched-pair correctness axis
    lda           shrunk LDA on step-mean states (correct vs wrong), the probe
    diff_pc{k}    top-k principal directions of the per-fork mean differences
    random{k}     random orthonormal k-frames (5 seeds, averaged)
    das_k{k}      the learned DAS subspaces already on disk (reproduction gate)
    full          the whole state (oracle)

Readout: held-out margin recovery = mean(m_patched - m_wrong) / mean(m_correct -
m_wrong), on the same train/val split das_fit.py used. Prespecified GO
(docs/das_probe_direction_v0_plan.md): mean_diff or lda (k=1) recovers >= 0.15 at
either layer and beats random k=1 with paired Wilcoxon p < 0.05.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def split_items(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Exactly das_fit.do_fit's split, so recoveries are comparable to §18.1."""
    train = [it for it in items if it["split"] == "train"]
    val = [it for it in items if it["split"] != "train"]
    if not train:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(items))
        cut = int(0.7 * len(items))
        train = [items[i] for i in idx[:cut]]
        val = [items[i] for i in idx[cut:]]
    return train, val


def orth(m: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(m)
    return q


def directions(train: list[dict], d: int) -> dict[str, np.ndarray]:
    diffs = np.stack([(it["donor"].float() - it["base"].float()).mean(0).numpy() for it in train])
    out = {"mean_diff": orth(diffs.mean(0)[:, None])}
    xc = np.stack([it["donor"].float().mean(0).numpy() for it in train])
    xw = np.stack([it["base"].float().mean(0).numpy() for it in train])
    mu_c, mu_w = xc.mean(0), xw.mean(0)
    resid = np.concatenate([xc - mu_c, xw - mu_w])
    cov = resid.T @ resid / len(resid)
    lam = 0.5
    cov = (1 - lam) * cov + lam * np.trace(cov) / d * np.eye(d)
    out["lda"] = orth(np.linalg.solve(cov, mu_c - mu_w)[:, None])
    u, _, _ = np.linalg.svd(diffs.T, full_matrices=False)
    for k in (1, 4, 16):
        out[f"diff_pc{k}"] = u[:, :k].copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path, default=Path("runs/das_train"))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--max_ctx", type=int, default=1024)
    ap.add_argument("--n_random", type=int, default=5)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    import torch
    from scipy import stats
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.das_branch.das_fit import _val_recovery, cache_path

    items = torch.load(cache_path(a.out_dir, a.layer), weights_only=False)
    items = [it for it in items
             if len(it["ctx_ids"]) + max(len(c) for c in it["cand_ids"]) <= a.max_ctx]
    train, val = split_items(items)
    d = items[0]["base"].shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_name_or_path, local_files_only=a.local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_name_or_path, torch_dtype=torch.bfloat16,
                                                 local_files_only=a.local_files_only).to(device).eval()
    pad = tok.pad_token_id or tok.eos_token_id

    Qs = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in directions(train, d).items()}
    for k in (1, 16):
        for s in range(a.n_random):
            g = torch.Generator().manual_seed(1000 + s)
            Qs[f"random{k}_s{s}"] = torch.linalg.qr(torch.randn(d, k, generator=g))[0].to(device)
    for f in sorted(a.out_dir.glob(f"U_L{a.layer}_k*_s0.pt")):
        w = torch.load(f, weights_only=True)["weight"]
        Qs[f"das_k{w.shape[1]}"] = torch.linalg.qr(w.float())[0].to(device)

    res = {}
    for name, Q in Qs.items():
        r = _val_recovery(model, val, pad, device, a.layer, lambda Q=Q: Q)
        res[name] = r
        print(f"[L{a.layer}] {name}: recovery {r['recovery']:+.4f}", flush=True)
    res["full"] = _val_recovery(model, val, pad, device, a.layer, None)
    print(f"[L{a.layer}] full: recovery {res['full']['recovery']:+.4f}", flush=True)

    def pooled_random(k: int) -> np.ndarray:
        return np.mean([res[f"random{k}_s{s}"]["deltas"] for s in range(a.n_random)], 0)

    summary = {"layer": a.layer, "n_train": len(train), "n_val": len(val),
               "recovery": {k: v["recovery"] for k, v in res.items()}}
    r1 = pooled_random(1)
    summary["random1_mean_recovery"] = float(np.mean([res[f"random1_s{s}"]["recovery"] for s in range(a.n_random)]))
    summary["p_vs_random1"] = {}
    for name in ("mean_diff", "lda", "diff_pc1"):
        dd = np.array(res[name]["deltas"])
        summary["p_vs_random1"][name] = float(stats.wilcoxon(dd, r1, alternative="greater").pvalue)
    summary["go"] = bool(any(summary["recovery"][n] >= 0.15 and summary["p_vs_random1"][n] < 0.05
                             for n in ("mean_diff", "lda")))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
