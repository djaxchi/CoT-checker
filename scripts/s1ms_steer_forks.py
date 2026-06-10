#!/usr/bin/env python3
"""Step 1 causal steering test on reserved PRM800K forks.

Tests whether the correctness direction at an intermediate layer is *causal* for
the model's own next-step preference, before investing in full generation.

Direction: d = mean(error activations) - mean(correct activations) at the chosen
layer, computed from the multi-layer PRM800K cache (probe_train_40k_L{idx}). We
steer TOWARD correct by adding -alpha*d to that layer's residual output (a
forward hook at the matching decoder block), for the whole forward.

Metric: for each reserved fork (shared prefix + correct/incorrect sibling), the
length-normalized log-prob of the correct continuation minus the incorrect one,
margin = logp(correct) - logp(incorrect). If steering toward "correct" raises the
margin (and a random direction of equal norm does not), the direction is causal.

Outputs <out_dir>/steer_forks_L{idx}.json and steer_forks_L{idx}.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from encode_prm800k_hidden_states import build_prompt_prefix  # noqa: E402


def get_decoder_layers(model):
    for attr in ("model", "transformer", "gpt_neox"):
        base = getattr(model, attr, None)
        if base is not None:
            for la in ("layers", "h"):
                layers = getattr(base, la, None)
                if layers is not None:
                    return layers
    raise RuntimeError("could not locate decoder layer list on model")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--ml_cache_dir", type=Path, required=True,
                   help="Multi-layer cache with probe_train_40k_L{idx}_h.npy + _y.npy.")
    p.add_argument("--layer_index", type=int, required=True,
                   help="hidden_states index probed (e.g. 20). Hook is placed on "
                        "decoder block (layer_index-1) whose output is that state.")
    p.add_argument("--steering_forks", type=Path, required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    li = args.layer_index

    # ---- correctness direction from the cache (error mean - correct mean) ----
    h = np.load(args.ml_cache_dir / f"probe_train_40k_L{li}_h.npy").astype(np.float32)
    y = np.load(args.ml_cache_dir / "probe_train_40k_y.npy").astype(int)
    d = (h[y == 1].mean(0) - h[y == 0].mean(0)).astype(np.float32)  # points toward "error"
    d_norm = float(np.linalg.norm(d))
    rng = np.random.default_rng(args.seed)
    r = rng.standard_normal(d.shape).astype(np.float32)
    r = r / (np.linalg.norm(r) + 1e-12) * d_norm  # random control, same norm
    print(f"[steer] layer hidden_states index {li} -> block {li-1}; |d|={d_norm:.3f}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    layers = get_decoder_layers(model)
    block = layers[li - 1]
    d_t = torch.tensor(d, device=device, dtype=torch.float16)
    r_t = torch.tensor(r, device=device, dtype=torch.float16)

    steer_holder = {"vec": None}  # current additive steering vector (hidden,) or None

    def hook(_module, _inp, out):
        if steer_holder["vec"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + steer_holder["vec"],) + tuple(out[1:])
        return out + steer_holder["vec"]

    handle = block.register_forward_hook(hook)

    @torch.no_grad()
    def cand_logprob(problem, prefix, cand):
        pre = tok(build_prompt_prefix(problem, prefix), add_special_tokens=True, truncation=False)["input_ids"]
        cand_ids = tok(cand, add_special_tokens=False, truncation=False)["input_ids"]
        if not cand_ids:
            return float("nan")
        ids = pre + cand_ids
        inp = torch.tensor([ids], device=device)
        logits = model(inp, use_cache=False).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = torch.tensor(ids, device=device)
        # token at position t is predicted by logits[t-1]; score the candidate tokens
        idxs = range(len(pre) - 1, len(ids) - 1)
        total = sum(lp[t, tgt[t + 1]].item() for t in idxs)
        return total / len(cand_ids)  # length-normalized

    forks = [json.loads(l) for l in args.steering_forks.read_text().splitlines() if l.strip()]
    print(f"[steer] {len(forks)} forks; alphas={args.alphas}", flush=True)

    results = {"layer_index": li, "alphas": args.alphas, "n_forks": len(forks),
               "steer": {}, "random": {}, "per_fork": []}

    def run_condition(vec_fn, label):
        per_alpha = {}
        for a in args.alphas:
            steer_holder["vec"] = None if a == 0 else vec_fn(a)
            margins, lp_pos, lp_neg = [], [], []
            for fk in forks:
                pp = cand_logprob(fk["problem"], fk["prefix"], fk["positive_step"])
                nn = cand_logprob(fk["problem"], fk["prefix"], fk["negative_step"])
                lp_pos.append(pp); lp_neg.append(nn); margins.append(pp - nn)
            per_alpha[a] = {"mean_margin": float(np.nanmean(margins)),
                            "mean_lp_pos": float(np.nanmean(lp_pos)),
                            "mean_lp_neg": float(np.nanmean(lp_neg)),
                            "frac_margin_pos": float(np.mean(np.array(margins) > 0))}
            print(f"[steer] {label} a={a}: margin={per_alpha[a]['mean_margin']:+.4f} "
                  f"frac>0={per_alpha[a]['frac_margin_pos']:.2f}", flush=True)
        steer_holder["vec"] = None
        return per_alpha

    results["steer"] = run_condition(lambda a: (-a) * d_t, "steer->correct")
    results["random"] = run_condition(lambda a: (-a) * r_t, "random-dir")
    handle.remove()

    (args.out_dir / f"steer_forks_L{li}.json").write_text(json.dumps(results, indent=2))

    # margin vs alpha: steer toward correct (expect rising) vs random control (flat)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    a = args.alphas
    ax.plot(a, [results["steer"][x]["mean_margin"] for x in a], "o-", color="#225522",
            label="steer toward correct")
    ax.plot(a, [results["random"][x]["mean_margin"] for x in a], "s--", color="#999999",
            label="random direction (control)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("steering strength alpha"); ax.set_ylabel("mean margin  logp(correct) - logp(incorrect)")
    ax.set_title(f"Fork steering causality at hidden_states L{li}  ({args.model_name_or_path.split('/')[-1]})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / f"steer_forks_L{li}.png", dpi=150); plt.close(fig)
    print(f"[steer] wrote {args.out_dir}/steer_forks_L{li}.json + .png", flush=True)


if __name__ == "__main__":
    main()
