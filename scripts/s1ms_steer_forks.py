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

Battery mode (--directions_npz): instead of the single cache mean-diff direction, sweep
EVERY direction in a directions_L{idx}.npz (built by build_steering_directions.py) over a
SIGNED alpha grid, and record both the fork margin (behavioural proxy) and the probe-logit
shift w.h_steered (readout). This is the cheap, no-generation Tier-0 gate that pins the
sign/usable-alpha range and gives an early causal-vs-diagnostic read before Tier-1 generation.
Outputs <out_dir>/steer_forks_battery_L{idx}.json and .png.
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


def run_battery(args, li, forks, cand_logprob, steer_holder, device) -> None:
    """Sweep every direction in args.directions_npz over a signed alpha grid.

    For each (direction, alpha) records the fork margin logp(correct)-logp(incorrect)
    (behavioural proxy) and the mean probe logit w.h_steered+b on the positive/negative
    candidate readout tokens (the diagnostic). Directions are stored toward-correct for
    treatments, so alpha>0 = toward correct; alpha<0 = toward incorrect (corruption).
    """
    z = np.load(args.directions_npz, allow_pickle=True)
    names = [str(s) for s in z["names"]]
    vectors = z["vectors"].astype(np.float32)               # (k, hidden) unit
    s_layer = float(z["s_layer"])
    w_score = torch.tensor(z["w_score"].astype(np.float32), device=device)  # toward incorrect
    b_score = float(z["b_score"])
    assert int(z["layer_index"]) == li, \
        f"directions built for L{int(z['layer_index'])} but --layer_index {li}"

    mags = sorted({abs(a) for a in args.alphas} - {0.0})
    alphas = [-m for m in reversed(mags)] + [0.0] + mags
    print(f"[battery] L{li} s_layer={s_layer:.3f} dirs={names} alphas={alphas}", flush=True)

    @torch.no_grad()
    def probe_logit():
        h = steer_holder["last_hidden"][0].float()
        return float(w_score @ h + b_score)

    out = {"layer_index": li, "s_layer": s_layer, "alphas": alphas,
           "n_forks": len(forks), "by_dir": {}}
    for di, name in enumerate(names):
        u = torch.tensor(vectors[di], device=device, dtype=torch.float16)
        per = {}
        for a in alphas:
            steer_holder["vec"] = None if a == 0 else (a * s_layer) * u
            margins, plog_pos, plog_neg = [], [], []
            for fk in forks:
                pp = cand_logprob(fk["problem"], fk["prefix"], fk["positive_step"])
                plog_pos.append(probe_logit())
                nn = cand_logprob(fk["problem"], fk["prefix"], fk["negative_step"])
                plog_neg.append(probe_logit())
                margins.append(pp - nn)
            per[f"{a:g}"] = {
                "mean_margin": float(np.nanmean(margins)),
                "frac_margin_pos": float(np.mean(np.array(margins) > 0)),
                "mean_probe_logit_pos": float(np.nanmean(plog_pos)),
                "mean_probe_logit_neg": float(np.nanmean(plog_neg)),
            }
            print(f"[battery] {name:18s} a={a:+.2g}: margin={per[f'{a:g}']['mean_margin']:+.4f} "
                  f"probe(pos)={per[f'{a:g}']['mean_probe_logit_pos']:+.3f}", flush=True)
        steer_holder["vec"] = None
        out["by_dir"][name] = per

    jpath = args.out_dir / f"steer_forks_battery_L{li}.json"
    jpath.write_text(json.dumps(out, indent=2))

    # margin vs alpha, one line per direction (treatment expected monotone up; controls flat)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for name in names:
        per = out["by_dir"][name]
        ys = [per[f"{a:g}"]["mean_margin"] for a in alphas]
        ps = [per[f"{a:g}"]["mean_probe_logit_pos"] for a in alphas]
        style = "o-" if name in ("probe", "sparse_restricted") else "s--"
        ax[0].plot(alphas, ys, style, label=name, alpha=0.85)
        ax[1].plot(alphas, ps, style, label=name, alpha=0.85)
    for a_, ttl, yl in ((ax[0], "fork margin vs alpha", "logp(correct)-logp(incorrect)"),
                        (ax[1], "probe logit (pos cand) vs alpha", "w.h_steered + b")):
        a_.axvline(0, color="k", lw=0.5); a_.axhline(0, color="k", lw=0.4)
        a_.set_xlabel("alpha  (+ = toward correct)"); a_.set_ylabel(yl)
        a_.set_title(ttl); a_.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Tier-0 fork steering battery, hidden_states L{li}  "
                 f"({args.model_name_or_path.split('/')[-1]})")
    fig.tight_layout(); fig.savefig(args.out_dir / f"steer_forks_battery_L{li}.png", dpi=150)
    plt.close(fig)
    print(f"[battery] wrote {jpath} + .png", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--ml_cache_dir", type=Path, default=None,
                   help="Multi-layer cache with probe_train_40k_L{idx}_h.npy + _y.npy "
                        "(legacy mean-diff mode; not needed with --directions_npz).")
    p.add_argument("--directions_npz", type=Path, default=None,
                   help="directions_L{idx}.npz from build_steering_directions.py -> "
                        "battery mode (sweep all directions, signed alpha).")
    p.add_argument("--layer_index", type=int, required=True,
                   help="hidden_states index probed (e.g. 20). Hook is placed on "
                        "decoder block (layer_index-1) whose output is that state.")
    p.add_argument("--steering_forks", type=Path, required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0],
                   help="alpha magnitudes; battery mode mirrors them to negatives + 0.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.directions_npz is None and args.ml_cache_dir is None:
        p.error("provide either --directions_npz (battery) or --ml_cache_dir (legacy)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    li = args.layer_index

    # ---- model + capturing steering hook -------------------------------
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

    # vec: current additive steering vector (hidden,) or None.
    # last_hidden: the (steered) block output at the last token of the most recent forward,
    # so a probe logit w.h + b can be read off the candidate's readout position.
    steer_holder = {"vec": None, "last_hidden": None}

    def hook(_module, _inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if steer_holder["vec"] is not None:
            o = o + steer_holder["vec"]
        steer_holder["last_hidden"] = o[:, -1, :].detach()
        if isinstance(out, tuple):
            return (o,) + tuple(out[1:])
        return o

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

    # ---- battery mode: every direction in the npz, signed alpha ---------
    if args.directions_npz is not None:
        run_battery(args, li, forks, cand_logprob, steer_holder, device)
        handle.remove()
        return

    # ---- legacy mean-diff direction from the cache (error - correct) ----
    h = np.load(args.ml_cache_dir / f"probe_train_40k_L{li}_h.npy").astype(np.float32)
    y = np.load(args.ml_cache_dir / "probe_train_40k_y.npy").astype(int)
    d = (h[y == 1].mean(0) - h[y == 0].mean(0)).astype(np.float32)  # points toward "error"
    d_norm = float(np.linalg.norm(d))
    rng = np.random.default_rng(args.seed)
    r = rng.standard_normal(d.shape).astype(np.float32)
    r = r / (np.linalg.norm(r) + 1e-12) * d_norm  # random control, same norm
    d_t = torch.tensor(d, device=device, dtype=torch.float16)
    r_t = torch.tensor(r, device=device, dtype=torch.float16)
    print(f"[steer] layer hidden_states index {li} -> block {li-1}; |d|={d_norm:.3f}", flush=True)

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
