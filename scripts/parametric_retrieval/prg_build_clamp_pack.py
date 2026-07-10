"""parametric_retrieval_geometry_v0: build a clamp-intervention pack.

Unlike steer_dirs (a bare direction), a clamp pack lets the hook READ a
feature's activation at a token and SET it to a target value:

    z   = relu(enc_row . (h - b_dec) + b_enc);  z = z if z > threshold else 0
    h'  = h + (target - z) * dec_col      # dec_col unit-norm in this SAE

so the edit norm equals |target - z| and the induced activation is exactly
`target` (verified in the run). This is the principled version of "put the
'I know this' feature to baseline / boost it", the intervention the targeted
causal tests need.

Targets are grounded in the decision-point (final_prompt_token) activation
distribution per retrieval class (so "baseline" = what a non_retrieved prompt
actually looks like, not an arbitrary 0). Presets per feature:
    ablate     0.0
    baseline   non_retrieved p50
    boost      direct_retrieval p50
    boost_hi   direct_retrieval p95
    boost_xl   2 * direct_retrieval max   (far out of range: stress test)

Also carries b_dec, threshold, a fixed random unit direction and the
dense_diff retrieval axis (matched-norm controls).

  python scripts/parametric_retrieval/prg_build_clamp_pack.py \
      --out_dir runs/parametric_retrieval_geometry_v0 \
      --sae_root data/public_sae/andyrdt-qwen2.5-7b-instruct --features 58264 88965
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--sae_root", type=Path,
                    default=Path("data/public_sae/andyrdt-qwen2.5-7b-instruct"))
    ap.add_argument("--hs", type=int, default=24)
    ap.add_argument("--trainer", type=int, default=1)
    ap.add_argument("--features", type=int, nargs="+", default=[58264, 88965])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from safetensors.numpy import load_file

    K, block = args.hs, args.hs - 1
    rng = np.random.default_rng(args.seed)
    sd = torch.load(args.sae_root / f"resid_post_layer_{block}"
                    / f"trainer_{args.trainer}" / "ae.pt", map_location="cpu")
    W_enc = sd["encoder.weight"].numpy()      # (dict, 3584)
    b_enc = sd["encoder.bias"].numpy()
    dec = sd["decoder.weight"].numpy()        # (3584, dict)
    b_dec = sd["b_dec"].numpy().astype(np.float32)
    thr = float(sd["threshold"].item())
    dim = dec.shape[0]

    # decision-point activations per class, to ground the targets
    hs_dir = args.out_dir / "hidden_states"
    m = pd.read_parquet(hs_dir / "hs_meta.parquet") \
        .reset_index().rename(columns={"index": "row_pos"})
    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    cls = grading.set_index("question_id").retrieval_class
    fp = m[(m.prompt_mode == "direct")
           & (m.position_name == "final_prompt_token")].copy()
    fp["cls"] = fp.question_id.map(cls)
    H = load_file(hs_dir / f"layer_{K:02d}.safetensors")["h"].astype(np.float32)
    Hfp = H[fp.row_pos.to_numpy()]

    feats, enc_rows, enc_bias, dec_cols, targets = [], [], [], [], {}
    for f in args.features:
        pre = (Hfp - b_dec) @ W_enc[f] + b_enc[f]
        z = np.maximum(pre, 0.0)
        z = z * (z > thr)
        dr = z[(fp.cls == "direct_retrieval").to_numpy()]
        nr = z[(fp.cls == "non_retrieved").to_numpy()]
        targets[f] = {
            "ablate": 0.0,
            "baseline": float(np.percentile(nr, 50)),
            "boost": float(np.percentile(dr, 50)),
            "boost_hi": float(np.percentile(dr, 95)),
            "boost_xl": float(2 * z.max()),
        }
        feats.append(f)
        enc_rows.append(W_enc[f].astype(np.float32))
        enc_bias.append(np.float32(b_enc[f]))
        dec_cols.append(dec[:, f].astype(np.float32))
        print(f"[clamp] feat {f}: dr p50 {targets[f]['boost']:.1f} / "
              f"nr p50 {targets[f]['baseline']:.1f} / max {z.max():.1f}",
              flush=True)

    mu_dr = Hfp[(fp.cls == "direct_retrieval").to_numpy()].mean(0)
    mu_nr = Hfp[(fp.cls == "non_retrieved").to_numpy()].mean(0)
    dense = (mu_dr - mu_nr)
    dense /= max(np.linalg.norm(dense), 1e-8)
    rand = rng.standard_normal(dim).astype(np.float32)
    rand /= np.linalg.norm(rand)

    out = args.out_dir / "sae" / f"clamp_pack_layer{block}.npz"
    np.savez(out,
             features=np.array(feats, np.int64),
             enc_rows=np.stack(enc_rows),
             b_enc=np.array(enc_bias, np.float32),
             dec_cols=np.stack(dec_cols),
             b_dec=b_dec, threshold=np.float32(thr),
             dense_diff=dense.astype(np.float32), random=rand,
             hs_idx=np.int64(K), block=np.int64(block),
             targets=np.array(json.dumps(targets)),
             resid_norm=np.float32(np.linalg.norm(Hfp, axis=1).mean()))
    print(f"[clamp] wrote {out}")


if __name__ == "__main__":
    main()
