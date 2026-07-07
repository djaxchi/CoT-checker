"""parametric_retrieval_geometry_v0: build steering directions for exp 1.

Assembles a tiny steering-vector file (no 3.5 GB ae.pt on the cluster) with
several directions at the block-23 residual (= hidden_states[24], the layer
whose SAE latents carry the answer-commitment signal):

  sae_dec_<feat>   unit SAE decoder column of a top feature (its "write"
                   direction; steering it = injecting that feature)
  dense_diff       unit (mean direct_retrieval - mean non_retrieved) of the
                   stored final-prompt-token states (the retrieval axis, a
                   NON-sparse control: does any retrieval direction move
                   behavior, or only the feature?)
  random           unit random gaussian (matched-norm perturbation control)

Also stores the mean residual norm at the decision-point positions so the
steering sweep can be calibrated in interpretable units.

  python scripts/parametric_retrieval/prg_build_steer_dirs.py \
      --out_dir runs/parametric_retrieval_geometry_v0 \
      --sae_root data/public_sae/andyrdt-qwen2.5-7b-instruct \
      --features 58264 88965
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


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


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

    K = args.hs
    block = K - 1
    rng = np.random.default_rng(args.seed)
    dirs, meta = {}, {"hs_idx": K, "block": block, "features": args.features}

    # ---- SAE decoder columns ----------------------------------------------
    sd = torch.load(args.sae_root / f"resid_post_layer_{block}"
                    / f"trainer_{args.trainer}" / "ae.pt", map_location="cpu")
    dec = sd["decoder.weight"].numpy()  # (3584, dict)
    dim = dec.shape[0]
    for f in args.features:
        dirs[f"sae_dec_{f}"] = unit(dec[:, f].astype(np.float32))
    del sd, dec

    # ---- dense retrieval axis from stored final-prompt states -------------
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
    mu_dr = H[fp[fp.cls == "direct_retrieval"].row_pos.to_numpy()].mean(0)
    mu_nr = H[fp[fp.cls == "non_retrieved"].row_pos.to_numpy()].mean(0)
    dirs["dense_diff"] = unit(mu_dr - mu_nr)
    dirs["random"] = unit(rng.standard_normal(dim).astype(np.float32))

    # decision-point residual norm (final prompt token) for calibration
    norms = np.linalg.norm(H[fp.row_pos.to_numpy()], axis=1)
    meta["resid_norm_final_prompt_mean"] = float(norms.mean())
    meta["resid_norm_final_prompt_p50"] = float(np.median(norms))

    # cosine table between directions (are the arms distinct?)
    names = list(dirs)
    cos = {a: {b: round(float(dirs[a] @ dirs[b]), 3) for b in names}
           for a in names}
    meta["cosines"] = cos

    out = args.out_dir / "sae" / f"steer_dirs_layer{block}.npz"
    np.savez(out, dim=np.int64(dim),
             names=np.array(names),
             mat=np.stack([dirs[n] for n in names]).astype(np.float32),
             meta=np.array(json.dumps(meta)))
    print(f"[steer-dirs] wrote {out}")
    print(f"[steer-dirs] resid norm at final prompt token: "
          f"mean {meta['resid_norm_final_prompt_mean']:.1f}")
    print("[steer-dirs] direction cosines:")
    print(pd.DataFrame(cos).to_string())


if __name__ == "__main__":
    main()
