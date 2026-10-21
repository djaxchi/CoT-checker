#!/usr/bin/env python3
"""DRV's pooled embedding as a vector store (read_token_v1, representation views).

DRV (Depth-Read Verifier, Stage F-B) is the d512 transformer over the read
token's states at blocks 3, 7, ..., 31, 34 and its final state (the rt_layers
store of scripts/read_token_store.py). Its head is one linear layer on the
masked mean of the encoder output, so that 512-d mean is the representation
DRV classifies. This writes it for every item of every rt_layers split, in the
split's own shard layout, order, meta and labels, so
scripts/analysis/prm_geometry.py reads it like any other store. The head's
logits are kept alongside (logits.npy per unit), and the test AUROC they give
is printed next to the cell's own, as a check that the model was rebuilt
exactly.

    python scripts/analysis/drv_embed.py \
        --cell_dir <stage_f>/cells_fab/step_tokens__transformer_d512_l2_f2048_h8__seed42 \
        --layers_root <L>/prm/rt_layers --out_root <L>/prm/drv_embed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.read_token_store import write_store  # noqa: E402
from src.harness.learners import build_learner  # noqa: E402
from src.repstore.store import VECTOR, RepSpec, RepSplit  # noqa: E402


def load_drv(cell_dir: Path, d: int, dev) -> torch.nn.Module:
    res = json.loads((cell_dir / "results.json").read_text())
    if res["protocol"].get("rescale", "none") != "none":
        raise SystemExit(f"{cell_dir} was trained with rescale {res['protocol']['rescale']!r}")
    model = build_learner(res["learner"], d, t_max=res["protocol"]["t_max"],
                          dropout=res["protocol"]["dropout"])
    model.load_state_dict(torch.load(cell_dir / "model.pt", map_location="cpu"))
    return model.to(dev).eval()


def embed(model: torch.nn.Module, X: np.ndarray, dev, batch: int = 4096):
    """(z (n, d_model), logits (n,)) for items X (n, T, d), all T rows real."""
    got = {}
    hook = model.head.register_forward_pre_hook(lambda _m, a: got.__setitem__("z", a[0]))
    Z, L = [], []
    try:
        with torch.no_grad():
            for i in range(0, len(X), batch):
                x = torch.from_numpy(np.array(X[i:i + batch], dtype=np.float32)).to(dev)
                L.append(model(x, torch.ones(x.shape[:2], device=dev)).float().cpu().numpy())
                Z.append(got["z"].float().cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(Z), np.concatenate(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell_dir", type=Path, required=True)
    p.add_argument("--layers_root", type=Path, required=True, help="rt_layers root with <split>/shard_*")
    p.add_argument("--out_root", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    for split in sorted(x for x in a.layers_root.iterdir() if x.is_dir()):
        ys, ls = [], []
        for u in sorted(x for x in split.iterdir() if x.is_dir() and x.name.startswith("shard_")):
            rs = RepSplit(u)
            R = np.unique(rs.lengths)
            if len(R) != 1:
                raise SystemExit(f"{u}: items of different lengths {R}")
            model = model or load_drv(a.cell_dir, rs.spec.dim, dev)
            # the cell's read_span: row 0 is the boundary slot, skipped
            X = np.asarray(rs.h).reshape(len(rs), int(R[0]), -1)[:, 1:]
            Z, L = embed(model, X, dev)
            out = a.out_root / split.name / u.name
            write_store(out, Z.astype(np.float32), np.ones(len(rs)), rs.y, rs.meta(),
                        RepSpec(name="drv_embed", kind=VECTOR, dim=Z.shape[1], layer=-1,
                                backbone=rs.spec.backbone, source_split=split.name,
                                prompt_style="verifier", readout=f"DRV pooled z ({a.cell_dir.name})",
                                git_commit=rs.spec.git_commit))
            np.save(out / "logits.npy", L)
            ys.append(np.asarray(rs.y))
            ls.append(L)
        if split.name == "test_2k":
            from sklearn.metrics import roc_auc_score
            res = json.loads((a.cell_dir / "results.json").read_text())
            print(f"[check] test_2k AUROC from rebuilt DRV "
                  f"{roc_auc_score(np.concatenate(ys), np.concatenate(ls)):.4f}, cell's own "
                  f"{res['in_domain']['auroc']:.4f}", flush=True)
        print(f"[drv_embed] {split.name}: {sum(len(y) for y in ys)} items", flush=True)


if __name__ == "__main__":
    main()
