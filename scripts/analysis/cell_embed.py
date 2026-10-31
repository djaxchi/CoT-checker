#!/usr/bin/env python3
"""A sequence cell's pooled embedding, written like an encode (read_token_v1 Stage K).

The leaderboard's best Qwen3-8B cell, `step_tokens x transformer:d512`, classifies a
step from the 512-d masked mean of its encoder output (the input of its one-layer
head). This writes that vector for every item of the given span-store splits, in the
npz/meta layout of scripts/encode_pooled_multilayer.py (key `z`, one layer, index -1),
so the class-separation views read it like any other representation. The head's
P(incorrect) is kept as `head_p_correct = 1 - P(incorrect)`, and the test AUROC it
gives is printed next to the cell's own, as a check that the model was rebuilt exactly.

    python scripts/analysis/cell_embed.py --cell_dir <cells>/step_tokens__transformer_d512_l2_f2048_h8__seed42 \
        --store <repstore>/step_spans --splits val:val_5k test:test_2k --out <dir>
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
from scripts.analysis.drv_embed import load_drv  # noqa: E402
from scripts.train_rep_learner_cell import build_handles  # noqa: E402
from src.harness.spanloader import SpanLoader  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402


def embed(model: torch.nn.Module, loader: SpanLoader, batch: int = 256):
    """(z (n, d_model), logits (n,)) in item order."""
    got = {}
    hook = model.head.register_forward_pre_hook(lambda _m, a: got.__setitem__("z", a[0]))
    Z, L = [], []
    n = len(loader.labels)
    try:
        with torch.no_grad():
            for i in range(0, n, batch):
                x, mask, _ = loader.collate(np.arange(i, min(n, i + batch)))
                L.append(model(x, mask).float().cpu().numpy())
                Z.append(got["z"].float().cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(Z), np.concatenate(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell_dir", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True, help="span store root with <stem>/shard_*")
    p.add_argument("--splits", nargs="+", required=True,
                   help="name:stem[:items.jsonl], the optional jsonl restricting the split to its uids")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.loads((a.cell_dir / "results.json").read_text())
    t_max = int(res["protocol"]["t_max"])
    model = None
    for spec in a.splits:
        name, stem, *keep = spec.split(":")
        view = ShardedRepSplit(a.store / stem)
        handles, meta = build_handles(view)
        if keep:
            want = {json.loads(l)["uid"] for l in open(keep[0]) if l.strip()}
            idx = [i for i, m in enumerate(meta) if m.get("uid") in want]
            if len(idx) != len(want):
                raise SystemExit(f"{stem}: {len(idx)} of {len(want)} uids found")
            handles, meta = [handles[i] for i in idx], [meta[i] for i in idx]
        model = model or load_drv(a.cell_dir, int(view.spec.dim), dev)
        loader = SpanLoader(handles, t_max, dev, preload=True)
        Z, L = embed(model, loader)
        y = loader.labels.astype(np.int8)
        p_inc = 1 / (1 + np.exp(-L))
        np.savez(a.out / f"{name}.shard0.npz", layers=np.array([-1]), y=y, kept=np.ones(len(y), bool),
                 head_p_correct=(1 - p_inc).astype(np.float32), z=Z[:, None].astype(np.float16))
        with open(a.out / f"{name}.shard0.meta.jsonl", "w") as f:
            for m, yy in zip(meta, y):
                f.write(json.dumps({"uid": m.get("uid"), "label": int(yy)}) + "\n")
        if stem == res["protocol"]["stems"]["test"]:
            from sklearn.metrics import roc_auc_score
            print(f"[check] {stem} AUROC from the rebuilt cell {roc_auc_score(y, L):.4f}, "
                  f"cell's own {res['in_domain']['auroc']:.4f}", flush=True)
        print(f"[cell_embed] {name}: {len(y)} items, z {Z.shape[1]}-d", flush=True)


if __name__ == "__main__":
    main()
