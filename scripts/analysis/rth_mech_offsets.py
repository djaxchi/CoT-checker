#!/usr/bin/env python3
"""M2 (CPU part): how large RTH's per-layer offsets are and what they write in vocabulary space.

RTH adds a trained vector deep[l] to decoder block l's output at the read position
(scripts/read_token_probe.py, deep_hooks). For every layer this reports

  norm          ||deep[l]||
  rel_read      ||deep[l]|| / median ||read state after block l|| over a bundle split
                (the read state includes the offset; row l of read.npy)
  rel_last      the same against the last step token's block l output, the state
                the offset would be compared with if the token were not there
  cos_head      cosine between deep[l] and the head weight w (w reads the final
                post-norm state, so this is only meaningful for late layers)
  top / bottom  the tokens whose unembedding rows have the largest and smallest dot
                product with g * deep[l] (g the final RMSNorm gain): a logit-lens
                reading of what the offset pushes toward and away from

plus the same vocabulary reading for the head weight (W_U w: tokens aligned with the
"incorrect" direction) and for the trained embedding's change from its start.

    python scripts/analysis/rth_mech_offsets.py --ckpt tokens/dq_correct/read_token.pt \
        --bundle <mech>/bundle --split test_2k --out m2_offsets.json
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
from scripts.read_token_mech_bundle import load_bundle  # noqa: E402


def median_norms(X: np.ndarray, chunk: int = 512) -> np.ndarray:
    """(N, rows, d) -> (rows,) median L2 norm over items, in float32 chunks."""
    norms = np.concatenate([np.linalg.norm(np.asarray(X[i:i + chunk], np.float32), axis=-1)
                            for i in range(0, len(X), chunk)])
    return np.median(norms, axis=0)


def offset_table(deep: np.ndarray, med_read: np.ndarray, med_last: np.ndarray,
                 w: np.ndarray) -> list[dict]:
    """deep (L, d), the per-layer medians (>= L rows), the head weight (d,)."""
    out = []
    wn = w / np.linalg.norm(w)
    for l, v in enumerate(deep):
        n = float(np.linalg.norm(v))
        out.append({"layer": l, "norm": n, "rel_read": n / float(med_read[l]),
                    "rel_last": n / float(med_last[l]),
                    "cos_head": float(v @ wn / n) if n > 0 else 0.0})
    return out


def vocab_top(V: torch.Tensor, W_U: torch.Tensor, decode, k: int, chunk: int = 8) -> list[dict]:
    """Top and bottom k tokens of W_U @ v for each row v of V (already gain-scaled)."""
    out = []
    for i in range(0, len(V), chunk):
        logits = V[i:i + chunk].float() @ W_U.float().T
        for row in logits:
            top, bot = torch.topk(row, k), torch.topk(-row, k)
            out.append({"top": [[decode(int(t)), round(float(s), 3)] for s, t in zip(top.values, top.indices)],
                        "bottom": [[decode(int(t)), round(float(-s), 3)]
                                   for s, t in zip(bot.values, bot.indices)]})
    return out


def load_unembed(model: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """lm_head weight (V, d), the final norm gain (d,) and the input embedding (V, d)
    from the cached checkpoint, without building the model."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    root = Path(snapshot_download(model, local_files_only=True))
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    names = {"lm_head.weight": None, "model.norm.weight": None, "model.embed_tokens.weight": None}
    for n in names:
        with safe_open(root / index[n], framework="pt") as f:
            names[n] = f.get_tensor(n)
    return names["lm_head.weight"], names["model.norm.weight"], names["model.embed_tokens.weight"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--split", default="test_2k")
    p.add_argument("--init_text", default=" correct", help="the text the embedding started from")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    deep = ck["deep"][:, -1].float()  # (L, d), the last read token's offsets
    w = ck["head"]["weight"].float().reshape(-1)
    b = load_bundle(a.bundle / a.split, arrays=("read", "last_step"))
    med_read, med_last = median_norms(b["read"]), median_norms(b["last_step"])
    table = offset_table(deep.numpy(), med_read, med_last, w.numpy())
    W_U, g, E = load_unembed(a.model)
    decode = lambda t: tok.decode([t])  # noqa: E731
    voc = vocab_top(deep * g.float(), W_U, decode, a.k)
    for row, v in zip(table, voc):
        row.update(v)
    init_ids = tok(a.init_text, add_special_tokens=False)["input_ids"]
    emb_change = ck["soft"][-1].float() - E[init_ids[-1]].float()
    head_voc, emb_voc = vocab_top(torch.stack([w, emb_change]), W_U, decode, a.k)
    windows = {f"{lo}-{hi}": float(np.linalg.norm(deep[lo:hi + 1].numpy(), axis=1).sum())
               for lo, hi in ((0, 8), (9, 17), (18, 26), (27, 35))}
    out = {"ckpt": str(a.ckpt), "split": a.split, "n_items": int(len(b["read"])), "layers": table,
           "window_norm_sum": windows, "head_vocab": head_voc,
           "embedding_change": {"norm": float(emb_change.norm()),
                                "rel_init": float(emb_change.norm() / E[init_ids[-1]].float().norm()),
                                **emb_voc},
           "median_read_norm": med_read.tolist(), "median_last_norm": med_last.tolist()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    for r in table:
        print(f"[m2] layer {r['layer']:2d} norm {r['norm']:8.3f} rel_read {r['rel_read']:.3f} "
              f"rel_last {r['rel_last']:.3f} cos_head {r['cos_head']:+.3f} top "
              f"{[t for t, _ in r['top'][:5]]}", flush=True)
    print(f"[m2] head W_U w top {[t for t, _ in head_voc['top'][:8]]} bottom "
          f"{[t for t, _ in head_voc['bottom'][:8]]}", flush=True)


if __name__ == "__main__":
    main()
