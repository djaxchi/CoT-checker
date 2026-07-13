"""parametric_retrieval_access_v1 stage 4 / Experiment A (method A):
logit-lens answer-content readout over all layers x positions.

For every extracted row (instance x position) and every hidden_states index,
apply the model's FINAL RMSNorm + unembedding to the stored residual state
and score the fact's candidate set (gold + matched hard negatives from
candidates.json) by the FIRST token of " <candidate>". Negatives whose first
token collides with the gold's are dropped from that group. Rows where the
gold answer is a single token (gold_n_tokens == 1) carry gold_single=True;
that subset supports the clean one-token claim, the rest is the first-token
approximation.

Reports, per row x layer: gold logit, gold-minus-best-negative margin, gold
rank, candidate count. The paired success-vs-fail comparison and heatmaps are
computed downstream from scores.parquet (aggregates.csv is a convenience
summary per layer x position x is_correct x gold_single).

Outputs (in --out_dir/logitlens):
  scores.parquet
  aggregates.csv

  python scripts/parametric_retrieval/prga_logitlens.py \
      --out_dir runs/parametric_retrieval_access_v1 \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct --local_files_only
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


def load_norm_and_unembed(model_name: str, local_files_only: bool,
                          device, dtype_str: str):
    """Final RMSNorm weight + lm_head weight (+eps), model freed after."""
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, dtype_str)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files_only, dtype=dtype,
            low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files_only, torch_dtype=dtype,
            low_cpu_mem_usage=True)
    norm_w = model.model.norm.weight.detach().to(device, torch.float32)
    w_u = model.lm_head.weight.detach().to(device, torch.float32)
    eps = float(model.config.rms_norm_eps)
    del model
    return norm_w, w_u, eps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--model_name_or_path", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--model_dtype",
                    choices=["bfloat16", "float16", "float32"],
                    default="bfloat16")
    ap.add_argument("--chunk_rows", type=int, default=2048)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ll_dir = args.out_dir / "logitlens"
    out_scores = ll_dir / "scores.parquet"
    if out_scores.exists() and not args.force:
        sys.exit(f"refusing to overwrite {out_scores}; pass --force")
    ll_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors.numpy import load_file
    from transformers import AutoTokenizer

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    hs_dir = args.out_dir / "hidden_states_v1"
    meta = pd.read_parquet(hs_dir / "hs_meta.parquet")
    layer_files = sorted(hs_dir.glob("layer_*.safetensors"))
    if not layer_files:
        sys.exit(f"no merged layer files in {hs_dir}")

    # gold token length per instance (single-token subset flag)
    gold_n = {}
    for ln in (args.out_dir / "generations.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            gold_n[r["instance_id"]] = r.get("gold_n_tokens")
    meta = meta.copy()
    meta["gold_single"] = meta.instance_id.map(gold_n) == 1

    # candidate first-token ids per (fact_id, direction)
    def first_tok(text: str) -> int:
        ids = tok(" " + str(text), add_special_tokens=False)["input_ids"]
        return int(ids[0]) if ids else -1

    groups: dict[tuple[str, str], dict] = {}
    for c in json.loads((args.out_dir / "candidates.json").read_text()):
        g_id = first_tok(c["gold"])
        neg_ids, seen = [], {g_id}
        for n in c["negatives"]:
            t = first_tok(n)
            if t >= 0 and t not in seen:
                seen.add(t)
                neg_ids.append(t)
        groups[(str(c["fact_id"]), c["direction"])] = {
            "gold_id": g_id, "neg_ids": neg_ids}

    key = list(zip(meta.fact_id.astype(str), meta.direction))
    known = np.array([k in groups for k in key])
    if not known.all():
        print(f"[logitlens] dropping {(~known).sum()} rows without candidate "
              f"sets", flush=True)
        meta = meta[known].reset_index(drop=True)
        key = [k for k, ok in zip(key, known) if ok]
    max_k = max(len(g["neg_ids"]) for g in groups.values()) + 1
    n_rows = len(meta)
    cand_ids = np.zeros((n_rows, max_k), dtype=np.int64)
    cand_mask = np.zeros((n_rows, max_k), dtype=bool)
    for i, k in enumerate(key):
        g = groups[k]
        ids = [g["gold_id"]] + g["neg_ids"]
        cand_ids[i, :len(ids)] = ids
        cand_mask[i, :len(ids)] = True
    cand_ids_t = torch.tensor(cand_ids, device=device)
    cand_mask_t = torch.tensor(cand_mask, device=device)

    norm_w, w_u, eps = load_norm_and_unembed(
        args.model_name_or_path, args.local_files_only, device,
        args.model_dtype)

    parts = []
    for lf in layer_files:
        k = int(lf.stem.split("_")[1])
        h_all = load_file(str(lf))["h"]
        if len(h_all) != len(known):
            sys.exit(f"row mismatch in {lf.name}: {len(h_all)} vs {len(known)}")
        h_all = h_all[known] if not known.all() else h_all
        gold_logit = np.empty(n_rows, dtype=np.float32)
        margin = np.empty(n_rows, dtype=np.float32)
        rank = np.empty(n_rows, dtype=np.int32)
        for s in range(0, n_rows, args.chunk_rows):
            e = min(s + args.chunk_rows, n_rows)
            x = torch.tensor(h_all[s:e], device=device,
                             dtype=torch.float32)
            rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            z = (rms * norm_w) @ w_u.T  # (chunk, vocab)
            zc = torch.gather(z, 1, cand_ids_t[s:e])
            zc = zc.masked_fill(~cand_mask_t[s:e], float("-inf"))
            g = zc[:, 0]
            neg = zc[:, 1:].max(dim=1).values
            gold_logit[s:e] = g.cpu().numpy()
            margin[s:e] = (g - neg).cpu().numpy()
            rank[s:e] = (1 + (zc[:, 1:] > g[:, None]).sum(dim=1)).cpu().numpy()
        part = meta[["row_id", "instance_id", "fact_id", "direction",
                     "position_name", "split", "is_correct",
                     "gold_single"]].copy()
        part["hs_idx"] = k
        part["gold_logit"] = gold_logit
        part["margin"] = margin
        part["gold_rank"] = rank
        part["n_candidates"] = cand_mask.sum(axis=1)
        parts.append(part)
        print(f"[logitlens] layer {k:02d} done", flush=True)
    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(out_scores, index=False)

    agg = (scores.groupby(["hs_idx", "position_name", "is_correct",
                           "gold_single"], dropna=False, observed=True)
           .agg(n=("margin", "size"),
                mean_margin=("margin", "mean"),
                hits_at_1=("gold_rank", lambda r: float((r == 1).mean())),
                hits_at_5=("gold_rank", lambda r: float((r <= 5).mean())),
                mrr=("gold_rank", lambda r: float((1.0 / r).mean())))
           .reset_index())
    agg.to_csv(ll_dir / "aggregates.csv", index=False)
    print(f"[logitlens] wrote {out_scores} ({len(scores)} rows) and "
          f"aggregates.csv ({len(agg)} cells)", flush=True)


if __name__ == "__main__":
    main()
