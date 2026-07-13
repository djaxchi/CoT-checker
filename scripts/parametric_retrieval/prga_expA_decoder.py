"""parametric_retrieval_access_v1 Experiment A method B: fact-general
candidate-ranking decoder s(h, a) = <A h, e(a)>.

e(a) is the FROZEN mean input embedding of the tokens of " <answer>", so the
decoder ranks full answer strings (not just first tokens) and is the
sequence-level complement to the first-token logit lens. A (d x d) is
trained with softmax cross-entropy over each row's matched candidate set,
on TRAIN-fact extraction rows only; evaluation is on unseen val/test facts
(and unseen answer identities to the extent the candidate pools differ).

Per cell (hs_idx x position) reports hits@1 / hits@5 / MRR split by
is_correct, plus the first-token logit-lens numbers on the same rows for
direct comparison.

Outputs: expA/decoder_results.csv, expA/decoder_manifest.json

  python scripts/parametric_retrieval/prga_expA_decoder.py --local_files_only
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

from scripts.parametric_retrieval import prga_common as C  # noqa: E402

CELLS = [(20, "final_prompt_token"), (24, "final_prompt_token"),
         (26, "final_prompt_token"), (27, "final_prompt_token"),
         (28, "final_prompt_token"), (27, "answer_prefix"),
         (28, "answer_prefix")]


def answer_embeddings(model_name: str, local_files_only: bool, device,
                      answers: list[str]):
    """Frozen mean input embedding of ' <answer>' per unique answer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name,
                                        local_files_only=local_files_only)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files_only,
            dtype=torch.bfloat16, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files_only,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    W = model.model.embed_tokens.weight.detach().to(torch.float32)
    del model
    E = torch.zeros(len(answers), W.shape[1])
    for i, a in enumerate(answers):
        ids = tok(" " + str(a), add_special_tokens=False)["input_ids"]
        if ids:
            E[i] = W[torch.tensor(ids)].mean(dim=0)
    return E.to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--model_name_or_path", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as tF

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = args.out_dir / "expA"
    exp_dir.mkdir(exist_ok=True)

    store = C.HSStore(args.out_dir)
    meta = store.meta
    cands_raw = json.loads((args.out_dir / "candidates.json").read_text())
    cands = {(str(c["fact_id"]), c["direction"]): c for c in cands_raw}

    # global answer vocabulary + per-group candidate index rows
    vocab: dict[str, int] = {}

    def vid(a: str) -> int:
        if a not in vocab:
            vocab[a] = len(vocab)
        return vocab[a]

    group_rows = {}
    max_k = 0
    for key, c in cands.items():
        ids = [vid(c["gold"])] + [vid(n) for n in c["negatives"]]
        group_rows[key] = ids
        max_k = max(max_k, len(ids))
    answers = [a for a, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    E = answer_embeddings(args.model_name_or_path, args.local_files_only,
                          device, answers)
    E = tF.normalize(E, dim=-1)
    print(f"[expA] {len(answers)} unique candidate answers, max_k={max_k}",
          flush=True)

    # logit-lens comparison numbers on identical rows
    lens = pd.read_parquet(args.out_dir / "logitlens" / "scores.parquet")

    results = []
    d_model = store.layer(CELLS[0][0]).shape[1]
    for hs_idx, position in CELLS:
        mask = (meta.position_name == position).to_numpy()
        rows = meta[mask].reset_index(drop=True)
        H = store.layer(hs_idx)[mask].astype(np.float32)
        keys = list(zip(rows.fact_id.astype(str), rows.direction))
        cand_idx = np.full((len(rows), max_k), -1, dtype=np.int64)
        for i, k in enumerate(keys):
            ids = group_rows[k]
            cand_idx[i, :len(ids)] = ids
        X = torch.tensor(H, device=device)
        X = tF.normalize(X, dim=-1)
        CI = torch.tensor(cand_idx, device=device)
        CM = CI >= 0
        CI = CI.clamp(min=0)
        split = rows.split.to_numpy()
        tr = np.flatnonzero(split == "train")

        A = torch.zeros(d_model, E.shape[1], device=device,
                        requires_grad=True)
        opt = torch.optim.AdamW([A], lr=args.lr,
                                weight_decay=args.weight_decay)
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        for ep in range(args.epochs):
            perm = torch.randperm(len(tr), generator=g).numpy()
            tot, nb = 0.0, 0
            for s in range(0, len(tr), args.batch_size):
                b = tr[perm[s:s + args.batch_size]]
                xb = X[b] @ A                                # (B, d)
                eb = E[CI[b]]                                # (B, K, d)
                logits = torch.einsum("bd,bkd->bk", xb, eb)
                logits = logits.masked_fill(~CM[b], float("-inf"))
                loss = tF.cross_entropy(
                    logits, torch.zeros(len(b), dtype=torch.long,
                                        device=device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
                nb += 1
            print(f"[expA] hs{hs_idx}/{position} epoch {ep}: "
                  f"loss {tot / nb:.3f}", flush=True)

        with torch.no_grad():
            xa = X @ A
            ranks = np.zeros(len(rows), dtype=np.int64)
            for s in range(0, len(rows), 4096):
                e = min(s + 4096, len(rows))
                logits = torch.einsum("bd,bkd->bk", xa[s:e], E[CI[s:e]])
                logits = logits.masked_fill(~CM[s:e], float("-inf"))
                gold = logits[:, 0]
                ranks[s:e] = (1 + (logits[:, 1:] > gold[:, None])
                              .sum(dim=1).cpu().numpy())
        rows = rows.assign(dec_rank=ranks)
        lens_cell = lens[(lens.hs_idx == hs_idx)
                         & (lens.position_name == position)][
            ["row_id", "gold_rank"]].set_index("row_id")
        rows = rows.join(lens_cell.rename(
            columns={"gold_rank": "lens_rank"}), on="row_id")
        for spl in ["val", "test"]:
            for corr in [True, False]:
                sub = rows[(rows.split == spl)
                           & (rows.is_correct == corr)]
                if not len(sub):
                    continue
                results.append({
                    "hs_idx": hs_idx, "position": position, "split": spl,
                    "is_correct": corr, "n": len(sub),
                    "dec_hits1": float((sub.dec_rank == 1).mean()),
                    "dec_hits5": float((sub.dec_rank <= 5).mean()),
                    "dec_mrr": float((1 / sub.dec_rank).mean()),
                    "lens_hits1": float((sub.lens_rank == 1).mean()),
                    "lens_hits5": float((sub.lens_rank <= 5).mean()),
                })
    res = pd.DataFrame(results)
    res.to_csv(exp_dir / "decoder_results.csv", index=False)
    (exp_dir / "decoder_manifest.json").write_text(json.dumps(
        {"cells": CELLS, "epochs": args.epochs, "lr": args.lr,
         "weight_decay": args.weight_decay, "seed": args.seed,
         "n_answers": len(answers)}, indent=2))
    print(res.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
