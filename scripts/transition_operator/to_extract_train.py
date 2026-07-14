"""transition_operator_v0 Stage 2a: extract training arrays per transition (L20).

Per FORK (pre context, shared by both branches):
  S_prev        h_L20 at the pre boundary "\n"        (encoder input, patch base)
  pre_logits    boundary next-token logits (fp16)     (dL = post - pre)
  belief_pre    8-cand softmax belief with the Stage-0 suffix
Per TRANSITION (branch):
  H_steps       h_L20 at step tokens, cap 192, fp16 + n_steps  (encoder input)
  S_post        h_L20 at the post boundary            (delta_true for audits)
  post_logits   boundary next-token logits (fp16)     (L_A target)
  belief_post   8-cand softmax belief                 (d_belief = post - pre)

Sharded over forks (--shard_idx/--num_shards), then --merge. Candidate sets use
stable per-fork seeds so extraction is reproducible across processes.

  python scripts/transition_operator/to_extract_train.py \
    --run_dir runs/transition_operator --shard_idx 0 --num_shards 4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.transition_operator import (  # noqa: E402
    SEP_TOKEN_ID,
    belief_from_scores,
    build_candidates,
    candidate_mean_logprobs,
    sep_join_ids,
    stable_seed,
)

SUFFIX = "\nSo the final answer is"  # Stage-0 gate 1 winner
MAX_STEP_TOKENS = 192


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, default=Path("runs/transition_operator"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--k_candidates", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or args.run_dir / "stage2" / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        _merge(out_dir)
        return

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype,
        local_files_only=args.local_files_only).to(device).eval()
    hidden = model.config.hidden_size
    L = args.layer
    pad = tok.pad_token_id or 0
    sfx = tok(SUFFIX, add_special_tokens=False)["input_ids"]

    forks = [json.loads(l) for l in open(args.run_dir / "forks.jsonl") if l.strip()]
    corpus_pool = sorted({fk["gt_answer"] for fk in forks})
    shard = forks[args.shard_idx::args.num_shards]
    print(f"[extract2] shard {args.shard_idx}/{args.num_shards}: {len(shard)} forks "
          f"on {device}", flush=True)

    fork_rows, trans_rows = [], []
    F = {"S_prev": [], "pre_logits": [], "belief_pre": []}
    T = {"H_steps": [], "n_steps": [], "S_post": [], "post_logits": [],
         "belief_post": []}
    skipped = 0
    for j, fk in enumerate(shard):
        pre = sep_join_ids(tok, [fk["question"], *fk["prefix_steps"]])
        step_ids = {b: tok(fk[b], add_special_tokens=False)["input_ids"] or [pad]
                    for b in ("correct", "wrong")}
        if max(len(pre) + len(step_ids[b]) + 1 for b in step_ids) > args.max_seq_len - 48:
            skipped += 1
            continue
        cands = build_candidates(fk["gt_answer"], fk["pre_generated_answer"],
                                 tuple(fk["wrong_finals"]), corpus_pool,
                                 k=args.k_candidates,
                                 seed=stable_seed(fk["fork_id"], args.seed))
        lead = "" if SUFFIX[-1].isspace() else " "
        cand_ids = [tok(lead + c, add_special_tokens=False)["input_ids"]
                    for c in cands]
        b = len(pre) - 1
        pre_t = torch.tensor([pre], device=device)
        with torch.no_grad():
            out_pre = model(input_ids=pre_t, output_hidden_states=True)
        F["S_prev"].append(out_pre.hidden_states[L][0, b].float().cpu().numpy())
        F["pre_logits"].append(out_pre.logits[0, b].float().cpu().numpy())
        F["belief_pre"].append(belief_from_scores(candidate_mean_logprobs(
            model, pre + sfx, cand_ids, pad_id=pad, device=device)).numpy())
        fork_rows.append({"fork_id": fk["fork_id"], "question": fk["question"],
                          "candidates": cands})
        for branch in ("correct", "wrong"):
            post = pre + step_ids[branch] + [SEP_TOKEN_ID]
            post_t = torch.tensor([post], device=device)
            with torch.no_grad():
                out = model(input_ids=post_t, output_hidden_states=True)
            h = out.hidden_states[L][0].float().cpu()
            steps = h[b + 1:len(post) - 1][:MAX_STEP_TOKENS]
            padded = np.zeros((MAX_STEP_TOKENS, hidden), np.float16)
            padded[:len(steps)] = steps.numpy().astype(np.float16)
            T["H_steps"].append(padded)
            T["n_steps"].append(len(steps))
            T["S_post"].append(h[-1].numpy())
            T["post_logits"].append(out.logits[0, -1].float().cpu().numpy())
            T["belief_post"].append(belief_from_scores(candidate_mean_logprobs(
                model, post + sfx, cand_ids, pad_id=pad, device=device)).numpy())
            trans_rows.append({"fork_id": fk["fork_id"], "branch": branch,
                               "question": fk["question"], "text": fk[branch]})
        if (j + 1) % 50 == 0:
            print(f"[extract2] {j + 1}/{len(shard)}", flush=True)

    tag = f"shard{args.shard_idx}of{args.num_shards}"
    np.savez(out_dir / f"fork_arrays_{tag}.npz",
             S_prev=np.stack(F["S_prev"]).astype(np.float16),
             pre_logits=np.stack(F["pre_logits"]).astype(np.float16),
             belief_pre=np.stack(F["belief_pre"]).astype(np.float32))
    np.savez(out_dir / f"trans_arrays_{tag}.npz",
             H_steps=np.stack(T["H_steps"]),
             n_steps=np.array(T["n_steps"], np.int32),
             S_post=np.stack(T["S_post"]).astype(np.float16),
             post_logits=np.stack(T["post_logits"]).astype(np.float16),
             belief_post=np.stack(T["belief_post"]).astype(np.float32))
    (out_dir / f"fork_rows_{tag}.json").write_text(json.dumps(fork_rows))
    (out_dir / f"trans_rows_{tag}.json").write_text(json.dumps(trans_rows))
    print(f"[extract2] wrote shard {tag} ({len(fork_rows)} forks, "
          f"{skipped} skipped overlength)", flush=True)


def _merge(out_dir: Path) -> None:
    import glob
    for kind in ("fork", "trans"):
        parts = sorted(glob.glob(str(out_dir / f"{kind}_arrays_shard*of*.npz")))
        n_shards = int(parts[0].split("of")[-1].split(".")[0])
        arrays = [np.load(out_dir / f"{kind}_arrays_shard{i}of{n_shards}.npz")
                  for i in range(n_shards)]
        rows = [json.loads((out_dir / f"{kind}_rows_shard{i}of{n_shards}.json")
                           .read_text()) for i in range(n_shards)]
        merged_rows = [r for rp in rows for r in rp]
        keys = arrays[0].files
        np.savez(out_dir / f"{kind}_arrays.npz",
                 **{k: np.concatenate([a[k] for a in arrays]) for k in keys})
        (out_dir / f"{kind}_rows.json").write_text(json.dumps(merged_rows))
        print(f"[merge] {kind}: {len(merged_rows)} rows")
    (out_dir / "extract2_manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "suffix": SUFFIX, "max_step_tokens": MAX_STEP_TOKENS}, indent=2))


if __name__ == "__main__":
    main()
