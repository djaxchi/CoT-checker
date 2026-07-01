#!/usr/bin/env python3
"""Per-token re-encode of the PRM800K 6k held-out test, with the L28 probe applied
to EVERY step token (not just first/last) + the model's per-token certainty.

The 6k mechanistic study (REPORT.md §15) read the dense L28 probe only at the last token
of each step. This script answers the finer question: *within a step labelled incorrect,
does one particular token make the probe fire, and does that token coincide with a dip in
the model's certainty?*

For each held-out item it reproduces the EXACT prompt/tokenization used to train and
evaluate the probe (encode_prm800k_multitoken_multilayer.tokenize_span), runs ONE
teacher-forced forward pass, and for every token of the candidate step keeps:
  - the L28 (and, for comparison, L20) hidden state projected through linear_probe.pt
    -> per-token probe score (higher = incorrect);
  - the per-token certainty from the predictive logits (nll, entropy, logit_gap,
    p_top1, p_realized), i.e. encode_fork_confidence's math, un-aggregated.

The probe is native to L28/last; ``probe_score_l20`` is the SAME L28 weights applied to
L20 activations (an off-layer diagnostic, kept because the user asked for the L20 span).

Outputs in --out_dir:
  {stem}_tokens.jsonl   one row per (step, token): uid, problem_id, step_idx, label,
                        rating, tok_pos, n_step_tokens, token, token_id, probe_score_l28,
                        probe_score_l20, nll, entropy, logit_gap, p_top1, p_realized
  {stem}_steps.jsonl    one row per step: uid, label, spike_stats(L28) + coincidence
                        (L28 probe vs nll and vs entropy)
  {stem}_tokens_manifest.json

Usage (TamIA, one GPU):
  python scripts/analysis/s3_token_incorrectness_extract.py \
    --items $SCRATCH/cot_mech/prestudy_v1/data/prm800k_heldout_test.jsonl \
    --probe runs/s1_model_size_dense/qwen2_5_7b/linear_probe.pt \
    --out_dir runs/s3_token_traj/qwen2_5_7b --stem prm800k_heldout_test \
    --model_name_or_path Qwen/Qwen2.5-7B --local_files_only \
    --layers 20 28 --probe_layer 28 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from encode_prm800k_hidden_states import git_commit, read_jsonl, write_jsonl  # noqa: E402
from encode_prm800k_multitoken_multilayer import tokenize_span  # noqa: E402
from inspect_margin_drivers import load_probe  # noqa: E402
from src.analysis.token_trajectory import coincidence, spike_stats  # noqa: E402


def _certainty_and_scores(out, b_idx, first, last, ids, hs_layers, w_t, bias, device):
    """Per-token certainty + per-layer probe score for one step, computed on device.

    Returns (token_ids, dict of (T,) numpy arrays). Mirrors
    src.analysis.token_trajectory.per_token_certainty / probe_scores.
    """
    import torch

    T = last - first + 1
    target = torch.tensor(ids[first:last + 1], dtype=torch.long, device=device)
    # predictive rows for step tokens [first, last] are logits[first-1 : last]
    pl = out.logits[b_idx, first - 1:last, :].float()
    logp = torch.log_softmax(pl, dim=-1)
    p = logp.exp()
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    entropy = -(p * logp).sum(dim=-1)
    p_top1 = p.max(dim=-1).values
    p_realized = p.gather(1, target.unsqueeze(1)).squeeze(1)
    top2 = pl.topk(2, dim=-1).values
    logit_gap = top2[:, 0] - top2[:, 1]

    scores = {}
    for li, hs in hs_layers.items():
        htok = hs[b_idx, first:last + 1, :].float()          # (T, hidden)
        scores[li] = (htok @ w_t + bias).detach().cpu().numpy()

    arrs = {
        "nll": nll.detach().cpu().numpy(),
        "entropy": entropy.detach().cpu().numpy(),
        "logit_gap": logit_gap.detach().cpu().numpy(),
        "p_top1": p_top1.detach().cpu().numpy(),
        "p_realized": p_realized.detach().cpu().numpy(),
    }
    assert all(v.shape == (T,) for v in arrs.values())
    return ids[first:last + 1], arrs, scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--stem", type=str, default="prm800k_heldout_test")
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 28],
                    help="hidden_states indices to project through the probe")
    ap.add_argument("--probe_layer", type=int, default=28,
                    help="native layer of the probe (used for the per-step summary)")
    ap.add_argument("--max_seq_len", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not (0 <= args.shard_idx < args.num_shards) or args.num_shards < 1:
        sys.exit(f"invalid shard config {args.shard_idx}/{args.num_shards}")
    if args.probe_layer not in args.layers:
        sys.exit(f"--probe_layer {args.probe_layer} must be one of --layers {args.layers}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_idx:02d}"
    tok_path = args.out_dir / f"{args.stem}{suffix}_tokens.jsonl"
    if tok_path.exists() and not args.force:
        sys.exit(f"refusing to overwrite {tok_path}; pass --force")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            dtype=dtype_map[args.model_dtype])
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only,
            torch_dtype=dtype_map[args.model_dtype])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    num_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)
    if args.max_seq_len <= 0:
        args.max_seq_len = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    for li in args.layers:
        if not (0 <= li <= num_layers):
            sys.exit(f"layer {li} out of range [0,{num_layers}]")

    w, bias = load_probe(args.probe, hidden)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[tok-traj] probe {args.probe.name}: w{w.shape} b={bias:.4f} "
          f"native L{args.probe_layer}", flush=True)

    examples = read_jsonl(args.items)
    for gi, ex in enumerate(examples):
        ex["global_index"] = gi
    n_total = len(examples)
    if args.limit is not None:
        examples = examples[:args.limit]
    if args.num_shards > 1:
        examples = [e for e in examples
                    if e["global_index"] % args.num_shards == args.shard_idx]
    n = len(examples)
    print(f"[tok-traj] {n}/{n_total} items (shard {args.shard_idx}/{args.num_shards}) "
          f"layers={args.layers} probe_layer={args.probe_layer}", flush=True)

    tok_rows: list[dict] = []
    step_rows: list[dict] = []
    n_tokens_total = 0
    t0 = time.perf_counter()
    i = 0
    while i < n:
        batch = examples[i:i + args.batch_size]
        ids_list, firsts, lasts = [], [], []
        for ex in batch:
            try:
                full, fi, la = tokenize_span(tok, ex["problem"], ex["prefix"],
                                             ex["candidate_step"], args.max_seq_len)
            except ValueError as e:
                sys.exit(f"[tok-traj] FATAL overlength uid={ex.get('uid','?')}: {e}")
            if fi < 1:
                sys.exit(f"[tok-traj] uid={ex.get('uid','?')} has no prefix token "
                         "before the step; cannot form predictive logits.")
            ids_list.append(full)
            firsts.append(fi)
            lasts.append(la)
        mx = max(len(x) for x in ids_list)
        inp_t = torch.tensor([x + [pad] * (mx - len(x)) for x in ids_list],
                             dtype=torch.long, device=device)
        att = torch.tensor([[1] * len(x) + [0] * (mx - len(x)) for x in ids_list],
                           dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inp_t, attention_mask=att, output_hidden_states=True,
                        use_cache=False)
        hs_layers = {li: out.hidden_states[li] for li in args.layers}

        for b, ex in enumerate(batch):
            first, last = firsts[b], lasts[b]
            step_ids, arrs, scores = _certainty_and_scores(
                out, b, first, last, ids_list[b], hs_layers, w_t, bias, device)
            toks = tok.convert_ids_to_tokens(step_ids)
            T = len(step_ids)
            for t in range(T):
                row = {
                    "uid": ex["uid"], "problem_id": ex["problem_id"],
                    "step_idx": ex["step_idx"], "label": int(ex["label"]),
                    "rating": ex.get("rating"),
                    "tok_pos": t, "n_step_tokens": T,
                    "token": toks[t], "token_id": int(step_ids[t]),
                    "nll": float(arrs["nll"][t]),
                    "entropy": float(arrs["entropy"][t]),
                    "logit_gap": float(arrs["logit_gap"][t]),
                    "p_top1": float(arrs["p_top1"][t]),
                    "p_realized": float(arrs["p_realized"][t]),
                }
                for li in args.layers:
                    row[f"probe_score_l{li}"] = float(scores[li][t])
                tok_rows.append(row)
            n_tokens_total += T

            s_probe = scores[args.probe_layer]
            summary = {"uid": ex["uid"], "label": int(ex["label"]),
                       "rating": ex.get("rating"), "n_step_tokens": T,
                       **{f"spike_{k}": v for k, v in spike_stats(s_probe).items()}}
            for unc_name in ("nll", "entropy"):
                for ck, cv in coincidence(s_probe, arrs[unc_name]).items():
                    summary[f"coin_{unc_name}_{ck}"] = cv
            step_rows.append(summary)

        del out, hs_layers
        i += len(batch)
        if (i // max(args.batch_size, 1)) % 8 == 0 or i == n:
            print(f"[tok-traj] {i}/{n} steps, {n_tokens_total} tokens "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)

    write_jsonl(tok_path, tok_rows)
    step_path = args.out_dir / f"{args.stem}{suffix}_steps.jsonl"
    write_jsonl(step_path, step_rows)
    manifest = {
        "model_name": args.model_name_or_path, "stem": args.stem,
        "probe": str(args.probe), "probe_layer": args.probe_layer,
        "layers": args.layers, "num_hidden_layers": num_layers,
        "hidden_size": hidden, "n_steps": n, "n_total_in_file": n_total,
        "n_tokens": n_tokens_total, "shard_idx": args.shard_idx,
        "num_shards": args.num_shards, "max_seq_len": args.max_seq_len,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }
    (args.out_dir / f"{args.stem}{suffix}_tokens_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[tok-traj] wrote {tok_path} ({len(tok_rows)} token rows) and {step_path} "
          f"({len(step_rows)} steps)", flush=True)


if __name__ == "__main__":
    main()
