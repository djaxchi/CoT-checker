#!/usr/bin/env python3
"""Per-step model-confidence sidecar for the PRM800K forks (Stage 0).

The fork audit ruled out length / numeric / position / answer-presence as the cause
of the L28 correctness probe. The leading remaining hypothesis is **model surprise**:
the human-written negative steps may simply be high-perplexity / OOD text that the
model "notices" as unlikely rather than as *wrong reasoning*. This script measures, for
the exact same fork items already encoded by ``encode_prm800k_forks.py``, the model's
teacher-forced confidence over the candidate-step tokens, so the battery
(``analyze_confidence_battery.py``) can test whether the probe is reducible to it.

It does NOT touch the validated hidden-state encode. It re-runs the same prompts (same
``build_prompt_prefix`` / ``tokenize_example``) and, instead of the last hidden, keeps
``out.logits`` over the candidate-step token span ``[len(prefix_ids):]`` to compute:

  nll_mean / nll_max / nll_last / nll_first   teacher-forced surprise of the written step
  entropy_mean                                next-token predictive entropy (uncertainty
                                              independent of what was written)
  logit_gap_mean                              mean top1-top2 logit (confidence margin)

Output (row-for-row aligned to {stem}_meta.jsonl produced by the hidden encode):
  {stem}_confidence.jsonl   one row per item: row, item_uid, fork_id, role, label,
                            n_step_tokens, nll_mean, nll_max, nll_last, nll_first,
                            entropy_mean, logit_gap_mean
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import (  # noqa: E402
    build_prompt_prefix,
    git_commit,
    read_jsonl,
    tokenize_example,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Pure, testable confidence math.  Operates on the *predictive* logits, i.e. the
# rows of out.logits that predict each candidate token.  For candidate token at
# sequence position j, the predicting distribution is logits[j-1]; so for a
# candidate span [prefix_len, T-1] the predictive rows are [prefix_len-1, T-2].
# ---------------------------------------------------------------------------

def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def step_confidence(pred_logits: np.ndarray, target_ids: np.ndarray) -> dict:
    """Confidence stats over one step's candidate tokens.

    pred_logits : (n_step_tokens, vocab) logits that *predict* each candidate token.
    target_ids  : (n_step_tokens,) realized candidate token ids.
    Returns nll_mean/max/last/first, entropy_mean, logit_gap_mean, n_step_tokens.
    NLL and entropy are in nats.
    """
    pred_logits = np.asarray(pred_logits, dtype=np.float64)
    target_ids = np.asarray(target_ids).astype(np.int64).reshape(-1)
    n = target_ids.shape[0]
    if n == 0 or pred_logits.shape[0] != n:
        nan = float("nan")
        return {"n_step_tokens": int(n), "nll_mean": nan, "nll_max": nan,
                "nll_last": nan, "nll_first": nan, "entropy_mean": nan,
                "logit_gap_mean": nan}
    logp = _log_softmax(pred_logits)
    nll = -logp[np.arange(n), target_ids]
    p = np.exp(logp)
    entropy = -(p * logp).sum(axis=-1)
    # top1 - top2 logit margin (partition avoids a full sort)
    part = np.partition(pred_logits, -2, axis=-1)
    gap = part[:, -1] - part[:, -2]
    return {
        "n_step_tokens": int(n),
        "nll_mean": float(nll.mean()),
        "nll_max": float(nll.max()),
        "nll_last": float(nll[-1]),
        "nll_first": float(nll[0]),
        "entropy_mean": float(entropy.mean()),
        "logit_gap_mean": float(gap.mean()),
    }


def prefix_token_len(tokenizer, problem: str, prefix: str) -> int:
    """Token length of build_prompt_prefix (with BOS), to locate the candidate span."""
    ids = tokenizer(build_prompt_prefix(problem, prefix),
                    add_special_tokens=True, truncation=False)["input_ids"]
    return len(ids)


# ---------------------------------------------------------------------------
# GPU loop
# ---------------------------------------------------------------------------

def encode_confidence(items, tokenizer, model, device, max_seq_len, batch_size,
                      pad_token_id) -> list[dict]:
    import torch

    n = len(items)
    rows: list[dict] = []
    t0 = time.perf_counter()
    i = 0
    while i < n:
        batch = items[i:i + batch_size]
        batch_ids: list[list[int]] = []
        batch_prefix_len: list[int] = []
        for ex in batch:
            if ex["role"] == "anchor":
                batch_ids.append(None)        # no candidate -> no confidence
                batch_prefix_len.append(-1)
                continue
            try:
                ids, _ = tokenize_example(tokenizer, ex["problem"], ex["prefix"],
                                          ex["candidate_step"], max_seq_len)
                plen = prefix_token_len(tokenizer, ex["problem"], ex["prefix"])
            except ValueError as e:
                sys.exit(f"[encode-conf] FATAL: {ex.get('item_uid','?')}: {e}")
            batch_ids.append(ids)
            batch_prefix_len.append(plen)

        real = [(b, ids) for b, ids in enumerate(batch_ids) if ids is not None]
        if real:
            max_len = max(len(ids) for _, ids in real)
            padded, masks = [], []
            for ids in batch_ids:
                seq = ids if ids is not None else [pad_token_id]
                padded.append(seq + [pad_token_id] * (max_len - len(seq)))
                masks.append([1] * len(seq) + [0] * (max_len - len(seq)))
            input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
            attn_tensor = torch.tensor(masks, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_tensor, attention_mask=attn_tensor,
                            output_hidden_states=False, use_cache=False)
            logits = out.logits
            for b, ids in real:
                plen = batch_prefix_len[b]
                L = len(ids)
                # predictive rows for candidate tokens [plen, L-1] are [plen-1, L-2]
                pred = logits[b, plen - 1:L - 1, :].float().cpu().numpy()
                targets = np.asarray(ids[plen:L])
                stats = step_confidence(pred, targets)
                ex = batch[b]
                rows.append({"row": i + b, "item_uid": ex["item_uid"],
                             "fork_id": ex["fork_id"], "role": ex["role"],
                             "label": ex["label"], **stats})
            del out, logits
        # anchors: still emit a placeholder row so alignment is exact
        for b, ids in enumerate(batch_ids):
            if ids is None:
                ex = batch[b]
                rows.append({"row": i + b, "item_uid": ex["item_uid"],
                             "fork_id": ex["fork_id"], "role": ex["role"],
                             "label": ex["label"], "n_step_tokens": 0,
                             "nll_mean": None, "nll_max": None, "nll_last": None,
                             "nll_first": None, "entropy_mean": None,
                             "logit_gap_mean": None})
        i += len(batch)
        if (i // batch_size) % 16 == 0 or i == n:
            print(f"[encode-conf] {i}/{n} ({time.perf_counter()-t0:.1f}s)", flush=True)

    rows.sort(key=lambda r: r["row"])
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Per-step confidence sidecar for fork items.")
    p.add_argument("--items", type=Path, required=True, help="forks_*_items.jsonl")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--stem", type=str, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--model_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    import torch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.stem}_confidence.jsonl"
    if out_path.exists() and not args.force:
        sys.exit(f"[encode-conf] Refusing to overwrite {out_path}. Pass --force.")

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    print(f"[encode-conf] Loading {args.model_name_or_path} ...", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, local_files_only=args.local_files_only)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, local_files_only=args.local_files_only,
                dtype=dtype_map[args.model_dtype])
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, local_files_only=args.local_files_only,
                torch_dtype=dtype_map[args.model_dtype])
    except OSError:
        sys.exit("Model not found locally. Pre-cache the model before running offline.")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        sys.exit("[encode-conf] Tokenizer has no pad/eos token; cannot pad.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    model_max = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        if model_max <= 0:
            sys.exit("[encode-conf] model has no max_position_embeddings; pass --max_seq_len.")
        args.max_seq_len = model_max
    if model_max and args.max_seq_len > model_max:
        sys.exit(f"[encode-conf] --max_seq_len={args.max_seq_len} exceeds context {model_max}.")

    items = read_jsonl(args.items)
    print(f"[encode-conf] {len(items)} items from {args.items}", flush=True)

    rows = encode_confidence(items, tokenizer, model, device,
                             args.max_seq_len, args.batch_size, pad_token_id)
    write_jsonl(out_path, rows)
    manifest = {
        "run_name": args.run_name, "model": args.model_name_or_path,
        "stem": args.stem, "n_items": len(items),
        "units": "nats", "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
    }
    (args.out_dir / f"{args.stem}_confidence_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[encode-conf] Saved {out_path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
