"""Extract residual-stream activations for the public-SAE audit (model-general).

Used by both arms of the audit:
  * Qwen2.5-7B-Instruct (Instruct-matched arm; there is no public base-7B SAE), and
  * Gemma-2-9B base (GemmaScope arm).
It reuses the EXACT prompt format, tokenization and last-token-of-candidate_step
readout of scripts/encode_prm800k_hidden_states.py.

Which residual readouts to save is set by --layer_map (name:hidden_states_index).
hidden_states[0]=embeddings, hidden_states[i]=output of block i-1. For an SAE whose
folder is 'layer_N' (residual stream = output of block N), use index N+1:
  Qwen2.5  L20<-resid_post_layer_19 => index 20 ; L28<-resid_post_layer_27 => 28
  Gemma-2-9B  L20<-GemmaScope layer_20 => index 21 ; L31<-layer_31 => 32

Sharding (4-GPU fan-out): a deterministic global_index is assigned over the full
file order; with --num_shards 4 only global_index %% 4 == shard_idx is encoded.

Per-shard output (npz): shards/shard_{g}.npz with arrays h_{name} (n,H) f16 per
readout, y (n,) i32, global_index (n,) i64, n_tokens (n,) i32 (+ a .meta.jsonl
sidecar). --merge combines shards into merged/heldout_{name}_h.npy + heldout_y.npy
+ heldout_meta.jsonl (sorted by global_index), the layout encode/probe consume.

Gemma-2 note: pass --attn_impl eager for faithful activations (soft-capping).
Offline: pass --local_files_only; the Slurm wrapper exports HF_HUB_OFFLINE=1 etc.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from encode_prm800k_hidden_states import (  # noqa: E402
    git_commit, read_jsonl, tokenize_example, write_jsonl,
)


def audit_readout(tokenizer, examples, max_seq_len, pad_token_id, n=6):
    """Prove the readout is the last NON-PAD token of candidate_step, and is NOT
    an EOS, a pad token, the batch-last token, or a chat-template suffix token.

    We feed build_prompt_prefix(...) text straight to the tokenizer (NO chat
    template), candidate tokenized with add_special_tokens=False (no appended
    EOS), and the readout index is len(prefix+candidate)-1, computed BEFORE any
    right-padding. This check fails the job loudly if any assumption breaks under
    the real (Instruct) tokenizer.
    """
    eos = tokenizer.eos_token_id
    print("[audit] readout = last token of candidate_step; no chat template; "
          f"candidate add_special_tokens=False; eos_id={eos} pad_id={pad_token_id}", flush=True)
    for ex in examples[:n]:
        ids, cand = tokenize_example(tokenizer, ex["problem"], ex["prefix"],
                                     ex["candidate_step"], max_seq_len)
        cand_id = ids[cand]
        cand_only = tokenizer(ex["candidate_step"], add_special_tokens=False)["input_ids"]
        is_last = (cand == len(ids) - 1)                 # not the post-padding batch-last
        matches_step_tail = (cand_id == cand_only[-1])   # genuinely the step's last token
        not_eos = (cand_id != eos)                       # not an EOS / chat-suffix marker
        tail = tokenizer.decode(ids[max(0, cand - 6):cand + 1])
        print(f"[audit] uid={ex.get('uid','?')} cand_idx={cand}/{len(ids)-1} id={cand_id} "
              f"tok={tokenizer.decode([cand_id])!r} last7={tail!r} | "
              f"is_last={is_last} matches_step_tail={matches_step_tail} not_eos={not_eos}",
              flush=True)
        if not (is_last and matches_step_tail and not_eos):
            sys.exit("[audit] FATAL: readout is not the last content token of the step "
                     "(EOS/pad/chat-suffix/index drift). Check tokenizer behaviour.")
    print("[audit] OK: readout token verified on the real tokenizer", flush=True)


def extract_shard(jsonl_path, out_path, tokenizer, model, device, max_seq_len,
                  batch_size, pad_token_id, shard_idx, num_shards, limit, layer_map):
    examples = read_jsonl(jsonl_path)
    for gi, ex in enumerate(examples):
        ex["global_index"] = gi
    if limit is not None:
        examples = examples[:limit]
    if num_shards > 1:
        examples = [e for e in examples if e["global_index"] % num_shards == shard_idx]
    n = len(examples)
    H = model.config.hidden_size
    h = {k: np.zeros((n, H), dtype=np.float16) for k in layer_map}
    y = np.zeros(n, dtype=np.int32)
    gidx = np.zeros(n, dtype=np.int64)
    ntok = np.zeros(n, dtype=np.int32)
    meta: list[dict] = []
    print(f"[extract] {Path(jsonl_path).name} shard {shard_idx}/{num_shards}: {n} ex", flush=True)
    if examples:
        audit_readout(tokenizer, examples, max_seq_len, pad_token_id)

    t0 = time.perf_counter()
    i = 0
    while i < n:
        batch = examples[i:i + batch_size]
        ids_list, cand_list = [], []
        for ex in batch:
            try:
                ids, cand = tokenize_example(tokenizer, ex["problem"], ex["prefix"],
                                             ex["candidate_step"], max_seq_len)
            except ValueError as e:
                sys.exit(f"[extract] FATAL overlength uid={ex.get('uid','?')}: {e}")
            ids_list.append(ids); cand_list.append(cand)
        mx = max(len(x) for x in ids_list)
        padded = [x + [pad_token_id] * (mx - len(x)) for x in ids_list]
        masks = [[1] * len(x) + [0] * (mx - len(x)) for x in ids_list]
        inp = torch.tensor(padded, dtype=torch.long, device=device)
        att = torch.tensor(masks, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inp, attention_mask=att, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple, len = n_blocks + 1
        for b, (ex, cand) in enumerate(zip(batch, cand_list)):
            # Readout must be a real, attended (non-pad) position and the last
            # token of this example's UNPADDED sequence (right-padding only).
            assert cand == len(ids_list[b]) - 1, "cand not last real token (padding/index drift)"
            assert masks[b][cand] == 1, "cand points at a pad position"
            for k, li in layer_map.items():
                h[k][i + b] = hs[li][b, cand, :].detach().to(torch.float16).cpu().numpy()
            y[i + b] = ex["label"]
            gidx[i + b] = ex["global_index"]
            ntok[i + b] = len(ids_list[b])
            meta.append({"uid": ex.get("uid"), "problem_id": ex.get("problem_id"),
                         "step_idx": ex.get("step_idx"), "label": int(ex["label"]),
                         "rating": ex.get("rating"), "global_index": ex["global_index"],
                         "n_tokens": len(ids_list[b])})
        del out, hs
        i += len(batch)
        if (i // max(batch_size, 1)) % 16 == 0 or i == n:
            print(f"[extract] {i}/{n} ({time.perf_counter()-t0:.0f}s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_arrays = {f"h_{k}": v for k, v in h.items()}
    np.savez(out_path, y=y, global_index=gidx, n_tokens=ntok, **save_arrays)
    # meta sidecar (jsonl) so merge can reconstruct uid/problem_id/step_idx
    write_jsonl(out_path.with_suffix(".meta.jsonl"), meta)
    print(f"[extract] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", flush=True)


def merge_shards(shard_dir: Path, out_dir: Path, n_shards: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_y, rows_g, rows_meta = [], [], []
    rows_h: dict[str, list] = {}
    layer_names: list[str] = []
    for g in range(n_shards):
        npz = shard_dir / f"shard_{g}.npz"
        if not npz.exists():
            sys.exit(f"[merge] FATAL missing shard {npz}")
        d = np.load(npz)
        if not layer_names:  # discover layer readouts from the first shard's h_* keys
            layer_names = [k[2:] for k in d.files if k.startswith("h_")]
            rows_h = {L: [] for L in layer_names}
        for L in layer_names:
            rows_h[L].append(d[f"h_{L}"])
        rows_y.append(d["y"]); rows_g.append(d["global_index"])
        rows_meta.extend(json.loads(l) for l in
                         npz.with_suffix(".meta.jsonl").read_text().splitlines() if l.strip())
    g_all = np.concatenate(rows_g)
    order = np.argsort(g_all, kind="stable")
    if not np.array_equal(g_all[order], np.unique(g_all)):
        sys.exit("[merge] FATAL global_index not a clean partition (dupes/gaps)")
    y_all = np.concatenate(rows_y)[order]
    meta_all = [rows_meta[k] for k in order]
    for L in layer_names:
        h_all = np.concatenate(rows_h[L])[order]
        np.save(out_dir / f"heldout_{L}_h.npy", h_all)
        print(f"[merge] heldout_{L}_h.npy {h_all.shape}", flush=True)
    np.save(out_dir / "heldout_y.npy", y_all.astype(np.int32))
    write_jsonl(out_dir / "heldout_meta.jsonl", meta_all)
    (out_dir / "extract_manifest.json").write_text(json.dumps({
        "n": int(len(y_all)), "n_correct": int((y_all == 0).sum()),
        "n_incorrect": int((y_all == 1).sum()), "n_shards": n_shards,
        "layers": layer_names, "git_commit": git_commit(),
    }, indent=2))
    print(f"[merge] n={len(y_all)} layers={layer_names} -> {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, help="held-out PRM800K steps jsonl")
    ap.add_argument("--out", type=Path, help="per-shard npz output path")
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--layer_map", nargs="+", default=["L20:20", "L28:28"],
                    help="readout name:hidden_states_index pairs. hidden_states[0]=embeddings, "
                         "hidden_states[i]=output of block i-1. For a GemmaScope/Qwen-Scope SAE "
                         "'layer_N' (output of block N), use index N+1. Default = Qwen2.5 L20/L28.")
    ap.add_argument("--attn_impl", type=str, default=None,
                    help="attn_implementation (e.g. 'eager' for Gemma-2 activation fidelity)")
    ap.add_argument("--max_seq_len", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--merge", action="store_true", help="merge mode (no model load)")
    ap.add_argument("--shard_dir", type=Path, help="merge: dir holding shard_*.npz")
    ap.add_argument("--merged_out", type=Path, help="merge: output dir")
    args = ap.parse_args()

    if args.merge:
        if not (args.shard_dir and args.merged_out):
            sys.exit("--merge needs --shard_dir and --merged_out")
        merge_shards(args.shard_dir, args.merged_out, args.num_shards)
        return

    if not (args.jsonl and args.out):
        sys.exit("extract mode needs --jsonl and --out")

    layer_map = {}
    for spec in args.layer_map:
        name, idx = spec.rsplit(":", 1)
        layer_map[name] = int(idx)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                        local_files_only=args.local_files_only)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    load_kw = dict(local_files_only=args.local_files_only)
    if args.attn_impl:
        load_kw["attn_implementation"] = args.attn_impl
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     dtype=torch.float16, **load_kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     torch_dtype=torch.float16, **load_kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    n_blocks = model.config.num_hidden_layers
    for k, li in layer_map.items():
        if li > n_blocks:
            sys.exit(f"[extract] {k}=hidden_states[{li}] but model has only "
                     f"{n_blocks} blocks ({n_blocks+1} hidden_states).")
    msl = args.max_seq_len
    if msl is not None and msl <= 0:
        msl = int(getattr(model.config, "max_position_embeddings", 32768) or 32768)

    extract_shard(args.jsonl, args.out, tok, model, device, msl, args.batch_size,
                  pad, args.shard_idx, args.num_shards, args.limit, layer_map)


if __name__ == "__main__":
    main()
