#!/usr/bin/env python3
"""Encode frozen trajectory manifests into a sharded full-token store.

bidirectional_token_probe_v1 (plan section 6). One causal pass of Qwen/Qwen3-8B
per complete trajectory, in the fixed solution-reading format of
scripts/encode_processbench_full_store.py::tokenize_solution (no chat template,
no verification request). Every input token's hidden_states[LAYER] row is kept.

Store layout (repstore TOKEN_SEQ contract, one item per TRAJECTORY):

    <rep_root>/shard_XX/h.npy          float16 (rows, d) all token states, packed
    <rep_root>/shard_XX/lengths.npy    int32 (N,) tokens per trajectory
    <rep_root>/shard_XX/y.npy          int8 (N,) TRACE-level flag only: 1 if any
                                       labeled step is incorrect. NOT step supervision.
    <rep_root>/shard_XX/meta.jsonl     trace_id, split, step_starts, step_ends, n_tokens
    <rep_root>/shard_XX/step_labels.jsonl   sidecar: per-step y (-1 unknown) and
                                       label_mask, joined from the frozen meta. The
                                       probe reads labels ONLY from here, never as input.
    <rep_root>/shard_XX/spec.json, encode_stats.json, DONE

Shards are written to shard_XX.tmp and renamed after validation, so a shard
directory without .tmp is complete. Existing complete shards are skipped, so a
re-run resumes. Token ids are re-derived and checked against the frozen manifest
hash; a mismatch aborts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.repstore.fingerprint import split_fingerprint  # noqa: E402
from src.repstore.store import TOKEN_SEQ, RepSpec  # noqa: E402


def tokenize_solution(tokenizer, problem: str, steps: list[str]):
    from encode_processbench_full_store import tokenize_solution as ts
    return ts(tokenizer, problem, steps)


def tokenize_view(tokenizer, row: dict):
    """Inputs row -> (ids, step_starts, step_ends). Rows with a 'header' (context
    views, context_ablation_v3) replace the "Problem: ... Solution:" template with
    that exact header; everything else is tokenize_solution, byte for byte."""
    if "header" not in row:
        return tokenize_solution(tokenizer, row["problem"], row["steps"])
    ids = list(tokenizer(row["header"], add_special_tokens=True)["input_ids"])
    sep_ids = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    ss, se = [], []
    for j, step in enumerate(row["steps"]):
        if j > 0:
            ids += sep_ids
        ss.append(len(ids))
        ids += tokenizer(step, add_special_tokens=False)["input_ids"]
        se.append(len(ids))
    return ids, ss, se


def fwd_len(r: dict) -> int:
    """Tokens in the backbone pass (compact views store fewer rows than they encode)."""
    return r.get("n_forward", r["n_tokens"])


def ids_hash(ids) -> str:
    return hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_manifest(mdir: Path):
    enc = load_jsonl(mdir / "encode_manifest.jsonl")
    inputs, labels = {}, {}
    for p in sorted((mdir / "inputs").glob("*.jsonl")):
        for r in load_jsonl(p):
            inputs[r["trace_id"]] = r
    for p in sorted((mdir / "meta").glob("*.jsonl")):
        for r in load_jsonl(p):
            labels[r["trace_id"]] = {"y": r["y"], "label_mask": r["label_mask"]}
    return enc, inputs, labels


def token_batches(rows: list[dict], budget: int):
    """Length-sorted batches with at most `budget` padded tokens."""
    order = sorted(range(len(rows)), key=lambda i: -fwd_len(rows[i]))
    batch, mx = [], 0
    for i in order:
        n = fwd_len(rows[i])
        if batch and max(mx, n) * (len(batch) + 1) > budget:
            yield batch
            batch, mx = [], 0
        batch.append(i)
        mx = max(mx, n)
    if batch:
        yield batch


class LayerCapture:
    """Capture the output of decoder block (layer-1) == hidden_states[layer]."""

    def __init__(self, model, layer: int):
        self.out = None
        blocks = model.model.layers
        self.h = blocks[layer - 1].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.out = out[0] if isinstance(out, tuple) else out


def forward_states(model, cap, ids_list, pad_id, device):
    maxlen = max(len(x) for x in ids_list)
    inp = torch.full((len(ids_list), maxlen), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids_list), maxlen), dtype=torch.long)
    for b, x in enumerate(ids_list):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    with torch.no_grad():
        model.model(input_ids=inp.to(device), attention_mask=att.to(device), use_cache=False)
    return cap.out


def encode_shard(shard: int, rows, inputs, labels, tok, model, cap, args, device, pad_id, spec_kw):
    final = args.rep_root / f"shard_{shard:02d}"
    if (final / "DONE").exists():
        print(f"[skip] shard {shard} complete", flush=True)
        return
    # --local_tmp: write the shard on node-local disk, then copy it to rep_root in
    # one sequential pass (row-by-row memmap writes on Lustre are ~20x slower)
    tmp_root = args.local_tmp if args.local_tmp is not None else args.rep_root
    tmp = tmp_root / f"shard_{shard:02d}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(args.rep_root / f"shard_{shard:02d}.tmp", ignore_errors=True)
    tmp.mkdir(parents=True)
    total = sum(r["n_tokens"] for r in rows)
    d = model.config.hidden_size
    h = np.lib.format.open_memmap(tmp / "h.npy", mode="w+", dtype=np.float16, shape=(total, d))
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([r["n_tokens"] for r in rows], out=offsets[1:])
    ids_all = []
    for r in rows:
        tr = inputs[r["trace_id"]]
        ids, ss, se = tokenize_view(tok, tr)
        if (ids_hash(ids) != r["ids_sha1"] or ss != r.get("fwd_step_starts", r["step_starts"])
                or se != r.get("fwd_step_ends", r["step_ends"]) or len(ids) != fwd_len(r)):
            raise SystemExit(f"[FATAL] tokenization drift for {r['trace_id']}")
        ids_all.append(ids)
    t0 = time.perf_counter()
    max_abs, n_nonfinite, rel_err_sum, rel_err_n = 0.0, 0, 0.0, 0
    for bi, batch in enumerate(token_batches(rows, args.batch_tokens)):
        hs = forward_states(model, cap, [ids_all[i] for i in batch], pad_id, device)
        for b, i in enumerate(batch):
            n = rows[i]["n_tokens"]
            if "keep" in rows[i]:  # compact view: store only the kept [a, b) row ranges
                x = torch.cat([hs[b, a:bb] for a, bb in rows[i]["keep"]]).float()
                assert x.shape[0] == n, rows[i]["trace_id"]
            else:
                x = hs[b, :n].float()
            x16 = x.to(torch.float16)
            n_nonfinite += int((~torch.isfinite(x16)).sum())
            max_abs = max(max_abs, float(x.abs().max()))
            if bi % 20 == 0 and b == 0:
                rel_err_sum += float((x16.float() - x).norm() / x.norm().clamp_min(1e-12))
                rel_err_n += 1
            h[offsets[i]:offsets[i + 1]] = x16.cpu().numpy()
        del hs
    h.flush()
    del h
    dt = time.perf_counter() - t0
    if n_nonfinite:
        raise SystemExit(f"[FATAL] shard {shard}: {n_nonfinite} non-finite fp16 values")
    lengths = np.array([r["n_tokens"] for r in rows], dtype=np.int32)
    np.save(tmp / "lengths.npy", lengths)
    yflag = [int(any(m and y == 1 for y, m in zip(labels[r["trace_id"]]["y"], labels[r["trace_id"]]["label_mask"])))
             for r in rows]
    np.save(tmp / "y.npy", np.asarray(yflag, dtype=np.int8))
    with open(tmp / "meta.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("trace_id", "split", "n_tokens", "n_steps",
                                                   "step_starts", "step_ends", "ids_sha1")}) + "\n")
    with open(tmp / "step_labels.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"trace_id": r["trace_id"], **labels[r["trace_id"]]}) + "\n")
    (tmp / "spec.json").write_text(RepSpec(source_split=f"shard_{shard:02d}", **spec_kw).to_json())
    stats = {"shard": shard, "traces": len(rows), "tokens": int(total), "seconds": dt,
             "tokens_per_s": total / max(dt, 1e-9), "max_abs_state": max_abs,
             "fp16_rel_err_mean": rel_err_sum / max(rel_err_n, 1),
             "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0}
    stats["fingerprint"] = split_fingerprint(tmp)
    (tmp / "encode_stats.json").write_text(json.dumps(stats, indent=2))
    (tmp / "DONE").write_text(stats["fingerprint"])
    if args.local_tmp is not None:
        staged = args.rep_root / f"shard_{shard:02d}.tmp"
        shutil.copytree(tmp, staged)
        shutil.rmtree(tmp)
        tmp = staged
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    print(f"[done] shard {shard}: {len(rows)} traces {total:,} tok {dt:.0f}s "
          f"{stats['tokens_per_s']:.0f} tok/s max|h|={max_abs:.1f}", flush=True)


def smoke_checks(rows, inputs, tok, model, cap, device, pad_id, layer, n=6) -> dict:
    """Index audit, prefix-only vs full equality, fp16 conversion error."""
    out = {}
    tr = inputs[rows[0]["trace_id"]]
    ids, ss, se = tokenize_view(tok, tr)
    with torch.no_grad():
        o = model.model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True, use_cache=False)
    out["n_hidden_states"] = len(o.hidden_states)
    out["num_hidden_layers"] = model.config.num_hidden_layers
    out["hook_vs_hidden_states_max_abs"] = float((cap.out[0].float() - o.hidden_states[layer][0].float()).abs().max())
    diffs = []
    for r in rows[:n]:
        tr = inputs[r["trace_id"]]
        ids, ss, se = tokenize_view(tok, tr)
        k = max(0, len(ss) // 2)
        cut = se[k]
        full = forward_states(model, cap, [ids], pad_id, device)[0, :cut].float()
        pref = forward_states(model, cap, [ids[:cut]], pad_id, device)[0, :cut].float()
        # also inside a padded batch with a longer partner
        pad = forward_states(model, cap, [ids[:cut], ids], pad_id, device)[0, :cut].float()
        scale = full.abs().mean()
        diffs.append({"cut": cut, "prefix_vs_full_max_abs": float((full - pref).abs().max()),
                      "prefix_vs_full_mean_rel": float((full - pref).abs().mean() / scale),
                      "padded_vs_full_mean_rel": float((full - pad).abs().mean() / scale),
                      "fp16_rel_err": float((full.half().float() - full).norm() / full.norm())})
    out["prefix_checks"] = diffs
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest_dir", type=Path, required=True)
    ap.add_argument("--rep_root", type=Path, required=True)
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--layer", type=int, default=35)
    ap.add_argument("--shards", type=int, nargs="+", required=True)
    ap.add_argument("--batch_tokens", type=int, default=32768)
    ap.add_argument("--model_dtype", default="bfloat16")
    ap.add_argument("--limit_per_shard", type=int, default=0, help="smoke only")
    ap.add_argument("--smoke_checks_out", type=Path, default=None)
    ap.add_argument("--local_tmp", type=Path, default=None,
                    help="node-local staging dir for shards (copied to rep_root when complete)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    enc, inputs, labels = load_manifest(args.manifest_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only,
        torch_dtype=getattr(torch, args.model_dtype), attn_implementation="sdpa").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cap = LayerCapture(model, args.layer)
    args.rep_root.mkdir(parents=True, exist_ok=True)
    try:
        import transformers
        rev = getattr(model.config, "_commit_hash", None)
        prov = {"model": args.model_name_or_path, "model_revision": rev,
                "tokenizer_revision": getattr(tok, "init_kwargs", {}).get("_commit_hash"),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "dtype": args.model_dtype, "attn": "sdpa",
                "layer": args.layer, "hidden_size": model.config.hidden_size,
                "num_hidden_layers": model.config.num_hidden_layers,
                "encoder_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        (args.rep_root / f"provenance_shard{args.shards[0]:02d}.json").write_text(json.dumps(prov, indent=2))
    except Exception as e:  # provenance must not kill the encode
        print("[warn] provenance:", e)
    spec_kw = dict(name=args.rep_root.name, kind=TOKEN_SEQ, dim=model.config.hidden_size,
                   layer=args.layer, backbone=Path(args.model_name_or_path).name,
                   readout="full_solution_tokens", prompt_style="solution_reading")
    for s in args.shards:
        rows = [r for r in enc if r["shard"] == s]
        if args.limit_per_shard:  # smoke: deterministic sample across splits
            rows = sorted(rows, key=lambda r: hashlib.sha1(r["trace_id"].encode()).hexdigest())
            rows = sorted(rows[:args.limit_per_shard], key=lambda r: (r["split"], r["trace_id"]))
        if args.smoke_checks_out is not None:
            res = smoke_checks(rows, inputs, tok, model, cap, device, pad_id, args.layer)
            args.smoke_checks_out.write_text(json.dumps(res, indent=2))
            print(json.dumps(res, indent=2), flush=True)
            args.smoke_checks_out = None
        encode_shard(s, rows, inputs, labels, tok, model, cap, args, device, pad_id, spec_kw)


if __name__ == "__main__":
    main()
