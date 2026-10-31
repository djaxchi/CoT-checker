#!/usr/bin/env python3
"""Pooled step vectors at several layers from one forward pass, no token store.

backbone_geometry_v0 (docs/backbone_geometry_v0_plan.md) compares backbones by
closed-form geometry (LDA), which needs only pooled vectors. The token stores of
the leaderboard cost ~160 GB per backbone; this writes, per item and per
requested `hidden_states` index:

    last      the step's last token
    mean      the mean over the step's tokens
    boundary  the token just before the step

and, under `--style prm_native`, also

    sep       the `<extra_0>` token that closes the current step, which is the
              position Qwen2.5-Math-PRM-7B's own head reads, plus that head's
              P(correct) there (`head_p_correct`), as a reproduction gate.

Styles:
    verifier    the leaderboard template (src/onpolicy/prompts.verifier_prefix)
    prm_native  the PRM's chat template: system prompt, the problem as the user
                turn, steps joined by `<extra_0>` with one after the current
                step (scripts/onpolicy/score_traces_with_prm.py). PRM800K
                prefixes are split on blank lines to recover the prior steps.

Items: PRM800K jsonl rows (problem, prefix, candidate_step, label with 1 =
incorrect) and ProcessBench traces flattened one item per step, all steps kept,
label 1 on the first-error step only, post-error steps kept with label 0 and the
trace's `first_error` in the meta (the grid_vectors convention). Output, per split and shard:
`{split}.shard{k}.npz` with float16 arrays of shape (n, n_layers, d) and a meta
jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.encode_processbench_token_store import flatten, load_traces  # noqa: E402
from src.onpolicy.prompts import verifier_prefix  # noqa: E402

SEP = "<extra_0>"
PRM_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def prm800k_items(path: Path, limit: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if limit and limit < len(rows):
        pick = np.sort(np.random.default_rng(seed).choice(len(rows), limit, replace=False))
        rows = [rows[i] for i in pick]
    return [{"uid": r["uid"], "problem": r["problem"], "prefix": r["prefix"],
             "prior_steps": [s for s in r["prefix"].split("\n\n") if s.strip()] if r["prefix"] else [],
             "step": r["candidate_step"], "label": int(r["label"]),
             # trace fields, when the file carries them (reprobe_comparison_v1 test and PB items)
             **{k: r[k] for k in ("id", "step_idx", "first_error") if k in r}} for r in rows]


def pb_items(path: Path, subset: str) -> list[dict]:
    out = []
    traces = {t["id"]: t for t in load_traces(path)}
    for r in flatten(list(traces.values()), subset):
        steps = traces[r["id"]]["steps"]
        fe = int(r["label"])
        out.append({"uid": f"{r['id']}::{r['step_idx']}", "id": r["id"], "step_idx": r["step_idx"],
                    "first_error": fe, "problem": r["problem"], "prefix": r["prefix"],
                    "prior_steps": steps[:r["step_idx"]], "step": r["current_step"],
                    "label": int(r["step_idx"] == fe)})
    return out


def load_prm_head(model_id: str, d: int) -> torch.nn.Module:
    """The PRM's 2-way head (Linear, ReLU, Linear on the post-norm final state),
    read straight from the checkpoint. The causal-LM load drops `score.*`; the
    released remote code needs transformers < 5, so the head is rebuilt here
    rather than loading Qwen2ForProcessRewardModel."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    snap = Path(snapshot_download(model_id, local_files_only=True))
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    head = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.ReLU(), torch.nn.Linear(d, 2))
    sd = {}
    for k in ("score.0.weight", "score.0.bias", "score.2.weight", "score.2.bias"):
        with safe_open(str(snap / index[k]), framework="pt") as f:
            sd[k.removeprefix("score.")] = f.get_tensor(k)
    head.load_state_dict(sd)
    return head.to(torch.bfloat16)


def tokenize(tok, it: dict, style: str) -> tuple[list[int], int, int, int]:
    """(ids, step_start, step_end, sep_pos). step_end is one past the step;
    sep_pos is the closing separator under prm_native, else -1."""
    if style == "verifier":
        pre = tok(verifier_prefix(it["problem"], it["prefix"]), add_special_tokens=True)["input_ids"]
        st = tok(it["step"], add_special_tokens=False)["input_ids"]
        return pre + st, len(pre), len(pre) + len(st), -1
    head = tok.apply_chat_template(
        [{"role": "system", "content": PRM_SYSTEM}, {"role": "user", "content": it["problem"]}],
        tokenize=False, add_generation_prompt=True)
    before = head + "".join(s + SEP for s in it["prior_steps"])
    pre = tok(before, add_special_tokens=False)["input_ids"]
    st = tok(it["step"], add_special_tokens=False)["input_ids"]
    sep = tok(SEP, add_special_tokens=False)["input_ids"]
    if len(sep) != 1:
        raise ValueError("separator is not one token")
    return pre + st + sep, len(pre), len(pre) + len(st), len(pre) + len(st)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--style", choices=["verifier", "prm_native"], default="verifier")
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--prm800k", nargs="*", default=[], help="name:path[:limit]")
    p.add_argument("--pb", nargs="*", default=[], help="subset:path")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_tokens_per_batch", type=int, default=24_000)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", type=int, default=0, help="items per split, for a quick test")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    full = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                local_files_only=True)
    head = load_prm_head(a.model, full.config.hidden_size) if a.style == "prm_native" else None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = full.model.to(dev).eval()
    if head is not None:
        head = head.to(dev).eval()
    n_hs = body.config.num_hidden_layers + 1
    layers = [l if l >= 0 else n_hs + l for l in a.layers]
    print(f"[pool] {a.model} style={a.style} layers={layers} of {n_hs}", flush=True)

    splits = []
    for spec in a.prm800k:
        name, path, *lim = spec.split(":")
        splits.append((name, prm800k_items(Path(path), int(lim[0]) if lim else 0, a.seed)))
    for spec in a.pb:
        sub, path = spec.split(":")
        splits.append((f"pb_{sub}", pb_items(Path(path), sub)))

    for name, items in splits:
        items = items[a.shard_idx::a.num_shards]
        if a.smoke:
            items = items[:a.smoke]
        toks = [tokenize(tok, it, a.style) for it in items]
        keep = [i for i, t in enumerate(toks) if len(t[0]) <= a.max_seq_len]
        order = sorted(keep, key=lambda i: len(toks[i][0]))
        d = body.config.hidden_size
        reps = ["last", "mean", "boundary"] + (["sep"] if a.style == "prm_native" else [])
        arr = {r: np.zeros((len(items), len(layers), d), np.float16) for r in reps}
        head_p = np.full(len(items), np.nan, np.float32)
        t0 = time.perf_counter()
        i = 0
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        while i < len(order):
            L = len(toks[order[i]][0])
            j = i
            while j < len(order) and (j - i + 1) * len(toks[order[j]][0]) <= a.max_tokens_per_batch:
                j += 1
            j = max(j, i + 1)
            batch = order[i:j]
            m = max(len(toks[b][0]) for b in batch)
            ids = torch.full((len(batch), m), pad, dtype=torch.long)
            att = torch.zeros((len(batch), m), dtype=torch.long)
            for r, b in enumerate(batch):
                t = toks[b][0]
                ids[r, :len(t)] = torch.tensor(t)
                att[r, :len(t)] = 1
            with torch.no_grad():
                out = body(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True,
                           use_cache=False)
                hs = torch.stack([out.hidden_states[l] for l in layers], 1)  # (B, nL, T, d)
                if head is not None:
                    seps = torch.tensor([toks[b][3] for b in batch], device=hs.device)
                    last_h = out.last_hidden_state[torch.arange(len(batch), device=hs.device), seps]
                    pc = torch.softmax(head(last_h).float(), -1)[:, 1].cpu().numpy()
            for r, b in enumerate(batch):
                _, s0, s1, sp_ = toks[b]
                arr["last"][b] = hs[r, :, s1 - 1].float().cpu().numpy()
                arr["mean"][b] = hs[r, :, s0:s1].float().mean(1).cpu().numpy()
                arr["boundary"][b] = hs[r, :, s0 - 1].float().cpu().numpy()
                if head is not None:
                    arr["sep"][b] = hs[r, :, sp_].float().cpu().numpy()
                    head_p[b] = pc[r]
            del out, hs
            i = j
            if (i // max(1, len(batch))) % 50 == 0:
                print(f"[pool] {name} {i}/{len(order)} ({time.perf_counter()-t0:.0f}s, L={L})", flush=True)
        for r in reps:
            if not np.isfinite(arr[r][keep]).all():
                raise ValueError(f"non-finite {r} in {name}")
        np.savez(a.out / f"{name}.shard{a.shard_idx}.npz", layers=np.array(layers),
                 y=np.array([it["label"] for it in items], np.int8),
                 kept=np.isin(np.arange(len(items)), keep), head_p_correct=head_p,
                 **{r: arr[r] for r in reps})
        with open(a.out / f"{name}.shard{a.shard_idx}.meta.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps({k: it[k] for k in ("uid", "id", "step_idx", "first_error", "label")
                                    if k in it}) + "\n")
        print(f"[pool] {name}: {len(keep)}/{len(items)} encoded in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
