#!/usr/bin/env python3
"""InStep paper, Figure "where the signal forms": ReProbe's hidden-state head retrained on each
layer of Qwen3-8B, so that ReProbe gets a per-layer curve instead of one last-layer dot.

ReProbe's released head reads the backbone's final post-norm states (`basic_hidden_states`,
layer -1). Here one head per block l in --blocks reads the output of block l passed through the
backbone's final RMSNorm, the same normalisation its released input has; at block 35 (the last of
36) this is exactly the released input (Qwen3's last_hidden_state = norm(output of the last
block)). Everything else is reprobe_probe.py's training recipe unchanged: same module
(ReProbeHead, released hyperparameters), future tokens visible as released, AdamW lr 5e-4 with 5%
warm-up and linear decay, weight decay 0.1, pos_weight 3, 128 traces per step, 5 epochs, best
epoch by validation AP (DeepSeek-R1 labels of rp/val.traces.jsonl).

Training steps: the labels of the steps listed in --train_keep (the 40k-step draw the InStep
states were encoded on, train40k.shard0.meta.jsonl), on the traces that contain them; every other
step of those traces is unlabelled (-1) and only serves as context, as for any ReProbe trace.

After training, one pass over --sets (name:traces.jsonl) writes, for the uids of --keep
(name:meta.jsonl, the InStep states' order), <out>/<name>.shard0.npz with y, kept and per block
`rp_block<l>` (n, 1, 512) float16, the head's pooled step representation (the input of its
classifier, as signal_states.py `reprobe_rep`), and `p_block<l>` (n,) the head's own
P(incorrect); plus <out>/<name>.shard0.meta.jsonl. All heads share each backbone forward pass.

    python scripts/instep_v1/reprobe_layers.py --train $D/rp/train.traces.jsonl --val $D/rp/val.traces.jsonl \\
        --train_keep $ST8/train40k.shard0.meta.jsonl --sets train40k:$D/rp/train.traces.jsonl val:... math:... \\
        --keep train40k:$ST8/train40k.shard0.meta.jsonl ... --seed 0 --out $SS/reprobe_layers/seed0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reprobe_cmp.reprobe_probe import (claim_tensors, decay_groups, labels_of, load_model,  # noqa: E402
                                               load_traces, micro_batches, run_head, set_ap)
from src.probes.reprobe_head import ReProbeHead, n_params  # noqa: E402


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def pairs(specs: list[str]) -> dict[str, Path]:
    return {s.split(":", 1)[0]: Path(s.split(":", 1)[1]) for s in specs}


def restrict_labels(traces: list[dict], keep: set[str], key: str) -> list[dict]:
    """Traces with at least one kept step; labels of the other steps set to -1."""
    out = []
    for t in traces:
        y = labels_of(t, key)
        y = [v if f"{t['trace_id']}::{k}" in keep else -1 for k, v in enumerate(y)]
        if any(v >= 0 for v in y):
            out.append({**t, "labels": {key: y}})
    return out


@torch.no_grad()
def layer_features(model, batch: list[dict], dev, pad: int, blocks: list[int]) -> tuple[dict, torch.Tensor]:
    """({block: (B, T-1, d) bf16 = norm(output of block)}, valid (B, T-1)), shifted by one as
    reprobe_probe.features (the feature of token t is the state at position t - 1)."""
    B, T = len(batch), max(len(t["ids"]) for t in batch)
    T = -(-T // 64) * 64
    ids = torch.full((B, T), pad, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    for b, t in enumerate(batch):
        ids[b, :len(t["ids"])] = torch.tensor(t["ids"])
        att[b, :len(t["ids"])] = 1
    cap = {}

    def make(l):
        def hook(_m, _i, out):
            cap[l] = out[0] if isinstance(out, tuple) else out
        return hook

    hs = [model.model.layers[l].register_forward_hook(make(l)) for l in blocks]
    try:
        model.model(input_ids=ids.to(dev), attention_mask=att.to(dev))
    finally:
        for h in hs:
            h.remove()
    norm = model.model.norm
    feats = {l: norm(cap[l])[:, :-1] for l in blocks}
    valid = torch.zeros((B, T - 1), dtype=torch.bool)
    for b, t in enumerate(batch):
        valid[b, :len(t["ids"]) - 1] = True
    return feats, valid.to(dev)


def score(model, heads: dict, traces: list[dict], dev, pad: int, a, reps: bool = False):
    """{block: {uid: p}} and, with reps, {block: {uid: pooled representation}}."""
    for h in heads.values():
        h.eval()
    cap: dict = {}
    hooks = []
    if reps:
        for l, h in heads.items():
            hooks.append(h.classifier.register_forward_hook(
                lambda m, inp, out, l=l: cap.__setitem__(l, inp[0].float())))
    P = {l: {} for l in heads}
    R = {l: {} for l in heads} if reps else None
    order = sorted(range(len(traces)), key=lambda i: len(traces[i]["ids"]))
    try:
        for mb in micro_batches(traces, order, a.max_tokens, a.max_claim_tokens):
            batch = [traces[i] for i in mb]
            feats, valid = layer_features(model, batch, dev, pad, list(heads))
            claims, owner, where = claim_tensors(batch, feats[next(iter(heads))].shape[1], dev)
            if claims is None:
                continue
            for l, h in heads.items():
                with torch.no_grad():
                    p = torch.sigmoid(run_head(h, feats[l], valid, claims, owner, "visible")).cpu().numpy()
                rep = cap[l].cpu().numpy() if reps else None
                for j, ((b, k), v) in enumerate(zip(where, p)):
                    u = f"{batch[b]['trace_id']}::{k}"
                    P[l][u] = float(v)
                    if reps:
                        R[l][u] = rep[j]
    finally:
        for h in hooks:
            h.remove()
    return P, R


def train(a, model, tok, pad, dev, blocks: list[int]) -> dict:
    keep = {r["uid"] for r in jl(a.train_keep)}
    tr = restrict_labels(load_traces(a.train, tok), keep, a.label)
    va = load_traces(a.val, tok)
    if a.smoke:
        tr, va = tr[:a.smoke], va[:a.smoke]
    n_lab = sum(sum(1 for v in labels_of(t, a.label) if v >= 0) for t in tr)
    torch.manual_seed(a.seed)
    heads, opts = {}, {}
    steps_per_epoch = math.ceil(len(tr) / a.batch_traces)
    total = a.epochs * steps_per_epoch
    warm = int(round(0.05 * total))
    for l in blocks:
        h = ReProbeHead(model.config.hidden_size).reprobe_init_().to(dev).train()
        opt = torch.optim.AdamW(decay_groups(h, a.wd), lr=a.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s, w=warm: (s + 1) / max(1, w) if s < w else max(0.0, (total - s) / max(1, total - w)))
        heads[l], opts[l] = h, (opt, sched)
    print(f"[rpl] seed {a.seed}: {len(tr)} train traces with {n_lab} labelled steps, {len(va)} val traces, "
          f"blocks {blocks}, {n_params(heads[blocks[0]])} params per head, {total} steps", flush=True)
    pw = torch.tensor([a.pos_weight], device=dev)
    best = {l: (-1.0, None, -1) for l in blocks}
    log, rng, t0 = [], np.random.default_rng(a.seed), time.perf_counter()
    for ep in range(a.epochs):
        order = rng.permutation(len(tr)).tolist()
        for st in range(steps_per_epoch):
            chunk = order[st * a.batch_traces:(st + 1) * a.batch_traces]
            nl = sum(sum(1 for v in labels_of(tr[i], a.label) if v >= 0) for i in chunk)
            for opt, _ in opts.values():
                opt.zero_grad(set_to_none=True)
            run = {l: 0.0 for l in blocks}
            for mb in micro_batches(tr, sorted(chunk, key=lambda i: len(tr[i]["ids"])), a.max_tokens,
                                    a.max_claim_tokens):
                batch = [tr[i] for i in mb]
                feats, valid = layer_features(model, batch, dev, pad, blocks)
                claims, owner, where = claim_tensors(batch, feats[blocks[0]].shape[1], dev)
                if claims is None:
                    continue
                y = torch.tensor([labels_of(batch[b], a.label)[k] for b, k in where], device=dev, dtype=torch.float32)
                m = y >= 0
                if not m.any():
                    continue
                for l, h in heads.items():
                    logit = run_head(h, feats[l], valid, claims, owner, "visible")
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logit[m], y[m], pos_weight=pw, reduction="sum") / max(nl, 1)
                    loss.backward()
                    run[l] += loss.item()
                del feats
            for l, (opt, sched) in opts.items():
                torch.nn.utils.clip_grad_norm_(heads[l].parameters(), 1.0)
                opt.step()
                sched.step()
            if (st + 1) % 25 == 0 or st + 1 == steps_per_epoch:
                print(f"[rpl] epoch {ep + 1} step {st + 1}/{steps_per_epoch} "
                      + " ".join(f"b{l} {v:.4f}" for l, v in run.items()) + f" ({time.perf_counter() - t0:.0f}s)",
                      flush=True)
        P, _ = score(model, heads, va, dev, pad, a)
        row = {"epoch": ep + 1}
        for l, h in heads.items():
            v = set_ap(va, P[l], a.label)
            row[f"block{l}"] = v
            if v > best[l][0]:
                best[l] = (v, copy.deepcopy(h.state_dict()), ep + 1)
            h.train()
        log.append(row)
        print(f"[rpl val] {json.dumps(row)}", flush=True)
    for l, h in heads.items():
        h.load_state_dict(best[l][1])
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({f"block{l}": {"state": best[l][1], "epoch": best[l][2], "val_ap": best[l][0]} for l in blocks}
               | {"_meta": {"seed": a.seed, "blocks": blocks, "n_train_traces": len(tr), "n_train_labels": n_lab,
                            "lr": a.lr, "epochs": a.epochs, "log": log}}, a.out / "heads.pt")
    (a.out / "train_log.json").write_text(json.dumps(log, indent=1))
    return heads


def dump(a, model, tok, pad, dev, heads: dict) -> dict:
    keeps, report = pairs(a.keep), {}
    for name, path in pairs(a.sets).items():
        keep = [r["uid"] for r in jl(keeps[name])]
        want = set(keep)
        tids = {u.rsplit("::", 1)[0] for u in keep}
        trs = [t for t in load_traces(path, tok) if t["trace_id"] in tids]
        if a.smoke:
            trs = trs[:a.smoke]
        t0 = time.perf_counter()
        P, R = score(model, heads, trs, dev, pad, a, reps=True)
        lab = {f"{t['trace_id']}::{k}": v for t in trs for k, v in enumerate(labels_of(t, a.label))}
        n = len(keep)
        arr = {"y": np.array([lab.get(u, -1) for u in keep], np.int8),
               "kept": np.array([all(u in P[l] for l in heads) for u in keep])}
        for l in heads:
            d = len(next(iter(R[l].values())))
            X = np.zeros((n, 1, d), np.float16)
            p = np.full(n, np.nan, np.float32)
            for i, u in enumerate(keep):
                if u in R[l]:
                    X[i, 0], p[i] = R[l][u], P[l][u]
            arr[f"rp_block{l}"], arr[f"p_block{l}"] = X, p
        np.savez(a.out / f"{name}.shard0.npz", **arr)
        with open(a.out / f"{name}.shard0.meta.jsonl", "w") as f:
            for u in keep:
                f.write(json.dumps({"uid": u}) + "\n")
        report[name] = {"traces": len(trs), "kept": int(arr["kept"].sum()), "of": n,
                        "seconds": time.perf_counter() - t0}
        print(f"[rpl dump] {name}: {report[name]}", flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--train_keep", type=Path, required=True, help="meta.jsonl of the training steps to label")
    p.add_argument("--sets", nargs="+", required=True, help="name:traces.jsonl to dump")
    p.add_argument("--keep", nargs="+", required=True, help="name:meta.jsonl, uids per dumped set, in order")
    p.add_argument("--blocks", default="3,7,11,15,19,23,27,31,35")
    p.add_argument("--label", default="ds")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--pos_weight", type=float, default=3.0)
    p.add_argument("--batch_traces", type=int, default=128)
    p.add_argument("--max_tokens", type=int, default=24000)
    p.add_argument("--max_claim_tokens", type=int, default=160000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)  # as reprobe_probe.py
    dev = torch.device("cuda")
    model, tok, pad = load_model(a.model, dev)
    blocks = [int(x) for x in a.blocks.split(",")]
    if max(blocks) >= model.config.num_hidden_layers:
        raise SystemExit(f"blocks {blocks} beyond {model.config.num_hidden_layers} layers")
    heads = train(a, model, tok, pad, dev, blocks)
    rep = dump(a, model, tok, pad, dev, heads)
    (a.out / "dump_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
