#!/usr/bin/env python3
"""ReProbe's hidden-state step head on frozen Qwen3-8B: released checkpoints and retraining
(reprobe_comparison_v1, docs/reprobe_comparison_v1_plan.md).

Features are Qwen3-8B's final post-norm hidden states over prompt + reply, read one
position late (token t at the state of position t - 1), exactly as ReProbe's
`basic_hidden_states` extractor with `layer_nums: [-1]`. The head is
src/probes/reprobe_head.py.

Phases
  released  score one released checkpoint dir (config.yaml + weights.pth) on trace files,
            with future tokens visible (as released) and masked
  train     train several heads on one backbone pass per batch: --heads name:label:future,
            e.g. rp_ds:ds:visible rpc_ds:ds:masked rp_self:self:visible rpc_self:self:masked.
            ReProbe's recipe: BCE pos_weight 3, AdamW lr 5e-4, weight decay 0.1 (not on
            biases and LayerNorm), 128 traces per step, 5 epochs, linear schedule with 5%
            warmup, grad clip 1.0, ReProbe's init; each head keeps its best epoch by
            validation AP
  score     score a trained heads.pt on trace files

Trace files (scripts/reprobe_cmp/build_{testsets,train}.py): one trace per line with
either input_ids + context_len (training, ProcessBench) or prompt_text + reply_ids (test
sets), steps[k].aligned_token_ids (reply-relative) and labels (a list, or a dict of label
sets). Scores go to <out>/scores/<head>/<set>.jsonl, one step per line:
{"uid": "<trace_id>::<k>", "p": P(incorrect)}.

    python scripts/reprobe_cmp/reprobe_probe.py released --ckpt <dir> --name rel_ds \\
        --sets math:<data>/test/math.traces.jsonl ... --out <dir>
    python scripts/reprobe_cmp/reprobe_probe.py train --train <data>/rp/train.traces.jsonl \\
        --val <data>/rp/val.traces.jsonl --seed 0 --out <dir> --sets ...
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
from src.probes.reprobe_head import ReProbeHead, n_params, step_feature_index  # noqa: E402


def load_traces(path: Path, tok, limit: int = 0) -> list[dict]:
    """Traces normalized to {trace_id, ids, ctx, steps: [[feature positions]], labels}."""
    out = []
    for l in open(path):
        if not l.strip():
            continue
        t = json.loads(l)
        if "input_ids" in t:
            ids, ctx = [int(x) for x in t["input_ids"]], int(t["context_len"])
        else:
            p = tok(t["prompt_text"], add_special_tokens=False)["input_ids"]
            ids, ctx = p + [int(x) for x in t["reply_ids"]], len(p)
        steps = [[q for q in step_feature_index(ctx, s["aligned_token_ids"]) if 0 <= q < len(ids) - 1]
                 for s in t["steps"]]
        out.append({"trace_id": t["trace_id"], "ids": ids, "ctx": ctx, "steps": steps, "labels": t["labels"]})
        if limit and len(out) >= limit:
            break
    return out


def labels_of(t: dict, key: str | None) -> list[int]:
    y = t["labels"]
    if isinstance(y, dict):
        y = y[key] if key in y else next(iter(y.values()))
    return [int(v) for v in y]


def micro_batches(traces: list[dict], order: list[int], max_tokens: int, max_claim_tokens: int) -> list[list[int]]:
    """Consecutive runs of `order`, each under a padded-token and a claims x length budget."""
    out, cur, T, C = [], [], 0, 0
    for i in order:
        L, c = len(traces[i]["ids"]), max(len(traces[i]["steps"]), 1)
        T2, C2 = max(T, L), C + c
        if cur and ((len(cur) + 1) * T2 > max_tokens or C2 * T2 > max_claim_tokens):
            out.append(cur)
            cur, T, C = [], 0, 0
            T2, C2 = L, c
        cur.append(i)
        T, C = T2, C2
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def features(model, batch: list[dict], dev, pad: int):
    """(feats (B, T-1, d) bf16, valid (B, T-1) bool) for right-padded traces."""
    B, T = len(batch), max(len(t["ids"]) for t in batch)
    T = -(-T // 64) * 64  # few distinct shapes: every new shape costs an attention-kernel setup
    ids = torch.full((B, T), pad, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    for b, t in enumerate(batch):
        ids[b, :len(t["ids"])] = torch.tensor(t["ids"])
        att[b, :len(t["ids"])] = 1
    h = model.model(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
    valid = torch.zeros((B, T - 1), dtype=torch.bool)
    for b, t in enumerate(batch):
        valid[b, :len(t["ids"]) - 1] = True
    return h[:, :-1], valid.to(dev)


def claim_tensors(batch: list[dict], T: int, dev):
    """(claims (C, T) long, trace_of (C,) long, (b, k) per claim) for steps with tokens."""
    rows, owner, where = [], [], []
    for b, t in enumerate(batch):
        for k, pos in enumerate(t["steps"]):
            if not pos:
                continue
            m = torch.zeros(T, dtype=torch.long)
            m[pos] = 1
            rows.append(m)
            owner.append(b)
            where.append((b, k))
    if not rows:
        return None, None, []
    return torch.stack(rows).to(dev), torch.tensor(owner, device=dev), where


def run_head(head, feats, valid, claims, owner, future):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=feats.is_cuda):
        return head(feats, valid, claims, owner, future).float()


@torch.no_grad()
def score_traces(model, heads: dict, traces: list[dict], dev, pad: int, max_tokens: int, max_claim_tokens: int) -> dict:
    """{head name: {uid: P(incorrect)}}; heads = {name: (module, future)}."""
    for h, _ in heads.values():
        h.eval()
    out = {n: {} for n in heads}
    order = sorted(range(len(traces)), key=lambda i: len(traces[i]["ids"]))
    for mb in micro_batches(traces, order, max_tokens, max_claim_tokens):
        batch = [traces[i] for i in mb]
        feats, valid = features(model, batch, dev, pad)
        claims, owner, where = claim_tensors(batch, feats.shape[1], dev)
        if claims is None:
            continue
        for n, (h, fut) in heads.items():
            p = torch.sigmoid(run_head(h, feats, valid, claims, owner, fut)).cpu().numpy()
            for (b, k), v in zip(where, p):
                out[n][f"{batch[b]['trace_id']}::{k}"] = float(v)
    return out


def ap(y, s) -> float:
    from sklearn.metrics import average_precision_score
    y, s = np.asarray(y), np.asarray(s)
    m = y >= 0
    return float(average_precision_score(y[m], s[m])) if m.any() and 0 < y[m].mean() < 1 else float("nan")


def write_scores(out: Path, name: str, set_name: str, scores: dict) -> None:
    d = out / "scores" / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{set_name}.jsonl", "w") as f:
        for uid, p in scores.items():
            f.write(json.dumps({"uid": uid, "p": p}) + "\n")


def set_ap(traces: list[dict], scores: dict, key: str | None) -> float:
    y, s = [], []
    for t in traces:
        for k, v in enumerate(labels_of(t, key)):
            uid = f"{t['trace_id']}::{k}"
            if uid in scores:
                y.append(v)
                s.append(scores[uid])
    return ap(y, s)


def load_model(name: str, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, local_files_only=True).to(dev).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return model, tok, pad


def head_from_release(d: Path, dev, feature_dim: int = 4096) -> ReProbeHead:
    import yaml
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    fe = cfg["feature_extractor"]
    if len(fe) != 1 or not fe[0]["name"].endswith("basic_hidden_states") or list(fe[0]["layer_nums"]) != [-1]:
        raise SystemExit(f"{d}: only the last-layer hidden-state probes are supported, got {fe}")
    u = cfg["uncertainty_head"]
    h = ReProbeHead(feature_dim, u["head_dim"], u["n_layers"], u["n_heads"], u["dropout"])
    sd = torch.load(d / "weights.pth", weights_only=True, map_location="cpu")
    h.load_state_dict({k: v.float() for k, v in sd.items()})
    return h.to(dev).eval()


def parse_sets(specs: list[str]) -> list[tuple[str, Path]]:
    return [(s.split(":", 1)[0], Path(s.split(":", 1)[1])) for s in specs]


def score_sets(model, tok, pad, heads, specs, out: Path, dev, a, label_key=None) -> dict:
    summary = {}
    for set_name, path in parse_sets(specs):
        traces = load_traces(path, tok, a.smoke)
        t0 = time.perf_counter()
        sc = score_traces(model, heads, traces, dev, pad, a.max_tokens, a.max_claim_tokens)
        for n, s in sc.items():
            write_scores(out, n, set_name, s)
            summary.setdefault(n, {})[set_name] = set_ap(traces, s, label_key)
        print(f"[score] {set_name}: {len(traces)} traces in {time.perf_counter() - t0:.0f}s "
              + " ".join(f"{n} AP {summary[n][set_name]:.4f}" for n in sc), flush=True)
    return summary


def decay_groups(h: torch.nn.Module, wd: float) -> list[dict]:
    """Hugging Face Trainer's grouping: no weight decay on biases and LayerNorm weights."""
    dec, nodec = [], []
    for mn, m in h.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            (nodec if pn.endswith("bias") or isinstance(m, torch.nn.LayerNorm) else dec).append(p)
    return [{"params": dec, "weight_decay": wd}, {"params": nodec, "weight_decay": 0.0}]


def train(a, model, tok, pad, dev) -> None:
    specs = [h.split(":") for h in a.heads]
    tr = load_traces(a.train, tok, a.smoke)
    va = load_traces(a.val, tok, a.smoke)
    torch.manual_seed(a.seed)
    heads, opts = {}, {}
    steps_per_epoch = math.ceil(len(tr) / a.batch_traces)
    total = a.epochs * steps_per_epoch
    for name, key, fut in specs:
        h = ReProbeHead(model.config.hidden_size).reprobe_init_().to(dev).train()
        opt = torch.optim.AdamW(decay_groups(h, a.wd), lr=a.lr)
        warm = int(round(0.05 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s, w=warm: (s + 1) / max(1, w) if s < w else max(0.0, (total - s) / max(1, total - w)))
        heads[name] = (h, fut, key)
        opts[name] = (opt, sched)
    print(f"[train] {len(tr)} train / {len(va)} val traces, heads {[s[0] for s in specs]}, "
          f"{n_params(next(iter(heads.values()))[0])} params each, {total} steps", flush=True)
    pw = torch.tensor([a.pos_weight], device=dev)
    best = {n: (-1.0, None, -1) for n in heads}
    log = []
    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    for ep in range(a.epochs):
        order = rng.permutation(len(tr)).tolist()
        for st in range(steps_per_epoch):
            chunk = order[st * a.batch_traces:(st + 1) * a.batch_traces]
            nlab = {n: sum(sum(1 for v in labels_of(tr[i], k) if v >= 0) for i in chunk)
                    for n, (_, _, k) in heads.items()}
            for opt, _ in opts.values():
                opt.zero_grad(set_to_none=True)
            run = {n: 0.0 for n in heads}
            for mb in micro_batches(tr, sorted(chunk, key=lambda i: len(tr[i]["ids"])), a.max_tokens,
                                    a.max_claim_tokens):
                batch = [tr[i] for i in mb]
                feats, valid = features(model, batch, dev, pad)
                claims, owner, where = claim_tensors(batch, feats.shape[1], dev)
                if claims is None:
                    continue
                for n, (h, fut, key) in heads.items():
                    y = torch.tensor([labels_of(batch[b], key)[k] for b, k in where], device=dev, dtype=torch.float32)
                    m = y >= 0
                    if not m.any():
                        continue
                    logit = run_head(h, feats, valid, claims, owner, fut)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logit[m], y[m], pos_weight=pw, reduction="sum") / max(nlab[n], 1)
                    loss.backward()
                    run[n] += loss.item()
            for n, (opt, sched) in opts.items():
                torch.nn.utils.clip_grad_norm_(heads[n][0].parameters(), 1.0)
                opt.step()
                sched.step()
            if (st + 1) % 25 == 0 or st + 1 == steps_per_epoch:
                print(f"[train] epoch {ep + 1} step {st + 1}/{steps_per_epoch} "
                      + " ".join(f"{n} {v:.4f}" for n, v in run.items())
                      + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        sc = score_traces(model, {n: (h, fut) for n, (h, fut, _) in heads.items()}, va, dev, pad,
                          a.max_tokens, a.max_claim_tokens)
        row = {"epoch": ep + 1}
        for n, (h, fut, key) in heads.items():
            v = set_ap(va, sc[n], key)
            row[n] = v
            if v > best[n][0]:
                best[n] = (v, copy.deepcopy(h.state_dict()), ep + 1)
            h.train()
        log.append(row)
        print(f"[val] {json.dumps(row)}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({n: {"state": best[n][1], "future": heads[n][1], "label": heads[n][2], "epoch": best[n][2],
                    "val_ap": best[n][0]} for n in heads} | {"_meta": {"seed": a.seed, "lr": a.lr,
                                                                       "epochs": a.epochs, "log": log}},
               a.out / "heads.pt")
    (a.out / "train_log.json").write_text(json.dumps(log, indent=1))
    for n in heads:
        heads[n][0].load_state_dict(best[n][1])
    final = {n: (h, fut) for n, (h, fut, _) in heads.items()}
    if a.sets:
        score_sets(model, tok, pad, final, a.sets, a.out, dev, a)
    if a.val_scores:
        sc = score_traces(model, final, va, dev, pad, a.max_tokens, a.max_claim_tokens)
        for n, s in sc.items():
            write_scores(a.out, n, "val", s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["released", "train", "score"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--ckpt", type=Path, help="released: dir with config.yaml + weights.pth; score: heads.pt")
    p.add_argument("--name", default="released", help="released: score name prefix")
    p.add_argument("--sets", nargs="*", default=[], help="name:path to trace files to score")
    p.add_argument("--train", type=Path)
    p.add_argument("--val", type=Path)
    p.add_argument("--val_scores", action="store_true", help="train: also write the best heads' val scores")
    p.add_argument("--heads", nargs="*", default=["rp_ds:ds:visible", "rpc_ds:ds:masked",
                                                  "rp_self:self:visible", "rpc_self:self:masked"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--pos_weight", type=float, default=3.0)
    p.add_argument("--batch_traces", type=int, default=128)
    p.add_argument("--max_tokens", type=int, default=24000, help="padded tokens per backbone micro-batch")
    p.add_argument("--max_claim_tokens", type=int, default=160000, help="steps x padded length per micro-batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", type=int, default=0, help="first N traces of every file")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cuDNN SDPA builds a graph per new shape (seconds each for the head's masked attention;
    # the smoke run spent up to 20 min on 40 traces), and its backward has given NaN
    # gradients on padded batches before (read_token_v1 Stage C)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model, tok, pad = load_model(a.model, dev)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.phase == "released":
        h = head_from_release(a.ckpt, dev, model.config.hidden_size)
        heads = {f"{a.name}": (h, "visible"), f"{a.name}_masked": (h, "masked")}
        summary = score_sets(model, tok, pad, heads, a.sets, a.out, dev, a)
    elif a.phase == "score":
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        heads = {}
        for n, v in ck.items():
            if n.startswith("_"):
                continue
            h = ReProbeHead(model.config.hidden_size)
            h.load_state_dict(v["state"])
            heads[n] = (h.to(dev).eval(), v["future"])
        summary = score_sets(model, tok, pad, heads, a.sets, a.out, dev, a)
    else:
        train(a, model, tok, pad, dev)
        return
    (a.out / f"summary_{a.phase}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
