#!/usr/bin/env python3
"""Score saved trajectories with RTH, in the codebase's score schema.

RTH (Read-Token Head, docs/read_token_v1_plan.md) is a trained read token on frozen
Qwen3-8B classified by its own linear head. It reads a step the way it was trained
to: the verifier template with the problem, the earlier steps of the same
trajectory as "Previous reasoning" and the step as "Current step", then the read
token. Every step is one item; the template puts the step after its own header,
so step k's text is not a prefix of step k+1's. By default each step is read with
its own prefix pass; `--branched` runs the trajectory's longest item once and gives
every step only the tokens after its longest common prefix with it
(`read_token_probe.read_branched`, the same computation at a fraction of the
tokens). `--inline` reads a token trained inside the trace (RTH-I,
scripts/read_token_inline.py): each trajectory once, in the chat layout it was
sampled under, with a read token after every step. `--compare_to` checks the
scores against an earlier score file.

Steps come from `split_into_steps`, the segmentation every other scorer of these
pools used, and `--verify_against` checks the step count per trace against an
existing score file (the PRM's) before anything is written. Scores are suspicion,
P(step incorrect) = sigmoid(head logit), as every consumer here expects.

    python scripts/onpolicy/score_traces_with_read_token.py --ckpt tokens/dq_correct/read_token.pt \
        --trajectories <pool>/tts_math500.shard0*_trajectories.jsonl \
        --verify_against <pool>/scores/tts_math500__prm_qwen25_math_7b.shard*.jsonl \
        --out <pool>/scores/tts_math500__rth_dq_correct.shard0.jsonl --shard_idx 0 --num_shards 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.generate_onpolicy_steps import split_into_steps  # noqa: E402
from scripts.read_token_probe import (  # noqa: E402
    batches, read_all_states, read_branched, reduce_states, tokenize)
from scripts.read_token_store import load_model, load_read_token  # noqa: E402
from src.onpolicy.prompts import chat_prompt, context_from_row  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def step_items(r: dict) -> list[dict]:
    steps = split_into_steps(r["solution"])
    return [{"problem": r["problem"], "prefix": "\n\n".join(steps[:k]), "step": s}
            for k, s in enumerate(steps)]


def score_items(model, tok, rt, items: list[dict], pad, dev, max_tokens: int,
                soft_attn) -> np.ndarray:
    """P(step incorrect) for each item."""
    toks = [tokenize(tok, it) for it in items]
    out = np.zeros(len(items), np.float64)
    for b in batches([len(t) for t in toks], max_tokens, None):
        with torch.no_grad():
            h = reduce_states(rt, read_all_states(model, rt, [toks[i] for i in b], pad, dev,
                                                  grad=False, soft_attn=soft_attn))
            logit = rt.logit(h).double()
        if not torch.isfinite(logit).all():
            raise SystemExit("non-finite RTH logit")
        out[b] = torch.sigmoid(logit).cpu().numpy()
    return out


def score_branched(model, tok, rt, items: list[dict], pad, dev, soft_attn) -> tuple[np.ndarray, int]:
    """P(step incorrect) for one trajectory's steps, and the tokens computed."""
    ids = [tokenize(tok, it) for it in items]
    h = reduce_states(rt, read_branched(model, rt, ids, pad, dev, soft_attn=soft_attn))
    with torch.no_grad():
        logit = rt.logit(h).double()
    if not torch.isfinite(logit).all():
        raise SystemExit("non-finite RTH logit")
    trunk = max(ids, key=len)
    from scripts.read_token_probe import common_prefix
    n_tok = len(trunk) + sum(len(t) - common_prefix(t, trunk) + rt.m for t in ids)
    return torch.sigmoid(logit).cpu().numpy(), n_tok


def score_inline(model, tok, rt, rows: list[dict], pad, dev, max_tokens: int,
                 soft_attn) -> tuple[list[np.ndarray], int]:
    """P(step incorrect) per trajectory, each read once in the chat layout with a read
    token after every step, and the tokens computed (text + read tokens)."""
    from scripts.read_token_inline import build_sequences, inline_read, seq_batches
    seqs, owner = [], []
    for t, r in enumerate(rows):
        if context_from_row(r) != chat_prompt(r["problem"]):
            raise SystemExit(f"{r['traj_uid']} was sampled under {r.get('prompt_style')!r}, "
                             f"not the chat layout an inline token reads")
        for q in build_sequences(tok, step_items(r), "chat")[0]:
            seqs.append(q)
            owner.append(t)
    out = [np.zeros(len(step_items(r)), np.float64) for r in rows]
    for b in seq_batches([len(q["ids"]) for q in seqs], [len(q["pos"]) for q in seqs],
                         max_tokens, None, None):
        with torch.no_grad():
            logit = rt.logit(inline_read(model, rt, [(seqs[i]["ids"], seqs[i]["pos"]) for i in b],
                                         pad, dev, grad=False, soft_attn=soft_attn)).double()
        if not torch.isfinite(logit).all():
            raise SystemExit("non-finite RTH-I logit")
        s = torch.sigmoid(logit).cpu().numpy()
        k = 0
        for i in b:
            for member in seqs[i]["members"]:
                out[owner[i]][member] = s[k]
                k += 1
    return out, sum(len(q["ids"]) + len(q["pos"]) for q in seqs)


def compare(out: list[dict], paths: list[Path]) -> dict:
    ref = {r["traj_uid"]: r["scores"] for p in paths for r in read_jsonl(p)}
    a, b = [], []
    for r in out:
        if r["traj_uid"] in ref and len(ref[r["traj_uid"]]) == len(r["scores"]):
            a += r["scores"]
            b += ref[r["traj_uid"]]
    a, b = np.array(a), np.array(b)
    d = np.abs(a - b)
    return {"n_steps": int(len(a)), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "pearson": float(np.corrcoef(a, b)[0, 1])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p.add_argument("--verify_against", type=Path, nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell_name", default=None, help="default: rth_<token dir name>")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=400, help="trajectories per batching pass")
    p.add_argument("--max_tokens", type=int, default=16000)
    p.add_argument("--soft_attn", choices=["no_cudnn"], default=None)
    p.add_argument("--branched", action="store_true", help="share each trajectory's common prefixes")
    p.add_argument("--inline", action="store_true",
                   help="an RTH-I token: each trajectory read once in the chat layout")
    p.add_argument("--compare_to", type=Path, nargs="*", default=[],
                   help="earlier score files of the same traces; prints the agreement")
    p.add_argument("--fork_ids", type=Path, default=None,
                   help="JSON list of fork_ids; only their trajectories are scored (before sharding)")
    a = p.parse_args()
    cell = a.cell_name or f"rth_{a.ckpt.parent.name}"

    rows = [r for path in a.trajectories for r in read_jsonl(path)]
    rows = [r for r in rows if r.get("gradeable") and (r.get("solution") or "").strip()]
    if a.fork_ids is not None:
        keep = set(json.loads(a.fork_ids.read_text()))
        rows = [r for r in rows if r.get("fork_id") in keep]
        print(f"[rth] --fork_ids: {len(keep)} questions, {len(rows)} traces", flush=True)
    mine = rows[a.shard_idx::a.num_shards]
    expect = {r["traj_uid"]: int(r["n_steps"]) for path in a.verify_against for r in read_jsonl(path)}
    bad = [r["traj_uid"] for r in mine
           if r["traj_uid"] in expect and expect[r["traj_uid"]] != len(split_into_steps(r["solution"]))]
    covered = sum(r["traj_uid"] in expect for r in mine)
    print(f"[rth] shard {a.shard_idx}/{a.num_shards}: {len(mine)} traces, step counts checked on "
          f"{covered}, mismatched {len(bad)}", flush=True)
    if bad:
        raise SystemExit(f"segmentation differs from --verify_against on {len(bad)} traces, e.g. {bad[:3]}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rt, ck = load_read_token(a.ckpt, model.config.hidden_size, dev, model)
    soft_attn = a.soft_attn or ck.get("soft_attn")

    if a.inline and a.branched:
        raise SystemExit("--inline and --branched are two different reads; pick one")
    out, t0, n_steps, n_tok = [], time.perf_counter(), 0, 0
    if a.inline:
        for c in range(0, len(mine), a.chunk):
            part = mine[c:c + a.chunk]
            sc, nt = score_inline(model, tok, rt, part, pad, dev, a.max_tokens, soft_attn)
            for r, s in zip(part, sc):
                out.append({"traj_uid": r["traj_uid"], "problem_id": r.get("fork_id"),
                            "dataset": r.get("dataset"), "split": r.get("split"),
                            "correct": bool(r["correct"]), "n_steps": len(s), "cell": cell,
                            "scores": [float(x) for x in s]})
                n_steps += len(s)
            n_tok += nt
            print(f"[rth-i] {min(c + a.chunk, len(mine))}/{len(mine)} traces, {n_steps} steps, "
                  f"{n_tok} tokens ({time.perf_counter() - t0:.0f}s)", flush=True)
    if a.branched:
        for i, r in enumerate(mine):
            items = step_items(r)
            s, nt = score_branched(model, tok, rt, items, pad, dev, soft_attn)
            out.append({"traj_uid": r["traj_uid"], "problem_id": r.get("fork_id"),
                        "dataset": r.get("dataset"), "split": r.get("split"),
                        "correct": bool(r["correct"]), "n_steps": len(items), "cell": cell,
                        "scores": [float(x) for x in s]})
            n_steps += len(items)
            n_tok += nt
            if (i + 1) % 200 == 0 or i + 1 == len(mine):
                print(f"[rth] {i + 1}/{len(mine)} traces, {n_steps} steps, {n_tok} tokens "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    for c in range(0, len(mine) if not (a.branched or a.inline) else 0, a.chunk):
        part = mine[c:c + a.chunk]
        per = [step_items(r) for r in part]
        flat = [it for items in per for it in items]
        s = score_items(model, tok, rt, flat, pad, dev, a.max_tokens, soft_attn)
        k = 0
        for r, items in zip(part, per):
            out.append({"traj_uid": r["traj_uid"], "problem_id": r.get("fork_id"),
                        "dataset": r.get("dataset"), "split": r.get("split"),
                        "correct": bool(r["correct"]), "n_steps": len(items), "cell": cell,
                        "scores": [float(x) for x in s[k:k + len(items)]]})
            k += len(items)
        n_steps += len(flat)
        print(f"[rth] {min(c + a.chunk, len(mine))}/{len(mine)} traces, {n_steps} steps "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    tmp.rename(a.out)
    (a.out.parent / f"{a.out.stem}_manifest.json").write_text(json.dumps({
        "cell": cell, "ckpt": str(a.ckpt), "model": a.model, "n_traces": len(out),
        "n_steps": n_steps, "seconds": round(time.perf_counter() - t0, 1),
        "branched": a.branched, "inline": a.inline,
        "tokens": n_tok if (a.branched or a.inline) else None,
        "score_schema": "suspicion_probability_v1",
        "template": "inline_chat" if a.inline else "verifier"}, indent=1))
    print(f"[rth] wrote {a.out}: {len(out)} traces", flush=True)
    if a.compare_to:
        print(f"[rth] against {len(a.compare_to)} earlier file(s): {compare(out, a.compare_to)}", flush=True)


if __name__ == "__main__":
    main()
