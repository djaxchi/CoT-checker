#!/usr/bin/env python3
"""InStep V1 P1: step scores from two Qwen PRMs on the ReProbe test sets, validation and ProcessBench.

PRMs (tags): prm7b = Qwen/Qwen2.5-Math-PRM-7B, prm800k = Qwen/Qwen2.5-Math-7B-PRM800K. Both are
Qwen2ForProcessRewardModel (remote code, transformers < 5) with a 2-way head read at <extra_0>.

Protocol (matches ReProbe's baselines/prm.py PRMStatCalculator, which produced the PRM scores
stored in ReProbe's released managers):
  * text: the items file of a set (uid, id, step_idx, problem, prefix, candidate_step). Items
    are grouped by id and ordered by step_idx; the trace's step list is the items'
    candidate_step, each stripped (LM-Polygraph's StepsExtractor keeps claim_text =
    line.strip(), with the "- Step k: " prefix and the final "<Answer>: ..." line as is).
    Every prefix is checked against the earlier candidate steps (whitespace-normalized).
  * input: chat template of the PRM, system "Please reason step by step, and put your final
    answer within \\boxed{}.", user = the problem (ReProbe parses {q} out of its prompt, i.e.
    the bare question), assistant = "<extra_0>".join(steps) + "<extra_0>". One forward per
    trace (traces batched by length, right padding, bf16), no truncation.
  * score: P(correct) = softmax(logits)[..., 1] at each <extra_0>; we write
    p = 1 - P(correct) (higher = incorrect). ReProbe stores -P(correct); same order.
Output: <out>/<backbone>/<tag>/<set>.jsonl, one line {"uid", "p"} per item uid (the `rp:`
arm format of scripts/reprobe_cmp/analyze.py).

Gate (merge): on every test set whose traces carry scores["PRM_<model>"], Spearman between
our p and the stored value and PR-AUC (fast_ap, labels v >= 0) of both; pass = Spearman >=
0.98 and |dAP| <= 0.01, computed on our scores rounded to bf16 like the stored ones (ReProbe's
softmax ran in bf16, so a saturated PRM-7B leaves ~20 distinct stored values per 100 steps and
the ties alone pull the raw Spearman down to ~0.92-0.98; the raw numbers are reported too, and
the written scores keep full float32 precision). Written to <out>/gate_prm.json.

    score: CUDA_VISIBLE_DEVICES=0 python scripts/instep_v1/score_prm.py score --tag prm7b \\
               --out $OUT/scores --inputs qwen3_8b:math:$R/data/test/math.items.jsonl ... \\
               --shard 0 --num_shards 2 [--smoke 20]
    merge: python scripts/instep_v1/score_prm.py merge --out $OUT/scores --inputs ... \\
               --tags prm7b prm800k --num_shards 2 --data qwen3_8b=$R ... \\
               [--copy qwen3_8b:pb_math:phi4:$R2/data/pb/math.items.jsonl]
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TAGS = {"prm7b": "Qwen/Qwen2.5-Math-PRM-7B", "prm800k": "Qwen/Qwen2.5-Math-7B-PRM800K"}
SEP = "<extra_0>"
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
TEST = ("math", "gsm8k", "proofnet", "trips", "meetings", "calendar", "strategyqa", "scienceqa")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def _ws(s: str) -> str:
    return " ".join(s.split())


# ---------------------------------------------------------------- traces and formatting
def load_traces(items_path: Path, smoke: int = 0) -> tuple[list[dict], int]:
    """Items -> [{"id", "problem", "uids", "steps"}] in first-appearance order of ids, and the
    number of items whose prefix disagrees with the earlier candidate steps (whitespace-
    normalized). Steps are candidate_step.strip(); uid k must equal the step position."""
    by: dict[str, list[dict]] = collections.OrderedDict()
    for r in jl(items_path):
        by.setdefault(r["id"], []).append(r)
    out, bad = [], 0
    for tid, rs in by.items():
        rs.sort(key=lambda r: int(r["step_idx"]))
        for i, r in enumerate(rs):
            if int(r["uid"].rsplit("::", 1)[1]) != i or r["problem"] != rs[0]["problem"]:
                raise ValueError(f"{items_path}: trace {tid} item {r['uid']} out of order or problem differs")
            if _ws(r["prefix"]) != _ws(" ".join(x["candidate_step"] for x in rs[:i])):
                bad += 1
        out.append({"id": tid, "problem": rs[0]["problem"], "uids": [r["uid"] for r in rs],
                    "steps": [r["candidate_step"].strip() for r in rs]})
    return (out[:smoke] if smoke else out), bad


def messages(problem: str, steps: list[str]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": SEP.join(steps) + SEP}]


def p_correct_at_seps(logits, ids, sep_id: int, lengths: list[int]) -> list[list[float]]:
    """P(correct) (softmax channel 1, float32) at every separator of each row of a right-padded
    batch; logits [B, T, 2], ids [B, T]."""
    import torch
    probs = torch.softmax(logits.float(), dim=-1)[..., 1]
    out = []
    for b, n in enumerate(lengths):
        m = ids[b, :n] == sep_id
        out.append(probs[b, :n][m].tolist())
    return out


def batches(lengths: list[int], max_tokens: int) -> list[list[int]]:
    """Indices grouped longest first so that len(batch) * longest <= max_tokens (a single
    over-long sequence gets its own batch)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > max_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def score_rows(traces: list[dict], p_correct: list[list[float]]) -> list[dict]:
    """{"uid", "p" = 1 - P(correct)} rows; raises if a trace's separator count differs."""
    rows = []
    for t, pc in zip(traces, p_correct):
        if len(pc) != len(t["steps"]):
            raise ValueError(f"{t['id']}: {len(pc)} separators for {len(t['steps'])} steps")
        rows += [{"uid": u, "p": 1.0 - float(x)} for u, x in zip(t["uids"], pc)]
    return rows


def parse_inputs(specs: list[str]) -> list[tuple[str, str, Path]]:
    out = []
    for s in specs:
        bb, name, path = s.split(":", 2)
        out.append((bb, name, Path(path)))
    return out


# ---------------------------------------------------------------- score
def cmd_score(a) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = a.model_path or TAGS[a.tag]
    units = []  # (bb, set, trace)
    for bb, name, path in parse_inputs(a.inputs):
        trs, bad = load_traces(path, a.smoke)
        if bad:
            print(f"[prm] {bb}/{name}: {bad} items with a prefix that differs from the earlier steps "
                  f"(whitespace-normalized); steps come from candidate_step", flush=True)
        units += [(bb, name, t) for t in trs]
    mine = units[a.shard::a.num_shards]
    print(f"[prm] {a.tag} shard {a.shard}/{a.num_shards}: {len(mine)} of {len(units)} traces", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model, info = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                            local_files_only=True, output_loading_info=True)
    if info.get("missing_keys"):
        sys.exit(f"[prm] FAILED: {len(info['missing_keys'])} weights not loaded, e.g. {info['missing_keys'][:3]}")
    model = model.to("cuda").eval()
    sep_id = tok.encode(SEP)[0]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    assert pad_id != sep_id

    ids = [tok.encode(tok.apply_chat_template(messages(t["problem"], t["steps"]), tokenize=False,
                                              add_generation_prompt=False)) for _, _, t in mine]
    lens = [len(x) for x in ids]
    print(f"[prm] tokens: total {sum(lens)}, max {max(lens)}, > 4096: {sum(l > 4096 for l in lens)}", flush=True)
    pc: list[list[float] | None] = [None] * len(mine)
    t0 = time.perf_counter()
    bs = batches(lens, a.max_tokens)
    with torch.no_grad():
        for bi, b in enumerate(bs):
            T = lens[b[0]]
            x = torch.full((len(b), T), pad_id, dtype=torch.long)
            att = torch.zeros((len(b), T), dtype=torch.long)
            for j, i in enumerate(b):
                x[j, :lens[i]] = torch.tensor(ids[i])
                att[j, :lens[i]] = 1
            x, att = x.cuda(), att.cuda()
            out = model(input_ids=x, attention_mask=att, use_cache=False)
            logits = out[0] if isinstance(out, tuple) else out.logits
            for i, r in zip(b, p_correct_at_seps(logits, x, sep_id, [lens[i] for i in b])):
                pc[i] = r
            if bi % 50 == 0 or bi == len(bs) - 1:
                print(f"[prm] batch {bi + 1}/{len(bs)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    per = collections.defaultdict(list)
    for (bb, name, t), r in zip(mine, pc):
        per[(bb, name)] += score_rows([t], [r])
    for (bb, name), rows in per.items():
        d = a.out / bb / a.tag
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{name}.shard{a.shard}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"[prm] {a.tag} shard {a.shard} done: {len(mine)} traces, {sum(lens)} tokens in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- merge + gate
def merge_set(d: Path, name: str, num_shards: int, uids: list[str]) -> int:
    """Concatenate <name>.shard*.jsonl into <name>.jsonl in item order; every uid exactly once.
    Without shard files an existing <name>.jsonl is re-checked (a rerun of merge for the gate)."""
    sc = {}
    shards = [d / f"{name}.shard{s}.jsonl" for s in range(num_shards)]
    if not any(p.exists() for p in shards) and (d / f"{name}.jsonl").exists():
        shards = [d / f"{name}.jsonl"]
    for p in shards:
        if p.exists():
            for r in jl(p):
                if r["uid"] in sc:
                    raise ValueError(f"{d}/{name}: duplicate uid {r['uid']}")
                sc[r["uid"]] = r["p"]
    miss = [u for u in uids if u not in sc]
    if miss:
        raise ValueError(f"{d}/{name}: {len(miss)} of {len(uids)} uids missing (e.g. {miss[0]})")
    with open(d / f"{name}.jsonl", "w") as f:
        for u in uids:
            f.write(json.dumps({"uid": u, "p": sc[u]}) + "\n")
    for s in range(num_shards):
        (d / f"{name}.shard{s}.jsonl").unlink(missing_ok=True)
    return len(uids)


def gate_set(traces_path: Path, scores: dict, key: str) -> dict | None:
    """Ours vs stored on the labeled (v >= 0) steps that both have; None if nothing stored."""
    from scipy.stats import spearmanr
    from scripts.reprobe_cmp.analyze import fast_ap
    ours, st, y = [], [], []
    for t in jl(traces_path):
        s = t["scores"].get(key)
        if not s:
            continue
        for k, v in enumerate(t["labels"]):
            u = f"{t['trace_id']}::{k}"
            if v >= 0 and u in scores:
                ours.append(scores[u]); st.append(float(s[k])); y.append(v)
    if not ours:
        return None
    ours, st, y = np.array(ours), np.array(st), np.array(y)
    ob = -bf16_round(1 - ours)  # ReProbe's value: -P(correct), P from a bf16 softmax
    rho, rho_b = float(spearmanr(ours, st).correlation), float(spearmanr(ob, st).correlation)
    ap_o, ap_b, ap_s = fast_ap(y, ours), fast_ap(y, ob), fast_ap(y, st)
    d = np.abs(ours - (1 + st))
    return {"n": int(len(y)), "prevalence": float(y.mean()), "n_unique_stored": int(len(np.unique(st))),
            "spearman": rho, "ap_ours": ap_o, "ap_stored": ap_s, "d_ap": ap_o - ap_s,
            "spearman_bf16": rho_b, "ap_ours_bf16": ap_b, "d_ap_bf16": ap_b - ap_s,
            "median_abs_diff": float(np.median(d)), "share_within_0.008": float((d <= 0.008).mean()),
            "max_abs_diff_p_vs_1_plus_stored": float(d.max()),
            "pass_raw": bool(rho >= 0.98 and abs(ap_o - ap_s) <= 0.01),
            "pass": bool(rho_b >= 0.98 and abs(ap_b - ap_s) <= 0.01)}


def bf16_round(x) -> np.ndarray:
    """Round float32 values to the nearest bfloat16 (ties to even), as float32. The stored
    ReProbe PRM scores come from a softmax in bf16 (2^-8 resolution on [0.5, 1)), so saturated
    PRMs tie many steps; the gate compares like with like on this rounded copy."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return b.astype(np.uint32).view(np.float32).astype(np.float64)


def cmd_merge(a) -> None:
    inputs = parse_inputs(a.inputs)
    for bb, name, path in inputs:
        trs, _ = load_traces(path, a.smoke)
        uids = [u for t in trs for u in t["uids"]]
        for tag in a.tags:
            n = merge_set(a.out / bb / tag, name, a.num_shards, uids)
            print(f"[merge] {bb}/{tag}/{name}: {n} steps", flush=True)
    for spec in a.copy:
        src_bb, name, dst_bb, dst_items = spec.split(":", 3)
        src_items = next(p for bb, n, p in inputs if bb == src_bb and n == name)
        if Path(dst_items).read_bytes() != src_items.read_bytes():
            raise SystemExit(f"[merge] --copy {spec}: items differ from {src_items}; score them instead")
        for tag in a.tags:
            (a.out / dst_bb / tag).mkdir(parents=True, exist_ok=True)
            shutil.copy(a.out / src_bb / tag / f"{name}.jsonl", a.out / dst_bb / tag / f"{name}.jsonl")
        print(f"[merge] copied {src_bb}/{name} -> {dst_bb} (identical items)", flush=True)

    gate, ok = {}, True
    for spec in a.data:
        bb, R = spec.split("=", 1)
        for tag in a.tags:
            for name in TEST:
                sp = a.out / bb / tag / f"{name}.jsonl"
                if not sp.exists():
                    continue
                scores = {r["uid"]: r["p"] for r in jl(sp)}
                g = gate_set(Path(R) / "data" / "test" / f"{name}.traces.jsonl", scores, f"PRM_{TAGS[tag]}")
                if g is None:
                    continue
                gate[f"{bb}/{tag}/{name}"] = g
                ok &= g["pass"]
                print(f"[gate] {bb:9s} {tag:7s} {name:10s} n={g['n']:5d} rho={g['spearman']:.4f} "
                      f"(bf16 {g['spearman_bf16']:.4f}) AP ours {g['ap_ours']:.4f} (bf16 {g['ap_ours_bf16']:.4f}) "
                      f"stored {g['ap_stored']:.4f} | med|d| {g['median_abs_diff']:.4f} "
                      f"<=.008 {g['share_within_0.008']:.3f} max {g['max_abs_diff_p_vs_1_plus_stored']:.3f} "
                      f"{'PASS' if g['pass'] else 'FAIL'}{'' if g['pass_raw'] else ' (raw FAIL)'}", flush=True)
    (a.out / "gate_prm.json").write_text(json.dumps({"all_pass": ok, "smoke": a.smoke, "cells": gate}, indent=1))
    print(f"[gate] {'ALL PASS' if ok else 'FAILED'} ({len(gate)} cells) -> {a.out / 'gate_prm.json'}", flush=True)
    if a.strict and not ok:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--tag", choices=sorted(TAGS), required=True)
    s.add_argument("--model_path", default=None)
    s.add_argument("--inputs", nargs="+", required=True, help="backbone:set:items.jsonl")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num_shards", type=int, default=1)
    s.add_argument("--smoke", type=int, default=0, help="first N traces per set")
    s.add_argument("--max_tokens", type=int, default=65536, help="padded tokens per batch")
    m = sub.add_parser("merge")
    m.add_argument("--inputs", nargs="+", required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--tags", nargs="+", default=sorted(TAGS))
    m.add_argument("--num_shards", type=int, default=1)
    m.add_argument("--smoke", type=int, default=0)
    m.add_argument("--data", nargs="*", default=[], help="backbone=R (data root) for the gate")
    m.add_argument("--copy", nargs="*", default=[], help="src_bb:set:dst_bb:dst_items (identical items)")
    m.add_argument("--strict", action="store_true", help="exit 1 if the gate fails")
    a = p.parse_args()
    cmd_score(a) if a.cmd == "score" else cmd_merge(a)


if __name__ == "__main__":
    main()
