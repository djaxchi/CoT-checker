#!/usr/bin/env python3
"""Freeze the risk_probe_v0 view manifest (docs/risk_probe_v0_plan.md, sections 2-3).

Four splits in the context_ablation_v3 `full` view format, so the store, the
probe and its scoring code are exactly §21.18's:

  v2_train_xfit  context_ablation_v3 v2_train views minus every fork problem
  v2_dev_xfit    v2_dev views minus every fork problem (the pilot forks are dev)
  fsl_forks      one view per factorized-label fork (steps 0..i of its trace),
                 built with build_context_views' own Tok/compact code
  gate_rp_test   the first --n_gate rp_test views by sha1(view id), for the
                 reproduction gate against the saved §21.18 predictions

Existing views are copied row for row (inputs, meta, encode_manifest); only the
split name and the shard assignment change. Writes <out_dir>/{inputs,meta}/,
encode_manifest.jsonl and build_audit.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_context_views import Q_HEADER, Tok, compact, ids_sha, view_steps  # noqa: E402
from src.data.reprobe_trajectories import strip_step  # noqa: E402


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wjl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fork_view(tk: Tok, fork: dict, trace: dict) -> tuple[dict, dict, dict]:
    """(inputs row, meta row, encode row) of a fork's `full` view."""
    i = fork["step_claim_idx"]
    steps = trace["steps"]
    want = strip_step(fork["step_text"])
    if steps[i] != want:
        raise ValueError(f"{fork['fork_id']}: manifest step {i} {steps[i]!r} != fork step {want!r}")
    header = Q_HEADER.format(p=trace["problem"])
    vs = view_steps("full", steps, i)
    ids, ss, se = tk.view(header, vs)
    keep, nst, st_s, st_e = compact(ss, se)
    vid = f"{fork['trace_id']}#{i}"
    inp = {"trace_id": vid, "header": header, "steps": vs}
    meta = {"trace_id": vid, "split": "fsl_forks", "orig_trace_id": fork["trace_id"], "orig_step": i,
            "orig_n_steps": len(steps), "problem_key": fork["problem_key"], "fork_id": fork["fork_id"],
            "validity": fork["validity"], "y": [int(fork["y_ds"])], "label_mask": [True]}
    enc = {"trace_id": vid, "split": "fsl_forks", "n_tokens": nst, "n_forward": len(ids), "n_steps": 1,
           "step_starts": st_s, "step_ends": st_e, "fwd_step_starts": ss, "fwd_step_ends": se,
           "keep": keep, "prefix_len": st_s[0], "ids_sha1": ids_sha(ids)}
    return inp, meta, enc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views_full", type=Path, required=True, help="context_ablation_v3 views/full")
    ap.add_argument("--v2_manifest", type=Path, required=True)
    ap.add_argument("--fork_splits", type=Path, nargs="+", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--n_gate", type=int, default=600)
    ap.add_argument("--num_shards", type=int, default=16)
    a = ap.parse_args()
    if (a.out_dir / "encode_manifest.jsonl").exists():
        sys.exit(f"[build] {a.out_dir} is frozen")

    forks = [f for p in a.fork_splits for f in json.loads(p.read_text())["forks"]]
    fork_pk = {f["problem_key"] for f in forks}
    assert len({f["fork_id"] for f in forks}) == len(forks)
    enc_by = {r["trace_id"]: r for r in jl(a.views_full / "encode_manifest.jsonl")}
    audit = {"views_full": str(a.views_full), "fork_splits": {str(p): sha256(p) for p in a.fork_splits},
             "n_forks": len(forks), "n_fork_problems": len(fork_pk), "splits": {}}

    out_in, out_meta, out_enc = {}, {}, []
    plan = [("v2_train_xfit", "v2_train", True), ("v2_dev_xfit", "v2_dev", True), ("gate_rp_test", "rp_test", False)]
    for new, old, drop in plan:
        ins = {r["trace_id"]: r for r in jl(a.views_full / "inputs" / f"{old}.jsonl")}
        metas = jl(a.views_full / "meta" / f"{old}.jsonl")
        c = Counter(src_views=len(metas))
        keep = []
        for m in metas:
            if drop and m["problem_key"] in fork_pk:
                c["dropped_fork_problem_views"] += 1
                continue
            keep.append(m)
        if not drop:
            keep = sorted(keep, key=lambda m: hashlib.sha1(m["trace_id"].encode()).hexdigest())[:a.n_gate]
        c["views"] = len(keep)
        c["traces"] = len({m.get("orig_trace_id") for m in keep})
        out_in[new] = [ins[m["trace_id"]] for m in keep]
        out_meta[new] = [{**m, "split": new} for m in keep]
        out_enc += [{**enc_by[m["trace_id"]], "split": new} for m in keep]
        audit["splits"][new] = dict(c)

    from transformers import AutoTokenizer
    tk = Tok(AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=a.local_files_only))
    traces = {}
    for sp in ("train", "dev"):
        for r in jl(a.v2_manifest / "inputs" / f"{sp}.jsonl"):
            traces[r["trace_id"]] = r
    out_in["fsl_forks"], out_meta["fsl_forks"] = [], []
    for f in forks:
        inp, meta, enc = fork_view(tk, f, traces[f["trace_id"]])
        out_in["fsl_forks"].append(inp); out_meta["fsl_forks"].append(meta); out_enc.append(enc)
    audit["splits"]["fsl_forks"] = {"views": len(forks)}

    ids = [r["trace_id"] for r in out_enc]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        sys.exit(f"[build] duplicate view ids across splits: {dup[:5]}")
    order = sorted(range(len(out_enc)), key=lambda i: (-out_enc[i]["n_forward"], out_enc[i]["trace_id"]))
    load = [0] * a.num_shards
    for i in order:
        s = int(np.argmin(load)); out_enc[i]["shard"] = s; load[s] += out_enc[i]["n_forward"]
    out_enc.sort(key=lambda r: (r["shard"], r["split"], r["trace_id"]))
    for sp in out_in:
        wjl(a.out_dir / "inputs" / f"{sp}.jsonl", out_in[sp])
        wjl(a.out_dir / "meta" / f"{sp}.jsonl", out_meta[sp])
    wjl(a.out_dir / "encode_manifest.jsonl", out_enc)
    audit["forward_tokens"] = int(sum(r["n_forward"] for r in out_enc))
    audit["stored_tokens"] = int(sum(r["n_tokens"] for r in out_enc))
    audit["store_gib"] = audit["stored_tokens"] * 8192 / 2**30
    audit["num_shards"] = a.num_shards
    audit["outputs"] = {str(p.relative_to(a.out_dir)): sha256(p) for p in sorted(a.out_dir.rglob("*.jsonl"))}
    (a.out_dir / "build_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: audit[k] for k in ("n_forks", "n_fork_problems", "splits", "forward_tokens",
                                            "store_gib")}, indent=1))


if __name__ == "__main__":
    main()
