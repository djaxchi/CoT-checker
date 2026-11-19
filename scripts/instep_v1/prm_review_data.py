#!/usr/bin/env python3
"""Inputs for the reviewer runs on PRM800K labels (InStep paper, reviewer Q1 and W2).

  draws   the 100,000-item training draw of InStep (PRM800K) for each seed, exactly as
          read_token_inline.py train draws it (encode_pooled_multilayer.prm800k_items with
          --n_train 100000 --seed s): <out>/draw_seed<s>.uids.json; the union of the draws as one
          items file (<out>/union.items.jsonl, rows of the full file) for a single encode; the
          problems of every ProcessBench subset (<out>/pb_problems.json, for --exclude_problems);
          and <out>/draws.json with how many drawn items belong to ProcessBench problems.
  traces  ReProbe trace files (scripts/reprobe_cmp/reprobe_probe.py format) from PRM800K items:
          the items are grouped into trajectories as InStep reads them (read_token_inline
          group_items, chat layout, steps separated by a blank line); a trace is the chat prompt
          plus the longest member's text, its steps are the member items (aligned_token_ids =
          the reply-relative tokens overlapping the step's characters), labels the items' labels.
          --uids restricts the items to one draw.

    python scripts/instep_v1/prm_review_data.py draws --full <prm800k_probe_train_full.jsonl> \\
        --pb <R>/data/pb --out <dir>
    python scripts/instep_v1/prm_review_data.py traces --items <full or val jsonl> \\
        --uids <dir>/draw_seed0.uids.json --out <dir>/rp_train_seed0.traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PB = ("gsm8k", "math", "olympiadbench", "omnimath")


def draw_indices(n_rows: int, limit: int, seed: int) -> np.ndarray:
    """The row indices prm800k_items keeps (same generator, same call)."""
    if not limit or limit >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, limit, replace=False))


def items_of(rows: list[dict]) -> list[dict]:
    return [{"uid": r["uid"], "problem": r["problem"], "prefix": r["prefix"], "step": r["candidate_step"],
             "label": int(r["label"])} for r in rows]


def build_traces(tok, items: list[dict], prefix: str = "prm") -> tuple[list[dict], int]:
    """ReProbe traces (input_ids, context_len, steps[].aligned_token_ids, labels) and the
    number of items whose step has no token (dropped)."""
    from scripts.read_token_inline import SEP, boundary_at_or_before, group_items, item_text
    from src.onpolicy.prompts import chat_prompt
    out, empty = [], 0
    for g in group_items(items, SEP):
        prompt = chat_prompt(g["problem"])
        enc = tok(prompt + g["text"], add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        ends = {e: j + 1 for j, (_, e) in enumerate(offs)}
        ctx = boundary_at_or_before(ends, len(prompt))
        steps, labels, uids = [], [], []
        for i in sorted(g["members"], key=lambda i: len(item_text(items[i], SEP))):
            it = items[i]
            lo = len(prompt) + (len(it["prefix"]) + len(SEP) if it["prefix"] else 0)
            hi = len(prompt) + len(item_text(it, SEP))
            ids = [j - ctx for j, (s, e) in enumerate(offs) if s < hi and e > lo and j >= ctx]
            if not ids:
                empty += 1
                continue
            steps.append({"aligned_token_ids": ids})
            labels.append(int(it["label"]))
            uids.append(it["uid"])
        if steps:
            out.append({"trace_id": f"{prefix}{len(out):06d}", "input_ids": list(enc["input_ids"]),
                        "context_len": ctx, "steps": steps, "labels": labels, "uids": uids})
    return out, empty


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["draws", "traces"])
    p.add_argument("--full", type=Path, help="draws: prm800k_probe_train_full.jsonl")
    p.add_argument("--pb", type=Path, help="draws: <R>/data/pb (the <subset>.items.jsonl files)")
    p.add_argument("--n_train", type=int, default=100000)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--items", type=Path, help="traces: PRM800K items file")
    p.add_argument("--uids", type=Path, help="traces: keep only these uids (a draw)")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.phase == "draws":
        from src.data.prm_trajectories import normalize_problem
        a.out.mkdir(parents=True, exist_ok=True)
        lines = [l for l in open(a.full) if l.strip()]
        pb_probs = sorted({json.loads(l)["problem"] for s in PB for l in open(a.pb / f"{s}.items.jsonl") if l.strip()})
        (a.out / "pb_problems.json").write_text(json.dumps(pb_probs))
        pbn = {normalize_problem(q) for q in pb_probs}
        union, info = set(), {"full_rows": len(lines), "n_train": a.n_train, "pb_problems": len(pbn), "seeds": {}}
        for s in [int(x) for x in a.seeds.split(",")]:
            idx = draw_indices(len(lines), a.n_train, s)
            rows = [json.loads(lines[i]) for i in idx]
            (a.out / f"draw_seed{s}.uids.json").write_text(json.dumps([r["uid"] for r in rows]))
            hit = sum(normalize_problem(r["problem"]) in pbn for r in rows)
            info["seeds"][s] = {"items": len(rows), "pb_problem_items": hit,
                                "pb_problems_hit": len({normalize_problem(r["problem"]) for r in rows} & pbn)}
            union.update(int(i) for i in idx)
            print(f"[draws] seed {s}: {len(rows)} items, {hit} of ProcessBench problems", flush=True)
        with open(a.out / "union.items.jsonl", "w") as f:
            for i in sorted(union):
                f.write(lines[i] if lines[i].endswith("\n") else lines[i] + "\n")
        info["union_items"] = len(union)
        (a.out / "draws.json").write_text(json.dumps(info, indent=1))
        print(f"[draws] union {len(union)} items -> {a.out}", flush=True)
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    rows = [json.loads(l) for l in open(a.items) if l.strip()]
    if a.uids is not None:
        keep = set(json.loads(a.uids.read_text()))
        rows = [r for r in rows if r["uid"] in keep]
        if len(rows) != len(keep):
            raise SystemExit(f"{a.items}: {len(rows)} of the {len(keep)} uids found")
    if a.smoke:
        rows = rows[:a.smoke]
    traces, empty = build_traces(tok, items_of(rows))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    n = sum(len(t["steps"]) for t in traces)
    print(f"[traces] {len(rows)} items -> {len(traces)} traces, {n} labelled steps ({empty} without tokens) "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
