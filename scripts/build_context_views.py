#!/usr/bin/env python3
"""Build per-context view manifests for context_ablation_v3 (docs/context_ablation_v3_plan.md).

Each target step i of a trace becomes one view, re-encoded by the backbone under a
context level:
  full   "Problem:\\n{p}\\n\\nSolution:\\n" + steps[0..i]
  prev1  same header + steps[i-1], steps[i]   (== q at i = 0)
  q      same header + steps[i]
  none   "Solution:\\n" + steps[i]
Compact storage keeps h header rows, then the pre-step token, then step i's
tokens, so the probe sees the SAME layout and positions under every context:
  single-step view: keep [0, e)            (pre-step token = last header token)
  multi-step view:  keep [0, h-1) + [s-1, e) (pre-step token = the separator)
Each view is a 1-step trace in the store (step_starts=[h], step_ends=[h+L]).

Splits written per context (trace_id of a view = "<orig_trace_id>#<i>"):
  v1_train v1_dev v1_calib   from manifest_v1 train/dev/calib  (labeled steps)
  v2_train v2_dev v2_calib   from manifest_v2 train/dev/calib  (labeled steps)
  rp_test                    manifest_v2 test                  (labeled steps)
  prm_human_test             manifest_v1 test, capped          (labeled steps)
  pb_*                       ProcessBench, every step
Caps are deterministic: traces ordered by sha1(trace_id).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CTX = ("full", "prev1", "q", "none")
PB = ("pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath")
Q_HEADER = "Problem:\n{p}\n\nSolution:\n"
N_HEADER = "Solution:\n"


def jl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def h16(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def ids_sha(ids) -> str:
    return hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def view_steps(ctx: str, steps: list[str], i: int) -> list[str]:
    if ctx == "full":
        return steps[:i + 1]
    if ctx == "prev1":
        return steps[max(0, i - 1):i + 1]
    return [steps[i]]


class Tok:
    """Tokenize each piece once; views are exact concatenations (tokenization is additive)."""

    def __init__(self, tokenizer):
        self.t = tokenizer
        self.sep = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        self.cache: dict = {}

    def header(self, text: str) -> list[int]:
        k = ("H", text)
        if k not in self.cache:
            self.cache[k] = list(self.t(text, add_special_tokens=True)["input_ids"])
        return self.cache[k]

    def step(self, text: str) -> list[int]:
        k = ("S", text)
        if k not in self.cache:
            self.cache[k] = list(self.t(text, add_special_tokens=False)["input_ids"])
        return self.cache[k]

    def view(self, header: str, steps: list[str]):
        ids = list(self.header(header))
        ss, se = [], []
        for j, s in enumerate(steps):
            if j:
                ids += self.sep
            ss.append(len(ids))
            ids += self.step(s)
            se.append(len(ids))
        return ids, ss, se


def compact(ss, se):
    """(keep ranges, stored length, stored step_starts, stored step_ends)."""
    h, s, e = ss[0], ss[-1], se[-1]
    if len(ss) == 1:
        keep = [[0, e]]
    else:
        keep = [[0, h - 1], [s - 1, e]]
    n = sum(b - a for a, b in keep)
    L = e - s
    assert n == h + L
    return keep, n, [h], [h + L]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1_manifest", type=Path, required=True)
    ap.add_argument("--v2_manifest", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--ctx", nargs="+", default=list(CTX))
    ap.add_argument("--cap_train", type=int, default=8000)
    ap.add_argument("--cap_devcal", type=int, default=1500)
    ap.add_argument("--cap_prm_test", type=int, default=3000)
    ap.add_argument("--num_shards", type=int, default=16)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tk = Tok(AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=a.local_files_only))

    # (out split, manifest, src split, cap, all_steps)
    plan = [("v1_train", a.v1_manifest, "train", a.cap_train, False),
            ("v1_dev", a.v1_manifest, "dev", a.cap_devcal, False),
            ("v1_calib", a.v1_manifest, "calib", a.cap_devcal, False),
            ("v2_train", a.v2_manifest, "train", a.cap_train, False),
            ("v2_dev", a.v2_manifest, "dev", a.cap_devcal, False),
            ("v2_calib", a.v2_manifest, "calib", a.cap_devcal, False),
            ("rp_test", a.v2_manifest, "test", 0, False),
            ("prm_human_test", a.v1_manifest, "test", a.cap_prm_test, False)] + \
           [(sp, a.v1_manifest, sp, 0, True) for sp in PB]

    sources = {}
    audit = {"caps": {"train": a.cap_train, "devcal": a.cap_devcal, "prm_test": a.cap_prm_test}, "splits": {}}
    for out_sp, man, sp, cap, all_steps in plan:
        ins = {r["trace_id"]: r for r in jl(man / "inputs" / f"{sp}.jsonl")}
        metas = [m for m in jl(man / "meta" / f"{sp}.jsonl") if m["trace_id"] in ins]
        metas.sort(key=lambda m: h16(m["trace_id"]))
        if cap:
            metas = metas[:cap]
        sources[out_sp] = (ins, metas, all_steps)
        audit["splits"][out_sp] = {"source": str(man), "src_split": sp, "traces": len(metas)}

    for ctx in a.ctx:
        out = a.out_root / ctx
        (out / "inputs").mkdir(parents=True, exist_ok=True)
        (out / "meta").mkdir(parents=True, exist_ok=True)
        enc = []
        n_views = {}
        for out_sp, (ins, metas, all_steps) in sources.items():
            fi = open(out / "inputs" / f"{out_sp}.jsonl", "w", encoding="utf-8")
            fm = open(out / "meta" / f"{out_sp}.jsonl", "w", encoding="utf-8")
            n = 0
            for m in metas:
                tr = ins[m["trace_id"]]
                steps = tr["steps"]
                header = N_HEADER if ctx == "none" else Q_HEADER.format(p=tr["problem"])
                for i in range(len(steps)):
                    if not all_steps and not m["label_mask"][i]:
                        continue
                    vs = view_steps(ctx, steps, i)
                    ids, ss, se = tk.view(header, vs)
                    keep, nst, st_s, st_e = compact(ss, se)
                    vid = f"{m['trace_id']}#{i}"
                    fi.write(json.dumps({"trace_id": vid, "header": header, "steps": vs}, ensure_ascii=False) + "\n")
                    fm.write(json.dumps({"trace_id": vid, "split": out_sp, "orig_trace_id": m["trace_id"],
                                         "orig_step": i, "orig_n_steps": len(steps),
                                         "problem_key": m["problem_key"], "y": [int(m["y"][i])],
                                         "label_mask": [bool(m["label_mask"][i])]}) + "\n")
                    enc.append({"trace_id": vid, "split": out_sp, "n_tokens": nst, "n_forward": len(ids),
                                "n_steps": 1, "step_starts": st_s, "step_ends": st_e,
                                "fwd_step_starts": ss, "fwd_step_ends": se, "keep": keep,
                                "prefix_len": st_s[0], "ids_sha1": ids_sha(ids)})
                    n += 1
            fi.close(); fm.close()
            n_views[out_sp] = n
        order = sorted(range(len(enc)), key=lambda i: (-enc[i]["n_forward"], enc[i]["trace_id"]))
        load = [0] * a.num_shards
        for i in order:
            s = int(np.argmin(load)); enc[i]["shard"] = s; load[s] += enc[i]["n_forward"]
        enc.sort(key=lambda r: (r["shard"], r["split"], r["trace_id"]))
        with open(out / "encode_manifest.jsonl", "w") as f:
            for r in enc:
                f.write(json.dumps(r) + "\n")
        stored = sum(r["n_tokens"] for r in enc)
        fwd = sum(r["n_forward"] for r in enc)
        audit[ctx] = {"views": n_views, "total_views": len(enc), "stored_tokens": stored,
                      "forward_tokens": fwd, "store_gib": stored * 8192 / 2**30}
        print(ctx, audit[ctx], flush=True)
    (a.out_root / "views_audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
