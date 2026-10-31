#!/usr/bin/env python3
"""ReProbe's training data and ProcessBench in ReProbe layout (reprobe_comparison_v1, S0).

Training rows: ReProbe's labeled parquets of one backbone, the first --labels set
defining the rows and every other set joined to it on (question, reply). Qwen3-8B
(v1): ds (DeepSeek-R1) and self, the same 32,484 replies; Qwen3-32B (v2): oss
(GPT-OSS-120B); Phi-4 (v2): qwen3 (Qwen3-8B as judge) and self. Splits come from the frozen
bidirectional_token_probe_v2 manifest by problem_key: `dev` is validation, `train`,
`calib` and `test` are training, and problems outside the manifest (the 752
ProcessBench-overlap problems) are dropped. Every labeled step ReProbe trains on is
kept, as ReProbe keeps it (a final step cut by the 256-token cap included).

Writes to --out:
  rp/{train,val}.traces.jsonl   ReProbe-head traces: input_ids (prompt + reply), context
                                length, steps' reply-relative token ids, labels {set: [...]}
  rt/<set>_{train,val}.items.jsonl  read-token items (read_token_inline.py --layout
                                reprobe); unlabeled steps are left out
  pb/<subset>.traces.jsonl, pb/<subset>.items.jsonl  ProcessBench rewritten as a ReProbe
                                reply: "- Step k: " + the step with newlines flattened,
                                one per line, ended by <|im_end|>; label = first error
  build_audit.json

    python scripts/reprobe_cmp/build_train.py --labels ds:$R2/raw/ds.parquet \\
        self:$R2/raw/self.parquet --v2_meta $R2/manifest_v2/meta --pb_dir $PB --out $R/data
    (v2: --backbone qwen3_32b --labels oss:<parquet>; --backbone phi4 --labels qwen3:<p> self:<p>)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.encode_processbench_token_store import load_traces  # noqa: E402
from src.data.prm_trajectories import problem_key  # noqa: E402
from src.data.reprobe_steps import split_to_steps, step_item  # noqa: E402
from scripts.reprobe_cmp.build_testsets import BACKBONES  # noqa: E402

TRAIN_SPLITS = ("train", "calib", "test")
PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def lab(v) -> int:
    return -1 if v is None or (isinstance(v, float) and math.isnan(v)) else int(v)


def wjl(p: Path, rows) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def char_spans(reply: str, sentences: list[str]) -> list[tuple[int, int]] | None:
    """Each claim sentence's span in the reply, in order; None if one is not found."""
    out, pos = [], 0
    for s in sentences:
        i = reply.find(s, pos)
        if i < 0:
            return None
        out.append((i, i + len(s)))
        pos = i + len(s)
    return out


def adjusted_ids(input_ids: list[int], ctx: int, ids: list[int], special: set[int]) -> list[int]:
    """ReProbe's `_adjust_claim_positions`, reply-relative: claim token ids index the
    reply's non-special tokens."""
    mapping = [i for i, t in enumerate(input_ids[ctx:]) if t not in special]
    return [mapping[i] for i in ids if len(mapping) > i]


def pb_reply(steps: list[str]) -> str:
    return "\n".join(f"- Step {k + 1}: " + re.sub(r"\s*\n\s*", " ", s).strip() for k, s in enumerate(steps))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", nargs="+", required=True, help="name:parquet; the first defines the rows")
    p.add_argument("--v2_meta", type=Path, required=True, help="manifest_v2/meta with {train,dev,calib,test}.jsonl")
    p.add_argument("--pb_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backbone", choices=sorted(BACKBONES), default="qwen3_8b")
    a = p.parse_args()
    import pandas as pd
    from transformers import AutoTokenizer
    tok_name, reprobe_prompt = BACKBONES[a.backbone][:2]
    tok = AutoTokenizer.from_pretrained(tok_name)
    names = [x.split(":", 1)[0] for x in a.labels]
    paths = [Path(x.split(":", 1)[1]) for x in a.labels]
    special = set(tok.all_special_ids)
    audit: dict = {"counts": Counter()}
    c = audit["counts"]

    split_of = {}
    for sp in ("train", "dev", "calib", "test"):
        for l in open(a.v2_meta / f"{sp}.jsonl"):
            if l.strip():
                split_of[json.loads(l)["problem_key"]] = sp
    cols = ["question", "answer", "input_ids", "reply", "claims", "verified"]
    ds = pd.read_parquet(paths[0], columns=cols).to_dict("records")
    others = {}
    for nm, pth in zip(names[1:], paths[1:]):
        rows = pd.read_parquet(pth, columns=["question", "reply", "claims", "verified"]).to_dict("records")
        others[nm] = {}
        for r in rows:
            others[nm].setdefault((r["question"], r["reply"]), []).append(r)
        audit[f"rows_{nm}"] = len(rows)
    audit[f"rows_{names[0]}"] = len(ds)

    rp = {"train": [], "val": []}
    rt = {(lb, sp): [] for lb in names for sp in ("train", "val")}
    all_val = []  # every validation step, labeled or not, for scoring (label = the first set's or -1)
    for i, r in enumerate(ds):
        k = problem_key(r["question"])
        sp = split_of.get(k)
        if sp is None:
            c["drop_not_in_v2_manifest"] += 1
            continue
        side = "val" if sp == "dev" else "train"
        claims = list(r["claims"])
        ys = {names[0]: [lab(v) for v in r["verified"]]}
        missing = False
        for nm, of in others.items():
            twins = of.get((r["question"], r["reply"]), [])
            if not twins:
                missing = True
                break
            s = twins.pop(0)
            ys[nm] = [lab(v) for v in s["verified"]]
            if [x["sentence"] for x in claims] != [x["sentence"] for x in s["claims"]]:
                # the annotation runs split this reply differently: the first set's steps
                # kept, this set's labels dropped for the trace
                c[f"{nm}_labels_dropped_claims_differ"] += 1
                ys[nm] = [-1] * len(claims)
        if missing:
            c["drop_no_twin"] += 1
            continue
        if any(len(y) != len(claims) for y in ys.values()):
            c["drop_label_length"] += 1
            continue
        yd = ys[names[0]]
        ids = [int(t) for t in r["input_ids"]]
        ptoks = tok(reprobe_prompt(r["question"]), add_special_tokens=False)["input_ids"]
        if ids[:len(ptoks)] != ptoks:
            c["drop_prompt_mismatch"] += 1
            continue
        ctx = len(ptoks)
        spans = char_spans(r["reply"], [x["sentence"] for x in claims])
        if spans is None:
            c["drop_sentence_not_in_reply"] += 1
            continue
        tid = f"rp{i:05d}"
        steps = [{"aligned_token_ids": adjusted_ids(ids, ctx, list(x["aligned_token_ids"]), special),
                  "start": a0, "end": b0} for x, (a0, b0) in zip(claims, spans)]
        if any(not st["aligned_token_ids"] for st in steps):
            c["drop_empty_alignment"] += 1
            continue
        try:
            bodies = [step_item(r["reply"], st) for st in steps]
        except ValueError:
            c["drop_step_not_at_line_start"] += 1
            continue
        rp[side].append({"trace_id": tid, "problem_key": k, "split": sp, "input_ids": ids, "context_len": ctx,
                         "steps": [{"aligned_token_ids": st["aligned_token_ids"]} for st in steps],
                         "labels": ys})
        c[f"traces_{side}"] += 1
        for j, (prefix, body) in enumerate(bodies):
            if side == "val":
                all_val.append({"uid": f"{tid}::{j}", "id": tid, "step_idx": j, "problem": r["question"],
                                "prefix": prefix, "candidate_step": body, "label": yd[j]})
            for lb, y in ((nm, ys[nm][j]) for nm in names):
                if y < 0:
                    c[f"unlabeled_{lb}"] += 1
                    continue
                rt[(lb, side)].append({"uid": f"{tid}::{j}", "id": tid, "step_idx": j, "problem": r["question"],
                                       "prefix": prefix, "candidate_step": body, "label": y})
    for side, rows in rp.items():
        wjl(a.out / "rp" / f"{side}.traces.jsonl", rows)
    for (lb, side), rows in rt.items():
        n = wjl(a.out / "rt" / f"{lb}_{side}.items.jsonl", rows)
        audit[f"rt_{lb}_{side}_items"] = n
        audit[f"rt_{lb}_{side}_incorrect_rate"] = sum(x["label"] for x in rows) / max(n, 1)
    audit["rt_all_val_items"] = wjl(a.out / "rt" / "all_val.items.jsonl", all_val)
    audit["problems"] = {side: len({t["problem_key"] for t in rows}) for side, rows in rp.items()}

    eot = tok.convert_tokens_to_ids("<|im_end|>")
    for sub in PB_SUBSETS:
        traces, items = [], []
        for tr in load_traces(a.pb_dir / f"processbench_{sub}.jsonl"):
            reply = pb_reply(tr["steps"])
            prompt = reprobe_prompt(tr["problem"])
            rids = tok(reply, add_special_tokens=False)["input_ids"] + [eot]
            steps = split_to_steps(reply, rids, tok.decode)
            if len(steps) != len(tr["steps"]):
                raise SystemExit(f"{sub}/{tr['id']}: {len(steps)} extracted steps for {len(tr['steps'])}")
            fe = int(tr["label"])
            ptoks = tok(prompt, add_special_tokens=False)["input_ids"]
            traces.append({"trace_id": str(tr["id"]), "set": f"pb_{sub}", "input_ids": ptoks + rids,
                           "context_len": len(ptoks), "first_error": fe,
                           "steps": [{"aligned_token_ids": s["aligned_token_ids"]} for s in steps],
                           "labels": {"pb": [int(k == fe) for k in range(len(steps))]}})
            for k, st in enumerate(steps):
                prefix, body = step_item(reply, st)
                items.append({"uid": f"{tr['id']}::{k}", "id": tr["id"], "step_idx": k, "first_error": fe,
                              "problem": tr["problem"], "prefix": prefix, "candidate_step": body,
                              "label": int(k == fe)})
        wjl(a.out / "pb" / f"{sub}.traces.jsonl", traces)
        audit[f"pb_{sub}_items"] = wjl(a.out / "pb" / f"{sub}.items.jsonl", items)
    audit["counts"] = dict(c)
    (a.out / "build_audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit, indent=1), flush=True)


if __name__ == "__main__":
    main()
