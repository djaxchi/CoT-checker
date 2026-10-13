#!/usr/bin/env python3
"""Freeze the factorized_step_labels_v1 pilot fork set and the reference set.

Plan: docs/factorized_step_labels_v1_plan.md, sections 2 and 3.

Inputs: the DeepSeek-labeled ReProbe parquet (text, token ids, claims) and the
frozen v2 manifest (split per trace, DeepSeek labels in meta/, self labels in
meta_self/), whose split assignment already excludes the 752 ProcessBench-overlap
problems. Every kept row is parsed token-exactly (src/analysis/factorized_step_labels
.parse_reprobe_row) against `reprobe_prompt`.

Writes to --out_dir:
  fork_split_v1.json       the pilot forks (2 validity x 3 positions x per_cell),
                           one per problem, from --fork_split
  reference_set_v1.json    --n_ref traces from --ref_split for the mean baseline
  build_audit.json         drop reasons, candidate counts per cell, checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.factorized_step_labels import (POSITIONS, fork_order_key,  # noqa: E402
                                                 gold_answer, grade_fork,
                                                 parse_reprobe_row, position_bucket,
                                                 select_forks)
from src.onpolicy.prompts import reprobe_prompt  # noqa: E402

VALIDITY = ("correct", "incorrect")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def check_manifest(manifest: Path, rels: list[str]) -> dict:
    sums = json.loads((manifest / "manifest_checksums.json").read_text())
    out = {}
    for rel in rels:
        got = sha256(manifest / rel)
        if sums.get(rel) != got:
            sys.exit(f"[build] checksum mismatch for {rel}: {got} vs {sums.get(rel)}")
        out[rel] = got
    return out


def parse_split(meta: list[dict], self_by: dict, rows, tok, eos_id, counters: Counter):
    """Yield (meta, self_meta, row, parsed) for every token-exact finished trace."""
    def decode(ids):
        return tok.decode(ids)
    for m in meta:
        if not m.get("retained", True) or not m.get("finished"):
            counters["skip_unfinished_or_unretained"] += 1
            continue
        s = self_by.get(m["trace_id"])
        if s is None:
            counters["skip_no_self_labels"] += 1
            continue
        ri = int(m["source_record_ids"][0].removeprefix("row"))
        row = rows[ri]
        pids = tok(reprobe_prompt(row["question"]), add_special_tokens=False)["input_ids"]
        parsed, why = parse_reprobe_row(row["input_ids"], pids, list(row["claims"]), decode, eos_id)
        counters[f"parse_{why}"] += 1
        if parsed is None:
            continue
        n = len(parsed["step_claim_idx"])
        if not n <= len(m["y"]) <= n + 1 or len(m["y"]) != len(s["y"]):
            counters["skip_label_length"] += 1
            continue
        yield m, s, row, parsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds_parquet", type=Path, required=True)
    ap.add_argument("--manifest_v2", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--fork_split", default="dev")
    ap.add_argument("--ref_split", default="calib")
    ap.add_argument("--per_cell", type=int, default=5)
    ap.add_argument("--per_cell_incorrect", type=int, default=None,
                    help="forks per incorrect cell (default --per_cell)")
    ap.add_argument("--per_cell_correct", type=int, default=None,
                    help="forks per correct cell (default --per_cell)")
    ap.add_argument("--correct_from_hard", action="store_true",
                    help="correct-step forks only from problems where some ReProbe trace of the "
                         "problem has a labeled error (either annotator) or a wrong final answer")
    ap.add_argument("--tag", default="v1", help="output names fork_split_<tag>.json, build_audit_<tag>.json")
    ap.add_argument("--skip_ref", action="store_true", help="do not (re)write the reference set")
    ap.add_argument("--count_only", action="store_true")
    ap.add_argument("--scarce_first", action="store_true",
                    help="fill the incorrect cells before the correct ones (full run); the pilot "
                         "filled cells in validity x position order")
    ap.add_argument("--n_ref", type=int, default=300)
    ap.add_argument("--min_steps", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out_fork = a.out_dir / f"fork_split_{a.tag}.json"
    if out_fork.exists() and not a.force:
        sys.exit(f"[build] {out_fork} exists; the split is frozen. Pass --force to rebuild.")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, revision=a.revision,
                                        local_files_only=a.local_files_only)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    rels = [f"{d}/{sp}.jsonl" for d in ("meta", "meta_self") for sp in (a.fork_split, a.ref_split)]
    sums = check_manifest(a.manifest_v2, rels)
    df = pd.read_parquet(a.ds_parquet, columns=["question", "answer", "input_ids", "claims"])
    rows = df.to_dict("records")
    audit = {"created": datetime.now(timezone.utc).isoformat(),
             "ds_parquet": {"path": str(a.ds_parquet), "sha256": sha256(a.ds_parquet)},
             "manifest_v2": str(a.manifest_v2), "manifest_checksums_verified": sums,
             "tokenizer": a.tokenizer, "revision": a.revision, "args": {k: str(v) for k, v in vars(a).items()}}

    def load(sp):
        meta = jl(a.manifest_v2 / "meta" / f"{sp}.jsonl")
        self_by = {s["trace_id"]: s for s in jl(a.manifest_v2 / "meta_self" / f"{sp}.jsonl")}
        assert all(m["split"] == sp for m in meta)
        return meta, self_by

    # --- fork candidates --------------------------------------------------------
    c_fork: Counter = Counter()
    cands = []
    meta, self_by = load(a.fork_split)
    hard: dict[str, bool] = {}
    for m in meta:
        sm = self_by.get(m["trace_id"], {"y": [], "label_mask": []})
        err = any(y == 1 and k for y, k in zip(m["y"], m["label_mask"])) or \
            any(y == 1 and k for y, k in zip(sm["y"], sm["label_mask"]))
        hard[m["problem_key"]] = hard.get(m["problem_key"], False) or err
    for m, s, row, p in parse_split(meta, self_by, rows, tok, eos_id, c_fork):
        gold = gold_answer(row["answer"])
        if gold is None:
            c_fork["skip_no_boxed_gold"] += 1
            continue
        n = len(p["step_spans"])
        if n < a.min_steps:
            c_fork["skip_short"] += 1
            continue
        g = grade_fork([{"text": tok.decode(p["reply_ids"], skip_special_tokens=True),
                         "truncated": False}], gold)[0]
        hard[m["problem_key"]] = hard[m["problem_key"]] or not g["correct"]
        for st in range(n):
            ks = p["step_claim_idx"][:st + 1]
            if not all(m["label_mask"][k] and s["label_mask"][k] for k in ks):
                break                               # an unlabeled step ends the clean prefix
            if any(m["y"][k] or s["y"][k] for k in ks[:-1]):
                break                               # prefix no longer clean in both
            yd, ys = m["y"][ks[-1]], s["y"][ks[-1]]
            if yd != ys:
                c_fork["step_annotator_disagree"] += 1
                continue
            lo, hi = p["step_spans"][st]
            cands.append({
                "fork_id": f"{m['trace_id']}#s{st}", "trace_id": m["trace_id"],
                "problem_key": m["problem_key"], "split": a.fork_split,
                "source_row": m["source_record_ids"][0], "problem": row["question"], "gold": gold,
                "step": st, "step_claim_idx": ks[-1], "n_steps": n,
                "position": position_bucket(st, n), "validity": VALIDITY[int(yd)],
                "y_ds": int(yd), "y_self": int(ys),
                "step_text": tok.decode(p["reply_ids"][lo:hi]),
                "orig_answer_text": p["answer_text"], "orig_pred": g["pred"],
                "orig_correct": g["correct"], "prompt_ids": p_ids(tok, row),
                "reply_ids": p["reply_ids"], "step_spans": p["step_spans"],
                "content_lo": p["content_lo"], "answer_span": list(p["answer_span"])})
            if yd == 1:
                break                               # the first error ends the clean prefix
    for c in cands:
        c["hard_problem"] = bool(hard[c["problem_key"]])
    if a.correct_from_hard:
        c_fork["correct_candidates_dropped_easy"] = sum(c["validity"] == "correct" and not c["hard_problem"]
                                                        for c in cands)
        cands = [c for c in cands if c["validity"] == "incorrect" or c["hard_problem"]]
    cell_counts = Counter((c["validity"], c["position"]) for c in cands)
    if a.count_only:
        print(json.dumps({"counters": dict(sorted(c_fork.items())),
                          "candidates_per_cell": {f"{v}/{pos}": n for (v, pos), n in sorted(cell_counts.items())},
                          "candidate_problems": {v: len({c["problem_key"] for c in cands if c["validity"] == v})
                                                 for v in VALIDITY}}, indent=1))
        return
    quota = {"incorrect": a.per_cell_incorrect or a.per_cell, "correct": a.per_cell_correct or a.per_cell}
    cells = [(v, pos) for v in VALIDITY for pos in POSITIONS]
    if a.scarce_first:
        forks = []
        for v in ("incorrect", "correct"):     # scarce cells first, so they get their problems
            forks += select_forks([c for c in cands if c["problem_key"] not in {f["problem_key"] for f in forks}],
                                  quota[v], [(v, pos) for pos in POSITIONS])
    else:                                      # the pilot's order (v1 split)
        forks = []
        for cell in cells:
            forks += select_forks([c for c in cands if c["problem_key"] not in {f["problem_key"] for f in forks}],
                                  quota[cell[0]], [cell])
    forks.sort(key=lambda f: (cells.index((f["validity"], f["position"])),
                              fork_order_key(f["trace_id"], f["step"])))
    for f in forks:
        f["order_key"] = fork_order_key(f["trace_id"], f["step"])

    # --- reference set ----------------------------------------------------------
    c_ref: Counter = Counter()
    rmeta, rself = load(a.ref_split)
    refs = []
    for m, s, row, p in ([] if a.skip_ref else parse_split(rmeta, rself, rows, tok, eos_id, c_ref)):
        refs.append({"trace_id": m["trace_id"], "problem_key": m["problem_key"],
                     "prompt_ids": p_ids(tok, row), "reply_ids": p["reply_ids"],
                     "step_spans": p["step_spans"], "content_lo": p["content_lo"]})
    refs.sort(key=lambda r: hashlib.sha1(r["trace_id"].encode()).hexdigest())
    refs = refs[:a.n_ref]
    assert not {r["problem_key"] for r in refs} & {f["problem_key"] for f in forks}

    fork_doc = {"experiment": "factorized_step_labels_v1", "kind": "fork_split",
                "fork_split": a.fork_split, "tag": a.tag, "per_cell": a.per_cell,
                "per_cell_incorrect": a.per_cell_incorrect, "per_cell_correct": a.per_cell_correct,
                "correct_from_hard": a.correct_from_hard, "scarce_first": a.scarce_first, "cells": cells,
                "n_forks": len(forks), "forks": forks}
    ref_doc = {"experiment": "factorized_step_labels_v1", "kind": "reference_set",
               "ref_split": a.ref_split, "n_traces": len(refs),
               "n_step_lines": sum(len(r["step_spans"]) for r in refs), "traces": refs}
    out_fork.write_text(json.dumps(fork_doc))
    outputs = {out_fork.name: sha256(out_fork)}
    if not a.skip_ref:
        (a.out_dir / "reference_set_v1.json").write_text(json.dumps(ref_doc))
        outputs["reference_set_v1.json"] = sha256(a.out_dir / "reference_set_v1.json")
    audit.update({"fork_counters": dict(sorted(c_fork.items())),
                  "ref_counters": dict(sorted(c_ref.items())),
                  "candidates_per_cell": {f"{v}/{pos}": cell_counts[(v, pos)] for v, pos in cells},
                  "selected_per_cell": {f"{v}/{pos}": sum((f["validity"], f["position"]) == (v, pos) for f in forks)
                                        for v, pos in cells},
                  "selected_orig_correct": sum(f["orig_correct"] for f in forks),
                  "selected_hard_problem": sum(f["hard_problem"] for f in forks),
                  "outputs": outputs})
    audit_name = "build_audit.json" if a.tag == "v1" else f"build_audit_{a.tag}.json"
    (a.out_dir / audit_name).write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: audit[k] for k in ("candidates_per_cell", "selected_per_cell",
                                            "selected_orig_correct", "outputs")}, indent=1))
    for f in forks:
        print(f"{f['fork_id']:28s} {f['validity']:9s} {f['position']:5s} step {f['step']+1}/{f['n_steps']} "
              f"orig_correct={f['orig_correct']} | {f['step_text'].strip()[:90]}")


def p_ids(tok, row) -> list[int]:
    return tok(reprobe_prompt(row["question"]), add_special_tokens=False)["input_ids"]


if __name__ == "__main__":
    main()
