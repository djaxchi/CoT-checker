#!/usr/bin/env python3
"""Freeze a step_value_labels_v1 trace set (docs/step_value_labels_v1_plan.md, section 3).

Two kinds of set, one JSON format (one record per trace, with the cut points to
sample):

  --from_fork_splits   the validation set: every factorized_step_labels_v1 fork
                       becomes one record with cuts [s, s + 1], so the labeler's
                       err(s) and err(s + 1) can be checked against the fork's
                       err(C) and err(A). Needs no parquet.
  --ds_parquet ...     the label set: whole ReProbe v2 traces (all cuts 0..n),
                       parsed token-exactly as in build_fork_split.py, with every
                       problem of --exclude_fork_splits removed and per-step
                       DeepSeek and self validity labels attached. With
                       --include_unfinished, traces that hit ReProbe's 256-token
                       cap join too, cut after their last complete step line
                       (src/analysis/step_value_labels.parse_unfinished_row). Traces fall in
                       three groups, filled in sha1(trace_id) order:
                         error       >= 1 step labeled incorrect by DeepSeek
                         hard_clean  no DeepSeek error, on a problem where some
                                     trace has a labeled error (either annotator)
                                     or a wrong final answer
                         easy        the rest

Writes <out_dir>/trace_set_<tag>.json and build_audit_<tag>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "factorized_step_labels"))

from src.analysis.factorized_step_labels import gold_answer, grade_fork  # noqa: E402
from src.analysis.step_value_labels import cut_offsets, parse_unfinished_row  # noqa: E402

GROUPS = ("error", "hard_clean", "easy")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def forks_of(paths: list[Path]) -> list[dict]:
    return [f for p in paths for f in json.loads(p.read_text())["forks"]]


def validation_records(forks: list[dict]) -> list[dict]:
    """One record per fork, cuts [s, s + 1] (the C and A contexts of track 1)."""
    out = []
    for f in forks:
        offs = cut_offsets(f["step_spans"], f["answer_span"][0])
        s = f["step"]
        out.append({"trace_id": f["fork_id"], "orig_trace_id": f["trace_id"], "problem_key": f["problem_key"],
                    "group": f"fork_{f['validity']}", "gold": f["gold"], "prompt_ids": f["prompt_ids"],
                    "reply_ids": f["reply_ids"], "n_steps": f["n_steps"], "cuts": [s, s + 1],
                    "cut_offsets": [offs[s], offs[s + 1]], "fork_step": s})
    return out


def select(records: list[dict], quota: dict[str, int | None], max_per_problem: int) -> list[dict]:
    """Fill each group in sha1(trace_id) order, at most `max_per_problem` traces per problem
    across all groups (the scarce error group is filled first). None = take all."""
    per_pk: Counter = Counter()
    out = []
    for g in GROUPS:
        q = quota.get(g)
        got = 0
        for r in sorted((r for r in records if r["group"] == g), key=lambda r: sha1(r["trace_id"])):
            if q is not None and got >= q:
                break
            if per_pk[r["problem_key"]] >= max_per_problem:
                continue
            out.append(r); per_pk[r["problem_key"]] += 1; got += 1
    return out


def label_records(a, audit: dict) -> list[dict]:
    import pandas as pd
    from transformers import AutoTokenizer
    from build_fork_split import check_manifest, p_ids, parse_split

    tok = AutoTokenizer.from_pretrained(a.tokenizer, revision=a.revision, local_files_only=a.local_files_only)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    audit["manifest_checksums_verified"] = check_manifest(
        a.manifest_v2, [f"{d}/{a.split}.jsonl" for d in ("meta", "meta_self")])
    audit["ds_parquet"] = {"path": str(a.ds_parquet), "sha256": sha256(a.ds_parquet)}
    rows = pd.read_parquet(a.ds_parquet, columns=["question", "answer", "input_ids", "claims"]).to_dict("records")
    meta = jl(a.manifest_v2 / "meta" / f"{a.split}.jsonl")
    self_by = {s["trace_id"]: s for s in jl(a.manifest_v2 / "meta_self" / f"{a.split}.jsonl")}
    excl = {f["problem_key"] for f in forks_of(a.exclude_fork_splits)}
    c: Counter = Counter()
    hard: dict[str, bool] = defaultdict(bool)
    for m in meta:
        sm = self_by.get(m["trace_id"], {"y": [], "label_mask": []})
        hard[m["problem_key"]] |= any(y == 1 and k for y, k in zip(m["y"], m["label_mask"])) or \
            any(y == 1 and k for y, k in zip(sm["y"], sm["label_mask"]))
    def unfinished():
        for m in meta:
            if m.get("finished") or not m.get("retained", True):
                continue
            s = self_by.get(m["trace_id"])
            if s is None:
                c["unfinished_skip_no_self_labels"] += 1
                continue
            row = rows[int(m["source_record_ids"][0].removeprefix("row"))]
            p, why = parse_unfinished_row(row["input_ids"], p_ids(tok, row), list(row["claims"]), tok.decode, eos_id)
            c[f"unfinished_parse_{why}"] += 1
            if p is None:
                continue
            if len(p["step_claim_idx"]) > len(m["y"]) or len(m["y"]) != len(s["y"]):
                c["unfinished_skip_label_length"] += 1
                continue
            p["answer_span"] = (p["end"], p["end"])
            yield m, s, row, p

    import itertools
    recs = []
    src = parse_split(meta, self_by, rows, tok, eos_id, c)
    if a.include_unfinished:
        src = itertools.chain(src, unfinished())
    for m, s, row, p in src:
        if m["problem_key"] in excl:
            c["skip_eval_fork_problem"] += 1
            continue
        gold = gold_answer(row["answer"])
        if gold is None:
            c["skip_no_boxed_gold"] += 1
            continue
        n = len(p["step_spans"])
        if n < a.min_steps:
            c["skip_short"] += 1
            continue
        finished = "end" not in p
        g = grade_fork([{"text": tok.decode(p["reply_ids"], skip_special_tokens=True), "truncated": not finished}],
                       gold)[0]
        if finished:                                # a capped trace has no answer to be wrong
            hard[m["problem_key"]] |= not g["correct"]
        ks = p["step_claim_idx"]
        y_ds = [int(m["y"][k]) for k in ks]
        mask_ds = [bool(m["label_mask"][k]) for k in ks]
        y_self = [int(s["y"][k]) for k in ks]
        mask_self = [bool(s["label_mask"][k]) for k in ks]
        first = next((i for i in range(n) if mask_ds[i] and y_ds[i]), None)
        recs.append({"trace_id": m["trace_id"], "problem_key": m["problem_key"], "gold": gold,
                     "prompt_ids": p_ids(tok, row), "reply_ids": p["reply_ids"], "n_steps": n,
                     "step_spans": p["step_spans"], "step_claim_idx": ks,
                     "cuts": list(range(n + 1)), "cut_offsets": cut_offsets(p["step_spans"], p["answer_span"][0]),
                     "y_ds": y_ds, "mask_ds": mask_ds, "y_self": y_self, "mask_self": mask_self,
                     "first_error_ds": first, "finished": finished,
                     "orig_pred": g["pred"], "orig_correct": g["correct"] if finished else None})
    for r in recs:
        r["hard_problem"] = bool(hard[r["problem_key"]])
        r["group"] = "error" if r["first_error_ds"] is not None else ("hard_clean" if r["hard_problem"] else "easy")
    audit["counters"] = dict(sorted(c.items()))
    audit["n_excluded_problems"] = len(excl)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--from_fork_splits", type=Path, nargs="+", default=None)
    ap.add_argument("--ds_parquet", type=Path)
    ap.add_argument("--manifest_v2", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--exclude_fork_splits", type=Path, nargs="+", default=[])
    ap.add_argument("--n_error", type=int, default=None, help="default: all")
    ap.add_argument("--n_hard_clean", type=int, default=None, help="default: as many as error traces")
    ap.add_argument("--n_easy", type=int, default=None, help="default: 10%% of error + hard_clean")
    ap.add_argument("--max_per_problem", type=int, default=2)
    ap.add_argument("--include_unfinished", action="store_true",
                    help="also traces that hit ReProbe's 256-token cap, cut after their last complete line")
    ap.add_argument("--min_steps", type=int, default=2)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--count_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = a.out_dir / f"trace_set_{a.tag}.json"
    if out.exists() and not a.force and not a.count_only:
        sys.exit(f"[build] {out} exists; the set is frozen. Pass --force to rebuild.")
    audit = {"created": datetime.now(timezone.utc).isoformat(), "args": {k: str(v) for k, v in vars(a).items()}}
    if a.from_fork_splits:
        audit["fork_splits"] = {str(p): sha256(p) for p in a.from_fork_splits}
        recs = validation_records(forks_of(a.from_fork_splits))
    else:
        if not (a.ds_parquet and a.manifest_v2):
            sys.exit("[build] the label set needs --ds_parquet and --manifest_v2")
        audit["exclude_fork_splits"] = {str(p): sha256(p) for p in a.exclude_fork_splits}
        pool = label_records(a, audit)
        audit["pool"] = {g: {"traces": sum(r["group"] == g for r in pool),
                             "steps": sum(r["n_steps"] for r in pool if r["group"] == g),
                             "incorrect_steps_ds": sum(sum(y and k for y, k in zip(r["y_ds"], r["mask_ds"]))
                                                       for r in pool if r["group"] == g),
                             "problems": len({r["problem_key"] for r in pool if r["group"] == g}),
                             "unfinished_traces": sum(not r["finished"] for r in pool if r["group"] == g)}
                         for g in GROUPS}
        n_err = sum(r["group"] == "error" for r in pool) if a.n_error is None else a.n_error
        n_hc = n_err if a.n_hard_clean is None else a.n_hard_clean
        n_easy = (n_err + n_hc) // 10 if a.n_easy is None else a.n_easy
        recs = select(pool, {"error": n_err, "hard_clean": n_hc, "easy": n_easy}, a.max_per_problem)
    audit["selected"] = {g: sum(r["group"] == g for r in recs) for g in sorted({r["group"] for r in recs})}
    audit["n_traces"] = len(recs)
    audit["n_cuts"] = sum(len(r["cuts"]) for r in recs)
    audit["n_problems"] = len({r["problem_key"] for r in recs})
    if not a.from_fork_splits:
        audit["selected_incorrect_steps_ds"] = sum(sum(y and k for y, k in zip(r["y_ds"], r["mask_ds"])) for r in recs)
    print(json.dumps({k: v for k, v in audit.items() if k not in ("args",)}, indent=1))
    if a.count_only:
        return
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiment": "step_value_labels_v1", "kind": "trace_set", "tag": a.tag,
                               "n_traces": len(recs), "traces": recs}))
    audit["outputs"] = {out.name: sha256(out)}
    (a.out_dir / f"build_audit_{a.tag}.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
