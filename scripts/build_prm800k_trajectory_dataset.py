#!/usr/bin/env python3
"""Build the frozen trajectory manifests for bidirectional_token_probe_v1.

Reads raw PRM800K (phase2_train + phase2_test; phase 1 is counted only) and the
four ProcessBench subsets, aligns labels to the ORIGINAL generated trajectories
(src/data/prm_trajectories.py), deduplicates, removes ProcessBench-overlapping
problems, assigns problem-disjoint splits, tokenizes every trace in the exact
encoder format to audit lengths, applies the frozen context cap, and writes:

  out_dir/inputs/<split>.jsonl     model inputs only: trace_id, problem, steps
  out_dir/meta/<split>.jsonl       labels, masks, ratings, alignment status, ids
  out_dir/encode_manifest.jsonl    one row per retained trace: split, n_tokens,
                                   step spans, shard assignment
  out_dir/data_audit.json          counts, exclusions, overlap, length stats
  out_dir/manifest_checksums.json  sha256 of every file above

The old train/val/test membership is NOT reused: the old builders keyed
problems by a per-record pseudo id (``p{sample_idx}_{hash}``), so one problem
text appears under many ids. data_audit.json records the evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.prm_trajectories import (  # noqa: E402
    align_record, assign_splits, merge_duplicates, normalize_problem,
    problem_key, processbench_trace,
)

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
PRM_SPLITS = ("train", "dev", "calib", "test")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_pb(pb_dir: Path, subset: str) -> list[dict]:
    p = pb_dir / f"{subset}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return read_jsonl(pb_dir / f"processbench_{subset}.jsonl")


def length_stats(xs) -> dict:
    if len(xs) == 0:
        return {"n": 0}
    a = np.asarray(xs)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": int(a.max())}


def trace_counts(traces: list[dict]) -> dict:
    c = Counter()
    for t in traces:
        T = len(t["steps"])
        c["traces"] += 1
        c["steps"] += T
        c["traces_with_error"] += t["first_error"] is not None
        last_labeled = max((k for k in range(T) if t["label_mask"][k]), default=-1)
        for k in range(T):
            if t["label_mask"][k]:
                c["labeled_incorrect" if t["y"][k] == 1 else "labeled_correct"] += 1
                if t.get("ratings") and t["ratings"][k] == 0:
                    c["labeled_neutral_rating0"] += 1
                if t.get("ratings") and t["ratings"][k] == 1:
                    c["labeled_positive_rating1"] += 1
                if k < T - 1:
                    c["labeled_targets_with_future"] += 1
                if k < T - 2:
                    c["labeled_targets_with_2plus_future"] += 1
            else:
                c["unlabeled_steps"] += 1
        c["unlabeled_after_last_label"] += T - 1 - last_labeled
    c["problems"] = len({t["problem_key"] for t in traces})
    return dict(c)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prm_raw_dir", type=Path, required=True)
    ap.add_argument("--pb_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--num_shards", type=int, default=32)
    ap.add_argument("--old_split_dir", type=Path, default=None,
                    help="dir with the old prm800k_{probe_train_full,val_5k,test_2k}.jsonl "
                         "to audit recoverability of the old membership")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from encode_processbench_full_store import tokenize_solution
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)

    audit: dict = {"seed": args.seed, "max_tokens": args.max_tokens, "sources": {}}
    counters: Counter = Counter()

    # --- raw PRM800K ------------------------------------------------------
    aligned = []
    records_per_problem = Counter()
    for fname in ("phase2_train.jsonl", "phase2_test.jsonl"):
        p = args.prm_raw_dir / fname
        audit["sources"][fname] = {"sha256": sha256_file(p)}
        n = 0
        for i, line in enumerate(open(p, encoding="utf-8")):
            rec = json.loads(line)
            n += 1
            counters["raw_records"] += 1
            counters["raw_quality_control"] += bool(rec.get("is_quality_control_question"))
            counters["raw_initial_screening"] += bool(rec.get("is_initial_screening_question"))
            counters[f"raw_finish_{rec.get('label', {}).get('finish_reason')}"] += 1
            q = rec.get("question") or {}
            if isinstance(q.get("problem"), str):
                records_per_problem[problem_key(q["problem"])] += 1
            a = align_record(rec, f"{fname}:{i}", counters)
            if a is not None:
                aligned.append(a)
        audit["sources"][fname]["records"] = n
    for fname in ("phase1_train.jsonl", "phase1_test.jsonl"):
        p = args.prm_raw_dir / fname
        if p.exists():
            audit["sources"][fname] = {"sha256": sha256_file(p),
                                       "records": sum(1 for _ in open(p)),
                                       "use": "counted only: no pre_generated_steps continuation"}
    counters["aligned_records"] = len(aligned)
    traces = merge_duplicates(aligned, counters)
    counters["unique_trajectories"] = len(traces)
    audit["records_per_problem"] = length_stats(list(records_per_problem.values()))
    audit["n_unique_problems_raw"] = len(records_per_problem)

    # --- ProcessBench + overlap ------------------------------------------
    pb_traces: dict[str, list[dict]] = {}
    pb_norm: dict[str, set[str]] = {}
    pb_exact: dict[str, set[str]] = {}
    for s in PB_SUBSETS:
        rows = load_pb(args.pb_dir, s)
        pb_traces[s] = [processbench_trace(r, s) for r in rows]
        pb_norm[s] = {normalize_problem(r["problem"]) for r in rows}
        pb_exact[s] = {r["problem"] for r in rows}
    prm_by_pk = {t["problem_key"]: t["problem"] for t in traces}
    overlap = {}
    excl: set[str] = set()
    for s in PB_SUBSETS:
        ex = {pk for pk, txt in prm_by_pk.items() if txt in pb_exact[s]}
        nm = {pk for pk, txt in prm_by_pk.items() if normalize_problem(txt) in pb_norm[s]}
        overlap[s] = {"exact_problems": len(ex), "normalized_problems": len(nm),
                      "pb_traces_on_overlapping_problems": sum(
                          1 for t in pb_traces[s] if t["problem_key"] in nm)}
        excl |= nm
    audit["pb_overlap"] = overlap
    audit["pb_overlap_excluded_problems"] = len(excl)
    audit["pb_overlap_excluded_traces"] = sum(1 for t in traces if t["problem_key"] in excl)
    audit["pb_overlap_note"] = ("exact and normalized (NFKC, lowercase, whitespace) problem-text "
                                "match only; does not establish absence of pretraining contamination")

    # --- old split recoverability ----------------------------------------
    if args.old_split_dir is not None:
        old = {}
        for stem, name in (("prm800k_probe_train_full.jsonl", "train"),
                           ("prm800k_val_5k.jsonl", "val"), ("prm800k_test_2k.jsonl", "test")):
            p = args.old_split_dir / stem
            if not p.exists():
                continue
            pks, pids = set(), set()
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    pks.add(problem_key(r["problem"]))
                    pids.add(r["problem_id"])
            old[name] = (pks, pids)
        if old:
            rec = {n: {"pseudo_problem_ids": len(v[1]), "canonical_problems": len(v[0])}
                   for n, v in old.items()}
            if "train" in old:
                for n in ("val", "test"):
                    if n in old:
                        rec[f"{n}_canonical_problems_also_in_train"] = len(old[n][0] & old["train"][0])
            audit["old_split_recoverability"] = rec

    # --- splits ------------------------------------------------------------
    assignment = assign_splits(traces, seed=args.seed, exclude_problem_keys=excl)
    for t in traces:
        t["split"] = assignment[t["problem_key"]]
    by_split: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        by_split[t["split"]].append(t)
    for s in PB_SUBSETS:
        by_split[f"pb_{s}"] = pb_traces[s]
    # leakage self-check
    seen: dict[str, str] = {}
    for sp, ts in by_split.items():
        if sp.startswith("pb_") or sp == "excluded_pb_overlap":
            continue
        for t in ts:
            assert seen.setdefault(t["problem_key"], sp) == sp, "split leakage"

    # --- tokenize + cap ---------------------------------------------------
    enc_rows = []
    len_audit: dict = {}
    cap_audit: dict = {}
    for sp in sorted(by_split):
        if sp == "excluded_pb_overlap":
            continue
        ntoks, step_lens = [], []
        kept_lab = Counter()
        lost_lab = Counter()
        for t in by_split[sp]:
            ids, ss, se = tokenize_solution(tok, t["problem"], t["steps"])
            n = len(ids)
            t["n_tokens"] = n
            ntoks.append(n)
            step_lens.extend(e - s for s, e in zip(ss, se))
            keep = n <= args.max_tokens and all(e > s for s, e in zip(ss, se))
            t["retained"] = keep
            for k, m in enumerate(t["label_mask"]):
                if m:
                    (kept_lab if keep else lost_lab)[int(t["y"][k])] += 1
            if keep:
                enc_rows.append({"trace_id": t["trace_id"], "split": sp, "n_tokens": n,
                                 "n_steps": len(t["steps"]), "step_starts": ss,
                                 "step_ends": se, "prefix_len": ss[0],
                                 "ids_sha1": hashlib.sha1(
                                     np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]})
        len_audit[sp] = {"trace_tokens": length_stats(ntoks), "step_tokens": length_stats(step_lens)}
        for cap in (4096, 8192, 16384):
            len_audit[sp][f"traces_over_{cap}"] = int(sum(n > cap for n in ntoks))
        cap_audit[sp] = {"traces_retained": sum(t["retained"] for t in by_split[sp]),
                         "traces_excluded": sum(not t["retained"] for t in by_split[sp]),
                         "labeled_correct_retained": kept_lab[0],
                         "labeled_incorrect_retained": kept_lab[1],
                         "labeled_correct_lost": lost_lab[0],
                         "labeled_incorrect_lost": lost_lab[1]}
    audit["token_lengths"] = len_audit
    audit["context_cap"] = cap_audit

    # deterministic shard assignment, balanced by tokens (greedy, longest first)
    order = sorted(range(len(enc_rows)), key=lambda i: (-enc_rows[i]["n_tokens"], enc_rows[i]["trace_id"]))
    load = [0] * args.num_shards
    for i in order:
        s = int(np.argmin(load))
        enc_rows[i]["shard"] = s
        load[s] += enc_rows[i]["n_tokens"]
    enc_rows.sort(key=lambda r: (r["shard"], r["split"], r["trace_id"]))
    total_tok = sum(r["n_tokens"] for r in enc_rows)
    audit["encode"] = {"traces": len(enc_rows), "tokens": total_tok,
                       "fp16_gib_at_d4096": total_tok * 4096 * 2 / 2**30,
                       "num_shards": args.num_shards,
                       "max_shard_tokens": max(load), "min_shard_tokens": min(load)}

    # --- write ------------------------------------------------------------
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    meta_fields = ("trace_id", "problem_key", "split", "y", "label_mask", "ratings",
                   "step_status", "first_error", "pb_label", "pb_subset", "finish_reasons",
                   "source_record_ids", "n_annotations", "n_tokens", "retained")
    split_counts = {}
    for sp, ts in sorted(by_split.items()):
        retained = [t for t in ts if t.get("retained", False)]
        split_counts[sp] = {"all": trace_counts(ts), "retained": trace_counts(retained)}
        if sp == "excluded_pb_overlap":
            continue
        write_jsonl(out / "inputs" / f"{sp}.jsonl",
                    ({"trace_id": t["trace_id"], "problem": t["problem"], "steps": t["steps"]}
                     for t in retained))
        write_jsonl(out / "meta" / f"{sp}.jsonl",
                    ({k: t[k] for k in meta_fields if k in t} for t in ts))
    write_jsonl(out / "encode_manifest.jsonl", enc_rows)
    audit["split_counts"] = split_counts
    audit["counters"] = dict(sorted(counters.items()))
    audit["tokenizer"] = args.tokenizer
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2))
    sums = {str(p.relative_to(out)): sha256_file(p)
            for p in sorted(out.rglob("*.json*")) if p.name != "manifest_checksums.json"}
    (out / "manifest_checksums.json").write_text(json.dumps(sums, indent=2))
    print(json.dumps({k: audit[k] for k in ("encode", "pb_overlap", "pb_overlap_excluded_problems")}, indent=2))
    for sp, c in split_counts.items():
        print(sp, c["retained"])


if __name__ == "__main__":
    main()
