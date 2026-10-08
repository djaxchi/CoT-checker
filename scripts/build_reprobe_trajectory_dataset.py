#!/usr/bin/env python3
"""Build the frozen v2 manifests (bidirectional_token_probe_v2, plan section 4).

Inputs: the DeepSeek-labeled ReProbe parquet (primary labels), optionally the
self-labeled parquet (written as a separate label sidecar for the sensitivity
arm), and the frozen v1 manifest (problem -> split assignment, ProcessBench and
PRM800K-human evaluation sets, copied unchanged).

Writes the v1 manifest schema to --out_dir:
  inputs/<split>.jsonl, meta/<split>.jsonl, encode_manifest.jsonl,
  data_audit.json, manifest_checksums.json
and, with --self_parquet, meta_self/<split>.jsonl (same traces, self labels).

ReProbe splits: train / dev / calib / test, inherited from v1 by problem_key.
ProcessBench-overlapping problems are excluded from every ReProbe split, so the
four pb_* sets stay intact. prm_human_test = v1 'test' (human first-error labels).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.prm_trajectories import normalize_problem, problem_key  # noqa: E402
from src.data.reprobe_trajectories import merge_duplicates, parse_row, post_error_slices  # noqa: E402

PB = ("pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath")
RP_SPLITS = ("train", "dev", "calib", "test")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wjl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_rows(parquet: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(parquet, columns=["question", "answer", "reply", "claims", "verified"])
    return df.to_dict("records")


def build(rows, counters):
    parsed = [p for i, r in enumerate(rows) if (p := parse_row(r, f"row{i}", counters)) is not None]
    counters["parsed"] = len(parsed)
    return merge_duplicates(parsed, counters)


def counts(ts: list[dict]) -> dict:
    c = Counter()
    for t in ts:
        c["traces"] += 1
        c["steps"] += len(t["steps"])
        c["finished"] += bool(t.get("finished", True))
        c["traces_with_error"] += t["first_error"] is not None
        pre, post = post_error_slices(t["y"], t["label_mask"])
        for k, m in enumerate(t["label_mask"]):
            if not m:
                c["unlabeled_steps"] += 1
                continue
            c["labeled_incorrect" if t["y"][k] == 1 else "labeled_correct"] += 1
            if post[k]:
                c["post_error_labeled"] += 1
                c["post_error_labeled_correct"] += t["y"][k] == 0
            if k < len(t["steps"]) - 1:
                c["labeled_with_future"] += 1
        if t["first_error"] is not None:
            after = [t["y"][k] for k in range(t["first_error"] + 1, len(t["steps"])) if t["label_mask"][k]]
            if after:
                c["err_traces_with_labeled_after"] += 1
                c["suffix_all_incorrect"] += all(v == 1 for v in after)
    c["problems"] = len({t["problem_key"] for t in ts})
    return dict(c)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds_parquet", type=Path, required=True)
    ap.add_argument("--self_parquet", type=Path, default=None)
    ap.add_argument("--v1_manifest", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--num_shards", type=int, default=32)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from encode_processbench_full_store import tokenize_solution
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=a.local_files_only)
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    audit: dict = {"sources": {"ds_parquet": {"path": str(a.ds_parquet), "sha256": sha256(a.ds_parquet)}},
                   "v1_manifest": str(a.v1_manifest), "seed": a.seed, "max_tokens": a.max_tokens}

    # --- v1 split assignment, PB problems, eval sets ---------------------------
    v1_split: dict[str, str] = {}
    for sp in ("train", "dev", "calib", "test"):
        for m in jl(a.v1_manifest / "meta" / f"{sp}.jsonl"):
            v1_split[m["problem_key"]] = sp
    pb_keys, pb_norm = set(), set()
    pb_inputs = {}
    for sp in PB:
        ins = {r["trace_id"]: r for r in jl(a.v1_manifest / "inputs" / f"{sp}.jsonl")}
        pb_inputs[sp] = ins
        for r in ins.values():
            pb_keys.add(problem_key(r["problem"]))
            pb_norm.add(normalize_problem(r["problem"]))

    # --- ReProbe trajectories (DeepSeek labels) ---------------------------------
    c_ds: Counter = Counter()
    ds = build(load_rows(a.ds_parquet), c_ds)
    c_ds["unique_trajectories"] = len(ds)
    rng = random.Random(a.seed)
    extra = sorted({t["problem_key"] for t in ds
                    if t["problem_key"] not in v1_split and t["problem_key"] not in pb_keys
                    and normalize_problem(t["problem"]) not in pb_norm})
    has_err = defaultdict(bool)
    for t in ds:
        has_err[t["problem_key"]] |= t["first_error"] is not None
    extra_assign = {}
    for stratum in (True, False):
        ks = [k for k in extra if has_err[k] == stratum]
        rng.shuffle(ks)
        n = len(ks); ntr = round(0.8 * n); nva = round(0.1 * n)
        for i, k in enumerate(ks):
            extra_assign[k] = ("train" if i < ntr else ("dev" if i < ntr + nva // 2 else
                               ("calib" if i < ntr + nva else "test")))
    split_of = {}
    for t in ds:
        k = t["problem_key"]
        if k in pb_keys or normalize_problem(t["problem"]) in pb_norm:
            split_of[t["trace_id"]] = "excluded_pb_overlap"
        elif k in v1_split:
            split_of[t["trace_id"]] = v1_split[k]
        else:
            split_of[t["trace_id"]] = extra_assign[k]
    audit["split_inheritance"] = {
        "from_v1": dict(Counter(v1_split[t["problem_key"]] for t in ds if t["problem_key"] in v1_split
                                and split_of[t["trace_id"]] != "excluded_pb_overlap")),
        "new_problems_assigned": dict(Counter(extra_assign.values())),
        "pb_overlap_problems_excluded": len({t["problem_key"] for t in ds
                                            if split_of[t["trace_id"]] == "excluded_pb_overlap"}),
        "pb_overlap_traces_excluded": sum(v == "excluded_pb_overlap" for v in split_of.values())}
    # leak check: no ReProbe training/dev/calib/test problem in PB
    for t in ds:
        if split_of[t["trace_id"]] in RP_SPLITS:
            assert t["problem_key"] not in pb_keys and normalize_problem(t["problem"]) not in pb_norm
    by: dict[str, list] = defaultdict(list)
    for t in ds:
        t["split"] = split_of[t["trace_id"]]
        by[t["split"]].append(t)

    # --- self labels on the same traces ----------------------------------------
    self_lab = None
    if a.self_parquet is not None:
        audit["sources"]["self_parquet"] = {"path": str(a.self_parquet), "sha256": sha256(a.self_parquet)}
        c_self: Counter = Counter()
        sl = build(load_rows(a.self_parquet), c_self)
        self_lab = {t["trace_id"]: t for t in sl}
        audit["self_counters"] = dict(sorted(c_self.items()))
        agree = Counter()
        for t in ds:
            s = self_lab.get(t["trace_id"])
            if s is None:
                agree["ds_trace_without_self_labels"] += 1
                continue
            for k in range(len(t["steps"])):
                if t["label_mask"][k] and s["label_mask"][k]:
                    agree[(t["y"][k], s["y"][k])] += 1
        n = sum(v for k, v in agree.items() if isinstance(k, tuple))
        po = (agree[(0, 0)] + agree[(1, 1)]) / max(n, 1)
        p1 = (agree[(1, 0)] + agree[(1, 1)]) / max(n, 1)
        p2 = (agree[(0, 1)] + agree[(1, 1)]) / max(n, 1)
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        audit["ds_vs_self_agreement"] = {"n_steps_both_labeled": n, "agreement": po,
                                         "kappa": (po - pe) / (1 - pe) if pe < 1 else None,
                                         "counts_ds_self": {f"{k[0]}{k[1]}": v for k, v in agree.items()
                                                            if isinstance(k, tuple)},
                                         "ds_trace_without_self_labels": agree["ds_trace_without_self_labels"]}

    # --- eval sets copied from v1 ---------------------------------------------
    eval_sets = {sp: (jl(a.v1_manifest / "inputs" / f"{sp}.jsonl"), jl(a.v1_manifest / "meta" / f"{sp}.jsonl"))
                 for sp in PB}
    eval_sets["prm_human_test"] = (jl(a.v1_manifest / "inputs" / "test.jsonl"),
                                   jl(a.v1_manifest / "meta" / "test.jsonl"))

    # --- tokenize, cap, write ---------------------------------------------------
    enc_rows = []
    len_audit = {}
    meta_fields = ("trace_id", "problem_key", "split", "y", "label_mask", "step_status", "first_error",
                   "finished", "source_record_ids", "n_annotations", "n_tokens", "retained")
    for sp in RP_SPLITS:
        ts = by[sp]
        lens = []
        for t in ts:
            ids, ss, se = tokenize_solution(tok, t["problem"], t["steps"])
            t["n_tokens"] = len(ids)
            t["retained"] = len(ids) <= a.max_tokens and all(e > s for s, e in zip(ss, se))
            lens.append(len(ids))
            if t["retained"]:
                enc_rows.append({"trace_id": t["trace_id"], "split": sp, "n_tokens": len(ids),
                                 "n_steps": len(t["steps"]), "step_starts": ss, "step_ends": se,
                                 "prefix_len": ss[0],
                                 "ids_sha1": hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]})
        len_audit[sp] = {"p50": float(np.median(lens)), "p99": float(np.percentile(lens, 99)), "max": int(max(lens)),
                         "excluded_over_cap": int(sum(not t["retained"] for t in ts))}
        kept = [t for t in ts if t["retained"]]
        wjl(out / "inputs" / f"{sp}.jsonl", ({"trace_id": t["trace_id"], "problem": t["problem"], "steps": t["steps"]} for t in kept))
        wjl(out / "meta" / f"{sp}.jsonl", ({k: t[k] for k in meta_fields if k in t} for t in ts))
        if self_lab is not None:
            rows = []
            for t in ts:
                s = self_lab.get(t["trace_id"])
                rows.append({"trace_id": t["trace_id"], "split": sp, "problem_key": t["problem_key"],
                             "y": s["y"] if s else [-1] * len(t["steps"]),
                             "label_mask": s["label_mask"] if s else [False] * len(t["steps"]),
                             "retained": t["retained"]})
            wjl(out / "meta_self" / f"{sp}.jsonl", rows)
    v1_enc = {r["trace_id"]: r for r in jl(a.v1_manifest / "encode_manifest.jsonl")}
    for sp, (ins, metas) in eval_sets.items():
        wjl(out / "inputs" / f"{sp}.jsonl", ins)
        wjl(out / "meta" / f"{sp}.jsonl", ({**m, "split": sp} for m in metas))
        for r in ins:
            e = dict(v1_enc[r["trace_id"]]); e["split"] = sp
            enc_rows.append(e)
    order = sorted(range(len(enc_rows)), key=lambda i: (-enc_rows[i]["n_tokens"], enc_rows[i]["trace_id"]))
    load = [0] * a.num_shards
    for i in order:
        s = int(np.argmin(load)); enc_rows[i]["shard"] = s; load[s] += enc_rows[i]["n_tokens"]
    enc_rows = [{k: v for k, v in r.items()} for r in enc_rows]
    enc_rows.sort(key=lambda r: (r["shard"], r["split"], r["trace_id"]))
    wjl(out / "encode_manifest.jsonl", enc_rows)
    tot = sum(r["n_tokens"] for r in enc_rows)
    audit.update({"counters_ds": dict(sorted(c_ds.items())), "token_lengths": len_audit,
                  "encode": {"traces": len(enc_rows), "tokens": tot, "fp16_gib_at_d4096": tot * 8192 / 2**30,
                             "num_shards": a.num_shards},
                  "split_counts": {sp: counts([t for t in by[sp] if t.get("retained", True)]) for sp in by},
                  "eval_sets_copied_from_v1": sorted(eval_sets)})
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    sums = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*.json*"))
            if p.name != "manifest_checksums.json"}
    (out / "manifest_checksums.json").write_text(json.dumps(sums, indent=2))
    print(json.dumps({k: audit[k] for k in ("split_inheritance", "encode")}, indent=1))
    for sp, c in audit["split_counts"].items():
        print(sp, c)
    if "ds_vs_self_agreement" in audit:
        print("agreement", audit["ds_vs_self_agreement"])


if __name__ == "__main__":
    main()
