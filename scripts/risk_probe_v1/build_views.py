#!/usr/bin/env python3
"""Freeze the risk_probe_v1 view manifest (docs/risk_probe_v1_plan.md, sections 3 and 8).

Splits, all in the context_ablation_v3 `full` view format so the store, the
validity probe and its scoring code are §21.18's:

  xf0..xf4      risk_probe_v0's v2_train_xfit views plus one view per label-set
                step, by problem fold (src/analysis/risk_probe_v1.fold_of); the
                cross-fitted validity probe xfit5_f<k> trains on the other four
  v2_dev_xfit   copied from risk_probe_v0 (early stopping)
  fsl_forks     copied from risk_probe_v0 (the 531 test forks, E1)
  gate_rp_test  copied from risk_probe_v0 (reproduction gate)

A label step whose view id (<trace_id>#<claim index>) is already a v0 view is
copied row for row; the others are built with build_context_views' Tok and
compact, as risk_probe_v0 built its fork views. Every label step's text must equal
its ReProbe step line after strip_step (the decoded reply span of the frozen
label set). Writes <out_dir>/{inputs,meta}/, encode_manifest.jsonl,
lab_views.jsonl (label step -> view id and fold) and build_audit.json.
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
from src.analysis.risk_probe_v1 import N_FOLDS, fold_of  # noqa: E402
from src.data.reprobe_trajectories import strip_step  # noqa: E402

COPIED = ("v2_dev_xfit", "fsl_forks", "gate_rp_test")
MIN_ALIGNED = 0.99


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wjl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def step_view(tk: Tok, vid: str, trace: dict, ci: int, problem_key: str, y: int, mask: bool,
              split: str) -> tuple[dict, dict, dict]:
    """(inputs row, meta row, encode row) of a label step's `full` view."""
    header = Q_HEADER.format(p=trace["problem"])
    vs = view_steps("full", trace["steps"], ci)
    ids, ss, se = tk.view(header, vs)
    keep, nst, st_s, st_e = compact(ss, se)
    inp = {"trace_id": vid, "header": header, "steps": vs}
    meta = {"trace_id": vid, "split": split, "orig_trace_id": trace["trace_id"], "orig_step": ci,
            "orig_n_steps": len(trace["steps"]), "problem_key": problem_key, "y": [int(y)],
            "label_mask": [bool(mask)]}
    enc = {"trace_id": vid, "split": split, "n_tokens": nst, "n_forward": len(ids), "n_steps": 1,
           "step_starts": st_s, "step_ends": st_e, "fwd_step_starts": ss, "fwd_step_ends": se,
           "keep": keep, "prefix_len": st_s[0], "ids_sha1": ids_sha(ids)}
    return inp, meta, enc


def build(v0: dict, label_traces: list[dict], manifest: dict[str, dict], tk: Tok, decode,
          fork_problems: set[str]) -> tuple[dict, dict, list[dict], list[dict], dict]:
    """Returns (inputs by split, meta by split, encode rows, lab_views rows, audit)."""
    c = Counter()
    pool_in, pool_meta, pool_enc = {}, {}, {}
    for m, i, e in zip(v0["v2_train_xfit"]["meta"], v0["v2_train_xfit"]["inputs"], v0["v2_train_xfit"]["enc"]):
        pool_meta[m["trace_id"]], pool_in[m["trace_id"]], pool_enc[m["trace_id"]] = m, i, e
    lab = []
    for t in label_traces:
        tr = manifest.get(t["trace_id"])
        if t["problem_key"] in fork_problems:
            sys.exit(f"[build] label trace {t['trace_id']} is on a test-fork problem")
        for i, ci in enumerate(t["step_claim_idx"]):
            c["label_steps"] += 1
            lo, hi = t["step_spans"][i]
            want = strip_step(decode(t["reply_ids"][lo:hi]))
            if tr is None or ci >= len(tr["steps"]):
                c["drop_no_manifest_step"] += 1
                continue
            if tr["steps"][ci] != want:
                c["drop_text_mismatch"] += 1
                continue
            vid = f"{t['trace_id']}#{ci}"
            y, mask = t["y_ds"][i], t["mask_ds"][i]
            if vid in pool_meta:
                c["copied_from_v0"] += 1
                old = pool_meta[vid]
                if old["label_mask"][0] and mask and int(old["y"][0]) != int(y):
                    c["v0_label_disagrees"] += 1
                src = "v0"
            else:
                inp, meta, enc = step_view(tk, vid, tr, ci, t["problem_key"], y, mask, "")
                pool_in[vid], pool_meta[vid], pool_enc[vid] = inp, meta, enc
                c["built"] += 1
                src = "built"
            lab.append({"view_id": vid, "trace_id": t["trace_id"], "step": i, "claim_idx": ci,
                        "problem_key": t["problem_key"], "fold": fold_of(t["problem_key"]), "source": src})
    aligned = len(lab) / max(1, c["label_steps"])
    c_split = Counter()
    out_in, out_meta, out_enc = {f"xf{k}": [] for k in range(N_FOLDS)}, {f"xf{k}": [] for k in range(N_FOLDS)}, []
    for vid in sorted(pool_meta):
        sp = f"xf{fold_of(pool_meta[vid]['problem_key'])}"
        out_in[sp].append(pool_in[vid])
        out_meta[sp].append({**pool_meta[vid], "split": sp})
        out_enc.append({**pool_enc[vid], "split": sp})
        c_split[sp] += 1
    for sp in COPIED:
        out_in[sp], out_meta[sp] = v0[sp]["inputs"], v0[sp]["meta"]
        out_enc += [{**e, "split": sp} for e in v0[sp]["enc"]]
        c_split[sp] = len(v0[sp]["meta"])
    lab_pk = {t["problem_key"] for t in label_traces}
    audit = {"counts": dict(c), "aligned_share": aligned, "views_by_split": dict(c_split),
             "dev_problems_in_label_set": len({m["problem_key"] for m in v0["v2_dev_xfit"]["meta"]} & lab_pk),
             "fork_problems_in_xf": len({m["problem_key"] for m in pool_meta.values()} & fork_problems)}
    return out_in, out_meta, out_enc, lab, audit


def load_v0(d: Path) -> dict:
    enc = jl(d / "encode_manifest.jsonl")
    out = {}
    for sp in ("v2_train_xfit",) + COPIED:
        meta = jl(d / "meta" / f"{sp}.jsonl")
        by = {e["trace_id"]: e for e in enc if e["split"] == sp}
        ins = {r["trace_id"]: r for r in jl(d / "inputs" / f"{sp}.jsonl")}
        out[sp] = {"meta": meta, "inputs": [ins[m["trace_id"]] for m in meta], "enc": [by[m["trace_id"]] for m in meta]}
    return out


def assign_shards(enc: list[dict], num_shards: int) -> None:
    order = sorted(range(len(enc)), key=lambda i: (-enc[i]["n_forward"], enc[i]["trace_id"]))
    load = [0] * num_shards
    for i in order:
        s = int(np.argmin(load)); enc[i]["shard"] = s; load[s] += enc[i]["n_forward"]
    enc.sort(key=lambda r: (r["shard"], r["split"], r["trace_id"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v0_views", type=Path, required=True, help="risk_probe_v0 views/ (frozen manifest)")
    ap.add_argument("--trace_set", type=Path, required=True, help="step_value_labels_v1 trace_set_label.json")
    ap.add_argument("--v2_manifest", type=Path, required=True)
    ap.add_argument("--fork_splits", type=Path, nargs="+", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--num_shards", type=int, default=16)
    a = ap.parse_args()
    if (a.out_dir / "encode_manifest.jsonl").exists():
        sys.exit(f"[build] {a.out_dir} is frozen")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=a.local_files_only)
    tk = Tok(tok)
    v0 = load_v0(a.v0_views)
    traces = json.loads(a.trace_set.read_text())["traces"]
    manifest = {r["trace_id"]: r for r in jl(a.v2_manifest / "inputs" / "train.jsonl")}
    forks = [f for p in a.fork_splits for f in json.loads(p.read_text())["forks"]]
    out_in, out_meta, out_enc, lab, audit = build(v0, traces, manifest, tk, tok.decode,
                                                  {f["problem_key"] for f in forks})
    ids = [r["trace_id"] for r in out_enc]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        sys.exit(f"[build] duplicate view ids across splits: {dup[:5]}")
    if audit["aligned_share"] < MIN_ALIGNED:
        sys.exit(f"[build] only {audit['aligned_share']:.4f} of label steps align (gate {MIN_ALIGNED})")
    if audit["fork_problems_in_xf"] or audit["dev_problems_in_label_set"]:
        sys.exit(f"[build] leakage: {audit}")
    assign_shards(out_enc, a.num_shards)
    for sp in out_in:
        wjl(a.out_dir / "inputs" / f"{sp}.jsonl", out_in[sp])
        wjl(a.out_dir / "meta" / f"{sp}.jsonl", out_meta[sp])
    wjl(a.out_dir / "encode_manifest.jsonl", out_enc)
    wjl(a.out_dir / "lab_views.jsonl", lab)
    audit.update({"v0_views": str(a.v0_views), "trace_set": str(a.trace_set), "trace_set_sha256": sha256(a.trace_set),
                  "fork_splits": {str(p): sha256(p) for p in a.fork_splits},
                  "forward_tokens": int(sum(r["n_forward"] for r in out_enc)),
                  "stored_tokens": int(sum(r["n_tokens"] for r in out_enc)), "num_shards": a.num_shards})
    audit["store_gib"] = audit["stored_tokens"] * 8192 / 2**30
    audit["outputs"] = {str(p.relative_to(a.out_dir)): sha256(p) for p in sorted(a.out_dir.rglob("*.jsonl"))}
    (a.out_dir / "build_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != "outputs"}, indent=1))


if __name__ == "__main__":
    main()
