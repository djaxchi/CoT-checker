"""Step texts aligned to the span stores (step_bert_probe_v1, Arm B).

Writes one JSON per collection, {trace_id: {"problem": str, "steps": [str, ...]}}:

* onpolicy: from the ReProbe items files (`candidate_step` of item
  `<trace_id>::step<k>`), the exact text each stored step span was encoded from;
* pb_<subset>: from Qwen/ProcessBench (`steps` list, ids like `math-0`).

Alignment report against each store's meta: every stored step must have a text
(missing steps abort), the number of texts must equal meta n_steps, and, with
--tokenizer, the Qwen token count of each step text is compared with the stored
span length (n_tokens - step_start_idx) and the exact-match rate is reported.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def onpolicy_texts(items_files: list[Path]) -> dict:
    steps, problem = defaultdict(dict), {}
    for f in items_files:
        with f.open() as fh:
            for line in fh:
                it = json.loads(line)
                tid = it["fork_id"]
                steps[tid][int(it["step_idx"])] = it["candidate_step"]
                problem[tid] = it["problem"]
    out = {}
    for tid, d in steps.items():
        n = max(d) + 1
        if sorted(d) != list(range(n)):
            raise ValueError(f"{tid}: non-contiguous step indices {sorted(d)[:10]}")
        out[tid] = {"problem": problem[tid], "steps": [d[k] for k in range(n)]}
    return out


def pb_texts(subset: str) -> dict:
    from datasets import load_dataset
    return {r["id"]: {"problem": r["problem"], "steps": list(r["steps"])}
            for r in load_dataset("Qwen/ProcessBench", split=subset)}


def store_meta(roots: list[Path]) -> list[dict]:
    rows = []
    for r in roots:
        for sh in sorted(r.glob("shard_*")):
            with (sh / "meta.jsonl").open() as fh:
                rows += [json.loads(line) for line in fh]
    return rows


def check(texts: dict, meta: list[dict], tokenizer=None) -> dict:
    missing, nsteps_bad, tok_eq, tok_n, tok_absdiff = 0, 0, 0, 0, []
    for m in meta:
        t = texts.get(m["id"])
        k = int(m["step_idx"])
        if t is None or k >= len(t["steps"]):
            missing += 1
            continue
        nsteps_bad += int(len(t["steps"]) != int(m["n_steps"]))
        if tokenizer is not None and tok_n < 20000:
            n_span = int(m["n_tokens"]) - int(m["step_start_idx"])
            n_txt = len(tokenizer(t["steps"][k], add_special_tokens=False)["input_ids"])
            tok_eq += int(n_txt == n_span)
            tok_absdiff.append(abs(n_txt - n_span))
            tok_n += 1
    rep = {"stored_steps": len(meta), "missing_text": missing, "n_steps_mismatch": nsteps_bad}
    if tok_n:
        tok_absdiff.sort()
        rep.update({"tok_checked": tok_n, "tok_exact_rate": tok_eq / tok_n,
                    "tok_absdiff_p50": tok_absdiff[tok_n // 2], "tok_absdiff_p99": tok_absdiff[int(tok_n * 0.99)]})
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", required=True, help="onpolicy or pb_<subset>")
    p.add_argument("--items", type=Path, nargs="*", default=[])
    p.add_argument("--store_roots", type=Path, nargs="+", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    texts = onpolicy_texts(args.items) if args.collection == "onpolicy" else pb_texts(args.collection[3:])
    meta = store_meta(args.store_roots)
    keep = {m["id"] for m in meta}
    texts = {k: v for k, v in texts.items() if k in keep}
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rep = check(texts, meta, tok)
    rep.update({"collection": args.collection, "traces": len(texts)})
    print(json.dumps(rep))
    if rep["missing_text"]:
        raise SystemExit(f"{rep['missing_text']} stored steps have no text")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(texts))
    args.out.with_suffix(".check.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
