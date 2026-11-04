#!/usr/bin/env python3
"""LLM-judge grading for the ReProbe test-time-scaling tables (Tables 2 and 3) on the sets
without a programmatic grader: ProofNet (proofs), Trips and Meetings (free-form plans).

ReProbe grades these with DeepSeek-R1 "based on both the problem statement and the reference
solution" (Sec. 4.2), with the chain-level prompt of its Fig. 11. DeepSeek-R1 does not fit a
TamIA node, so the judge here is a local model (default Qwen/Qwen3-32B, non-thinking, greedy,
through Hugging Face generate: the cluster's vLLM wheel does not install), with the Fig. 11
prompt plus the reference solution. A validation slice of candidates from the
programmatically graded sets (MATH, GSM8K, Calendar, StrategyQA, ScienceQA) is judged too,
and `merge` reports the judge's agreement with the programmatic grade there.

  build  judge items: best-of-N candidates (the first N per question of the released pools,
         N = 10, 5 for planning, as build_bon.py) of --sets; every beam-search answer in
         --beam/<set>*.jsonl of those sets; --val_per_set candidates per graded set.
         One JSON line {jid, set, problem, reference, solution, [prog]}.
         --only picks.json limits the best-of-N items to the (qi, k) listed per set.
  picks  the candidates the tables need when not every candidate is judged: per question,
         candidate 0 (an unbiased pass@1 sample) and the candidate each scorer picks (every
         seed of every --rt/--rp arm and every stored pool column of Table 3), as picks.json.
  judge  one shard of the items (resumable): {jid, correct (true/false/null), tail}.
  merge  <bon>/<set>.graded.jsonl ({qi, k, ok, key: null}) for the judged best-of-N sets,
         <out>/beam_judged.json {set: {"qi|arm": correct}}, and <out>/judge_agreement.json.

    python scripts/instep_v1/tts_judge.py build --pools $SCRATCH/cot_mech/reprobe_bon/pools \\
        --beam $V1/beam --out $OUT/tts/judge
    CUDA_VISIBLE_DEVICES=0 python scripts/instep_v1/tts_judge.py judge --out $OUT/tts/judge --shard 0 --num_shards 4
    python scripts/instep_v1/tts_judge.py merge --out $OUT/tts/judge --bon $V1/bon
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

JUDGED = ("proofnet", "trips", "meetings")
VALSETS = ("math", "gsm8k", "calendar", "strategyqa", "scienceqa")
N_BON = {"trips": 5, "meetings": 5, "calendar": 5}

# ReProbe Fig. 11 (chain-level correctness), with the reference solution the text says the
# judge sees (Sec. 4.2: "based on both the problem statement and the reference solution")
PROMPT = """You will be given a <Problem>, its <Reference Solution> and a proposed <Solution>. Your task is to assess whether the proposed solution is **correct** or **incorrect**.

Respond using the **exact format** below, do not include any text outside this template.
Output format:
<start of response>
Solution comments:
... your comments on the solution, explaining reasoning, pointing out any errors or confirming correctness ...
<Grade>: (Correct | Incorrect)
<end of response>

<Problem>: {problem}

<Reference Solution>: {reference}

<Solution>: {solution}"""
GRADE = re.compile(r"<Grade>\s*:\s*\(?\s*(Correct|Incorrect)", re.I)


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def parse_grade(text: str) -> bool | None:
    """The last <Grade>: verdict after any thinking block; None if there is none."""
    text = text.split("</think>")[-1]
    m = GRADE.findall(text)
    return None if not m else m[-1].lower() == "correct"


def cmd_build(a) -> None:
    from scripts.reprobe_cmp.bon_d0 import load_pool
    from src.eval.reprobe_bon_grade import grade
    a.out.mkdir(parents=True, exist_ok=True)
    items = []
    pools = {}
    what = set(a.what.split(","))
    only = None
    if a.only is not None:
        only = {s: {tuple(x) for x in v} for s, v in json.loads(a.only.read_text()).items()}
    for s in a.sets.split(","):
        df = load_pool(a.pools, s)
        pools[s] = df
        n = N_BON.get(s, 10)
        for r in (df[df.k < n].to_dict("records") if "bon" in what else []):
            if only is not None and (int(r["qi"]), int(r["k"])) not in only[s]:
                continue
            items.append({"jid": f"bon|{s}|{r['qi']}|{r['k']}", "set": s, "problem": r["q"],
                          "reference": str(r["answer"]), "solution": r["reply"]})
    if a.beam is not None and "beam" in what:
        for s in a.sets.split(","):
            gold = {int(r["qi"]): r for r in pools[s][pools[s].k == 0].to_dict("records")}
            seen = set()
            for f in sorted(a.beam.glob(f"{s}*.jsonl")):
                for r in jl(f):
                    if r.get("set") != s or (r["qi"], r["arm"]) in seen:
                        continue
                    seen.add((r["qi"], r["arm"]))
                    g = gold[int(r["qi"])]
                    items.append({"jid": f"beam|{s}|{r['qi']}|{r['arm']}", "set": s, "problem": g["q"],
                                  "reference": str(g["answer"]), "solution": "\n".join(r["steps"])})
    if a.val_per_set and "val" in what:
        import numpy as np
        rng = np.random.default_rng(0)
        for s in VALSETS:
            df = load_pool(a.pools, s)
            df = df[df.k < N_BON.get(s, 10)].reset_index(drop=True)
            for i in rng.choice(len(df), size=min(a.val_per_set, len(df)), replace=False):
                r = df.iloc[int(i)]
                ok, _ = grade(s, r["reply"], str(r["answer"]), r["q"])
                items.append({"jid": f"val|{s}|{r['qi']}|{r['k']}", "set": s, "problem": r["q"],
                              "reference": str(r["answer"]), "solution": r["reply"], "prog": bool(ok)})
    with open(a.out / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = defaultdict(int)
    for it in items:
        c[it["jid"].rsplit("|", 2)[0]] += 1
    print(f"[judge build] {len(items)} items: {dict(c)}", flush=True)


def needed_candidates(meta: list[dict], scores: list) -> set:
    """(qi, k) of candidate 0 of every question and of each score array's pick (lowest score)."""
    import numpy as np
    from scripts.reprobe_cmp.bon_analyze import pick
    qi = np.array([m["qi"] for m in meta])
    keep = {(m["qi"], m["k"]) for m in meta if m["k"] == 0}
    for sc in scores:
        keep.update((meta[i]["qi"], meta[i]["k"]) for i in pick(qi, np.asarray(sc, dtype=float)).values())
    return keep


def cmd_picks(a) -> None:
    import numpy as np
    from scripts.instep_v1.tts_tables import T3_ROWS, parse_arms
    from scripts.reprobe_cmp.analyze import seed_scores
    from scripts.reprobe_cmp.bon_analyze import trace_scores
    arms = {**parse_arms(a.rt, "rt"), **parse_arms(a.rp, "rp")}
    stored = sorted({src for _, _, kind, src in T3_ROWS if kind == "stored"})
    out = {}
    for s in a.sets.split(","):
        meta = jl(a.bon / "inputs" / f"{s}.meta.jsonl")
        tid = [m["trace_id"] for m in meta]
        scores = [[m["stored"][c] for m in meta] for c in stored if c in meta[0]["stored"]]
        for name, (kind, dirs) in arms.items():
            for d in dirs:
                sc = seed_scores(kind, d, f"bon_{s}", a.bon)
                if sc is None:
                    raise SystemExit(f"[picks] {name}: no bon_{s} scores in {d}")
                ts = trace_scores(sc)
                scores.append([ts.get(t, np.inf) for t in tid])
        keep = needed_candidates(meta, scores)
        out[s] = sorted(keep)
        print(f"[picks] {s}: {len(scores)} scorers, {len(keep)} of {len(meta)} candidates "
              f"({len(keep) / len(meta):.0%})", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out))


def _prompts(tok, items: list[dict], thinking: bool) -> list[str]:
    # bool(): Qwen3's template tests `enable_thinking is false`, which the integer 0 fails (thinking stays on)
    return [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(**{k: it[k] for k in
                                     ("problem", "reference", "solution")})}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=bool(thinking)) for it in items]


def cmd_judge(a) -> None:
    items = jl(a.out / "items.jsonl")[a.shard::a.num_shards]
    if a.smoke:
        items = items[:a.smoke]
    outp = a.out / f"verdicts.shard{a.shard}.jsonl"
    done = {r["jid"] for r in jl(outp)} if outp.exists() else set()
    todo = [it for it in items if it["jid"] not in done]
    print(f"[judge] shard {a.shard}/{a.num_shards}: {len(items)} items, {len(todo)} to do, backend {a.backend}",
          flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    if a.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.92,
                  enable_prefix_caching=True, seed=0)
        tok = llm.get_tokenizer()
        sp = SamplingParams(temperature=0.6 if a.thinking else 0.0, top_p=0.95, top_k=20, max_tokens=a.max_new_tokens,
                            seed=0)
        with open(outp, "a") as f:
            for i in range(0, len(todo), a.chunk):
                part = todo[i:i + a.chunk]
                outs = llm.generate(_prompts(tok, part, a.thinking), sp)
                for it, o in zip(part, outs):
                    txt = o.outputs[0].text
                    f.write(json.dumps({"jid": it["jid"], "correct": parse_grade(txt), "tail": txt[-300:],
                                        "n_tokens": len(o.outputs[0].token_ids)}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[judge] shard {a.shard}: {min(i + a.chunk, len(todo))}/{len(todo)} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
        todo.sort(key=lambda it: len(it["solution"]) + len(it["reference"]))  # less padding per batch
        with open(outp, "a") as f:
            for i in range(0, len(todo), a.batch):
                part = todo[i:i + a.batch]
                enc = tok(_prompts(tok, part, a.thinking), return_tensors="pt", padding=True,
                          add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    kw = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20) if a.thinking else \
                        dict(do_sample=False, temperature=None, top_p=None, top_k=None)
                    out = model.generate(**enc, max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
                for it, r in zip(part, out[:, enc["input_ids"].shape[1]:]):
                    txt = tok.decode(r, skip_special_tokens=True)
                    f.write(json.dumps({"jid": it["jid"], "correct": parse_grade(txt), "tail": txt[-300:]},
                                       ensure_ascii=False) + "\n")
                f.flush()
                print(f"[judge] shard {a.shard}: {min(i + a.batch, len(todo))}/{len(todo)} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[judge] shard {a.shard} done in {time.perf_counter() - t0:.0f}s", flush=True)


def cmd_merge(a) -> None:
    items = {it["jid"]: it for it in jl(a.out / "items.jsonl")}
    v = {}
    for f in sorted(a.out.glob("verdicts.shard*.jsonl")):
        for r in jl(f):
            v[r["jid"]] = r["correct"]
    missing = [j for j in items if j not in v]
    unparsed = [j for j in items if j in v and v[j] is None]
    print(f"[judge merge] {len(v)}/{len(items)} judged, {len(missing)} missing, {len(unparsed)} without a verdict",
          flush=True)
    bon, beam, agree = defaultdict(list), defaultdict(dict), defaultdict(lambda: [0, 0, 0, 0])  # tp fp fn tn
    for jid, it in items.items():
        kind, s, qi, x = jid.split("|")
        c = v.get(jid)
        if kind == "bon":
            bon[s].append({"qi": int(qi), "k": int(x), "ok": bool(c), "key": None, "judged": c is not None})
        elif kind == "beam":
            beam[s][f"{qi}|{x}"] = c
        elif c is not None:
            agree[s][(0 if it["prog"] else 2) + (0 if c else 1)] += 1
    if a.bon is not None:
        for s, rows in bon.items():
            with open(a.bon / f"{s}.graded.jsonl", "w") as f:
                for r in sorted(rows, key=lambda r: (r["qi"], r["k"])):
                    f.write(json.dumps(r) + "\n")
            print(f"[judge merge] bon {s}: {len(rows)} candidates, accuracy {sum(r['ok'] for r in rows) / len(rows):.3f}",
                  flush=True)
    (a.out / "beam_judged.json").write_text(json.dumps(beam, indent=0))
    ag = {}
    for s, (tp, fn, fp, tn) in agree.items():  # index: prog correct & judge correct, prog correct & judge wrong, ...
        n = tp + fn + fp + tn
        ag[s] = {"n": n, "agreement": (tp + tn) / n, "prog_correct_judge_wrong": fn, "prog_wrong_judge_correct": fp}
        print(f"[judge merge] agreement with the programmatic grade on {s}: {(tp + tn) / n:.3f} (n {n}; "
              f"judge rejects {fn} programmatic-correct, accepts {fp} programmatic-wrong)", flush=True)
    (a.out / "judge_agreement.json").write_text(json.dumps(ag, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pools", type=Path, required=True)
    b.add_argument("--beam", type=Path, default=None)
    b.add_argument("--sets", default=",".join(JUDGED))
    b.add_argument("--val_per_set", type=int, default=200)
    b.add_argument("--what", default="bon,beam,val", help="item kinds to build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--only", type=Path, default=None, help="picks.json: judge only these best-of-N candidates")
    k = sub.add_parser("picks")
    k.add_argument("--bon", type=Path, required=True, help="best-of-N dir with inputs/<set>.meta.jsonl")
    k.add_argument("--sets", default=",".join(JUDGED))
    k.add_argument("--rt", action="append", help="NAME=DIR[,DIR...] read-token arm (repeatable)")
    k.add_argument("--rp", action="append", help="NAME=DIR[,DIR...] ReProbe-format arm (repeatable)")
    k.add_argument("--out", type=Path, required=True)
    j = sub.add_parser("judge")
    j.add_argument("--out", type=Path, required=True)
    j.add_argument("--model", default="Qwen/Qwen3-32B")
    j.add_argument("--backend", choices=["vllm", "hf"], default="hf")
    j.add_argument("--thinking", type=int, default=0)
    j.add_argument("--max_new_tokens", type=int, default=768)
    j.add_argument("--max_model_len", type=int, default=16384)
    j.add_argument("--chunk", type=int, default=256)
    j.add_argument("--batch", type=int, default=8)
    j.add_argument("--shard", type=int, default=0)
    j.add_argument("--num_shards", type=int, default=1)
    j.add_argument("--smoke", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--bon", type=Path, default=None)
    a = p.parse_args()
    {"build": cmd_build, "picks": cmd_picks, "judge": cmd_judge, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
