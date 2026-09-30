#!/usr/bin/env python3
"""Compare what a "step" is in PRM800K, ProcessBench, the Instruct pools and the online rollouts.

Token lengths (Qwen tokenizer), steps per solution, the share of each step kind
(prose, markdown header, horizontal rule, display math only, lead-in ending with a
colon), and each checker's mean suspicion by step kind on the online smoke drafts.
"""
import json, glob, re, random, statistics as st, sys, itertools
from collections import defaultdict
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from datasets import load_dataset
from scripts.generate_onpolicy_steps import split_into_steps
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
def kind(s):
    t = s.strip()
    if re.fullmatch(r"[-*_]{3,}", t): return "horizontal rule"
    if t.startswith("#"): return "markdown header"
    if re.fullmatch(r"(\$\$.*\$\$|\\\[.*\\\])", t, re.S): return "display math only"
    if "\\boxed" in t and n(t) < 30: return "boxed-answer line"
    if t.endswith(":") and n(t) < 30: return "lead-in ending ':'"
    if n(t) < 8: return "fragment <8 tok"
    return "prose / derivation"
random.seed(0)
out = {}
def summ(name, steps, per=None):
    L = sorted(n(s) for s in steps); p = lambda x: L[int(x*(len(L)-1))]
    K = defaultdict(int)
    for s in steps: K[kind(s)] += 1
    out[name] = {"n": len(steps), "median": p(.5), "p10": p(.1), "p90": p(.9), "mean": st.mean(L),
                 "steps_per_solution": st.median(per) if per else None,
                 "kinds": {k: v/len(steps) for k, v in K.items()}}
    print(f"== {name}: n={len(steps)} tokens median {p(.5)} (p10 {p(.1)}, p90 {p(.9)}), mean {st.mean(L):.1f}" + (f", steps/solution median {st.median(per)}" if per else ""))
    for k, v in sorted(K.items(), key=lambda x: -x[1]): print(f"     {k:20s} {100*v/len(steps):5.1f}%")
prm, per = [], []
for i, l in enumerate(open("/Users/djadja/.cache/huggingface/hub/datasets--tasksource--PRM800K/snapshots/547b19506677a59037ee888838834b65e9b1ddd4/phase2_test.jsonl")):
    if i >= 1500: break
    r = json.loads(l); txt = []
    for st_ in r["label"]["steps"]:
        c = st_.get("chosen_completion")
        comp = st_["completions"][c] if c is not None else (st_["completions"][0] if st_["completions"] else None)
        if comp: txt.append(comp["text"])
    prm += txt; per.append(len(txt))
summ("PRM800K (training steps)", random.sample(prm, min(3000, len(prm))), per)
pb = load_dataset("Qwen/ProcessBench", split="math")
summ("ProcessBench math", random.sample([s for r in pb for s in r["steps"]], 3000), [len(r["steps"]) for r in pb])
for dsn in ["math500", "gsm8k"]:
    rows = [json.loads(l) for f in glob.glob(f"cot-checker-results/tts_instruct_v1/tts_{dsn}.shard*_trajectories.jsonl") for l in open(f)]
    rows = random.sample(rows, 400); S = [split_into_steps(r["solution"]) for r in rows]
    summ(f"Instruct pool {dsn}", [s for x in S for s in x], [len(x) for x in S])
sm = [json.loads(l) for f in glob.glob("results/online_v2/smoke/*.jsonl") for l in open(f)]
summ("online smoke (kept steps)", [s for r in sm for s in r["steps"]], [r["n_steps"] for r in sm])
# probe and PRM scores by step kind, from every smoke draft
sc = defaultdict(lambda: defaultdict(list))
for r in sm:
    for texts, scores in zip(r.get("draft_texts") or [[s] for s in r["steps"]], r["draft_scores"]):
        for t, s in zip(texts, scores):
            for k, v in s.items(): sc[kind(t)][k].append(v)
print("\nMean suspicion by step kind (smoke drafts):")
for kd, d in sorted(sc.items(), key=lambda x: -len(next(iter(x[1].values())))):
    print(f"  {kd:20s} n={len(next(iter(d.values()))):4d} " + "  ".join(f"{k.split('__')[0][:14]} {st.mean(v):.3f}" for k, v in d.items()))
rows = [json.loads(l) for f in glob.glob("cot-checker-results/tts_instruct_v1/tts_math500.shard*_trajectories.jsonl") for l in open(f)]
print("\nOne Instruct MATH-500 solution, as steps:")
for i, s in enumerate(split_into_steps(rows[7]["solution"])[:16]): print(f"  [{i:2d}] {n(s):3d} tok  {kind(s):20s} {s[:95]!r}")
print("\nPRM800K steps, first solution:")
for s in prm[:6]: print(f"  {n(s):3d} tok  {s[:110]!r}")
json.dump(out, open("results/online_v2/step_definition_audit.json", "w"), indent=1)
