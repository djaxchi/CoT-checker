"""parametric_retrieval_steer_v1: Golden-Gate-style causal test of the
flip-neurons found in prgm (expF).

prgm showed a sparse set of MLP neurons at layer 27 whose donor-swap flips a
fact's answer, often a single neuron per fact. This asks the stronger,
concept-level question in the style of Anthropic's Scaling Monosemanticity: if
we CLAMP a fact's neuron to a large value on EVERY token of UNRELATED prompts,
does the model start talking about that fact's object/subject?

Two informative outcomes:
  concept-like    steering the neuron makes the fact appear on unrelated
                  prompts, and does so specifically for its own fact (diagonal
                  of the specificity matrix) -> a fact "knowledge neuron".
  contextual gate steering does nothing off-context -> the neuron is a
                  "this fact is relevant here" switch, not a concept feature
                  (the honest prior for a raw, polysemantic neuron).

CAVEAT: these are RAW MLP neurons, not SAE features. Scaling Monosemanticity
steered monosemantic SAE features; raw neurons are expected to be polysemantic.
A clean null here is the trigger to escalate to the SAE-feature version.

Neuron per fact = the most frequent top-attribution neuron across that fact's
pairs (from expF/run.parquet). Clamp value = alpha * max activation of the
neuron over the extraction set (NeuronStore). Controls: a random neuron, and
no-clamp baseline. Own-fact question is generated too as a positive control.

Phases: --phase run (sharded by fact) -> --analyze (curves + specificity
matrix + coherence).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.parametric_retrieval import prga_common as C  # noqa: E402
from scripts.parametric_retrieval.prga_generate import (  # noqa: E402
    encode_prompt,
)

UNRELATED = [
    "Tell me about your day.",
    "Write a short story about a garden.",
    "What should I cook for dinner tonight?",
    "Describe your favorite color.",
    "Give me some advice for staying motivated.",
    "What is the weather like where you are?",
    "Recommend a hobby to try this weekend.",
    "Explain how to tie a shoelace.",
    "Write two sentences about the ocean.",
    "What makes a good friend?",
]
ALPHAS = ["none", "0", "3", "6", "10"]
MATRIX_ALPHA = "6"


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested)
# --------------------------------------------------------------------------- #

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def mentions(gen: str, entity: str) -> bool:
    """Normalized substring test; entities shorter than 3 chars are skipped
    (too collision-prone to attribute)."""
    e = normalize(entity)
    if len(e) < 3:
        return False
    return e in normalize(gen)


def build_fact_neuron(run_df: pd.DataFrame) -> dict:
    """fact_id -> neuron id, the rank-weighted most frequent top-attribution
    neuron across that fact's pairs (rank r contributes weight 1/(r+1))."""
    out = {}
    tn = run_df[run_df.task == "top_neurons"]
    for fid, g in tn.groupby("fact_id"):
        score: Counter = Counter()
        for ids in g.neuron_ids.dropna():
            for r, nid in enumerate(ids):
                score[int(nid)] += 1.0 / (r + 1)
        if score:
            out[str(fid)] = max(score.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def run(args):
    model, tok, device = C.load_model_and_tok(
        args.model_name_or_path, args.local_files_only)
    out_dir = args.out_dir
    sel = json.loads((out_dir / "expE" / "selection.json").read_text())
    mlp_layer = int(sel["mlp"]["layer"])

    nstore = C.NeuronStore(out_dir)
    g_all = nstore.layer(mlp_layer)                       # (n_inst, inter)
    max_act = g_all.max(axis=0).astype(np.float32)        # per-neuron max
    n_neurons = len(max_act)

    run_df = pd.read_parquet(out_dir / "expF" / "run.parquet")
    fact_neuron = build_fact_neuron(run_df)

    meta = pd.read_parquet(out_dir / "metadata.parquet")
    meta["fact_id"] = meta.fact_id.astype(str)
    facts_info = {}
    for fid, g in meta[meta.prompt_mode == "direct"].groupby("fact_id"):
        r0 = g.iloc[0]
        facts_info[fid] = {"object": str(r0.gold_answer),
                           "subject": str(r0.subject),
                           "direction": str(r0.direction),
                           "own_msg": str(r0.user_message)}

    facts = sorted(f for f in fact_neuron
                   if f in facts_info
                   and len(normalize(facts_info[f]["object"])) >= 3)
    facts = facts[:args.n_facts]
    facts = facts[args.shard_idx::args.num_shards]

    rng = np.random.default_rng(args.seed)
    clamp = C.ClampNeuron(model, mlp_layer)
    rows = []
    for fid in facts:
        nid = int(fact_neuron[fid])
        rnd = int(rng.integers(n_neurons))
        info = facts_info[fid]
        battery = [("unrelated", i, encode_prompt(tok, m))
                   for i, m in enumerate(UNRELATED)]
        battery.append(("own_question", 0, encode_prompt(tok, info["own_msg"])))
        p_ids = [b[2] for b in battery]
        for arm, neuron in (("target", nid), ("random", rnd)):
            for a in ALPHAS:
                with clamp:
                    if a == "none":
                        clamp.set([], [])
                    else:
                        clamp.set([neuron], [float(a) * float(max_act[neuron])])
                    gens = C.greedy_generate(model, tok, device, p_ids,
                                             args.max_new_tokens, clamp)
                for (ptype, pidx, _), gen in zip(battery, gens):
                    rows.append({"fact_id": fid, "object": info["object"],
                                 "subject": info["subject"], "arm": arm,
                                 "neuron_id": neuron, "alpha": a,
                                 "prompt_type": ptype, "prompt_idx": pidx,
                                 "gen_text": gen})
        print(f"[run] fact {fid} (neuron {nid}) done", flush=True)

    exp = out_dir / "expG"
    exp.mkdir(exist_ok=True)
    suf = "" if args.num_shards == 1 else f"_shard{args.shard_idx:02d}"
    p = exp / f"steer{suf}.parquet"
    if p.exists() and not args.force:
        sys.exit(f"refusing to overwrite {p}; pass --force")
    pd.DataFrame(rows).to_parquet(p, index=False)
    print(f"[run] wrote {p} ({len(rows)} rows)", flush=True)


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #

def analyze(args):
    exp = args.out_dir / "expG"
    parts = []
    for s in range(args.num_shards):
        suf = "" if args.num_shards == 1 else f"_shard{s:02d}"
        f = exp / f"steer{suf}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(exp / "steer.parquet", index=False)

    df["hit_object"] = [mentions(g, o)
                        for g, o in zip(df.gen_text, df.object)]
    df["hit_subject"] = [mentions(g, s)
                         for g, s in zip(df.gen_text, df.subject)]
    df["hit_own"] = df.hit_object | df.hit_subject
    df["uniq_ratio"] = [len(set(normalize(g).split()))
                        / max(len(normalize(g).split()), 1) for g in df.gen_text]

    unrel = df[df.prompt_type == "unrelated"]
    # mention rate vs alpha, by arm (the Golden Gate curve)
    curve = (unrel.groupby(["arm", "alpha"])
             .agg(hit_own=("hit_own", "mean"),
                  hit_object=("hit_object", "mean"),
                  uniq_ratio=("uniq_ratio", "mean"),
                  n=("hit_own", "size")).reset_index())
    own = (df[df.prompt_type == "own_question"]
           .groupby("alpha").hit_own.mean().rename("own_question_hit")
           .reset_index())
    curve = curve.merge(own, on="alpha", how="left")
    order = {a: i for i, a in enumerate(ALPHAS)}
    curve = curve.sort_values(["arm", "alpha"], key=lambda s: s.map(
        lambda x: order.get(x, 99)) if s.name == "alpha" else s)
    curve.to_csv(exp / "steer_curves.csv", index=False)

    # specificity matrix: steer fact i's neuron (target, MATRIX_ALPHA),
    # rate object j appears in fact i's unrelated generations
    tgt = unrel[(unrel.arm == "target") & (unrel.alpha == MATRIX_ALPHA)]
    facts = sorted(tgt.fact_id.unique())
    objs = {f: tgt[tgt.fact_id == f].object.iloc[0] for f in facts}
    M = np.zeros((len(facts), len(facts)))
    gens_by = {f: tgt[tgt.fact_id == f].gen_text.tolist() for f in facts}
    for i, fi in enumerate(facts):
        for j, fj in enumerate(facts):
            M[i, j] = np.mean([mentions(g, objs[fj]) for g in gens_by[fi]])
    np.save(exp / "specificity_matrix.npy", M)
    (exp / "specificity_labels.json").write_text(json.dumps(
        {"facts": facts, "objects": [objs[f] for f in facts]}))
    diag = float(np.mean(np.diag(M)))
    off = float((M.sum() - np.trace(M)) / max(len(facts) ** 2 - len(facts), 1))

    summary = {
        "n_facts": len(facts),
        "diag_mention_rate": diag,
        "offdiag_mention_rate": off,
        "specificity_ratio": diag / off if off > 0 else None,
        "target_hit_by_alpha": {a: float(
            unrel[(unrel.arm == "target") & (unrel.alpha == a)].hit_own.mean())
            for a in ALPHAS},
        "random_hit_by_alpha": {a: float(
            unrel[(unrel.arm == "random") & (unrel.alpha == a)].hit_own.mean())
            for a in ALPHAS},
        "matrix_alpha": MATRIX_ALPHA,
    }
    (exp / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(curve.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--model_name_or_path", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--phase", choices=["run"], default=None)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n_facts", type=int, default=80)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze(args)
    elif args.phase == "run":
        run(args)
    else:
        sys.exit("pass --phase run or --analyze")


if __name__ == "__main__":
    main()
