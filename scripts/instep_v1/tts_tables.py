#!/usr/bin/env python3
"""ReProbe's Table 2 (beam search) and Table 3 (best-of-N), Qwen3-8B, rebuilt in one harness
with InStep inside: same rows, columns and averages as the paper, every number recomputed.

Why recomputed: ReProbe's released best-of-N pools do not reproduce its printed Table 3 (e.g.
MATH pass@1 68.6 under our grading against 92.4 printed; 25 of 25 sampled "wrong" MATH answers
are wrong, e.g. "2" for gold 4), so printed numbers and ours cannot share a table. Here every
row is scored on the same released pools / the same beam-search harness and graded the same way:
programmatic graders (src/eval/reprobe_bon_grade.py) on MATH, GSM8K, Calendar, StrategyQA,
ScienceQA (GSM8K by exact match, as the paper), the Qwen3-32B judge (tts_judge.py) on ProofNet,
Trips, Meetings.

Best-of-N (Table 3): the first N candidates per question (10; 5 for planning). A candidate's
score is its worst step (highest P(incorrect); for the stored PRM and ReProbe columns, the
pool's stored trace score), the lowest-scoring candidate is picked (bon_analyze.py). Rows:
pass@1, pass@N, majority vote (sets with a programmatic answer key), the pool's stored PRMs and
ReProbe Attn+Logit DeepSeek column, the released ReProbe hidden-state probes, InStep (3 seeds).
Beam search (Table 2): B = 5, N = 5 (beam_search.py), arms on the same questions with the same
sampling seed, accuracy over the questions every arm finished.
ID = mean of MATH, GSM8K, ProofNet; OOD = mean of the other sets; Overall = mean of all sets
(the paper's averages). Writes tts_tables.{json,md,tex}.

    python scripts/instep_v1/tts_tables.py --bon $V1/bon --beam $V1/beam \\
        --beam_judged $OUT/tts/judge_beam/beam_judged.json --out $OUT/tts/tables \\
        --rt InStep-DS=$OUT/rt/instep_ds_seed0,... --rt InStep-self=$V1/runs/rt/rt_self_seed0,... \\
        --rp "ReProbe HS self-anno=$V1/runs/released/scores/rel_self" ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reprobe_cmp.analyze import jl, seed_scores  # noqa: E402
from scripts.reprobe_cmp.bon_analyze import pick, trace_scores  # noqa: E402

T3_SETS = ("math", "gsm8k", "proofnet", "trips", "meetings", "calendar", "strategyqa", "scienceqa")
T2_SETS = ("math", "gsm8k", "proofnet", "trips", "calendar", "strategyqa", "scienceqa")
ID = ("math", "gsm8k", "proofnet")
LBL = {"math": "MATH", "gsm8k": "GSM8K", "proofnet": "ProofNet", "trips": "Trips", "meetings": "Meetings",
       "calendar": "Calendar", "strategyqa": "StrQA", "scienceqa": "SciQA"}
UHEAD = "JingweiNi/uhead_claim_Qwen3-8B_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3"
# Table 3 rows in the paper's order: (section, row label, kind, source); kind "stored" = a pool column
T3_ROWS = [
    ("Pass@N, majority voting", "Qwen3-8B pass@1 (lower bound)", "base", "pass@1"),
    ("Pass@N, majority voting", "Qwen3-8B pass@N (upper bound)", "base", "pass@N"),
    ("Pass@N, majority voting", "Qwen3-14B pass@1", "missing", "not generated"),
    ("Pass@N, majority voting", "Majority voting", "base", "majority"),
    ("PRMs", "Skywork-PRM-1.5B", "missing", "not in the released pools"),
    ("PRMs", "H4-Qwen2.5-PRM-1.5B-0.2", "stored", "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2"),
    ("PRMs", "Math-Shepherd-PRM-7B", "stored", "peiyi9979/math-shepherd-mistral-7b-prm"),
    ("PRMs", "RLHFlow-PRM-Deepseek-Data", "stored", "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"),
    ("PRMs", "RLHFlow-PRM-Mistral-Data", "stored", "RLHFlow/Llama3.1-8B-PRM-Mistral-Data"),
    ("PRMs", "Universal-PRM-Qwen2.5-Math-7B", "stored", "universalprm/Universal-PRM"),
    ("PRMs", "Qwen2.5-Math-7B-PRM800k", "stored", "Qwen/Qwen2.5-Math-7B-PRM800K"),
    ("PRMs", "Qwen2.5-Math-PRM-7B", "stored", "Qwen/Qwen2.5-Math-PRM-7B"),
    ("ReProbes", "ReProbe, Attn+Logit, Self-anno", "missing", "not released"),
    ("ReProbes", "ReProbe, Attn+Logit, DeepSeek-anno", "stored", UHEAD),
    ("ReProbes", "ReProbe, Hidden States, Self-anno", "arm", "ReProbe HS self-anno"),
    ("ReProbes", "ReProbe, Hidden States, DeepSeek-anno", "arm", "ReProbe HS DeepSeek-anno"),
    ("InStep (ours)", "InStep, Self-anno", "arm", "InStep-self"),
    ("InStep (ours)", "InStep, DeepSeek-anno", "arm", "InStep-DS"),
]
T2_ROWS = [
    ("No verifier", "Random scores (same search)", "random"),
    ("PRMs", "Qwen2.5-Math-7B-PRM800k", "prm800k"),
    ("PRMs", "Qwen2.5-Math-PRM-7B", "prm7b"),
    ("ReProbes", "ReProbe, Attn+Logit, Self-anno", None),
    ("ReProbes", "ReProbe, Attn+Logit, DeepSeek-anno", None),
    ("ReProbes", "ReProbe, Hidden States, Self-anno", "rp_self"),
    ("ReProbes", "ReProbe, Hidden States, DeepSeek-anno", "rp_ds"),
    ("InStep (ours)", "InStep, Self-anno", "rt_self"),
    ("InStep (ours)", "InStep, DeepSeek-anno", "instep_ds"),
]


def averages(row: dict, sets: tuple) -> dict:
    ood = [s for s in sets if s not in ID]
    out = {}
    for name, ss in (("ID", ID), ("OOD", ood), ("Overall", sets)):
        v = [row.get(s) for s in ss]
        out[name] = float(np.mean(v)) if v and all(x is not None for x in v) else None
    return out


def bon_table(bon: Path, arms: dict, sets: tuple, n_boot: int, pairs: list) -> dict:
    perq = defaultdict(dict)
    stored_cols = {src for _, _, kind, src in T3_ROWS if kind == "stored"}
    acc = defaultdict(dict)  # name -> set -> accuracy
    notes = {}
    for s in sets:
        gp = bon / f"{s}.graded.jsonl"
        if not gp.exists() or not (bon / "inputs" / f"{s}.meta.jsonl").exists():
            notes[s] = "not graded yet"
            continue
        g = {(r["qi"], r["k"]): r for r in jl(gp)}
        meta = jl(bon / "inputs" / f"{s}.meta.jsonl")
        meta = [m for m in meta if (m["qi"], m["k"]) in g]
        tid = [m["trace_id"] for m in meta]
        qi = np.array([m["qi"] for m in meta])
        y = np.array([g[(m["qi"], m["k"])]["ok"] for m in meta], dtype=float)
        keys = [g[(m["qi"], m["k"])].get("key") for m in meta]
        keys = [k if isinstance(k, str) else None for k in keys]
        judged = any("judged" in g[(m["qi"], m["k"])] for m in meta)
        qs = sorted(set(qi.tolist()))
        by_q = defaultdict(list)
        for i, q in enumerate(qi):
            by_q[int(q)].append(i)
        base = {"pass@1": np.array([y[by_q[q]].mean() for q in qs]), "pass@N": np.array([y[by_q[q]].max() for q in qs])}
        if not judged:
            maj = []
            for q in qs:
                c = Counter(keys[i] for i in by_q[q] if keys[i] is not None)
                maj.append(y[next(i for i in by_q[q] if keys[i] == c.most_common(1)[0][0])] if c else 0.0)
            base["majority"] = np.array(maj)
        for k, v in base.items():
            acc[k][s], perq[k][s] = 100 * float(v.mean()), v
        for col in stored_cols:
            if col in meta[0]["stored"]:
                score = np.array([m["stored"][col] for m in meta])
                best = pick(qi, score)
                v = np.array([y[best[q]] for q in qs])
                acc[col][s], perq[col][s] = 100 * float(v.mean()), v
        for name, (kind, dirs) in arms.items():
            per_seed = []
            for d in dirs:
                sc = seed_scores(kind, d, f"bon_{s}", bon)
                if sc is None:
                    break
                ts = trace_scores(sc)
                score = np.array([ts.get(t, np.inf) for t in tid])
                best = pick(qi, score)
                per_seed.append(np.array([y[best[q]] for q in qs]))
            if len(per_seed) == len(dirs):
                v = np.mean(per_seed, 0)
                acc[name][s], perq[name][s] = 100 * float(v.mean()), v
                acc[name][f"{s}_seeds"] = [100 * float(x.mean()) for x in per_seed]
    table = []
    for sec, label, kind, src in T3_ROWS:
        r = {"section": sec, "row": label, "note": src if kind == "missing" else None}
        if kind != "missing":
            r.update({s: acc[src].get(s) for s in sets})
            r.update(averages(r, sets))
        table.append(r)
    return {"rows": table, "pairs": boot_pairs(perq, pairs, sets, n_boot), "notes": notes,
            "seeds": {n: {s: acc[n].get(f"{s}_seeds") for s in sets} for n in arms}}


def beam_table(beam: Path, judged: dict, sets: tuple, n_boot: int, pairs: list) -> dict:
    rows = defaultdict(lambda: defaultdict(dict))
    for f in sorted(beam.glob("*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            c = r["correct"]
            if c is None:
                c = judged.get(r["set"], {}).get(f"{r['qi']}|{r['arm']}")
            rows[r["set"]][r["arm"]][r["qi"]] = c
    arms = [a for _, _, a in T2_ROWS if a]
    acc, perq, info = defaultdict(dict), defaultdict(dict), {}
    for s in sets:
        have = [a for a in arms if rows[s].get(a)]
        if not have:
            continue
        common = sorted(set.intersection(*(set(q for q, c in rows[s][a].items() if c is not None) for a in have)))
        info[s] = {"questions": len(common), "arms": have}
        for a in have:
            v = np.array([float(rows[s][a][q]) for q in common])
            acc[a][s], perq[a][s] = 100 * float(v.mean()) if len(v) else None, v
    table = []
    for sec, label, a in T2_ROWS:
        r = {"section": sec, "row": label, "note": None if a else "not released as a runnable probe"}
        if a:
            r.update({s: acc[a].get(s) for s in sets})
            r.update(averages(r, sets))
        table.append(r)
    return {"rows": table, "info": info, "pairs": boot_pairs(perq, pairs, sets, n_boot)}


def boot_pairs(perq: dict, pairs: list, sets: tuple, n_boot: int) -> list:
    rng = np.random.default_rng(0)
    out = []
    for A, B in pairs:
        ss = [s for s in sets if s in perq.get(A, {}) and s in perq.get(B, {})
              and len(perq[A][s]) == len(perq[B][s]) and len(perq[A][s])]
        if not ss:
            continue
        res, dm = {"pair": f"{A} - {B}", "sets": ss}, np.zeros(n_boot)
        for s in ss:
            d = perq[A][s] - perq[B][s]
            bs = 100 * d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
            dm += bs / len(ss)
            res[s] = [100 * float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res["mean"] = [float(np.mean([res[s][0] for s in ss])), float(np.percentile(dm, 2.5)),
                       float(np.percentile(dm, 97.5))]
        out.append(res)
    return out


def fmt(v) -> str:
    return "–" if v is None else f"{v:.1f}"


def md(t: dict, sets: tuple, title: str) -> list[str]:
    cols = [LBL[s] for s in sets] + ["ID", "OOD", "Overall"]
    L = [f"### {title}", "", "| Method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    sec = None
    for r in t["rows"]:
        if r["section"] != sec:
            sec = r["section"]
            L.append(f"| *{sec}* |" + " |" * len(cols))
        if r.get("note"):
            L.append(f"| {r['row']} | " + " | ".join("–" for _ in cols) + f" |  <!-- {r['note']} -->")
        else:
            L.append(f"| {r['row']} | " + " | ".join(fmt(r.get(c)) for c in list(sets) + ["ID", "OOD", "Overall"]) + " |")
    if t.get("pairs"):
        L += ["", "| pair (paired bootstrap over questions) | mean [95% CI] |", "|---|---|"]
        for p in t["pairs"]:
            L.append(f"| {p['pair']} ({len(p['sets'])} sets) | {p['mean'][0]:+.1f} [{p['mean'][1]:+.1f}, {p['mean'][2]:+.1f}] |")
    return L + [""]


def tex(t: dict, sets: tuple, caption: str) -> str:
    cols = [LBL[s] for s in sets] + ["ID", "OOD", "Overall"]
    L = ["\\begin{table*}[t]", "\\centering\\small", f"\\caption{{{caption}}}",
         "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", "Method & " + " & ".join(cols) + " \\\\",
         "\\midrule"]
    sec = None
    for r in t["rows"]:
        if r["section"] != sec:
            if sec is not None:
                L.append("\\midrule")
            sec = r["section"]
            L.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{sec}}}}} \\\\")
        vals = ["--" for _ in cols] if r.get("note") else [fmt(r.get(c)).replace("–", "--")
                                                          for c in list(sets) + ["ID", "OOD", "Overall"]]
        L.append(f"{r['row']} & " + " & ".join(vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(L)


def parse_arms(specs: list, kind: str) -> dict:
    out = {}
    for sp in specs or []:
        name, _, dirs = sp.partition("=")
        out[name] = (kind, dirs.split(","))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bon", type=Path, required=True)
    p.add_argument("--beam", type=Path, required=True)
    p.add_argument("--beam_judged", type=Path, default=None)
    p.add_argument("--rt", action="append", help="NAME=DIR[,DIR] read-token arm (bon_<set>.shard0.npz)")
    p.add_argument("--rp", action="append", help="NAME=DIR[,DIR] {uid, p} arm (bon_<set>.jsonl)")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arms = {**parse_arms(a.rt, "rt"), **parse_arms(a.rp, "rp")}
    t3 = bon_table(a.bon, arms, T3_SETS, a.n_boot,
                   [("InStep-DS", "ReProbe HS DeepSeek-anno"), ("InStep-self", "ReProbe HS self-anno"),
                    ("InStep-DS", "Qwen/Qwen2.5-Math-PRM-7B"), ("InStep-DS", "majority"), ("InStep-DS", "pass@1")])
    judged = json.loads(a.beam_judged.read_text()) if a.beam_judged and a.beam_judged.exists() else {}
    t2 = beam_table(a.beam, judged, T2_SETS, a.n_boot,
                    [("instep_ds", "rp_ds"), ("rt_self", "rp_self"), ("instep_ds", "prm7b"), ("instep_ds", "random")])
    (a.out / "tts_tables.json").write_text(json.dumps({"table3_bon": t3, "table2_beam": t2}, indent=1))
    L = md(t3, T3_SETS, "Table 3: Best-of-N accuracy (Qwen3-8B), ReProbe's released pools, one harness")
    L += [f"- Best-of-N sets not graded yet: {t3['notes']}", ""] if t3["notes"] else []
    L += md(t2, T2_SETS, "Table 2: Beam search accuracy (Qwen3-8B, B = 5, N = 5), one harness")
    L += [f"- beam questions per set (common to every arm run): "
          + ", ".join(f"{s} {v['questions']} ({len(v['arms'])} arms)" for s, v in t2["info"].items()), ""]
    (a.out / "tts_tables.md").write_text("\n".join(L))
    (a.out / "tts_tables.tex").write_text(
        tex(t3, T3_SETS, "Best-of-$N$ decoding accuracy (Qwen3-8B), recomputed on ReProbe's released pools.") + "\n\n"
        + tex(t2, T2_SETS, "Beam search decoding accuracy (Qwen3-8B, $B=5$, $N=5$)."))
    print("\n".join(L))


if __name__ == "__main__":
    main()
