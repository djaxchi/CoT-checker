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
Beam search (Table 2): ReProbe's printed rows (PRMs and ReProbes, its own search and grading,
cited as printed: no beam code is released) above our search, B = 5, N = 5 (beam_search.py):
random scores, the released ReProbe hidden-state probes as the anchor, and InStep, on the same
questions with the same sampling seed, accuracy over the questions every arm finished. Our search
runs at a lower level than the printed one (ReProbe HS DeepSeek-anno 59.6 against 76.6 printed),
so InStep is compared with the anchor row, never with a printed row.
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
OURS = "Our search (B = 5, N = 5), same questions and sampling seeds"
T2_ROWS = [
    (OURS, "Random scores (same search)", "random"),
    (OURS, "ReProbe, Hidden States, Self-anno", "rp_self"),
    (OURS, "ReProbe, Hidden States, DeepSeek-anno", "rp_ds"),
    (OURS, "InStep, Self-anno", "rt_self"),
    (OURS, "InStep, DeepSeek-anno", "instep_ds"),
]
# ReProbe's Table 2 as printed (arXiv 2511.06209v5 HTML, parsed 2026-11-09), Qwen3-8B; ID, OOD and
# Overall are the printed averages
T2_PRINTED_SEC = "Printed in ReProbe (Table 2, its search and grading)"
T2_PRINTED = [
    ("Qwen2.5-Math-7B-PRM800k", (89.8, 80.4, 95.2, 13.1, 45.0, 86.7, 91.2), (88.5, 59.0, 71.6)),
    ("Qwen2.5-Math-PRM-7B", (88.1, 95.4, 93.6, 13.1, 41.5, 79.2, 83.6), (92.4, 54.4, 70.7)),
    ("ReProbe, Attn+Logit, Self-anno", (90.3, 95.4, 95.1, 15.0, 41.5, 94.8, 96.2), (93.6, 61.9, 75.5)),
    ("ReProbe, Attn+Logit, DeepSeek-anno", (81.8, 93.3, 79.1, 11.8, 42.4, 90.7, 90.8), (84.7, 58.9, 70.0)),
    ("ReProbe, Hidden States, Self-anno", (84.1, 97.3, 90.6, 15.6, 40.5, 91.2, 92.8), (90.7, 60.0, 73.2)),
    ("ReProbe, Hidden States, DeepSeek-anno", (86.8, 98.8, 95.6, 13.4, 48.0, 96.7, 96.7), (93.7, 63.7, 76.6)),
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
        # a judged set may grade only the candidates the table needs (tts_judge.py picks): every
        # scorer still picks among all candidates; ungraded candidates are NaN
        partial = any((m["qi"], m["k"]) not in g for m in meta)
        if not partial or not any((m["qi"], m["k"]) in g for m in meta):
            meta = [m for m in meta if (m["qi"], m["k"]) in g]
        tid = [m["trace_id"] for m in meta]
        qi = np.array([m["qi"] for m in meta])
        kk = np.array([m["k"] for m in meta])
        gr = [g.get((m["qi"], m["k"])) for m in meta]
        y = np.array([np.nan if r is None else r["ok"] for r in gr], dtype=float)
        keys = [None if r is None else r.get("key") for r in gr]
        keys = [k if isinstance(k, str) else None for k in keys]
        judged = any(r is not None and "judged" in r for r in gr)
        qs = sorted(set(qi.tolist()))
        by_q = defaultdict(list)
        for i, q in enumerate(qi):
            by_q[int(q)].append(i)
        if partial:  # pass@1 from candidate 0 alone (an unbiased sample); pass@N needs every candidate
            base = {"pass@1": np.array([y[[i for i in by_q[q] if kk[i] == 0][0]] for q in qs])}
            notes[s] = "partially judged: pass@1 from candidate 0, pass@N not available"
        else:
            base = {"pass@1": np.array([y[by_q[q]].mean() for q in qs]),
                    "pass@N": np.array([y[by_q[q]].max() for q in qs])}
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
                if np.isnan(v).any():
                    notes[f"{s}:{col}"] = "picks not judged"
                    continue
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
            if len(per_seed) == len(dirs) and any(np.isnan(x).any() for x in per_seed):
                notes[f"{s}:{name}"] = "picks not judged"
            elif len(per_seed) == len(dirs):
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


def bon_curve(bon: Path, arms: dict, sets: tuple, ns: tuple) -> dict:
    """Best-of-N accuracy as N grows: for each N, only the first N candidates (k < N) of every
    question, picked by majority vote, by each stored pool column and by each arm (mean over
    seeds), with the same rules as bon_table. Averaged over the graded sets whose pools hold at
    least max(ns) candidates per question, so every point of a curve averages the same sets."""
    stored_cols = {src for _, _, kind, src in T3_ROWS if kind == "stored"}
    per_set, used = defaultdict(list), []
    for s in sets:
        gp = bon / f"{s}.graded.jsonl"
        if not gp.exists() or not (bon / "inputs" / f"{s}.meta.jsonl").exists():
            continue
        g = {(r["qi"], r["k"]): r for r in jl(gp)}
        allmeta = jl(bon / "inputs" / f"{s}.meta.jsonl")
        meta = [m for m in allmeta if (m["qi"], m["k"]) in g]
        if len(meta) < len(allmeta):  # partially judged: picks among the first n need every grade
            continue
        size = Counter(m["qi"] for m in meta)
        if min(size.values()) < max(ns):
            continue
        used.append(s)
        tid = [m["trace_id"] for m in meta]
        qi = np.array([m["qi"] for m in meta])
        kk = np.array([m["k"] for m in meta])
        y = np.array([g[(m["qi"], m["k"])]["ok"] for m in meta], dtype=float)
        keys = [g[(m["qi"], m["k"])].get("key") for m in meta]
        keys = [k if isinstance(k, str) else None for k in keys]
        scores = {col: [np.array([m["stored"][col] for m in meta])] for col in stored_cols if col in meta[0]["stored"]}
        for name, (kind, dirs) in arms.items():
            seeds = [seed_scores(kind, d, f"bon_{s}", bon) for d in dirs]
            if all(sc is not None for sc in seeds):
                ts = [trace_scores(sc) for sc in seeds]
                scores[name] = [np.array([t_.get(t, np.inf) for t in tid]) for t_ in ts]
        by_q = defaultdict(list)
        for i, q in enumerate(qi):
            by_q[int(q)].append(i)
        acc = defaultdict(list)
        for n in ns:
            rows = np.flatnonzero(kk < n)
            q_n = qi[rows]
            qs = sorted(set(q_n.tolist()))
            maj = []
            for q in qs:
                idx = [i for i in by_q[q] if kk[i] < n]
                c = Counter(keys[i] for i in idx if keys[i] is not None)
                maj.append(y[next(i for i in idx if keys[i] == c.most_common(1)[0][0])] if c else 0.0)
            acc["majority"].append(100 * float(np.mean(maj)))
            for name, seeds in scores.items():
                vals = []
                for sc in seeds:
                    best = pick(q_n, sc[rows])
                    vals.append(np.mean([y[rows[best[q]]] for q in qs]))
                acc[name].append(100 * float(np.mean(vals)))
        for name, v in acc.items():
            per_set[name].append(v)
    curves = {name: [float(x) for x in np.mean(v, 0)] for name, v in per_set.items() if len(v) == len(used)}
    return {"sets": used, "n": list(ns), "curves": curves}


def beam_table(beam: Path, judged: dict, sets: tuple, n_boot: int, pairs: list) -> dict:
    rows = defaultdict(lambda: defaultdict(dict))
    secs = defaultdict(lambda: defaultdict(dict))  # wall-clock seconds of the whole search per question
    for f in sorted(beam.glob("*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            c = r["correct"]
            if c is None:
                c = judged.get(r["set"], {}).get(f"{r['qi']}|{r['arm']}")
            rows[r["set"]][r["arm"]][r["qi"]] = c
            secs[r["set"]][r["arm"]][r["qi"]] = r.get("seconds")
    arms = [a for _, _, a in T2_ROWS if a]
    acc, perq, info, seconds = defaultdict(dict), defaultdict(dict), {}, defaultdict(dict)
    for s in sets:
        have = [a for a in arms if rows[s].get(a)]
        if not have:
            continue
        common = sorted(set.intersection(*(set(q for q, c in rows[s][a].items() if c is not None) for a in have)))
        info[s] = {"questions": len(common), "arms": have}
        for a in have:
            v = np.array([float(rows[s][a][q]) for q in common])
            acc[a][s], perq[a][s] = 100 * float(v.mean()) if len(v) else None, v
            t = [secs[s][a][q] for q in common if secs[s][a].get(q) is not None]
            seconds[a][s] = float(np.mean(t)) if t and len(t) == len(common) else None
    table = []
    for sec, label, a in T2_ROWS:
        r = {"section": sec, "row": label, "note": None}
        r.update({s: acc[a].get(s) for s in sets})
        r.update(averages(r, sets))
        table.append(r)
    printed = [{"section": T2_PRINTED_SEC, "row": label, "note": None, **dict(zip(T2_SETS, v)),
                **dict(zip(("ID", "OOD", "Overall"), avg))} for label, v, avg in T2_PRINTED]
    return {"rows": table, "printed": printed, "info": info, "pairs": boot_pairs(perq, pairs, sets, n_boot),
            "seconds": {a: dict(v) for a, v in seconds.items()}}


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
    for r in t.get("printed", []) + t["rows"]:
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
    for r in t.get("printed", []) + t["rows"]:
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
                    [("instep_ds", "rp_ds"), ("rt_self", "rp_self"), ("instep_ds", "random"), ("rt_self", "random")])
    curve = bon_curve(a.bon, arms, T3_SETS, (1, 2, 4, 6, 8, 10))
    (a.out / "tts_tables.json").write_text(json.dumps({"table3_bon": t3, "table2_beam": t2, "bon_curve": curve},
                                                      indent=1))
    L = md(t3, T3_SETS, "Table 3: Best-of-N accuracy (Qwen3-8B), ReProbe's released pools, one harness")
    L += [f"- Best-of-N sets not graded yet: {t3['notes']}", ""] if t3["notes"] else []
    L += md(t2, T2_SETS, "Table 2: Beam search accuracy (Qwen3-8B): ReProbe's printed rows, then our search")
    L += [f"- beam questions per set (common to every arm run): "
          + ", ".join(f"{s} {v['questions']} ({len(v['arms'])} arms)" for s, v in t2["info"].items()), ""]
    (a.out / "tts_tables.md").write_text("\n".join(L))
    (a.out / "tts_tables.tex").write_text(
        tex(t3, T3_SETS, "Best-of-$N$ decoding accuracy (Qwen3-8B), recomputed on ReProbe's released pools.") + "\n\n"
        + tex(t2, T2_SETS, "Beam search decoding accuracy (Qwen3-8B). Top: ReProbe's printed rows. Bottom: our "
              "search ($B=5$, $N=5$), where InStep is compared with the released ReProbe probe."))
    print("\n".join(L))


if __name__ == "__main__":
    main()
