#!/usr/bin/env python3
"""InStep paper appendix: candidate example traces where InStep, Qwen2.5-Math-PRM-7B and ReProbe
disagree, with every step's text, label and each method's P(incorrect).

Methods (--arm name=kind:dir, analyze.seed_scores kinds, one seed each), in this order: InStep
first, then the two it is compared with. Each method flags a step when P(incorrect) >= its
threshold:
  ReProbe test sets (DeepSeek-R1 labels)  the method's best-F1 threshold on the validation split
                                          (the paper's val-threshold F1), 0.5 when it has no val
  ProcessBench (human labels)             0.5, the PRM's deployed rule (InStep-DS's F1_PB at 0.5,
                                          0.641, is level with its calib-20 0.632)
Categories:
  catch   test sets: a trace's FIRST incorrect step that InStep flags, with no InStep flag before
          it, while both other methods leave it unflagged
          ProcessBench: an erroneous trace whose first InStep flag is exactly the labelled first
          error, while each other method's first flag is elsewhere (earlier, later or none)
  quiet   a fully correct trace that InStep never flags while both other methods flag a step
  miss    (failure analysis) a first incorrect step / first error that InStep leaves unflagged
          while both other methods flag it
Within a category, candidates are ranked by the smallest of the three methods' margins to their
thresholds (all three must agree clearly), shorter traces first on ties; --top per set.

Writes <out>/showcase.md (readable) and <out>/showcase.json.

    python scripts/instep_v1/showcase.py --data $V1/data --out $OUT/showcase \\
        --arm instep=rt:$OUT/rt/instep_ds_seed0 prm7b=rp:$OUT/scores/qwen3_8b/prm7b \\
              reprobe=rp:$V1/runs/released/scores/rel_ds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reprobe_cmp.analyze import TEST, best_f1_threshold, jl, seed_scores, val_set  # noqa: E402

PB = ("gsm8k", "math", "olympiadbench", "omnimath")


def threshold(kind: str, d: str, data: Path, val_label: str) -> float:
    sc = seed_scores(kind, d, "val", data)
    if not sc:
        return 0.5
    v = val_set(data, val_label)
    keep = [i for i, u in enumerate(v["uids"]) if u in sc]
    return best_f1_threshold(v["y"][keep], np.array([sc[v["uids"][i]] for i in keep]))


def load_set(data: Path, folder: str, name: str, arms: dict) -> list[dict]:
    """Traces with step text, labels and every method's scores; steps without a label get -1."""
    items = {}
    for r in jl(data / folder / f"{name}.items.jsonl"):
        items[r["uid"]] = r
    sc = {k: seed_scores(kind, d, name if folder == "test" else f"pb_{name}", data) for k, (kind, d) in arms.items()}
    out = []
    for t in jl(data / folder / f"{name}.traces.jsonl"):
        tid = t["trace_id"]
        n = len(t["steps"])
        uids = [f"{tid}::{k}" for k in range(n)]
        if folder == "test":
            labels = [int(v) for v in t["labels"]]
            question = t.get("question", "")
        else:
            fe = int(t["first_error"])
            labels = [1 if k == fe else (0 if fe < 0 or k < fe else -1) for k in range(n)]
            question = items.get(uids[0], {}).get("problem", "")
        if any(sc[k] is None or any(u not in sc[k] for u in uids) for k in arms):
            continue
        text = [items[u]["candidate_step"] if u in items else t["steps"][k].get("sentence", "")
                for k, u in enumerate(uids)]
        out.append({"set": name if folder == "test" else f"pb_{name}", "trace_id": tid, "question": question,
                    "steps": text, "labels": labels,
                    "p": {k: [float(sc[k][u]) for u in uids] for k in arms}})
    return out


def first(xs) -> int:
    i = np.flatnonzero(np.asarray(xs))
    return int(i[0]) if len(i) else -1


def classify(tr: dict, thr: dict, names: list[str]) -> list[tuple[str, float, int]]:
    """[(category, strength, step)] this trace qualifies for."""
    a, b, c = names
    lab = np.array(tr["labels"])
    fl = {k: np.array(tr["p"][k]) >= thr[k] for k in names}
    mg = {k: np.array(tr["p"][k]) - thr[k] for k in names}
    out = []
    if (lab == 1).any():
        k = first(lab == 1)
        if tr["set"].startswith("pb_"):
            fa, fb, fc = first(fl[a]), first(fl[b]), first(fl[c])
            if fa == k and fb != k and fc != k:
                out.append(("catch", float(min(mg[a][k], -mg[b][k], -mg[c][k])), k))
            if fa != k and fb == k and fc == k and (fa == -1 or fa > k):
                out.append(("miss", float(min(-mg[a][k], mg[b][k], mg[c][k])), k))
        else:
            if fl[a][k] and not fl[a][:k].any() and not fl[b][k] and not fl[c][k]:
                out.append(("catch", float(min(mg[a][k], -mg[b][k], -mg[c][k])), k))
            if not fl[a][k] and fl[b][k] and fl[c][k]:
                out.append(("miss", float(min(-mg[a][k], mg[b][k], mg[c][k])), k))
    elif (lab == 0).all() and not fl[a].any() and fl[b].any() and fl[c].any():
        out.append(("quiet", float(min(-mg[a].max(), mg[b].max(), mg[c].max())), first(fl[b] | fl[c])))
    return out


def md(c: dict, names: list[str], thr: dict) -> str:
    L = [f"### {c['set']} / {c['trace_id']} ({c['category']}, step {c['step'] + 1}, strength {c['strength']:.3f})", "",
         f"**Question.** {c['question'][:700]}", "",
         "| step | label | " + " | ".join(f"{k} (thr {thr[k]:.2f})" for k in names) + " | text |",
         "|---|---|" + "---|" * len(names) + "---|"]
    for k, (s, lab) in enumerate(zip(c["steps"], c["labels"])):
        cells = []
        for n in names:
            p = c["p"][n][k]
            cells.append(f"**{p:.2f}**" if p >= thr[n] else f"{p:.2f}")
        txt = " ".join(s.split())[:400].replace("|", "\\|")
        mark = {1: "WRONG", 0: "ok", -1: "-"}[lab]
        L.append(f"| {k + 1}{' <' if k == c['step'] else ''} | {mark} | " + " | ".join(cells) + f" | {txt} |")
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--arm", nargs=3, required=True, help="name=kind:dir, InStep first")
    p.add_argument("--val_label", default="ds")
    p.add_argument("--top", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=14, help="skip longer traces (readability)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    arms = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        kind, d = rest.split(":", 1)
        arms[name] = (kind, d)
    names = list(arms)
    thr_test = {k: threshold(kind, d, a.data, a.val_label) for k, (kind, d) in arms.items()}
    thr_pb = {k: 0.5 for k in names}
    print("[showcase] val thresholds " + ", ".join(f"{k} {v:.3f}" for k, v in thr_test.items()), flush=True)
    res, counts = [], {}
    for folder, sets, thr in (("test", TEST, thr_test), ("pb", PB, thr_pb)):
        for s in sets:
            trs = load_set(a.data, folder, s, arms)
            cands = []
            for tr in trs:
                if len(tr["steps"]) > a.max_steps:
                    continue
                for cat, strength, k in classify(tr, thr, names):
                    cands.append({**tr, "category": cat, "strength": strength, "step": k})
            for cat in ("catch", "quiet", "miss"):
                cs = sorted([c for c in cands if c["category"] == cat],
                            key=lambda c: (-round(c["strength"], 2), len(c["steps"])))
                counts[f"{folder}/{s}/{cat}"] = len(cs)
                res += [{**c, "thr": thr} for c in cs[:a.top]]
            print(f"[showcase] {folder}/{s}: {len(trs)} traces, "
                  + ", ".join(f"{cat} {counts[f'{folder}/{s}/{cat}']}" for cat in ("catch", "quiet", "miss")), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "showcase.json").write_text(json.dumps({"thresholds_test": thr_test, "counts": counts, "cases": res}, indent=1))
    with open(a.out / "showcase.md", "w") as f:
        f.write("# InStep showcase candidates\n\nBold: flagged (P(incorrect) >= the method's threshold). "
                "`<` marks the step the case is about.\n\n")
        for cat in ("catch", "quiet", "miss"):
            f.write(f"## {cat}\n\n")
            for c in [c for c in res if c["category"] == cat]:
                f.write(md(c, names, c["thr"]) + "\n")


if __name__ == "__main__":
    main()
