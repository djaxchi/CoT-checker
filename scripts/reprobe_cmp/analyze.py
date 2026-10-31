#!/usr/bin/env python3
"""reprobe_comparison_v1 S3: every arm on ReProbe's eight test sets and on ProcessBench.

An arm is NAME=KIND:DIR[,DIR...], one DIR per seed:
  rp:<run>/scores/<head>   reprobe_probe.py score files, <set>.jsonl {"uid", "p"}
  rt:<encode dir>          read_token_inline.py encode outputs, <set>.shard0.{npz,meta.jsonl}
  stored:<estimator>       scores stored in the test managers (one "seed"; test sets only)

Per test set: AP (PR-AUC of "incorrect", the published metric), AUROC, F1 at the
threshold that maximizes F1 on validation (ReProbe val split, --val_label), oracle F1
and the always-positive F1. Seeds: metrics are computed per seed and averaged.
ID = MATH, GSM8K, ProofNet; OOD = the five others; overall = all eight.

--pairs A-B: paired bootstrap over traces within each set (--n_boot), recomputing every
seed's AP on the resample; reported for every set and for the ID, OOD and overall means.

ProcessBench (arms with pb_* files): calib-20 F1_PB through the leaderboard code, and the
same paired bootstrap as scripts/analysis/stage_f_bootstrap.py.

    python scripts/reprobe_cmp/analyze.py --data <R>/data --out <R>/analysis \\
        --arm RT-ds=rt:<R>/rt/rt_ds_seed0,<R>/rt/rt_ds_seed1 RP-ds=rp:<R>/rp/seed0/scores/rp_ds,... \\
        --pairs RT-ds-RP-ds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TEST = ("math", "gsm8k", "proofnet", "trips", "meetings", "calendar", "strategyqa", "scienceqa")
ID, OOD = TEST[:3], TEST[3:]
PB = ("gsm8k", "math", "olympiadbench", "omnimath")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def fast_ap(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (step-wise, as sklearn's average_precision_score, ties grouped)."""
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    tp = np.cumsum(y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # last index of each tie group
    tp, n = tp[last], last + 1
    prec, rec = tp / n, tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    o = np.argsort(-s, kind="mergesort")
    ys, ss = y[o], s[o]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum())
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    i = last[np.argmax(f1[last])]
    return float(ss[i])


def f1_at(y: np.ndarray, s: np.ndarray, t: float) -> float:
    pred = s >= t
    tp = float((pred & (y == 1)).sum())
    return 2 * tp / max(pred.sum() + y.sum(), 1)


# ---------------------------------------------------------------- loading
def test_sets(data: Path, exclude: set[str] = frozenset()) -> dict:
    """{set: {"uids": [...], "y": array, "trace": array of trace index}} over labeled steps,
    without the traces in `exclude`."""
    out = {}
    for s in TEST:
        uids, y, tr = [], [], []
        for ti, t in enumerate(jl(data / "test" / f"{s}.traces.jsonl")):
            if t["trace_id"] in exclude:
                continue
            for k, v in enumerate(t["labels"]):
                if v >= 0:
                    uids.append(f"{t['trace_id']}::{k}")
                    y.append(v)
                    tr.append(ti)
        out[s] = {"uids": uids, "y": np.array(y), "trace": np.array(tr)}
    return out


def val_set(data: Path, label: str) -> dict:
    uids, y = [], []
    for t in jl(data / "rp" / "val.traces.jsonl"):
        for k, v in enumerate(t["labels"][label]):
            if v >= 0:
                uids.append(f"{t['trace_id']}::{k}")
                y.append(v)
    return {"uids": uids, "y": np.array(y)}


def seed_scores(kind: str, d: str, name: str, data: Path) -> dict | None:
    """{uid: P(incorrect)} of one seed on one set, None if the arm has no such file."""
    if kind == "rp":
        p = Path(d) / f"{name}.jsonl"
        return {r["uid"]: r["p"] for r in jl(p)} if p.exists() else None
    if kind == "rt":
        z, m = Path(d) / f"{name}.shard0.npz", Path(d) / f"{name}.shard0.meta.jsonl"
        if not z.exists():
            return None
        pc = np.load(z)["head_p_correct"]
        return {r["uid"]: float(1 - pc[i]) for i, r in enumerate(jl(m)) if np.isfinite(pc[i])}
    if kind == "stored":
        if name not in TEST:
            return None
        out = {}
        for t in jl(data / "test" / f"{name}.traces.jsonl"):
            for k, v in enumerate(t["scores"].get(d, [])):
                out[f"{t['trace_id']}::{k}"] = float(v)
        return out or None
    raise ValueError(kind)


def parse_arm(spec: str) -> tuple[str, str, list[str]]:
    name, rest = spec.split("=", 1)
    kind, dirs = rest.split(":", 1)
    return name, kind, dirs.split(",")


def aligned(sc: dict, uids: list[str], what: str) -> np.ndarray:
    miss = [u for u in uids if u not in sc]
    if miss:
        raise SystemExit(f"{what}: {len(miss)} of {len(uids)} labeled steps have no score (e.g. {miss[0]})")
    return np.array([sc[u] for u in uids])


# ---------------------------------------------------------------- ProcessBench
def write_pb(kind: str, d: str, data: Path, out: Path) -> bool:
    """pb_step_scores_<subset>.jsonl in data/pb order; False if the seed has no PB scores."""
    rows = {}
    for sub in PB:
        sc = seed_scores(kind, d, f"pb_{sub}", data)
        if sc is None:
            return False
        rows[sub] = []
        for t in jl(data / "pb" / f"{sub}.traces.jsonl"):
            n = len(t["steps"])
            rows[sub].append({"id": t["trace_id"], "label": int(t["first_error"]), "n_steps": n,
                              "scores": [sc[f"{t['trace_id']}::{k}"] for k in range(n)]})
    out.mkdir(parents=True, exist_ok=True)
    for sub, rs in rows.items():
        with open(out / f"pb_step_scores_{sub}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    return True


def pb_calib20(dirs: list[str]) -> dict:
    from scripts.analysis.judge_prompt_report import PB_SUBSETS, calib20_subset, load_traces
    per = [{s: calib20_subset(load_traces(Path(d) / f"pb_step_scores_{s}.jsonl")) for s in PB_SUBSETS}
           for d in dirs]
    return {"calib20_mean": float(np.mean([np.mean(list(p.values())) for p in per])),
            "per_subset": {s: float(np.mean([p[s] for p in per])) for s in PB_SUBSETS},
            "seeds": [float(np.mean(list(p.values()))) for p in per]}


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", nargs="+", required=True)
    p.add_argument("--pairs", nargs="*", default=[], help="A-B with arm names A and B (names may not contain '-' twice)")
    p.add_argument("--val_label", default="ds")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--pb_boot", type=int, default=200)
    p.add_argument("--exclude_traces", type=Path, help="JSON list of test trace ids to leave out (overlap audit)")
    p.add_argument("--no_pb", action="store_true", help="skip ProcessBench")
    p.add_argument("--group", nargs="*", default=[],
                   help="extra mean groups name:set,set (v2: Phi-4's five Table 9 sets)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    extra = [(g.split(":", 1)[0], tuple(g.split(":", 1)[1].split(","))) for g in a.group]
    grouping = [("id", ID), ("ood", OOD), ("overall", TEST)] + extra
    tests = test_sets(a.data, set(json.loads(a.exclude_traces.read_text())) if a.exclude_traces else set())
    val = val_set(a.data, a.val_label) if (a.data / "rp" / "val.traces.jsonl").exists() else None

    arms, S = {}, {}
    for spec in a.arm:
        name, kind, dirs = parse_arm(spec)
        arms[name] = (kind, dirs)
        S[name] = {s: [aligned(sc, tests[s]["uids"], f"{name}/{s}") for d in dirs
                       if (sc := seed_scores(kind, d, s, a.data)) is not None] for s in TEST}
    table = {}
    for name, (kind, dirs) in arms.items():
        row = {}
        thr = []
        if val is not None:
            for d in dirs:
                sc = seed_scores(kind, d, "val", a.data)
                if sc is not None:
                    thr.append(best_f1_threshold(val["y"], aligned(sc, val["uids"], f"{name}/val")))
        for s in TEST:
            seeds, y = S[name][s], tests[s]["y"]
            if not seeds:
                continue
            aps = [fast_ap(y, x) for x in seeds]
            r = {"ap": float(np.mean(aps)), "ap_sd": float(np.std(aps)), "auroc": float(np.mean([auroc(y, x) for x in seeds])),
                 "oracle_f1": float(np.mean([f1_at(y, x, best_f1_threshold(y, x)) for x in seeds])),
                 "trivial_f1": float(2 * y.mean() / (1 + y.mean())), "n_seeds": len(seeds)}
            if len(thr) == len(seeds):
                r["val_f1"] = float(np.mean([f1_at(y, x, t) for x, t in zip(seeds, thr)]))
            row[s] = r
        for grp, sets in grouping:
            if all(s in row for s in sets):
                row[grp] = {"ap": float(np.mean([row[s]["ap"] for s in sets]))}
        pbd = []
        for i, d in enumerate(dirs):
            if kind != "stored" and not a.no_pb and write_pb(kind, d, a.data, a.out / "pb_scores" / name / f"seed{i}"):
                pbd.append(str(a.out / "pb_scores" / name / f"seed{i}"))
        if pbd:
            row["processbench"] = pb_calib20(pbd)
            row["processbench"]["dirs"] = pbd
        table[name] = row

    boots = []
    rng = np.random.default_rng(0)
    for pair in a.pairs:
        A, B = next((x, y) for x in arms for y in arms if f"{x}-{y}" == pair)
        res = {"pair": pair}
        deltas = {s: np.empty(a.n_boot) for s in TEST}
        point = {}
        for s in TEST:
            if not S[A][s] or not S[B][s]:
                continue
            y, tr = tests[s]["y"], tests[s]["trace"]
            groups = [np.nonzero(tr == t)[0] for t in np.unique(tr)]
            point[s] = np.mean([fast_ap(y, x) for x in S[A][s]]) - np.mean([fast_ap(y, x) for x in S[B][s]])
            for b in range(a.n_boot):
                idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                deltas[s][b] = (np.mean([fast_ap(y[idx], x[idx]) for x in S[A][s]])
                                - np.mean([fast_ap(y[idx], x[idx]) for x in S[B][s]]))
            res[s] = {"delta": float(point[s]), "ci95": [float(np.nanpercentile(deltas[s], 2.5)),
                                                         float(np.nanpercentile(deltas[s], 97.5))]}
        for grp, sets in grouping:
            if all(s in point for s in sets):
                d = np.mean([deltas[s] for s in sets], axis=0)
                res[grp] = {"delta": float(np.mean([point[s] for s in sets])),
                            "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                            "share_le0": float((d <= 0).mean())}
        if "processbench" in table.get(A, {}) and "processbench" in table.get(B, {}):
            from scripts.analysis.judge_prompt_report import boot_delta
            res["processbench"] = boot_delta(table[A]["processbench"]["dirs"], table[B]["processbench"]["dirs"],
                                             a.pb_boot)
        boots.append(res)

    (a.out / "summary.json").write_text(json.dumps({"arms": {k: {"kind": v[0], "dirs": v[1]} for k, v in arms.items()},
                                                    "table": table, "pairs": boots}, indent=1))
    gnames = ["id", "ood", "overall"] + [g for g, _ in extra]
    lines = ["| arm | " + " | ".join(TEST) + " | ID | OOD | overall | " + "".join(f"{g} | " for g, _ in extra)
             + "PB calib-20 |", "|---" * (len(TEST) + 5 + len(extra)) + "|"]
    for name, row in table.items():
        cells = [f"{row[s]['ap']:.3f}" if s in row else "-" for s in TEST]
        cells += [f"{row[g]['ap']:.3f}" if g in row else "-" for g in gnames]
        cells.append(f"{row['processbench']['calib20_mean']:.3f}" if "processbench" in row else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "| pair | ID | OOD | overall | " + "".join(f"{g} | " for g, _ in extra) + "PB calib-20 |",
              "|---" * (5 + len(extra)) + "|"]
    for r in boots:
        def c(k):
            return f"{r[k]['delta']:+.3f} [{r[k]['ci95'][0]:+.3f}, {r[k]['ci95'][1]:+.3f}]" if k in r else "-"
        pb = r.get("processbench")
        pbs = f"{pb['delta']:+.3f} [{pb['ci95_normal'][0]:+.3f}, {pb['ci95_normal'][1]:+.3f}]" if pb else "-"
        lines.append(f"| {r['pair']} | " + " | ".join(c(g) for g in gnames) + f" | {pbs} |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
