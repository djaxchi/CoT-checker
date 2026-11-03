#!/usr/bin/env python3
"""InStep paper V1, final aggregation (run C1 of docs/instep_v1_runbook.md).

Every arm that exists is scored on every backbone; any arm, seed or set that is missing is
skipped and listed in the "what is missing" section, never a crash. Outputs in --final:
results.md (all tables), tables.tex, final.json (every number), fig*.png/pdf.

Per backbone (ReProbe test sets, metrics as scripts/reprobe_cmp/analyze.py): per-set AP
(PR-AUC of "incorrect"), AUROC, F1 at the validation-picked threshold, oracle F1 and the
always-positive F1, seed-averaged; ID / OOD / overall (and Phi-4's t9) means. Paired trace
bootstrap (shared resamples per set, --n_boot): InStep minus each Tier A comparator, the ladder
rungs (each minus the previous), and a CI for every arm's own overall AP.
ProcessBench: threshold-free first-error AUROC / AP (scripts/instep_v1/pb_metrics.py) with the
same paired bootstrap, and calib-20 (analyze.py's code) for the appendix.
Tier B: the ReProbe paper's printed PR-AUC (paper_numbers.py), with a consistency check
against the scores stored in ReProbe's released managers.

    python scripts/instep_v1/paper_v1.py --out_root $OUT --final $OUT/final --workers 48
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reprobe_cmp.analyze import (  # noqa: E402
    ID, OOD, TEST, best_f1_threshold, f1_at, fast_ap, jl, seed_scores, test_sets, val_set,
)
from scripts.instep_v1 import paper_numbers as PN  # noqa: E402
from scripts.instep_v1 import pb_metrics as PBM  # noqa: E402

T9 = PN.T9
USER = os.environ.get("USER", "dchikhi")
PROJ = Path(f"/project/aip-azouaq/{USER}/cot_mech")
BACKBONES = ("qwen3_8b", "phi4", "qwen3_32b")
VAL_LABEL = {"qwen3_8b": "ds", "phi4": "qwen3", "qwen3_32b": "oss"}
PRIMARY = {"qwen3_8b": "instep_ds", "phi4": "instep", "qwen3_32b": "instep"}
BB_LABEL = {"qwen3_8b": "Qwen3-8B", "phi4": "Phi-4", "qwen3_32b": "Qwen3-32B"}
SETS_LABEL = {"math": "MATH", "gsm8k": "GSM8K", "proofnet": "ProofNet", "trips": "Trips",
              "meetings": "Meetings", "calendar": "Calendar", "strategyqa": "StrQA", "scienceqa": "SciQA"}
LADDER = [("ptrue", "P(True), zero-shot"), ("passive", "passive linear"), ("emb_ds", "read token, embedding only"),
          ("off_ds", "+ per-layer offsets"), ("chain_ds", "+ chained reads"), ("instep_ds", "+ marker (= InStep)")]
LADDER_SIDE = ("nooff_ds", "InStep without offsets")

# cost constants (existing measurements, Qwen3-8B, MATH-500 trajectories, batch 1)
COST = [
    {"key": "instep_ds", "method": "InStep", "params": 307201, "ms": 8.8,
     "note": "online read over the live KV cache; offline (one text pass + read) 35.3 ms; +0.27 ms per decode step",
     "ms_offline": 35.3, "ms_decode_step": 0.27},
    {"key": "rp_ds", "method": "ReProbe (hidden states)", "params": 8140801, "ms": 9.4, "note": ""},
    {"key": "prm7b", "method": "Qwen2.5-Math-PRM-7B", "params": 7.6e9, "ms": 21.8, "note": "second 7B model"},
    {"key": "passive", "method": "passive linear", "params": 4097, "ms": 0.0,
     "note": "~4k params (4096 weights + bias); reads states the policy already computed"},
    {"key": "maxprob", "method": "MaxProb", "params": 0, "ms": 0.0, "note": "free signal"},
    {"key": "ptrue", "method": "P(True)", "params": 0, "ms": None,
     "note": "one extra prompted forward per step, not timed"},
]
PARAMS = {"instep_ds": 307201, "instep_prm": 307201, "instep_self": 307201, "instep": 307201,
          "rp_ds": 8140801, "rpc_ds": 8140801, "rp_rel_ds": 8140801, "rp_self": 8140801,
          "rp_rel_self": 8140801, "prm7b": 7.6e9, "prm800k": 7.6e9, "passive": 4097,
          "maxprob": 0, "ptrue": 0}
MS = {"instep_ds": 8.8, "rp_ds": 9.4, "rp_rel_ds": 9.4, "prm7b": 21.8, "passive": 0.0, "maxprob": 0.0}

# ReProbe-paper method each of our arms corresponds to (for "paper number" cells)
PAPER_MAP = {
    "qwen3_8b": {"maxprob": PN.MAXPROB, "maxprob_stored": PN.MAXPROB, "maxent": PN.MAXENT, "ppl": PN.PPL,
                 "ptrue": PN.PTRUE, "prm7b": PN.PRM7B, "prm800k": PN.PRM800K, "rp_rel_ds": PN.RP_HS_DS,
                 "rp_rel_self": PN.RP_HS_SELF, "rp_al": PN.RP_AL_DS},
    "phi4": {"maxprob": PN.MAXPROB, "maxprob_stored": PN.MAXPROB, "maxent": PN.MAXENT, "ppl": PN.PPL,
             "prm7b": PN.PRM7B, "prm800k": PN.PRM800K, "prm7b_stored": PN.PRM7B,
             "prm800k_stored": PN.PRM800K, "rp_al": PN.RP_AL_Q3},
    "qwen3_32b": {"prm7b": PN.PRM7B, "prm800k": PN.PRM800K, "prm7b_stored": PN.PRM7B,
                  "prm800k_stored": PN.PRM800K, "rp_stored": PN.RP_HS_OSS, "rp_rel": PN.RP_HS_OSS,
                  "maxprob_stored": None},
}


# ---------------------------------------------------------------------------- arm specs
def seeds(base: str, n: int = 3) -> list[str]:
    return [f"{base}{i}" for i in range(n)]


def candidate_arms(bb: str, out: Path, v1: Path, v2: Path) -> tuple[Path, list[dict]]:
    """(data root R, ordered list of candidate arms). paths: seed dirs; stored: estimator."""
    S = out / "scores" / bb
    conf = [dict(key="maxprob", label="MaxProb", family="confidence", paths=[str(S / "maxprob")]),
            dict(key="maxent", label="MaxEntropy", family="confidence", paths=[str(S / "maxent")]),
            dict(key="ppl", label="Perplexity", family="confidence", paths=[str(S / "ppl")]),
            dict(key="maxprob_stored", label="MaxProb (stored)", family="confidence",
                 stored="MaximumClaimProbability"),
            dict(key="passive", label="passive linear", family="passive", paths=[str(S / "passive")]),
            dict(key="prm7b", label="Qwen2.5-Math-PRM-7B", family="prm", paths=[str(S / "prm7b")]),
            dict(key="prm800k", label="Qwen2.5-Math-7B-PRM800K", family="prm", paths=[str(S / "prm800k")])]
    if bb == "qwen3_8b":
        R, U, rt = v1, v1 / "runs", out / "rt"
        rp = lambda h: [str(U / "rp" / f"seed{i}" / "scores" / h) for i in range(3)]  # noqa: E731
        arms = [dict(key="instep_ds", label="InStep (DS labels)", family="instep", paths=seeds(f"{rt}/instep_ds_seed")),
                dict(key="instep_prm", label="InStep (PRM800K labels, zero-shot)", family="instep",
                     paths=seeds(f"{rt}/instep_prm_seed")),
                dict(key="instep_self", label="InStep (self labels)", family="instep",
                     paths=seeds(f"{U}/rt/rt_self_seed")),
                dict(key="rp_ds", label="ReProbe retrained (DS)", family="reprobe", paths=rp("rp_ds")),
                dict(key="rpc_ds", label="ReProbe retrained, causal (DS)", family="reprobe", paths=rp("rpc_ds")),
                dict(key="rp_self", label="ReProbe retrained (self)", family="reprobe", paths=rp("rp_self")),
                dict(key="rpc_self", label="ReProbe retrained, causal (self)", family="reprobe", paths=rp("rpc_self")),
                dict(key="rp_rel_ds", label="ReProbe released (DS)", family="reprobe",
                     paths=[str(U / "released" / "scores" / "rel_ds")]),
                dict(key="rp_rel_self", label="ReProbe released (self)", family="reprobe",
                     paths=[str(U / "released" / "scores" / "rel_self")]),
                dict(key="rp_al", label="ReProbe Attn+Logit (stored, DS)", family="reprobe",
                     stored="LuqClaimEstimatorDummy_claim"),
                dict(key="ptrue", label="P(True)", family="ptrue", paths=[str(S / "ptrue")])] + conf
        arms += [dict(key=f"{v}_ds", label=lab, family="ladder", paths=seeds(f"{rt}/{v}_ds_seed"))
                 for v, lab in (("emb", "read token, embedding only (DS)"), ("off", "+ per-layer offsets (DS)"),
                                ("chain", "+ chained reads (DS)"), ("nooff", "InStep without offsets (DS)"))]
        arms.append(dict(key="emb_prm", label="read token, embedding only (PRM800K)", family="ladder",
                         paths=seeds(f"{rt}/emb_prm_seed")))
        return R, arms
    if bb == "phi4":
        R = v2 / "phi4"
        U = R / "runs"
        rp = lambda h: [str(U / "rp" / f"seed{i}" / "scores" / h) for i in range(3)]  # noqa: E731
        arms = [dict(key="instep", label="InStep (Qwen3-8B labels)", family="instep",
                     paths=seeds(f"{U}/rt/rt_qwen3_seed")),
                dict(key="instep_self", label="InStep (self labels)", family="instep",
                     paths=seeds(f"{U}/rt/rt_self_seed")),
                dict(key="rp_q3", label="ReProbe retrained (Qwen3-8B labels)", family="reprobe", paths=rp("rp_qwen3")),
                dict(key="rpc_q3", label="ReProbe retrained, causal (Qwen3-8B labels)", family="reprobe",
                     paths=rp("rpc_qwen3")),
                dict(key="rp_self", label="ReProbe retrained (self)", family="reprobe", paths=rp("rp_self")),
                dict(key="rpc_self", label="ReProbe retrained, causal (self)", family="reprobe", paths=rp("rpc_self")),
                dict(key="rp_al", label="ReProbe Attn+Logit (stored, Qwen3-8B labels)", family="reprobe",
                     stored="LuqClaimEstimatorDummy_claim"),
                dict(key="prm7b_stored", label="Qwen2.5-Math-PRM-7B (stored)", family="prm",
                     stored="PRM_Qwen/Qwen2.5-Math-PRM-7B"),
                dict(key="prm800k_stored", label="Qwen2.5-Math-7B-PRM800K (stored)", family="prm",
                     stored="PRM_Qwen/Qwen2.5-Math-7B-PRM800K")] + conf
        return R, arms
    R = v2 / "qwen3_32b"
    U = R / "runs"
    rp = lambda h: [str(U / "rp" / f"seed{i}" / "scores" / h) for i in range(3)]  # noqa: E731
    arms = [dict(key="instep", label="InStep (GPT-OSS labels)", family="instep", paths=seeds(f"{U}/rt/rt_oss_seed")),
            dict(key="rp_oss", label="ReProbe retrained (GPT-OSS)", family="reprobe", paths=rp("rp_oss")),
            dict(key="rpc_oss", label="ReProbe retrained, causal (GPT-OSS)", family="reprobe", paths=rp("rpc_oss")),
            dict(key="rp_rel", label="ReProbe released (GPT-OSS), rescored", family="reprobe",
                 paths=[str(U / "released" / "scores" / "rel_oss")]),
            dict(key="rp_stored", label="ReProbe hidden states (stored, GPT-OSS)", family="reprobe",
                 stored="LuqClaimEstimatorDummy_claim"),
            dict(key="prm7b_stored", label="Qwen2.5-Math-PRM-7B (stored)", family="prm",
                 stored="PRM_Qwen/Qwen2.5-Math-PRM-7B"),
            dict(key="prm800k_stored", label="Qwen2.5-Math-7B-PRM800K (stored)", family="prm",
                 stored="PRM_Qwen/Qwen2.5-Math-7B-PRM800K")] + conf
    return R, arms


def pairs_for(bb: str) -> list[tuple[str, str]]:
    P = PRIMARY[bb]
    if bb == "qwen3_8b":
        return ([(P, c) for c in ("rp_ds", "rpc_ds", "rp_rel_ds", "rp_al", "prm7b", "prm800k", "passive",
                                  "maxprob", "ptrue", "instep_prm", "instep_self")]
                + [("instep_self", c) for c in ("rp_self", "rpc_self", "rp_rel_self")]
                + [("instep_prm", c) for c in ("rp_rel_ds", "prm800k", "prm7b")])
    if bb == "phi4":
        return ([(P, c) for c in ("rp_q3", "rpc_q3", "rp_al", "prm7b", "prm800k", "prm7b_stored",
                                  "prm800k_stored", "passive", "maxprob", "maxprob_stored", "instep_self")]
                + [("instep_self", c) for c in ("rp_self", "rpc_self")])
    return [(P, c) for c in ("rp_oss", "rpc_oss", "rp_rel", "rp_stored", "prm7b", "prm800k", "prm7b_stored",
                             "prm800k_stored", "passive", "maxprob", "maxprob_stored")]


PB_PAIRS = [("instep_prm", c) for c in ("prm800k", "prm7b", "rp_rel_ds", "rp_ds", "rpc_ds", "passive", "maxprob",
                                        "ptrue", "emb_prm", "instep_ds", "instep_self")] + \
           [("instep_ds", c) for c in ("prm800k", "prm7b", "rp_rel_ds", "rp_ds", "rpc_ds", "passive", "maxprob",
                                       "ptrue")]


def groups_for(bb: str) -> list[tuple[str, tuple]]:
    g = [("id", ID), ("ood", OOD), ("overall", TEST)]
    return g + [("t9", T9)] if bb == "phi4" else g


# ---------------------------------------------------------------------------- loading
def resolve(path: str) -> tuple[str | None, list[str]]:
    """(kind, [seed dirs]) of a score dir: rt if it has *.shard0.npz, rp if *.jsonl; a dir
    without score files but with seed*/ subdirs expands to those."""
    p = Path(path)
    if not p.is_dir():
        return None, []
    if any(p.glob("*.shard0.npz")):
        return "rt", [str(p)]
    if any(p.glob("*.jsonl")):
        return "rp", [str(p)]
    subs = sorted(d for d in p.iterdir() if d.is_dir() and d.name.startswith("seed"))
    kinds = [resolve(str(d)) for d in subs]
    kinds = [k for k in kinds if k[0]]
    if kinds and len({k[0] for k in kinds}) == 1:
        return kinds[0][0], [d for k in kinds for d in k[1]]
    return None, []


def load_stored(data: Path, names: set[str]) -> dict:
    """{estimator: {set: {uid: score}}} from data/test/<set>.traces.jsonl, read once."""
    out = {n: {} for n in names}
    for s in TEST:
        p = data / "test" / f"{s}.traces.jsonl"
        if not p.exists():
            continue
        for t in jl(p):
            for n in names:
                v = t.get("scores", {}).get(n)
                if v:
                    d = out[n].setdefault(s, {})
                    for k, x in enumerate(v):
                        if x is not None:
                            d[f"{t['trace_id']}::{k}"] = float(x)
    return out


def short(d: str) -> str:
    return "/".join(Path(d).parts[-3:])


def align(sc: dict | None, uids: list[str]) -> tuple[np.ndarray | None, int]:
    if sc is None:
        return None, -1
    miss = sum(u not in sc for u in uids)
    if miss:
        return None, miss
    a = np.array([sc[u] for u in uids], dtype=float)
    return (a, 0) if np.all(np.isfinite(a)) else (None, int((~np.isfinite(a)).sum()))


def load_arm(spec: dict, data: Path, tests: dict, val: dict | None, pb_items: dict, stored: dict,
             missing: list) -> dict | None:
    """Arm with aligned per-seed arrays: test[set] = [arrays], thr = [val thresholds or None]
    per seed (dir order), pb = {sub: [arrays]} if complete. None if nothing exists."""
    key = spec["key"]
    if "stored" in spec:
        st = stored.get(spec["stored"], {})
        test = {}
        for s in TEST:
            a, m = align(st.get(s), tests[s]["uids"])
            if a is not None:
                test[s] = [a]
            elif s in st:
                missing.append(f"{key}: stored scores on {s} miss {m} labeled steps")
        if not test:
            missing.append(f"{key}: no stored column {spec['stored']}")
            return None
        return dict(spec, kind="stored", dirs=[spec["stored"]], test=test, seed_idx={s: [0] for s in test},
                    thr=[None], pb=None)
    kind, dirs = None, []
    for p in spec["paths"]:
        k, ds = resolve(p)
        if k is None:
            continue
        if kind not in (None, k):
            missing.append(f"{key}: mixed score formats, {p} skipped")
            continue
        kind = k
        dirs += ds
    if not dirs:
        missing.append(f"{key}: not found ({', '.join(spec['paths'])})")
        return None
    test, seed_idx = {}, {}
    for s in TEST:
        for i, d in enumerate(dirs):
            try:
                a, m = align(seed_scores(kind, d, s, data), tests[s]["uids"])
            except Exception as e:  # noqa: BLE001
                a, m = None, f"error {e!r}"
            if a is not None:
                test.setdefault(s, []).append(a)
                seed_idx.setdefault(s, []).append(i)
            elif m != -1:
                missing.append(f"{key}: {short(d)} {s} unusable ({m} labeled steps missing or non-finite)")
            else:
                missing.append(f"{key}: {short(d)} has no {s} scores")
    thr = []
    for i, d in enumerate(dirs):
        t = None
        if val is not None:
            try:
                a, m = align(seed_scores(kind, d, "val", data), val["uids"])
            except Exception:  # noqa: BLE001
                a, m = None, -1
            if a is not None:
                t = best_f1_threshold(val["y"], a)
            else:
                missing.append(f"{key}: {short(d)} has no usable val scores (no val-threshold F1)")
        thr.append(t)
    pb = None
    if pb_items:
        per_seed = []
        for d in dirs:
            sc = {}
            for sub in pb_items:
                try:
                    x = seed_scores(kind, d, f"pb_{sub}", data)
                except Exception:  # noqa: BLE001
                    x = None
                if x is not None:
                    sc[sub] = x
            if len(sc) == len(pb_items):
                per_seed.append(sc)
            elif sc:
                missing.append(f"{key}: {short(d)} has ProcessBench scores on {sorted(sc)} only")
        if per_seed:
            pb = PBM.seed_arrays(pb_items, per_seed)
            if pb is None:
                missing.append(f"{key}: ProcessBench scores do not cover every first-error item")
    if not test and pb is None:
        missing.append(f"{key}: found {dirs} but no usable test or ProcessBench scores")
        return None
    return dict(spec, kind=kind, dirs=dirs, test=test, seed_idx=seed_idx, thr=thr, pb=pb)


# ---------------------------------------------------------------------------- metrics
def arm_metrics(arm: dict, tests: dict, groups: list) -> dict:
    from scripts.instep_v1.pb_metrics import fast_auroc
    row = {}
    for s, arrs in arm["test"].items():
        y = tests[s]["y"]
        aps = [fast_ap(y, x) for x in arrs]
        r = {"ap": float(np.mean(aps)), "ap_sd": float(np.std(aps)), "ap_seeds": [float(a) for a in aps],
             "auroc": float(np.mean([fast_auroc(y, x) for x in arrs])),
             "oracle_f1": float(np.mean([f1_at(y, x, best_f1_threshold(y, x)) for x in arrs])),
             "trivial_f1": float(2 * y.mean() / (1 + y.mean())), "prevalence": float(y.mean()),
             "n_steps": int(len(y)), "n_seeds": len(arrs)}
        thr = [arm["thr"][i] for i in arm["seed_idx"][s]]
        if thr and all(t is not None for t in thr):
            r["val_f1"] = float(np.mean([f1_at(y, x, t) for x, t in zip(arrs, thr)]))
        row[s] = r
    for g, sets in groups:
        if all(s in row for s in sets):
            r = {"ap": float(np.mean([row[s]["ap"] for s in sets])),
                 "auroc": float(np.mean([row[s]["auroc"] for s in sets])),
                 "trivial_f1": float(np.mean([row[s]["trivial_f1"] for s in sets])),
                 "oracle_f1": float(np.mean([row[s]["oracle_f1"] for s in sets]))}
            if all("val_f1" in row[s] for s in sets):
                r["val_f1"] = float(np.mean([row[s]["val_f1"] for s in sets]))
            n = {row[s]["n_seeds"] for s in sets}
            if len(n) == 1:  # per-seed group AP (seed i of every set)
                r["ap_seeds"] = [float(np.mean([row[s]["ap_seeds"][i] for s in sets])) for i in range(n.pop())]
                r["ap_seed_sd"] = float(np.std(r["ap_seeds"]))
            row[g] = r
    return row


def trace_groups(tr: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(tr, kind="mergesort")
    _, start = np.unique(tr[order], return_index=True)
    end = np.r_[start[1:], len(tr)]
    return [order[a:b] for a, b in zip(start, end)]


def make_test_boot(tests: dict, n_boot: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for s in TEST:
        g = trace_groups(tests[s]["trace"])
        out[s] = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).astype(np.int32)
                  for _ in range(n_boot)]
    return out


_G: dict = {}  # shared with forked workers


def _boot_test_arm(task: tuple) -> tuple:
    """(bb, key) -> {set: {"ap": (n_boot,), "auroc": (n_boot,)}} seed-averaged on the resamples."""
    from scripts.instep_v1.pb_metrics import fast_auroc
    bb, key = task
    G = _G[bb]
    arm, tests, idx = G["arms"][key], G["tests"], G["boot"]
    out = {}
    for s, arrs in arm["test"].items():
        y = tests[s]["y"]
        ap = np.empty(len(idx[s]))
        au = np.empty(len(idx[s]))
        for b, ii in enumerate(idx[s]):
            yy = y[ii]
            ap[b] = np.mean([fast_ap(yy, x[ii]) for x in arrs])
            au[b] = np.mean([fast_auroc(yy, x[ii]) for x in arrs])
        out[s] = {"ap": ap, "auroc": au}
    return (bb, key), out


def _boot_pb_arm(task: tuple) -> tuple:
    bb, key = task
    G = _G[bb]
    return (bb, key), PBM.boot_arrays(G["pb_items"], G["arms"][key]["pb"], G["pb_boot"])


def run_pool(fn, tasks: list, workers: int) -> dict:
    if not tasks:
        return {}
    if workers <= 1 or len(tasks) == 1:
        return dict(fn(t) for t in tasks)
    with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
        return dict(pool.map(fn, tasks, chunksize=1))


def ci(d: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def pair_result(A: str, B: str, mA: dict, mB: dict, bA: dict, bB: dict, groups: list) -> dict:
    res = {"a": A, "b": B}
    for s in TEST:
        if s in bA and s in bB:
            for met in ("ap", "auroc"):
                d = bA[s][met] - bB[s][met]
                res.setdefault(s, {})[met] = {"delta": mA[s][met] - mB[s][met], "ci95": ci(d)}
    for g, sets in groups:
        if all(s in bA and s in bB for s in sets):
            for met in ("ap", "auroc"):
                d = np.mean([bA[s][met] - bB[s][met] for s in sets], axis=0)
                res.setdefault(g, {})[met] = {"delta": float(np.mean([mA[s][met] - mB[s][met] for s in sets])),
                                              "ci95": ci(d), "share_le0": float((d <= 0).mean())}
    return res


def arm_ci(m: dict, b: dict, groups: list) -> dict:
    out = {}
    for g, sets in groups:
        if all(s in b for s in sets):
            out[g] = {met: ci(np.mean([b[s][met] for s in sets], axis=0)) for met in ("ap", "auroc")}
    return out


# ---------------------------------------------------------------------------- abstention
def abstention(arm: dict, tests: dict) -> dict | None:
    """Risk-coverage on the 8 test sets pooled: trace flawed if any labelled step is incorrect,
    trace score = max step score; keep the lowest-scoring traces. Per seed, then averaged."""
    if not all(s in arm["test"] for s in TEST):
        return None
    ns = min(len(arm["test"][s]) for s in TEST)
    curves, aurc = [], []
    for i in range(ns):
        sc, fl = [], []
        for s in TEST:
            tr, y, x = tests[s]["trace"], tests[s]["y"], arm["test"][s][i]
            order = np.argsort(tr, kind="mergesort")
            trs = tr[order]
            start = np.r_[0, np.nonzero(np.diff(trs))[0] + 1]
            sc.append(np.maximum.reduceat(x[order], start))
            fl.append(np.maximum.reduceat(y[order], start))
        sc, fl = np.concatenate(sc), np.concatenate(fl).astype(float)
        o = np.argsort(sc, kind="mergesort")
        risk = np.cumsum(fl[o]) / np.arange(1, len(o) + 1)
        curves.append(risk)
        aurc.append(float(risk.mean()))
    n = len(curves[0])
    risk = np.mean(curves, axis=0)
    grid = np.linspace(0.01, 1.0, 100)
    k = np.clip(np.ceil(grid * n).astype(int) - 1, 0, n - 1)
    return {"coverage": grid.tolist(), "risk": risk[k].tolist(), "aurc": float(np.mean(aurc)),
            "aurc_seeds": aurc, "base_rate": float(risk[-1]), "n_traces": int(n), "n_seeds": ns,
            "risk_at": {f"{c:.1f}": float(risk[min(n - 1, int(math.ceil(c * n)) - 1)]) for c in (0.5, 0.8)}}


# ---------------------------------------------------------------------------- consistency
def consistency(bb: str, tests: dict, stored: dict) -> list[dict]:
    """Recomputed AP of ReProbe's stored estimator columns vs the transcribed paper cells."""
    rows = []
    for method, col in PN.STORED.get(bb, {}).items():
        paper, printed = PN.per_set(bb, method), PN.per_set(bb, method, corrected=False)
        for s in TEST:
            sc = stored.get(col, {}).get(s)
            a, _ = align(sc, tests[s]["uids"])
            if a is None or s not in paper:
                continue
            ap = fast_ap(tests[s]["y"], a)
            d = ap - paper[s]
            r = {"backbone": bb, "method": method, "set": s, "stored_col": col, "ours": ap, "paper": paper[s],
                 "paper_printed_column": printed.get(s), "diff": d, "ok": abs(d) <= 0.002}
            if not r["ok"]:
                r["known"] = PN.KNOWN_ISSUES.get((bb, method, s), "")
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------- per-layer
def parse_per_layer(obj) -> dict:
    """Loose reader for per_layer_<bb>.json: returns {series: [(layer, ap), ...]}. Accepts
    {name: {layer: ap | {"ap"/"val_ap": x}}}, {name: [{"layer"/"block": l, "ap"/"val_ap": x}]},
    or a list of records with a name key (kind / state / which / series / arm)."""
    apk = ("val_ap", "ap", "AP", "pr_auc")
    lk = ("layer", "block", "l", "idx")
    nk = ("kind", "state", "which", "series", "arm", "name", "view")

    def num(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, dict):
            for k in apk:
                if isinstance(v.get(k), (int, float)):
                    return float(v[k])
        return None

    out = {}

    def recs(lst, name):
        for r in lst:
            if not isinstance(r, dict):
                continue
            lay = next((r[k] for k in lk if k in r), None)
            v = num(r)
            if lay is None or v is None:
                continue
            nm = next((str(r[k]) for k in nk if k in r), name)
            out.setdefault(nm, []).append((int(lay), v))

    def walk(o, name):
        if isinstance(o, list):
            recs(o, name)
        elif isinstance(o, dict):
            ks = list(o)
            if ks and all(str(k).lstrip("-").isdigit() for k in ks) and all(num(o[k]) is not None for k in ks):
                out.setdefault(name, []).extend((int(k), num(o[k])) for k in ks)
                return
            for k, v in o.items():
                walk(v, f"{name}/{k}" if name else str(k))

    walk(obj, "")
    return {k: sorted(v) for k, v in out.items() if len(v) >= 2}


# ---------------------------------------------------------------------------- formatting
def f3(x, nd=3) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def fd(r: dict | None) -> str:
    if not r:
        return "-"
    return f"{r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"


def md_table(t: dict) -> str:
    lines = [f"**{t['title']}**", "", "| " + " | ".join(t["header"]) + " |", "|" + "---|" * len(t["header"])]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in t["rows"]]
    lines += [""] + [f"- {n}" for n in t.get("notes", [])]
    return "\n".join(lines) + "\n"


def tex_escape(s: str) -> str:
    s = str(s).replace("\\", "/")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    for a, b in (("Δ", r"$\Delta$"), ("−", "-"), ("±", r"$\pm$"), ("→", r"$\to$"), ("≈", r"$\approx$")):
        s = s.replace(a, b)
    return s


def tex_table(t: dict) -> str:
    n = len(t["header"])
    lines = [f"% {t['id']}", r"\begin{table*}[t]", r"\centering\small", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "r" * (n - 1) + "}", r"\toprule",
             " & ".join(tex_escape(h) for h in t["header"]) + r" \\", r"\midrule"]
    for r in t["rows"]:
        if r and r[0].startswith("**") and all(c == "" for c in r[1:]):
            lines.append(r"\midrule \multicolumn{" + str(n) + "}{l}{\\textit{" + tex_escape(r[0].strip("*")) + r"}} \\")
            continue
        lines.append(" & ".join(tex_escape(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\caption{" + tex_escape(t["title"]) + "}",
              r"\label{tab:" + t["id"] + "}", r"\end{table*}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- tables
def table1(bb: str, R: dict) -> dict:
    P = PRIMARY[bb]
    M, arms, pairs = R["metrics"], R["arms"], {(p["a"], p["b"]): p for p in R["pairs"]}
    groups = [g for g, _ in groups_for(bb)]
    hdr = ["method", "tier"] + [SETS_LABEL[s] for s in TEST] + [g.upper() if g != "overall" else "overall"
                                                                for g in groups] + \
          ["AUROC (overall)", "F1 val-thr (mean)", "F1 always-pos (mean)", f"Δ PR-AUC {arms.get(P, {}).get('label', 'InStep')} − row [95% CI]"]
    rows, notes = [], []
    # Tier B
    ours_by_method = {}
    for k, m in PAPER_MAP.get(bb, {}).items():
        if m and k in M:
            ours_by_method.setdefault(m, []).append(k)
    if PN.methods(bb):
        rows.append(["**Tier B: ReProbe paper " + PN.TABLES[bb]["table"] + " (PR-AUC as printed)**"] + [""] * (len(hdr) - 1))
        flagged = False
        for meth in PN.methods(bb):
            p = PN.per_set(bb, meth)
            cand = sorted(ours_by_method.get(meth, []), key=lambda k: arms[k]["kind"] == "stored")
            cells = []
            for s in TEST:
                c = f3(p.get(s))
                ok = next((k for k in cand if s in M[k]), None)
                if ok and abs(M[ok][s]["ap"] - p[s]) > 0.01:
                    c += f" [{M[ok][s]['ap']:.3f}]"
                    flagged = True
                cells.append(c)
            gcells = [f3(p.get(g)) for g in groups]
            rows.append([meth, "B"] + cells + gcells + ["-", "-", "-", "-"])
        if flagged:
            notes.append("[x]: our re-score of the same method on the same released traces, where it differs from the "
                         "printed cell by more than 0.01.")
        if PN.TABLES[bb]["column_fix"]:
            notes.append("Table 5 columns Meetings/Calendar and StrategyQA/ScienceQA are printed swapped; they are "
                         "shown here under the set the released scores identify (see the consistency check).")
    rows.append(["**Tier A: our scores (seed mean)**"] + [""] * (len(hdr) - 1))
    order = [k for k in arms if k in M and any(s in M[k] for s in TEST)]
    order = [P] + [k for k in order if k != P] if P in order else order
    for k in order:
        m = M[k]
        sd = m.get("overall", {}).get("ap_seed_sd")
        ov = f3(m.get("overall", {}).get("ap"))
        cells = [f3(m[s]["ap"]) if s in m else "-" for s in TEST]
        gcells = [f3(m[g]["ap"]) if g in m else "-" for g in groups]
        if sd is not None and m["overall"].get("ap_seeds") and len(m["overall"]["ap_seeds"]) > 1:
            gcells[groups.index("overall")] = f"{ov} ± {sd:.3f}"
        pr = pairs.get((P, k), {}).get("overall", {}).get("ap") if k != P else None
        delta = "ref" if k == P else (fd(pr) if pr else "-")
        meth = PAPER_MAP.get(bb, {}).get(k)
        lab = arms[k]["label"] + (f" (paper {PN.per_set(bb, meth)['overall']:.3f})"
                                  if meth and PN.per_set(bb, meth) else "")
        rows.append([lab, "A"] + cells + gcells + [f3(m.get("overall", {}).get("auroc")),
                                                  f3(m.get("overall", {}).get("val_f1")),
                                                  f3(m.get("overall", {}).get("trivial_f1")), delta])
    notes += ["Overall = mean of the 8 per-set PR-AUCs; ± = SD over seeds of the overall PR-AUC.",
              "F1 columns are means over the 8 sets of per-set F1 at the threshold that maximises F1 on the "
              f"validation split ({VAL_LABEL[bb]} labels) and of the always-positive F1 (same prevalence for every row).",
              f"Δ: paired trace bootstrap ({R['n_boot']} resamples, shared across arms), mean over the 8 sets."]
    if bb == "phi4":
        notes.append("t9 = the five released sets of the paper's Table 9 (MATH, GSM8K, ProofNet, StrategyQA, ScienceQA); "
                     "the paper's planning columns come from traces that are not released.")
    return {"id": f"table1_{bb}", "title": f"ReProbe step benchmark, {BB_LABEL[bb]}: PR-AUC of incorrect steps",
            "header": hdr, "rows": rows, "notes": notes}


def table_pb(bb: str, R: dict, metric: str) -> dict | None:
    PB_ = R.get("pb", {})
    if not PB_:
        return None
    pp = {(p["a"], p["b"]): p for p in R.get("pb_pairs", [])}
    subs = list(R["pb_subsets"])
    refs = [r for r in ("instep_prm", "instep_ds") if r in PB_]
    hdr = ["method"] + subs + ["mean", "95% CI"] + [f"Δ {R['arms'][r]['label']} − row" for r in refs]
    rows = []
    order = sorted(PB_, key=lambda k: -PB_[k][f"{metric}_mean"])
    for k in order:
        r = PB_[k]
        cells = [f3(r["per_subset"][s][metric]) for s in subs]
        c = R.get("pb_ci", {}).get(k, {}).get(f"{metric}_mean")
        row = [R["arms"][k]["label"]] + cells + [f3(r[f"{metric}_mean"]), f"[{c[0]:.3f}, {c[1]:.3f}]" if c else "-"]
        for ref in refs:
            row.append("ref" if k == ref else fd(pp.get((ref, k), {}).get(f"{metric}_mean")))
        rows.append(row)
    prev = {s: R["pb_subset_stats"][s] for s in subs}
    notes = ["Items: every step of correct traces and the steps of erroneous traces up to and including the "
             "labelled first error (later steps dropped); positive = the first-error step.",
             "Positives / items per subset: " + ", ".join(f"{s} {v['n_pos']}/{v['n_items']}" for s, v in prev.items()),
             f"CIs and Δ: trace bootstrap within subset ({R['pb_n_boot']} resamples, shared across arms)."]
    name = "AUROC" if metric == "auroc" else "PR-AUC"
    return {"id": f"table2_{bb}_{metric}", "title": f"ProcessBench first-error {name}, {BB_LABEL[bb]} backbone",
            "header": hdr, "rows": rows, "notes": notes}


def table_calib20(bb: str, R: dict) -> dict | None:
    C = R.get("calib20", {})
    if not C:
        return None
    subs = list(next(iter(C.values()))["per_subset"])
    rows = [[R["arms"][k]["label"]] + [f3(v["per_subset"][s]) for s in subs] + [f3(v["calib20_mean"])]
            for k, v in sorted(C.items(), key=lambda kv: -kv[1]["calib20_mean"])]
    return {"id": f"tableA_calib20_{bb}", "title": f"ProcessBench F1_PB at calib-20 (appendix), {BB_LABEL[bb]}",
            "header": ["method"] + subs + ["mean"], "rows": rows,
            "notes": ["calib-20: threshold grid-maximised on 20 held-out traces per subset, applied to the rest, "
                      "mean over splits (scripts/merge_rep_grid_leaderboard.py)."]}


def table_ladder(R: dict) -> dict | None:
    M = R["metrics"]
    present = [(k, lab) for k, lab in LADDER if k in M and "overall" in M[k]]
    if not present:
        return None
    lp = {(p["a"], p["b"]): p for p in R.get("ladder_pairs", [])}
    rows = []
    prev = None
    for k, lab in present + ([LADDER_SIDE] if LADDER_SIDE[0] in M and "overall" in M[LADDER_SIDE[0]] else []):
        m = M[k]["overall"]
        c = R["arm_ci"].get(k, {}).get("overall", {}).get("ap")
        if k == LADDER_SIDE[0]:
            d = fd(lp.get(("instep_ds", k), {}).get("overall", {}).get("ap"))
            dl = f"InStep − this: {d}"
        else:
            d = lp.get((k, prev), {}).get("overall", {}).get("ap") if prev else None
            dl = f"vs {R['arms'][prev]['label']}: {fd(d)}" if prev else "-"
            prev = k
        rows.append([lab, f3(m["ap"]) + (f" ± {m['ap_seed_sd']:.3f}" if m.get("ap_seeds") and len(m["ap_seeds"]) > 1 else ""),
                     f"[{c[0]:.3f}, {c[1]:.3f}]" if c else "-", f3(m["auroc"]), f3(m.get("val_f1")),
                     str(M[k]["math"]["n_seeds"]) if "math" in M[k] else "-", dl])
    PB_ = R.get("pb", {})
    lpp = {(p["a"], p["b"]): p for p in R.get("pb_pairs", [])}
    for k, lab in (("emb_prm", "PB: read token, embedding only (PRM800K)"), ("instep_prm", "PB: InStep (PRM800K)")):
        if k in PB_:
            c = R.get("pb_ci", {}).get(k, {}).get("auroc_mean")
            d = lpp.get(("instep_prm", "emb_prm"), {}).get("auroc_mean") if k == "instep_prm" else None
            rows.append([lab, f3(PB_[k]["auroc_mean"]) + " (AUROC)", f"[{c[0]:.3f}, {c[1]:.3f}]" if c else "-",
                         "-", "-", str(PB_[k]["n_seeds"]), f"vs embedding only: {fd(d)}" if d else "-"])
    return {"id": "table_ladder", "title": "Ladder on Qwen3-8B (DS training data): ReProbe overall PR-AUC",
            "header": ["rung", "overall PR-AUC ± seed SD", "95% CI", "AUROC", "F1 val-thr", "seeds", "Δ PR-AUC [95% CI]"],
            "rows": rows, "notes": ["Δ: paired trace bootstrap of each rung minus the previous rung that exists.",
                                    "PB rows: ProcessBench mean first-error AUROC of the PRM800K-trained tokens."]}


def table_f1(bb: str, R: dict) -> dict:
    M = R["metrics"]
    any_ = next((M[k] for k in M if all(s in M[k] for s in TEST)), None)
    rows = []
    if any_:
        rows.append(["always-positive F1"] + [f3(any_[s]["trivial_f1"]) for s in TEST])
        rows.append(["prevalence (incorrect)"] + [f3(any_[s]["prevalence"]) for s in TEST])
    for k in M:
        if any(s in M[k] for s in TEST):
            rows.append([R["arms"][k]["label"]] + [
                (f"{M[k][s]['val_f1']:.3f} / {M[k][s]['oracle_f1']:.3f}" if "val_f1" in M[k][s]
                 else f"- / {M[k][s]['oracle_f1']:.3f}") if s in M[k] else "-" for s in TEST])
    return {"id": f"tableA_f1_{bb}", "title": f"Per-set F1 of the incorrect class, {BB_LABEL[bb]}: val-threshold / oracle",
            "header": ["method"] + [SETS_LABEL[s] for s in TEST], "rows": rows,
            "notes": ["Compare F1 only within a column, against that column's always-positive F1 (prevalence differs by set)."]}


def table_cost() -> dict:
    rows = []
    for c in COST:
        p = c["params"]
        ps = "0" if p == 0 else (f"~{p / 1e9:.1f}B" if p >= 1e9 else f"{p:,}")
        rows.append([c["method"], ps, "-" if c["ms"] is None else f"{c['ms']:.1f}", c["note"]])
    return {"id": "table3_cost", "title": "Cost on Qwen3-8B (trainable parameters added; ms per MATH-500 trajectory, batch 1)",
            "header": ["method", "params", "ms / trajectory", "note"], "rows": rows,
            "notes": ["Existing measurements (read_token_v1 Stage H, ReProbe speed runs); not re-timed in V1."]}


def table_consistency(rows: list) -> dict:
    bad = [r for r in rows if not r["ok"]]
    out = [[BB_LABEL[r["backbone"]], r["method"], r["set"], f3(r["ours"]), f3(r["paper"]), f"{r['diff']:+.3f}",
            r.get("known", "") or "UNEXPLAINED"] for r in bad]
    return {"id": "tableA_consistency", "title": "Transcribed paper PR-AUC vs AP recomputed from ReProbe's stored scores",
            "header": ["backbone", "method", "set", "stored → AP", "paper", "diff", "reason"], "rows": out,
            "notes": [f"{sum(r['ok'] for r in rows)} of {len(rows)} (method, set) cells agree within 0.002; "
                      "the rows above are the ones that do not."]}


def table_abstention(R: dict) -> dict | None:
    A = R.get("abstention", {})
    if not A:
        return None
    rows = [[R["arms"][k]["label"], f3(v["aurc"]), f3(v["risk_at"]["0.5"]), f3(v["risk_at"]["0.8"]),
             f3(v["base_rate"]), str(v["n_seeds"])] for k, v in sorted(A.items(), key=lambda kv: kv[1]["aurc"])]
    return {"id": "table_abstention", "title": "Abstention on the 8 test sets pooled, Qwen3-8B (trace level)",
            "header": ["method", "AURC (lower better)", "risk @ 50% coverage", "risk @ 80% coverage",
                       "risk @ 100%", "seeds"], "rows": rows,
            "notes": ["A trace is flawed if any labelled step is incorrect; trace score = max step P(incorrect); "
                      "the lowest-scoring traces are kept. AURC = mean selective risk over all coverages."]}


# ---------------------------------------------------------------------------- best-of-N
def run_bon(R: dict, v1: Path, final: Path, n_boot: int, missing: list) -> dict | None:
    bon = v1 / "bon"
    sets = ["math", "gsm8k", "strategyqa", "scienceqa", "calendar"]
    if not all((bon / f"{s}.graded.jsonl").exists() and (bon / "inputs" / f"{s}.meta.jsonl").exists() for s in sets):
        missing.append("best-of-N: graded pools not found")
        return None
    specs = []
    for k, name in (("instep_ds", "InStep-DS"), ("instep_prm", "InStep-PRM800K"), ("instep_self", "InStep-self"),
                    ("rp_ds", "RP-ds"), ("rp_rel_ds", "RP-rel-ds")):
        a = R["arms"].get(k)
        if not a or a["kind"] not in ("rp", "rt"):
            continue
        ok = [d for d in a["dirs"] if all(
            (Path(d) / (f"bon_{s}.jsonl" if a["kind"] == "rp" else f"bon_{s}.shard0.npz")).exists() for s in sets)]
        if ok:
            specs.append((name, f"{name}={a['kind']}:{','.join(ok)}"))
        else:
            missing.append(f"best-of-N: {k} has no complete bon_* scores")
    meta0 = jl(bon / "inputs" / f"{sets[0]}.meta.jsonl")[0].get("stored", {})
    for col, name in (("Qwen/Qwen2.5-Math-PRM-7B", "PRM-7B"), ("Qwen/Qwen2.5-Math-7B-PRM800K", "PRM800K-7B")):
        if col in meta0:
            specs.append((name, f"{name}=stored:{col}"))
    names = [n for n, _ in specs]
    if not any(n.startswith("InStep") for n in names):
        missing.append("best-of-N: no InStep arm with bon scores")
        return None
    pairs = [f"{a}-{b}" for a in ("InStep-DS", "InStep-PRM800K") if a in names
             for b in ("RP-ds", "RP-rel-ds", "PRM-7B", "majority") if b in names + ["majority"]]
    out = final / "bon"
    cmd = [sys.executable, str(ROOT / "scripts/reprobe_cmp/bon_analyze.py"), "--bon", str(bon), "--out", str(out),
           "--n_boot", str(n_boot), "--arm", *[s for _, s in specs]] + (["--pairs", *pairs] if pairs else [])
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if p.returncode != 0:
        missing.append(f"best-of-N: bon_analyze.py failed: {p.stderr.strip()[-400:]}")
        return None
    return {"summary": json.loads((out / "summary.json").read_text()), "md": (out / "summary.md").read_text(),
            "arms": [s for _, s in specs]}


# ---------------------------------------------------------------------------- main
def analyse_backbone(bb: str, a, missing: list, log) -> dict | None:
    R_root, specs = candidate_arms(bb, a.out_root, a.v1, a.v2)
    for o in a.override:
        b, rest = o.split(":", 1)
        k, paths = rest.split("=", 1)
        if b != bb:
            continue
        hit = [s for s in specs if s["key"] == k]
        if hit:
            hit[0]["paths"] = paths.split(",")
            hit[0].pop("stored", None)
            hit[0]["label"] += " [override]"
        else:
            specs.append(dict(key=k, label=k + " [override]", family="other", paths=paths.split(",")))
    data = R_root / "data"
    if not (data / "test").is_dir():
        missing.append(f"{bb}: data root {data} not found, backbone skipped")
        return None
    t0 = time.time()
    tests = test_sets(data)
    val = val_set(data, VAL_LABEL[bb]) if (data / "rp" / "val.traces.jsonl").exists() else None
    if val is None:
        missing.append(f"{bb}: no validation traces, no val-threshold F1")
    pb_items = PBM.load_items(data) if (bb == "qwen3_8b" or a.pb_all) else {}
    if bb == "qwen3_8b" and len(pb_items) != 4:
        missing.append(f"{bb}: ProcessBench traces incomplete ({sorted(pb_items)})")
        pb_items = {}
    stored = load_stored(data, {s["stored"] for s in specs if "stored" in s} | set(PN.STORED.get(bb, {}).values()))
    arms = {}
    for s in specs:
        arm = load_arm(s, data, tests, val, pb_items, stored, missing)
        if arm is not None:
            arms[s["key"]] = arm
    log(f"[{bb}] loaded {len(arms)} arms in {time.time() - t0:.0f}s: {', '.join(arms)}")
    groups = groups_for(bb)
    metrics = {k: arm_metrics(v, tests, groups) for k, v in arms.items() if v["test"]}
    # bootstraps (shared resamples)
    t0 = time.time()
    _G[bb] = {"arms": arms, "tests": tests, "boot": make_test_boot(tests, a.n_boot)}
    boots = run_pool(_boot_test_arm, [(bb, k) for k in metrics], a.workers)
    log(f"[{bb}] test bootstrap of {len(boots)} arms x {a.n_boot} in {time.time() - t0:.0f}s")
    B = {k: boots[(bb, k)] for k in metrics}
    pairs = []
    for A_, B_ in pairs_for(bb):
        if A_ in metrics and B_ in metrics:
            pairs.append(pair_result(A_, B_, metrics[A_], metrics[B_], B[A_], B[B_], groups))
        else:
            missing.append(f"{bb}: pair {A_} − {B_} skipped ({', '.join(x for x in (A_, B_) if x not in metrics)} missing)")
    ladder_pairs = []
    if bb == "qwen3_8b":
        present = [k for k, _ in LADDER if k in metrics and "overall" in metrics[k]]
        for x, y in zip(present[1:], present[:-1]):
            ladder_pairs.append(pair_result(x, y, metrics[x], metrics[y], B[x], B[y], groups))
        if "instep_ds" in metrics and LADDER_SIDE[0] in metrics:
            ladder_pairs.append(pair_result("instep_ds", LADDER_SIDE[0], metrics["instep_ds"],
                                            metrics[LADDER_SIDE[0]], B["instep_ds"], B[LADDER_SIDE[0]], groups))
        for k, _ in LADDER + [LADDER_SIDE]:
            if k not in metrics:
                missing.append(f"{bb}: ladder rung {k} missing")
    arm_cis = {k: arm_ci(metrics[k], B[k], groups) for k in metrics}
    res = {"data": str(data), "val_label": VAL_LABEL[bb], "n_boot": a.n_boot,
           "arms": {k: {"key": k, "label": v["label"], "family": v["family"], "kind": v["kind"], "dirs": v["dirs"],
                        "seeds_per_set": {s: len(x) for s, x in v["test"].items()},
                        "val_thresholds": v["thr"], "has_pb": v["pb"] is not None} for k, v in arms.items()},
           "metrics": metrics, "pairs": pairs, "ladder_pairs": ladder_pairs, "arm_ci": arm_cis,
           "test_stats": {s: {"n_steps": int(len(tests[s]["y"])), "n_traces": int(len(np.unique(tests[s]["trace"]))),
                              "prevalence": float(tests[s]["y"].mean())} for s in TEST}}
    # ProcessBench
    pb_arms = {k: v for k, v in arms.items() if v["pb"] is not None}
    if pb_items and pb_arms:
        t0 = time.time()
        res["pb_subsets"] = list(pb_items)
        res["pb_subset_stats"] = {s: {"n_items": int(len(v["y"])), "n_pos": int(v["y"].sum())} for s, v in pb_items.items()}
        res["pb"] = {k: PBM.point(pb_items, v["pb"]) for k, v in pb_arms.items()}
        _G[bb]["pb_items"] = pb_items
        _G[bb]["pb_boot"] = PBM.boot_indices(pb_items, a.pb_boot)
        pbb = run_pool(_boot_pb_arm, [(bb, k) for k in pb_arms], a.workers)
        pbb = {k: pbb[(bb, k)] for k in pb_arms}
        res["pb_n_boot"] = a.pb_boot
        res["pb_ci"] = {k: {m: PBM.ci(v[m]) for m in v} for k, v in pbb.items()}
        res["pb_pairs"] = []
        for A_, B_ in PB_PAIRS:
            if A_ in pbb and B_ in pbb:
                r = PBM.pair_from_arrays(res["pb"][A_], res["pb"][B_], pbb[A_], pbb[B_])
                res["pb_pairs"].append({"a": A_, "b": B_, **r})
        log(f"[{bb}] ProcessBench: {len(pb_arms)} arms, bootstrap in {time.time() - t0:.0f}s")
        if a.calib20:
            t0 = time.time()
            from scripts.reprobe_cmp.analyze import pb_calib20, write_pb
            res["calib20"] = {}
            for k, v in pb_arms.items():
                if v["kind"] not in ("rp", "rt"):
                    continue
                dd = []
                for i, d in enumerate(v["dirs"]):
                    o = a.final / "pb_scores" / bb / k / f"seed{i}"
                    try:
                        if write_pb(v["kind"], d, data, o):
                            dd.append(str(o))
                    except Exception as e:  # noqa: BLE001
                        missing.append(f"{bb}: calib-20 {k} seed {i} skipped ({e!r})")
                if dd:
                    try:
                        res["calib20"][k] = pb_calib20(dd)
                    except Exception as e:  # noqa: BLE001
                        missing.append(f"{bb}: calib-20 {k} failed ({e!r})")
            log(f"[{bb}] calib-20 in {time.time() - t0:.0f}s")
    for k in ("instep_ds", "instep_prm", "prm800k", "prm7b", "passive", "maxprob", "ptrue") if bb == "qwen3_8b" else ():
        if k in arms and arms[k]["pb"] is None and pb_items:
            missing.append(f"{bb}: {k} has no complete ProcessBench scores")
    # abstention
    if bb == "qwen3_8b":
        res["abstention"] = {}
        best_prm = best_of(metrics, ("prm7b", "prm800k"))
        for k in ("instep_ds", "rp_ds", "rp_rel_ds", best_prm, "passive", "maxprob", "maxprob_stored"):
            if k and k in arms and k not in res["abstention"]:
                r = abstention(arms[k], tests)
                if r:
                    res["abstention"][k] = r
    res["consistency"] = consistency(bb, tests, stored)
    pl = a.out_root / "analysis" / f"per_layer_{bb}.json"
    if pl.exists():
        try:
            raw = json.loads(pl.read_text())
            res["per_layer"] = {"file": str(pl), "series": parse_per_layer(raw)}
            if not res["per_layer"]["series"]:
                missing.append(f"{bb}: {pl} has no series the loose reader understands (Fig 5 skipped)")
        except Exception as e:  # noqa: BLE001
            missing.append(f"{bb}: {pl} unreadable ({e!r})")
    elif bb == "qwen3_8b":
        missing.append(f"{bb}: per-layer file {pl} not found (Fig 5 skipped)")
    return res


def best_of(metrics: dict, keys) -> str | None:
    c = [k for k in keys if k in metrics and "overall" in metrics[k]]
    return max(c, key=lambda k: metrics[k]["overall"]["ap"]) if c else None


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", type=Path, default=PROJ / "instep_v1")
    p.add_argument("--v1", type=Path, default=PROJ / "reprobe_cmp_v1")
    p.add_argument("--v2", type=Path, default=PROJ / "reprobe_cmp_v2")
    p.add_argument("--final", type=Path, default=None, help="default: <out_root>/final")
    p.add_argument("--backbones", default=",".join(BACKBONES))
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--pb_boot", type=int, default=2000)
    p.add_argument("--bon_boot", type=int, default=2000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--no_calib20", dest="calib20", action="store_false")
    p.add_argument("--no_bon", dest="bon", action="store_false")
    p.add_argument("--no_figures", dest="figures", action="store_false")
    p.add_argument("--pb_all", action="store_true", help="ProcessBench on every backbone with data/pb (default: Qwen3-8B)")
    p.add_argument("--override", nargs="*", default=[],
                   help="bb:key=dir1,dir2 replaces (or adds) an arm's seed dirs, e.g. for a dry run")
    a = p.parse_args(argv)
    a.final = a.final or a.out_root / "final"
    a.final.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    logf = open(a.final / "paper_v1.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    missing, errors = [], []
    final = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "args": {k: str(v) for k, v in vars(a).items()},
             "git": _git(), "backbones": {}, "paper_numbers": {bb: {m: PN.per_set(bb, m) for m in PN.methods(bb)}
                                                                for bb in BACKBONES}}
    for bb in a.backbones.split(","):
        try:
            r = analyse_backbone(bb, a, missing, log)
            if r is not None:
                final["backbones"][bb] = r
        except Exception:  # noqa: BLE001
            errors.append(f"{bb}: {traceback.format_exc()}")
            log(f"[{bb}] FAILED\n{traceback.format_exc()}")
        _G.pop(bb, None)
    q = final["backbones"].get("qwen3_8b")
    if q and a.bon:
        try:
            final["bon"] = run_bon(q, a.v1, a.final, a.bon_boot, missing)
        except Exception:  # noqa: BLE001
            errors.append(f"bon: {traceback.format_exc()}")
    beam = a.v1 / "beam" / "analysis" / "summary.md"
    final["beam_md"] = beam.read_text() if beam.exists() else None
    if final["beam_md"] is None:
        missing.append(f"beam search summary {beam} not found")
    final["cost"] = COST
    final["missing"] = missing
    final["errors"] = errors

    tables = []
    for bb, R in final["backbones"].items():
        for fn in (lambda: table1(bb, R), lambda: table_pb(bb, R, "auroc"), lambda: table_pb(bb, R, "ap"),
                   lambda: table_calib20(bb, R), lambda: table_ladder(R) if bb == "qwen3_8b" else None,
                   lambda: table_abstention(R) if bb == "qwen3_8b" else None, lambda: table_f1(bb, R)):
            try:
                t = fn()
                if t:
                    tables.append(t)
            except Exception:  # noqa: BLE001
                errors.append(f"table ({bb}): {traceback.format_exc()}")
    tables.append(table_cost())
    cons = [r for R in final["backbones"].values() for r in R.get("consistency", [])]
    tables.append(table_consistency(cons))
    final["tables"] = tables

    figs = []
    if a.figures:
        try:
            from scripts.instep_v1 import figures as FIG
            figs, ferr = FIG.make_all(final, a.final)
            missing.extend(ferr)
        except Exception:  # noqa: BLE001
            errors.append(f"figures: {traceback.format_exc()}")
    final["figures"] = figs
    final["runtime_s"] = time.time() - t_start
    (a.final / "final.json").write_text(json.dumps(to_jsonable(final), indent=1))
    (a.final / "results.md").write_text(results_md(final, tables, figs))
    (a.final / "tables.tex").write_text("% generated by scripts/instep_v1/paper_v1.py\n% needs booktabs, graphicx\n\n"
                                        + "\n".join(tex_table(t) for t in tables))
    log(f"done in {time.time() - t_start:.0f}s: {len(tables)} tables, {len(figs)} figure files, "
        f"{len(missing)} missing items, {len(errors)} errors -> {a.final}")
    return 0


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=str(ROOT)).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def headline(final: dict) -> list[str]:
    out = []
    for bb, R in final["backbones"].items():
        M, P = R["metrics"], PRIMARY[bb]
        if P not in M or "overall" not in M[P]:
            out.append(f"- {BB_LABEL[bb]}: primary InStep arm ({P}) missing.")
            continue
        s = f"- {BB_LABEL[bb]}: {R['arms'][P]['label']} overall PR-AUC {M[P]['overall']['ap']:.3f}"
        for p in R["pairs"]:
            if p["a"] == P and p["b"] in ("rp_ds", "rp_q3", "rp_oss", "prm7b", "prm800k") and "overall" in p:
                d = p["overall"]["ap"]
                s += f"; − {R['arms'][p['b']]['label']} {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
        out.append(s + ".")
        if R.get("pb"):
            for k in ("instep_prm", "instep_ds"):
                if k in R["pb"]:
                    out.append(f"  ProcessBench {R['arms'][k]['label']}: mean first-error AUROC "
                               f"{R['pb'][k]['auroc_mean']:.3f}, PR-AUC {R['pb'][k]['ap_mean']:.3f}.")
    return out


def results_md(final: dict, tables: list, figs: list) -> str:
    L = ["# InStep paper V1: results", "",
         f"Generated {final['generated']} (git {final['git']}) by `scripts/instep_v1/paper_v1.py`; "
         f"runtime {final.get('runtime_s', 0):.0f} s. Every number is in `final.json`.", "", "## Headline", ""]
    L += headline(final) + [""]
    L += ["## What is missing", ""]
    if final["missing"] or final["errors"]:
        L += [f"- {m}" for m in final["missing"]]
        L += [f"- ERROR: {e.strip().splitlines()[-1]}" for e in final["errors"]]
    else:
        L.append("- nothing")
    L += ["", "## Tables", ""]
    L += [md_table(t) for t in tables]
    if final.get("bon"):
        L += ["## Best-of-N (Qwen3-8B pools, our grading)", "", final["bon"]["md"],
              "Arms: " + "; ".join(final["bon"]["arms"]), ""]
    if final.get("beam_md"):
        L += ["## Beam search (existing result, reprobe_cmp_v1/beam/analysis/summary.md)", "", final["beam_md"], ""]
    L += ["## Figures", ""] + [f"- `{f}`" for f in figs] + [""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
