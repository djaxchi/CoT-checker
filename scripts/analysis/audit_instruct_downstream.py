#!/usr/bin/env python3
"""Audit rescue accounting, score failures and held-out CPU-only selector repairs.

Uses the fixed-grader pools. Labels select policies only on other question folds.
All variants and cross-validation are exploratory, not a new untouched benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.downstream_audit import (AGGREGATIONS, RULES, aggregate, describe_pool,
                                         outcome, stable_fold, within_auc)
from scripts.generate_onpolicy_steps import split_into_steps

PRM = "prm_qwen25_math_7b"
BOUNDARY = "boundary_stats__mlp_h1024x2__seed42__gen"
STEPSTATS = "step_stats__mlp_h1024__seed43__gen"
D512 = "step_tokens__transformer_d512_l2_f2048_h8__seed44__gen"
LAST = "last_token__mlp_h1024__seed42__gen"
REPAIR_PROBES = (BOUNDARY, STEPSTATS, D512, LAST)
AUDIT_AGGREGATIONS = (*AGGREGATIONS, "worst_no_separators", "tail_half_worst")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_pool(root: Path, ds: str, cells: list[str]) -> tuple[list[dict], list[Path]]:
    stem = "tts_" + ds
    files = sorted(root.glob(stem + ".shard*_trajectories.jsonl"))
    rows = [r for f in files for r in read_rows(f) if r["gradeable"]]
    scores = {}
    for cell in cells:
        paths = sorted((root / "scores").glob(stem + "__" + cell + ".shard*.jsonl"))
        files += paths
        values = {}
        for f in paths:
            for r in read_rows(f):
                if r["traj_uid"] in values:
                    raise ValueError(f"Duplicate score: {cell} {r['traj_uid']}")
                values[r["traj_uid"]] = r
        if not values:
            raise ValueError(f"No scores: {cell}")
        scores[cell] = values
    conf_files = sorted(root.glob(stem + ".shard*_conf.jsonl"))
    files += conf_files
    conf = {r["traj_uid"]: r["rules"]["mean_token_conf"] for f in conf_files for r in read_rows(f)}
    by = defaultdict(list)
    for r in rows:
        if "answer_key" not in r:
            raise ValueError("Use the regraded_v2 pools with stored answer identity")
        by[r["fork_id"]].append(r)
    problems = []
    seen = set()
    for pid, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["traj_uid"])
        if len(rs) != 10 or len({r["traj_uid"] for r in rs}) != 10:
            raise ValueError(f"Incomplete/duplicate pool: {pid}")
        qhash = rs[0]["question_hash"]
        if qhash in seen:
            raise ValueError("Duplicate question hashes require a clustered extension")
        seen.add(qhash)
        p = describe_pool([r["answer_key"] for r in rs], [r["correct"] for r in rs])
        p.update({"pid": pid, "question_hash": qhash, "fold": stable_fold(qhash), "rows": rs,
                  "scores": {}, "span_bad": {}, "aggregates": {},
                  "tokens": np.array([r["n_gen_tokens"] for r in rs], float)})
        steps = [split_into_steps(r["solution"]) for r in rs]
        content = [np.array([not re.fullmatch(r"[-*_]{3,}|\$\$|\\[\[\]]", text.strip()) for text in span])
                   for span in steps]
        for cell in cells:
            atoms = [scores[cell][r["traj_uid"]] for r in rs]
            p["scores"][cell] = [a["scores"] for a in atoms]
            p["span_bad"][cell] = [not a.get("span_ok", True) for a in atoms]
            for agg in AGGREGATIONS:
                p["aggregates"][(cell, agg)] = np.array([aggregate(a["scores"], agg) for a in atoms])
            if any(len(a["scores"]) != len(span) for a, span in zip(atoms, steps)):
                raise ValueError("Step count mismatch")
            p["aggregates"][(cell, "worst_no_separators")] = np.array([
                max(np.array(a["scores"])[mask]) if mask.any() else max(a["scores"])
                for a, mask in zip(atoms, content)])
            p["aggregates"][(cell, "tail_half_worst")] = np.array([
                max(a["scores"][len(a["scores"])//2:]) for a in atoms])
        p["aggregates"][("confidence", "worst")] = -np.array([conf[r["traj_uid"]] for r in rs])
        problems.append(p)
    return problems, files


def interval(values: list | np.ndarray, seed: int = 731, draws: int = 4000) -> dict:
    """Problem bootstrap, conditional on the fixed fitted/selected policies."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = np.concatenate([x[rng.integers(len(x), size=(min(200, draws-start), len(x)))].mean(1)
                            for start in range(0, draws, 200)])
    return {"mean": float(x.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "n": len(x)}


def policy_metrics(ps: list[dict], values: np.ndarray) -> dict:
    base = np.array([p["majority"] for p in ps])
    diff = values-base
    return {"accuracy": float(values.mean()), "gain_pp": interval(100*diff),
            "rescues_expected": float(np.maximum(diff, 0).sum()),
            "damage_expected": float(np.maximum(-diff, 0).sum())}


def score_diagnostic(ps: list[dict], cell: str) -> dict:
    output = {}
    for subset in ["strict_minority_correct", "vote_tie_with_correct", "majority_correct_mixed", "legacy_rescuable"]:
        chosen = [p for p in ps if p["case"] == subset or
                  (subset == "legacy_rescuable" and p["case"] in ("strict_minority_correct", "vote_tie_with_correct"))]
        output[subset] = {"n": len(chosen),
                          "top1": interval([outcome(p, p["aggregates"][(cell, "worst")], "rerank") for p in chosen]),
                          "auc": interval([within_auc(p["correct"], p["aggregates"][(cell, "worst")]) for p in chosen]),
                          "random_sample": float(np.mean([p["correct"].mean() for p in chosen])) if chosen else None}
    lengths, risks, labels, first, last, saturated, span_bad = [], [], [], [], [], [], []
    for p in ps:
        for i, s in enumerate(p["scores"][cell]):
            arr = np.array(s)
            worst = arr.max()
            tied = np.flatnonzero(arr == worst)
            lengths.append(len(s)); risks.append(worst); labels.append(p["correct"][i])
            first.append(float(0 in tied)/len(tied)); last.append(float(len(s)-1 in tied)/len(tied))
            saturated.append(worst == 1.); span_bad.append(p["span_bad"][cell][i])
    labels = np.asarray(labels)
    output["scores"] = {"fraction_max_exactly_one": float(np.mean(saturated)),
                        "fraction_worst_at_first": float(np.mean(first)),
                        "fraction_worst_at_last": float(np.mean(last)),
                        "bad_span_fraction": float(np.mean(span_bad)),
                        "length_risk_spearman_correct": float(spearmanr(np.array(lengths)[labels], np.array(risks)[labels]).statistic),
                        "mean_steps_correct": float(np.mean(np.array(lengths)[labels])),
                        "mean_steps_wrong": float(np.mean(np.array(lengths)[~labels])),
                        "correct_fraction_max_exactly_one": float(np.mean(np.array(saturated)[labels]))}
    return output


def build_policies(ps: list[dict]) -> tuple[dict, dict]:
    values = {"majority": np.array([p["majority"] for p in ps])}
    calls = {"majority": np.zeros(len(ps))}
    for cell in [*REPAIR_PROBES, PRM]:
        for agg in AUDIT_AGGREGATIONS:
            for rule in RULES:
                key = f"{cell}|{agg}|{rule}"
                values[key] = np.array([outcome(p, p["aggregates"][(cell, agg)], rule) for p in ps])
                if cell == PRM:
                    calls[key] = np.array([
                        0 if len(p["groups"]) <= 1 else
                        (sum(p["counts"][p["top"]]) if len(p["top"]) > 1 else 0)
                        if rule == "tiebreak" else
                        (10 if len(p["groups"]) > 1 and (rule != "gate_margin02" or p["margin"] <= .2) else 0)
                        for p in ps], float)
    # Fair PRM baseline can skip unanimous answer pools without affecting accuracy.
    # These read counts are hypothetical deployment calls, not measured wall time.
    for agg in ["worst"]:
        for rule in ["rerank", "wvote", "tiebreak"]:
            calls[f"{PRM}|{agg}|{rule}"] = np.array([
                0 if len(p["groups"]) <= 1 else
                (sum(p["counts"][p["top"]]) if len(p["top"]) > 1 else 0) if rule == "tiebreak" else
                (10 if len(p["groups"]) > 1 else 0) for p in ps], float)
    for margin in [.0, .2, .4, 1.]:
        for gate in ["vote_only", "probe_disagrees"]:
            for prm_rule in ["rerank", "wvote"]:
                key = f"cascade|{gate}|margin{margin:g}|{prm_rule}"
                chosen, reads = [], []
                for p in ps:
                    s = p["aggregates"][(BOUNDARY, "worst")]
                    best_indices = np.flatnonzero(s == s.min())
                    majority_indices = set(np.concatenate([p["groups"][i] for i in p["top"]]))
                    disagrees = any(i not in majority_indices for i in best_indices) or len(p["top"]) > 1
                    call = len(p["groups"]) > 1 and p["margin"] <= margin and (gate == "vote_only" or disagrees)
                    chosen.append(outcome(p, p["aggregates"][(PRM, "worst")], prm_rule) if call else p["majority"])
                    reads.append(10 if call else 0)
                values[key] = np.array(chosen); calls[key] = np.array(reads, float)
    return values, calls


def crossfit(pools: dict[str, list[dict]], matrices: dict, calls: dict) -> dict:
    """Select on four question folds, evaluate the fifth in both temperatures.

    Both temperatures contribute training data; all copies of a held-out question
    stay out. Include majority as a fallback, and use a small fixed candidate menu.
    """
    keys = list(matrices["10"])
    groups = {
        "probe_aggregation_rerank": [k for k in keys if any(k.startswith(c+"|") for c in REPAIR_PROBES) and k.endswith("|rerank")],
        "probe_conservative": ["majority"] + [k for k in keys if any(k.startswith(c+"|") for c in REPAIR_PROBES) and
                                                           k.split("|")[-1] in ("tiebreak", "safe_wvote", "gate_margin02")],
        "prm_aggregation": ["majority"] + [k for k in keys if k.startswith(PRM+"|")],
        "cascade_cost005": ["majority"] + [k for k in keys if k.startswith("cascade|")],
    }
    result = {}
    for name, candidates in groups.items():
        pred = {t: np.zeros(len(ps)) for t, ps in pools.items()}
        reads = {t: np.zeros(len(ps)) for t, ps in pools.items()}
        choices = []
        for fold in range(5):
            utility = []
            for k in candidates:
                chunks = []
                for t, ps in pools.items():
                    mask = np.array([p["fold"] != fold for p in ps])
                    y = matrices[t][k][mask]
                    if name == "cascade_cost005":
                        y = y - .005*calls[t][k][mask]/10
                    chunks.append(y)
                utility.append(np.concatenate(chunks).mean())
            choice = candidates[int(np.argmax(utility))]
            choices.append({"fold": fold, "policy": choice})
            for t, ps in pools.items():
                mask = np.array([p["fold"] == fold for p in ps])
                pred[t][mask] = matrices[t][choice][mask]
                if choice in calls[t]:
                    reads[t][mask] = calls[t][choice][mask]
        result[name] = {"choices": choices, "n_candidates": len(candidates), "pools": {}}
        for t, ps in pools.items():
            prm = matrices[t][f"{PRM}|worst|rerank"]
            result[name]["pools"][t] = {**policy_metrics(ps, pred[t]),
                                        "vs_prm_rerank_pp": interval(100*(pred[t]-prm)),
                                        "prm_traces_per_problem": float(reads[t].mean())}
    return result


def pair_analysis(ps: list[dict]) -> dict:
    """Exhaust all 45 two-sample subsets; bootstrap problems, not pairs."""
    cols = defaultdict(list)
    for p in ps:
        local = defaultdict(list)
        for pair in itertools.combinations(range(10), 2):
            ii = list(pair)
            sub = describe_pool([p["rows"][i]["answer_key"] for i in ii], p["correct"][ii].tolist())
            local["majority"].append(sub["majority"])
            for cell in [BOUNDARY, STEPSTATS, D512, PRM, "confidence"]:
                s = p["aggregates"][(cell, "worst")][ii]
                local[cell].append(outcome(sub, s, "tiebreak"))
            local["prm_reads"].append(2 if len(sub["top"]) > 1 else 0)
        for k, v in local.items():
            cols[k].append(np.mean(v))
    result = {}
    for cell in [BOUNDARY, STEPSTATS, D512, PRM, "confidence"]:
        y = np.array(cols[cell])
        result[cell] = {"accuracy": float(y.mean()), "gain_pp": interval(100*(y-np.array(cols["majority"]))),
                        "vs_prm_pp": interval(100*(y-np.array(cols[PRM])))}
    result["majority"] = float(np.mean(cols["majority"]))
    result["prm_reads_per_problem"] = float(np.mean(cols["prm_reads"]))
    return result


def pairwise_repair(pools: dict[str, list[dict]]) -> dict:
    """Fit a small within-question ranker, with length/vote ablations.

    Every mixed training pool contributes equal total pair weight. Mirrored
    pairs balance the binary labels. C=1 is fixed, without held-out tuning.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    names = [f"{c}|{a}" for c in REPAIR_PROBES for a in ("worst", "mean", "q90", "last")]
    names += ["log_steps", "log_tokens", "confidence", "vote_share"]
    rows = [(t, p) for t, ps in pools.items() for p in ps]
    features = []
    for _, p in rows:
        columns = []
        for c in REPAIR_PROBES:
            for a in ("worst", "mean", "q90", "last"):
                s = np.clip(p["aggregates"][(c, a)], 1e-7, 1-1e-7)
                columns.append(np.log(s)-np.log1p(-s))
        share = np.zeros(10)
        for g in p["groups"]:
            share[g] = len(g)/10
        columns += [np.log1p([len(s) for s in p["scores"][BOUNDARY]]),
                    np.log1p(p["tokens"]), p["aggregates"][("confidence", "worst")], share]
        features.append(np.column_stack(columns))
    variants = {"length_only": [16, 17], "scores_only": list(range(16)),
                "scores_length_conf": list(range(19)), "scores_length_conf_votes": list(range(20))}
    result = {}
    for name, use in variants.items():
        pred = np.zeros(len(rows)); coef = []
        for fold in range(5):
            X, y, w = [], [], []
            for j, (_, p) in enumerate(rows):
                if p["fold"] == fold:
                    continue
                pos, neg = np.flatnonzero(p["correct"]), np.flatnonzero(~p["correct"])
                if not len(pos) or not len(neg):
                    continue
                diff = (features[j][pos][:, None, use]-features[j][neg][None, :, use]).reshape(-1, len(use))
                X.extend([diff, -diff]); y.extend([np.ones(len(diff)), np.zeros(len(diff))])
                w.extend([np.full(len(diff), .5/len(diff))]*2)
            X, y, w = np.vstack(X), np.concatenate(y), np.concatenate(w)
            scaler = StandardScaler(with_mean=False).fit(X)
            model = LogisticRegression(C=1., fit_intercept=False, max_iter=2000, random_state=731)
            model.fit(scaler.transform(X), y, sample_weight=w)
            coef.append(dict(zip([names[k] for k in use], model.coef_[0].tolist())))
            for j, (_, p) in enumerate(rows):
                if p["fold"] == fold:
                    quality = model.decision_function(scaler.transform(features[j][:, use]))
                    pred[j] = outcome(p, -quality, "rerank")
        result[name] = {"features": [names[k] for k in use], "fold_coefficients_standardized": coef, "pools": {}}
        for t, ps in pools.items():
            mask = np.array([tag == t for tag, _ in rows])
            reference = np.array([outcome(p, p["aggregates"][(PRM, "worst")], "rerank") for p in ps])
            result[name]["pools"][t] = {**policy_metrics(ps, pred[mask]),
                                        "vs_prm_rerank_pp": interval(100*(pred[mask]-reference))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, default=ROOT/"cot-checker-results/tts_regraded_v2")
    parser.add_argument("--out", type=Path, default=ROOT/"results/instruct_downstream_v1/audit_v1")
    parser.add_argument("--allow-baseline-change", action="store_true",
                        help="For a newly regraded pool; record the change from original majority.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    best = json.loads((ROOT/"results/instruct_downstream_v1/best_seed.json").read_text())
    cells = [f"{c['rep']}__{c['learner'].replace(':','_').replace(',','_')}__seed{c['seed']}__gen" for c in best]+[PRM]
    report, sources, prediction_rows = {}, set(), []
    for ds in ["math500", "gsm8k"]:
        pools, matrices, calls = {}, {}, {}
        report[ds] = {}
        for t in ["10", "07"]:
            print(f"Loading {ds} t{t}", flush=True)
            ps, files = load_pool(args.pool_root/f"instruct_t{t}", ds, cells)
            sources.update(f.resolve() for f in files)
            pools[t] = ps
            matrices[t], calls[t] = build_policies(ps)
            base = np.array([p["majority"] for p in ps])
            oracle = np.array([p["oracle"] for p in ps])
            diagnostics = {c: score_diagnostic(ps, c) for c in cells}
            summary = json.loads((ROOT/f"results/instruct_downstream_v1/summary_t{t}.json").read_text())
            if not args.allow_baseline_change and not np.isclose(base.mean(), summary["majority"][f"{ds}|10"]):
                raise ValueError("N=10 majority does not reproduce saved summary")
            r = {"n_problems": len(ps), "cases": dict(Counter(p["case"] for p in ps)),
                 "majority": float(base.mean()), "oracle": float(oracle.mean()),
                 "majority_change_vs_original_pp": float(100*(base.mean()-summary["majority"][f"{ds}|10"])),
                 "oracle_gap_problem_equivalents": float((oracle-base).sum()),
                 "scorer_diagnostics": diagnostics,
                 "policies": {k: policy_metrics(ps, y) for k, y in matrices[t].items()},
                 "prm_traces_per_problem": {k: float(v.mean()) for k, v in calls[t].items()},
                 "exact_N2": pair_analysis(ps)}
            report[ds][t] = r
            for i, p in enumerate(ps):
                prediction_rows.append({"dataset": ds, "temperature": t, "problem_id": p["pid"],
                                        "question_hash": p["question_hash"], "fold": p["fold"], "case": p["case"],
                                        "correct_samples": int(p["correct"].sum()),
                                        "majority": p["majority"], "oracle": p["oracle"],
                                        "policies": {k: float(v[i]) for k, v in matrices[t].items()}})
            print(ds, t, r["cases"], "oracle gap", r["oracle_gap_problem_equivalents"], flush=True)
        report[ds]["crossfit"] = crossfit(pools, matrices, calls)
        print(f"Fitting grouped pairwise repairs: {ds}", flush=True)
        report[ds]["pairwise_repair"] = pairwise_repair(pools)
    meta = {"created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "bootstrap_seed": 731, "bootstrap_draws": 4000, "folds": 5,
            "source_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(sources)},
            "code_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                            [Path(__file__), ROOT/"src/analysis/downstream_audit.py"]},
            "limitations": ["Exploratory policies chosen after prior downstream inspection.",
                            "Same question is held out across both temperatures during policy selection.",
                            "Crossfit bootstrap conditions on selected fold policies, excluding selection uncertainty.",
                            "Exact score ties averaged, so small differences from 32 random order estimates are expected.",
                            "Zero weighted-vote totals fall back to majority in this audit.",
                            "Cascade costs count hypothetical PRM trace reads, not measured GPU latency."]}
    (args.out/"audit.json").write_text(json.dumps({"metadata": meta, "results": report}, indent=2)+"\n")
    (args.out/"per_problem.jsonl").write_text("\n".join(json.dumps(r) for r in prediction_rows)+"\n")
    print(args.out/"audit.json", flush=True)


if __name__ == "__main__":
    main()
