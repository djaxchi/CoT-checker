#!/usr/bin/env python3
"""Mixed-pair accuracy of verifiers, token confidence and their sum (pair_complementarity_v0).

Plan and prespecified reading: docs/pair_complementarity_v0_plan.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.math_grade import answer_key  # noqa: E402


def load_scores(pattern: str) -> dict[str, float]:
    """traj_uid -> worst-step suspicion turned into quality (1 - max suspicion)."""
    out = {}
    for f in glob.glob(pattern):
        for r in map(json.loads, open(f)):
            if r.get("scores"):
                out[r["traj_uid"]] = 1.0 - max(r["scores"])
    return out


def zwithin(vals: dict[str, float], groups: dict[str, list[str]]) -> dict[str, float]:
    out = {}
    for ids in groups.values():
        v = np.array([vals[i] for i in ids])
        sd = v.std()
        for i, x in zip(ids, v):
            out[i] = 0.0 if sd < 1e-12 else (x - v.mean()) / sd
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("cot-checker-results/tts_regraded_v2/instruct_t10"))
    p.add_argument("--probe", default="boundary_stats__mlp_h1024x2__seed42__gen")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--datasets", nargs="+", default=["math500", "gsm8k"])
    p.add_argument("--out", type=Path, default=Path("results/pair_complementarity_v0/summary.json"))
    a = p.parse_args()
    out = {}
    for ds in a.datasets:
        trajs = {r["traj_uid"]: r for f in glob.glob(str(a.root / f"tts_{ds}.shard*_trajectories.jsonl"))
                 for r in map(json.loads, open(f))}
        conf = {r["traj_uid"]: r["rules"] for f in glob.glob(str(a.root / f"tts_{ds}.shard*_conf.jsonl"))
                for r in map(json.loads, open(f))}
        S = {"prm": load_scores(str(a.root / f"scores/tts_{ds}__prm_qwen25_math_7b.shard*.jsonl")),
             "probe": load_scores(str(a.root / f"scores/tts_{ds}__{a.probe}.shard*.jsonl")),
             "conf_mean": {u: c["mean_token_conf"] for u, c in conf.items()},
             "conf_b10": {u: c["bottom10_group_w32"] for u, c in conf.items()}}
        groups = defaultdict(list)
        for u, r in trajs.items():
            if r.get("gradeable") and all(u in s for s in S.values()):
                groups[r["fork_id"]].append(u)
        # orientation of confidence: pick the sign that ranks correct traces higher overall
        for c in ("conf_mean", "conf_b10"):
            us = [u for g in groups.values() for u in g]
            y = np.array([trajs[u]["correct"] for u in us], float); x = np.array([S[c][u] for u in us])
            if np.corrcoef(x, y)[0, 1] < 0:
                S[c] = {u: -v for u, v in S[c].items()}
        pairs = []  # (question, i_correct, j_wrong)
        for q, ids in groups.items():
            for i in ids:
                for j in ids:
                    if trajs[i]["correct"] and not trajs[j]["correct"] and \
                            answer_key(trajs[i]["pred"]) != answer_key(trajs[j]["pred"]):
                        pairs.append((q, i, j))
        qs = sorted({q for q, _, _ in pairs})
        qidx = {q: k for k, q in enumerate(qs)}

        def pref(sc):
            return np.array([1.0 if sc[i] > sc[j] else 0.5 if sc[i] == sc[j] else 0.0 for _, i, j in pairs])
        P = {k: pref(v) for k, v in S.items()}
        acc = {k: float(v.mean()) for k, v in P.items()}
        best_conf = max(("conf_mean", "conf_b10"), key=lambda c: acc[c])
        zp, zc = zwithin(S["probe"], groups), zwithin(S[best_conf], groups)
        zr = zwithin(S["prm"], groups)
        P["probe+conf"] = pref({u: zp[u] + zc[u] for u in zp})
        P["prm+conf"] = pref({u: zr[u] + zc[u] for u in zr})
        P["probe+prm"] = pref({u: zp[u] + zr[u] for u in zr})
        qa = np.array([qidx[q] for q, _, _ in pairs])
        # per-question mean preference, then mean over questions (each question weighted equally)
        def per_q(v):
            s = np.bincount(qa, v, len(qs)); n = np.bincount(qa, None, len(qs)); return s / n
        Q = {k: per_q(v) for k, v in P.items()}
        rng = np.random.default_rng(0)
        boot = rng.integers(0, len(qs), (a.n_boot, len(qs)))
        res = {"n_pairs": len(pairs), "n_questions": len(qs), "pair_questions": qs, "best_conf": best_conf,
               "acc_question_weighted": {k: float(v.mean()) for k, v in Q.items()}}
        part = "probe" if Q["probe"].mean() >= Q[best_conf].mean() else best_conf
        d = (Q["probe+conf"] - Q[part])[boot].mean(1)
        res["probe+conf_minus_best_part"] = [float((Q["probe+conf"] - Q[part]).mean()),
                                             float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        d2 = (Q["prm+conf"] - Q["prm"])[boot].mean(1)
        res["prm+conf_minus_prm"] = [float((Q["prm+conf"] - Q["prm"]).mean()),
                                     float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))]
        wrong_conf = P[best_conf] == 0.0
        res["on_pairs_conf_prefers_wrong"] = {"n": int(wrong_conf.sum()),
                                              **{k: float(P[k][wrong_conf].mean()) for k in ("prm", "probe")}}
        agree = (P["probe"] == 1) == (P[best_conf] == 1)
        for k in ("probe", "prm"):
            qw, v = qa[wrong_conf], P[k][wrong_conf]
            arr = np.array([v[qw == q].mean() for q in np.unique(qw)])
            bb = rng.integers(0, len(arr), (a.n_boot, len(arr)))
            res.setdefault("wrong_conf_question_weighted", {})[k] = [
                float(arr.mean()), float(np.percentile(arr[bb].mean(1), 2.5)),
                float(np.percentile(arr[bb].mean(1), 97.5))]
        res["probe_conf_agreement"] = float(agree.mean())
        res["go"] = bool(ds == "math500" and res["probe+conf_minus_best_part"][0] >= 0.02
                         and res["probe+conf_minus_best_part"][1] > 0)
        out[ds] = res
        print(ds, json.dumps({k: v for k, v in res.items() if k != "pair_questions"}, indent=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
