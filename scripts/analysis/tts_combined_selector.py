#!/usr/bin/env python3
"""Can the PRM and the hidden-state probes, used together, close the gap to the oracle?

Every scorer is already saved per candidate, so combining them needs no GPU. Three
selectors are compared on the same random subsets of each problem's ten samples:

  zsum      a fixed, unfitted combination: standardise each score within the
            subset, add the PRM and the chosen probes, pick the best candidate
            by the sum, then return its answer (a rerank by the combined score)
  learned   a logistic regression on candidate correctness, fitted by 5-fold
            cross-validation over PROBLEMS (a problem's candidates never inform
            its own selection). Features per candidate, computed inside the
            subset so training and inference see the same thing: the answer's
            vote share, each scorer's worst-step score and its rank within the
            subset, and length. Each answer group is scored by its members' mean
            predicted probability and the best group wins.
  learned_noprm  the same without the PRM, to price what the PRM adds

against majority vote, the PRM weighted vote and the oracle. Draws are fixed by
seed, so every selector sees identical subsets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.math_grade import answer_key  # noqa: E402

PRM = "prm_qwen25_math_7b"


def load(run_root: Path, stem: str, cells: list[str]):
    tr = [json.loads(l) for f in sorted(run_root.glob(f"{stem}.shard*_trajectories.jsonl"))
          for l in f.read_text().splitlines() if l.strip()]
    conf = {}
    for f in sorted(run_root.glob(f"{stem}.shard*_conf.jsonl")):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l); conf[r["traj_uid"]] = r["rules"]
    sc = {}
    for c in cells:
        sc[c] = {}
        for f in sorted((run_root / "scores").glob(f"{stem}__{c}.shard*.jsonl")):
            for l in f.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); sc[c][r["traj_uid"]] = r["scores"]
    by = defaultdict(list)
    for r in tr:
        if not r.get("gradeable"):
            continue
        feats = []
        for c in cells:
            s = sc[c].get(r["traj_uid"])
            feats.append(max(s) if s else np.nan)          # worst-step suspicion
        by[r["fork_id"]].append({
            "key": r.get("answer_key") or answer_key(r.get("pred")),
            "correct": bool(r["correct"]),
            "s": np.array(feats, float),
            "conf": conf.get(r["traj_uid"], {}).get("mean_token_conf", np.nan),
            "len": np.log1p(r.get("n_gen_tokens", 0))})
    return by


def subset_features(cands: list[dict], use: list[int]) -> np.ndarray:
    keys = [c["key"] for c in cands]
    share = np.array([keys.count(k) / len(keys) for k in keys])
    S = np.stack([c["s"][use] for c in cands])
    S = np.where(np.isnan(S), np.nanmean(S, 0) if np.isfinite(S).any() else 0.5, S)
    rank = np.argsort(np.argsort(S, 0), 0) / max(len(cands) - 1, 1)
    conf = np.array([c["conf"] for c in cands], float)
    conf = np.nan_to_num(conf, nan=np.nanmean(conf) if np.isfinite(conf).any() else 0.0)
    ln = np.array([c["len"] for c in cands])
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)  # noqa: E731
    return np.column_stack([share, S, rank, z(conf), z(ln)])


def pick_group(cands, p) -> bool:
    g = defaultdict(list)
    for c, pi in zip(cands, p):
        g[c["key"]].append((pi, c["correct"]))
    best = max(g.values(), key=lambda v: np.mean([x for x, _ in v]))
    return best[0][1]


def majority(cands, rng) -> bool:
    g = defaultdict(list)
    for c in cands:
        g[c["key"]].append(c["correct"])
    m = max(len(v) for v in g.values())
    tied = [v for v in g.values() if len(v) == m]
    return tied[rng.integers(len(tied))][0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", type=Path, required=True)
    p.add_argument("--probes", nargs="+", required=True, help="cell names (as in scores/)")
    p.add_argument("--stems", nargs="+", default=["tts_gsm8k", "tts_math500"])
    p.add_argument("--ns", type=int, nargs="+", default=[2, 3, 4, 6, 8, 10])
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from sklearn.linear_model import LogisticRegression

    cells = [PRM] + a.probes
    results = {}
    for stem in a.stems:
        by = load(a.run_root, stem, cells)
        pids = sorted(by)
        rng = np.random.default_rng(0)
        fold = {pid: i % 5 for i, pid in enumerate(rng.permutation(pids))}
        for n in a.ns:
            draws = {pid: [rng.choice(len(by[pid]), size=min(n, len(by[pid])), replace=False)
                           for _ in range(a.draws)] for pid in pids}
            variants = {"learned": list(range(len(cells))), "learned_noprm": list(range(1, len(cells)))}
            acc = defaultdict(list)
            for name, use in variants.items():
                preds = {}
                for f in range(5):
                    X, y = [], []
                    for pid in pids:
                        if fold[pid] == f:
                            continue
                        for d in draws[pid]:
                            c = [by[pid][i] for i in d]
                            X.append(subset_features(c, use)); y += [x["correct"] for x in c]
                    clf = LogisticRegression(max_iter=2000, C=1.0).fit(np.vstack(X), y)
                    for pid in pids:
                        if fold[pid] != f:
                            continue
                        preds[pid] = [pick_group([by[pid][i] for i in d],
                                                 clf.predict_proba(subset_features(
                                                     [by[pid][i] for i in d], use))[:, 1])
                                      for d in draws[pid]]
                acc[name] = [np.mean(preds[pid]) for pid in pids]
            r2 = np.random.default_rng(1)
            for pid in pids:
                ds = [[by[pid][i] for i in d] for d in draws[pid]]
                acc["majority"].append(np.mean([majority(c, r2) for c in ds]))
                acc["oracle"].append(np.mean([any(x["correct"] for x in c) for c in ds]))
                zs = []
                pw = []
                for c in ds:
                    S = np.stack([x["s"] for x in c])
                    S = (S - np.nanmean(S, 0)) / (np.nanstd(S, 0) + 1e-9)
                    tot = np.nansum(S, 1)                      # suspicion: lower is better
                    zs.append(c[int(np.argmin(tot))]["correct"])
                    w = defaultdict(float)
                    for x in c:
                        w[x["key"]] += 1.0 - (x["s"][0] if np.isfinite(x["s"][0]) else 0.5)
                    k = max(w, key=w.get)
                    pw.append(next(x["correct"] for x in c if x["key"] == k))
                acc["zsum_rerank"].append(np.mean(zs))
                acc["prm_weighted_vote"].append(np.mean(pw))
            results[f"{stem}|{n}"] = {k: float(np.mean(v)) for k, v in acc.items()}
            print(f"{stem:11s} N={n:<2d} " + "  ".join(f"{k} {100*v:.2f}" for k, v in results[f"{stem}|{n}"].items()), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"probes": a.probes, "draws": a.draws, "results": results}, indent=1))


if __name__ == "__main__":
    main()
