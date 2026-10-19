#!/usr/bin/env python3
"""Mixed-pair accuracy of LDA readers on different backbones (external_reader_v0).

Plan and prespecified reading: docs/external_reader_v0_plan.md. Each backbone's
LDA is fit on its own PRM800K training vectors, applied to every step of the
MATH-500 TTS candidates, aggregated by worst step, and compared on the mixed
pairs of §21.34 (all of them, and the ones where token confidence prefers the
wrong answer).
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
from scripts.analysis.backbone_geometry import load_split  # noqa: E402
from scripts.analysis.cross_model_direction import lda_fit  # noqa: E402
from scripts.analysis.sae_feature_localization import load_dense  # noqa: E402
from src.eval.math_grade import answer_key  # noqa: E402


def lda_from_vectors(d: Path, layer: int, rep: str):
    tr, va = load_split(d, "train")[0], load_split(d, "val")[0]
    li = [int(l) for l in tr["layers"]].index(layer)
    x, y = tr[rep][:, li].astype(np.float64), tr["y"]
    return lda_fit(x, y, va[rep][:, li].astype(np.float64), va["y"])


def lda_from_grid(cache: Path, rep: str, n_fit: int = 60000):
    x, y, _ = load_dense(cache, rep, "probe_train_full")
    idx = np.sort(np.random.default_rng(0).choice(len(y), n_fit, replace=False))
    xv, yv, _ = load_dense(cache, rep, "val_5k")
    return lda_fit(np.asarray(x[idx], np.float64), y[idx], np.asarray(xv, np.float64), yv)


def pair_stats(quality: dict, trajs: dict, groups: dict, conf: dict, rng, n_boot=2000):
    pairs = [(q, i, j) for q, ids in groups.items() for i in ids for j in ids
             if trajs[i]["correct"] and not trajs[j]["correct"]
             and answer_key(trajs[i]["pred"]) != answer_key(trajs[j]["pred"])]
    pref = np.array([1.0 if quality[i] > quality[j] else 0.5 if quality[i] == quality[j] else 0.0 for _, i, j in pairs])
    cw = np.array([conf[i] < conf[j] for _, i, j in pairs])
    qs = np.array([q for q, _, _ in pairs])
    out = {}
    for name, m in (("all", np.ones(len(pairs), bool)), ("conf_wrong", cw)):
        uq = np.unique(qs[m]); arr = np.array([pref[m][qs[m] == q].mean() for q in uq])
        bb = rng.integers(0, len(arr), (n_boot, len(arr))).astype(int)
        out[name] = [float(arr.mean()), float(np.percentile(arr[bb].mean(1), 2.5)),
                     float(np.percentile(arr[bb].mean(1), 97.5)), int(m.sum())]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tts_root", type=Path, required=True)
    p.add_argument("--vectors", type=Path, required=True, help="external_reader_v0/vectors")
    p.add_argument("--train_vectors", type=Path, required=True, help="backbone_geometry_v0/vectors")
    p.add_argument("--base_grid", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    trajs = {r["traj_uid"]: r for f in glob.glob(str(a.tts_root / "tts_math500.shard*_trajectories.jsonl"))
             for r in map(json.loads, open(f)) if r.get("gradeable")}
    conf = {r["traj_uid"]: r["rules"]["mean_token_conf"] for f in glob.glob(str(a.tts_root / "tts_math500.shard*_conf.jsonl"))
            for r in map(json.loads, open(f))}
    readers = {"q3": ("q3_ver", [(35, "mean"), (35, "last")]), "parent": ("parent_ver", [(27, "mean"), (28, "last"), (27, "last"), (28, "mean")]),
               "prm": ("prm_ver", [(27, "mean"), (28, "last"), (27, "last"), (28, "mean")]), "q3base": (None, [(35, "mean"), (35, "last")])}
    results = {}
    for name, (tv, blocks) in readers.items():
        S, meta = load_split(a.vectors / name, "tts")
        layers = [int(l) for l in S["layers"]]
        for layer, rep in blocks:
            if tv is None:
                if rep == "last":
                    rep_g = "last_token"
                else:
                    rep_g = "step_mean"
                dirn, mu, _ = lda_from_grid(a.base_grid, rep_g)
            else:
                dirn, mu, _ = lda_from_vectors(a.train_vectors / tv, layer, rep)
            sc = (S[rep][:, layers.index(layer)].astype(np.float64) - mu) @ dirn  # suspicion
            worst = defaultdict(lambda: -np.inf)
            for m, s in zip(meta, sc):
                t = m["uid"].rsplit("::", 1)[0]
                worst[t] = max(worst[t], s)
            quality = {t: -v for t, v in worst.items()}
            groups = defaultdict(list)
            for t in quality:
                if t in trajs and t in conf:
                    groups[trajs[t]["fork_id"]].append(t)
            r = pair_stats(quality, trajs, groups, conf, rng)
            results[f"{name}_L{layer}_{rep}"] = r
            print(name, layer, rep, json.dumps(r), flush=True)
    par = {k: v for k, v in results.items() if k.startswith("parent_")}
    best = max(par, key=lambda k: par[k]["all"][0])
    results["go"] = bool(par[best]["conf_wrong"][0] >= 0.55 and par[best]["conf_wrong"][1] > 0.5)
    results["parent_best_block"] = best
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1))
    print("GO", results["go"], best)


if __name__ == "__main__":
    main()
