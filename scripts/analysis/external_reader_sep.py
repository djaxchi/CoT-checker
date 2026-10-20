#!/usr/bin/env python3
"""Does a linear readout at the PRM's separator keep the PRM's independence from confidence?

external_reader_v1 (docs/external_reader_v1_plan.md). §21.34 found the PRM's head
at 0.629 on the MATH-500 mixed pairs where token confidence prefers the wrong
answer; §21.37 found LDA on the PRM's states under the verifier template at
0.557. §21.31 found that LDA at the native `<extra_0>` separator matches the
head on ProcessBench. This scores the same pairs with the head's P(correct) and
with LDA on the separator state, both from one native-template pass, fit on that
pass's own PRM800K training vectors. Worst step per trace, question-weighted
pair accuracy, as in external_reader_pairs.py.
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
from scripts.analysis.external_reader_pairs import pair_stats  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tts_root", type=Path, required=True)
    p.add_argument("--vectors", type=Path, required=True, help="train, val and tts shards of one native pass")
    p.add_argument("--blocks", nargs="+", default=["28:sep", "20:sep", "28:last"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    trajs = {r["traj_uid"]: r for f in glob.glob(str(a.tts_root / "tts_math500.shard*_trajectories.jsonl"))
             for r in map(json.loads, open(f)) if r.get("gradeable")}
    conf = {r["traj_uid"]: r["rules"]["mean_token_conf"] for f in glob.glob(str(a.tts_root / "tts_math500.shard*_conf.jsonl"))
            for r in map(json.loads, open(f))}
    tr, va = load_split(a.vectors, "train")[0], load_split(a.vectors, "val")[0]
    S, meta = load_split(a.vectors, "tts")
    layers = [int(l) for l in S["layers"]]
    scores = {"head": 1.0 - S["head_p_correct"].astype(np.float64)}  # suspicion
    for spec in a.blocks:
        layer, rep = int(spec.split(":")[0]), spec.split(":")[1]
        li = layers.index(layer)
        dirn, mu, _ = lda_fit(tr[rep][:, li].astype(np.float64), tr["y"],
                              va[rep][:, li].astype(np.float64), va["y"])
        scores[f"lda_L{layer}_{rep}"] = (S[rep][:, li].astype(np.float64) - mu) @ dirn
    results = {}
    for name, sc in scores.items():
        worst = defaultdict(lambda: -np.inf)
        for m, s in zip(meta, sc):
            t = m["uid"].rsplit("::", 1)[0]
            worst[t] = max(worst[t], s)
        quality = {t: -v for t, v in worst.items()}
        groups = defaultdict(list)
        for t in quality:
            if t in trajs and t in conf:
                groups[trajs[t]["fork_id"]].append(t)
        results[name] = pair_stats(quality, trajs, groups, conf, rng)
        print(name, json.dumps(results[name]), flush=True)
    sep = results["lda_L28_sep"]["conf_wrong"]
    results["gate_head_all"] = results["head"]["all"][0]
    results["go"] = bool(sep[0] >= 0.60 and sep[1] > 0.5)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1))
    print("GO", results["go"])


if __name__ == "__main__":
    main()
