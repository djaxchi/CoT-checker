#!/usr/bin/env python3
"""Stack pooled readouts across layers (layer_stack_v0).

§16 found, on Qwen2.5-7B, that concatenating layers lifted held-out AUROC by
about 0.05 offline but not ProcessBench F1. §21.31 found Qwen3-8B's layers 24 to
36 flat one at a time. This asks whether they are complementary: one
standardized L2 logistic probe on the concatenation of several (layer, readout)
blocks against the same probe on each block alone, from the pooled vectors of
backbone_geometry_v0. C by val AUROC. Metrics as in backbone_geometry.py.
Prespecified GO: the best stack beats the best single block by >= 0.03
ProcessBench calib-20.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.backbone_geometry import SUBSETS, evaluate, load_split  # noqa: E402

CS = (1e-4, 1e-3, 1e-2, 1e-1)


def block(S, key, blocks, layers):
    return np.concatenate([S[key][0][r][:, layers.index(l)].astype(np.float32) for l, r in blocks], 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--singles", nargs="+", required=True, help="layer:rep")
    p.add_argument("--stacks", nargs="+", required=True, help="layer:rep,layer:rep,...")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    S = {k: load_split(a.dir, k) for k in ("train", "val", "test")}
    for s in SUBSETS:
        S[s] = load_split(a.dir, f"pb_{s}")
    layers = [int(l) for l in S["train"][0]["layers"]]
    parse = lambda spec: [(int(x.split(":")[0]), x.split(":")[1]) for x in spec.split(",")]  # noqa: E731
    rows = []
    for spec in a.singles + a.stacks:
        blocks = parse(spec)
        X = {k: block(S, k, blocks, layers) for k in S}
        mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
        Z = {k: (v - mu) / sd for k, v in X.items()}
        best = None
        for c in CS:
            clf = LogisticRegression(C=c, max_iter=3000).fit(Z["train"], S["train"][0]["y"])
            va = roc_auc_score(S["val"][0]["y"], clf.decision_function(Z["val"]))
            if best is None or va > best[0]:
                best = (va, c, clf)
        clf = best[2]
        data = {"test": (Z["test"], S["test"][0]["y"]), **{s: (Z[s], S[s][0]["y"], S[s][1]) for s in SUBSETS}}
        r = evaluate(clf.decision_function, data) | {"spec": spec, "C": best[1], "val_auroc": best[0]}
        rows.append(r)
        print(f"{spec}: test={r['test_auroc']:.4f} pb_auroc={r['pb_auroc']:.4f} "
              f"pb_calib20={r['pb_calib20']:.4f} C={best[1]}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))
    single = max(r["pb_calib20"] for r in rows if r["spec"] in a.singles)
    stack = max(r["pb_calib20"] for r in rows if r["spec"] in a.stacks)
    print(f"best single {single:.4f} best stack {stack:.4f} GO={stack - single >= 0.03}")


if __name__ == "__main__":
    main()
