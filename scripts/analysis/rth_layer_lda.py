#!/usr/bin/env python3
"""M1: where along the layers the read token's correctness signal forms (read_token_v1).

For one representation of the mechanistic bundle (scripts/read_token_mech_bundle.py):

  last_step  Qwen3-8B, last step token, rows 0..35 = block outputs, row 36 = final state
  read       the read token right after the step, same rows
  prm        Qwen2.5-Math-PRM-7B, last step token, rows 0..27 = block outputs, row 28 = final

and every row (layer), LDA is fit on the bundle's train sample (class means and the
shrunk within-class covariance, shrinkage picked on val AUROC) and scored on test
(AUROC) and ProcessBench (AUROC of the first erroneous step against the steps before
it and the steps of correct traces; calib-20 and oracle F1_PB), with
scripts/analysis/prm_geometry.py's own functions.

    python scripts/analysis/rth_layer_lda.py --bundle <mech>/bundle --rep read --out m1_read.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.prm_geometry import (SHRINKS, auroc, evaluate, moments,  # noqa: E402
                                           pb_step_class, project, shrink)
from scripts.read_token_mech_bundle import load_bundle  # noqa: E402

PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def load(bundle: Path, rep: str, split: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    b = load_bundle(bundle / split, arrays=(rep, "y"), mmap=False)
    return b[rep], np.asarray(b["y"], np.int64), b["meta"]


def layer_lda(Xtr, ytr, Xva, yva, Xte, yte, pb: dict, dev) -> dict:
    m = moments(Xtr, ytr, dev)
    dmu = m["mu1"] - m["mu0"]
    best = (-1.0, None, None)
    for a in SHRINKS:
        w = torch.linalg.solve(shrink(m["sw"], a), dmu)
        av = auroc(yva, project(Xva, w, dev))
        if av > best[0]:
            best = (av, a, w)
    w = best[2]
    s_pb = {k: project(v[0], w, dev) for k, v in pb.items()}
    res = evaluate("lda", project(Xva, w, dev), project(Xte, w, dev), s_pb, yva, yte,
                   {k: v[1] for k, v in pb.items()})
    pb_auc = {}
    for k, (_, meta) in pb.items():
        c = pb_step_class(meta)
        pb_auc[k] = auroc(c[c >= 0], s_pb[k][c >= 0])
    return {"shrink": best[1], "val_auroc": best[0], "test_auroc": res["auroc_test"],
            "pb_auroc": pb_auc, "pb_auroc_mean": float(np.mean(list(pb_auc.values()))),
            "pb_calib20": res["pb_avg_F1_PB_calib20"], "pb_oracle": res["pb_avg_F1_PB_oracle"],
            "pb_calib20_by_subset": {k: v["F1_PB_calib20"] for k, v in res["pb"].items()}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--rep", choices=["last_step", "read", "prm"], required=True)
    p.add_argument("--layers", default="all", help="comma list of rows, or all")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    Xtr, ytr, _ = load(a.bundle, a.rep, "probe_train_full")
    Xva, yva, _ = load(a.bundle, a.rep, "val_5k")
    Xte, yte, _ = load(a.bundle, a.rep, "test_2k")
    pb = {s: load(a.bundle, a.rep, s) for s in PB_SUBSETS}
    print(f"[m1] {a.rep}: loaded {Xtr.shape} train in {time.time() - t0:.0f}s", flush=True)
    rows = range(Xtr.shape[1]) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    out = {"rep": a.rep, "n_rows": int(Xtr.shape[1]), "n_train": int(len(ytr)), "layers": {}}
    for l in rows:
        r = layer_lda(Xtr[:, l], ytr, Xva[:, l], yva, Xte[:, l], yte,
                      {k: (v[0][:, l], v[2]) for k, v in pb.items()}, dev)
        out["layers"][str(l)] = r
        print(f"[m1] {a.rep} row {l}: test AUROC {r['test_auroc']:.4f} PB AUROC {r['pb_auroc_mean']:.4f} "
              f"calib-20 {r['pb_calib20']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
