#!/usr/bin/env python3
"""Map context-view predictions back to (trace, step) for evaluation (context_ablation_v3).

Fits live at <fits_root>/<ctx>_<source>_s<seed>/local/ and score view splits.
For one source this writes, under <out>/<source>/:
  fits/<ctx>_s<seed>/<ctx>/predictions.jsonl.gz   (arm name = ctx), done.json
  meta/<split>.jsonl                               original trace metas, renamed
so scripts/eval_contextual_token_probe.py runs unchanged on it, with
calib/dev/test = the source's own splits and the other sets as OOD step sets.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

PB = ("pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath")
# source -> {eval split name: (view split, original manifest key, original split)}
MAPS = {
    "v1human": {"calib": ("v1_calib", "v1", "calib"), "dev": ("v1_dev", "v1", "dev"),
                "test": ("prm_human_test", "v1", "test"), "rp_test": ("rp_test", "v2", "test")},
    "v2ds": {"calib": ("v2_calib", "v2", "calib"), "dev": ("v2_dev", "v2", "dev"),
             "test": ("rp_test", "v2", "test"), "prm_human_test": ("prm_human_test", "v1", "test")},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits_root", type=Path, required=True)
    ap.add_argument("--views_meta", type=Path, required=True, help="any context's views meta dir")
    ap.add_argument("--v1_manifest", type=Path, required=True)
    ap.add_argument("--v2_manifest", type=Path, required=True)
    ap.add_argument("--source", choices=sorted(MAPS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    mp = dict(MAPS[a.source])
    for sp in PB:
        mp[sp] = (sp, "v1", sp)
    view_to_eval = {v: e for e, (v, _, _) in mp.items()}
    vmeta = {}
    for f in a.views_meta.glob("*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            vmeta[r["trace_id"]] = r
    out = a.out / a.source
    (out / "meta").mkdir(parents=True, exist_ok=True)
    mans = {"v1": a.v1_manifest, "v2": a.v2_manifest}
    for e, (v, mk, osp) in mp.items():
        keep = {vmeta[k]["orig_trace_id"] for k in vmeta if vmeta[k]["split"] == v}
        with open(out / "meta" / f"{e}.jsonl", "w") as f:
            for l in open(mans[mk] / "meta" / f"{osp}.jsonl"):
                m = json.loads(l)
                if m["trace_id"] in keep:
                    f.write(json.dumps({**m, "split": e}) + "\n")
    n_runs = 0
    for fd in sorted(a.fits_root.glob(f"*_{a.source}_s*")):
        ctx, _, seed = fd.name.rsplit("_", 2)
        seed = int(seed[1:])
        src = fd / "local"
        done = json.loads((src / "done.json").read_text())
        dst = out / "fits" / f"{ctx}_s{seed}" / ctx
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        with gzip.open(src / "predictions.jsonl.gz", "rt") as fi, gzip.open(dst / "predictions.jsonl.gz", "wt") as fo:
            for l in fi:
                r = json.loads(l)
                if r["split"] not in view_to_eval:
                    continue
                m = vmeta[r["trace_id"]]
                fo.write(json.dumps({"split": view_to_eval[r["split"]], "trace_id": m["orig_trace_id"],
                                     "step": m["orig_step"], "n_steps": m["orig_n_steps"],
                                     "logit": r["logit"]}) + "\n")
                n += 1
        (dst / "done.json").write_text(json.dumps({**done, "arm": ctx, "seed": seed, "n_prediction_rows": n,
                                                   "source": a.source}))
        n_runs += 1
    print(f"[collect] {a.source}: {n_runs} runs -> {out}")


if __name__ == "__main__":
    main()
