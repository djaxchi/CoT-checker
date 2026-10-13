#!/usr/bin/env python3
"""Track 2 of factorized_step_labels_v1: necessity by activation patching.

Plan: docs/factorized_step_labels_v1_plan.md, section 2.

Two modes.

  --mode ref    Encode the frozen reference set (300 calib traces) and write the
                position-matched mean of step-content states at every layer:
                ref_means.pt. Run once, before the shards.
  --mode patch  For every fork of this shard, teacher-force the ORIGINAL trajectory
                and record per-token log-probs of everything after step i under:
                  clean, clean2 (a second clean pass: the numeric floor)
                  self   step i's content states replaced by themselves (no-op gate)
                  mean   replaced by the position-matched reference mean
                  rand   replaced by random directions, norm-matched per position
                at each layer L of --layers, hidden_states[L] indexing (1-based,
                block L-1 output). The patch is das_span.make_span_patch_hook, so
                every block above L recomputes K/V from the patched span. A landing
                check reads block L's input on the span in every patched pass, and
                one clean pass checks that block L-1's output equals hidden_states[L].

Output per shard: track2.shardNN.jsonl, one row per fork with per-layer sums for
the continuation, step i+1 and the answer line, the gate quantities, and the raw
per-token log-probs (so any other segment can be recomputed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.factorized_step_labels import (OffsetMeanAccumulator,  # noqa: E402
                                                 clean_states, mean_baseline,
                                                 necessity_from_lp,
                                                 norm_matched_random, patched_logprobs,
                                                 segment_sums)

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def load_model(a):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model_name_or_path, revision=a.revision,
                                                 local_files_only=a.local_files_only,
                                                 dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), device


def run_ref(a) -> None:
    import torch
    out = a.out_dir / "ref_means.pt"
    if out.exists() and not a.force:
        sys.exit(f"[t2] refusing to overwrite {out}")
    ref = json.loads(a.reference_set.read_text())["traces"]
    if a.limit:
        ref = ref[:a.limit]
    model, device = load_model(a)
    d = model.config.hidden_size
    acc = {L: OffsetMeanAccumulator(d, a.max_offset, a.min_count) for L in a.layers}
    t0 = time.perf_counter()
    for n, r in enumerate(ref):
        ids = torch.tensor([r["prompt_ids"] + r["reply_ids"]], device=device)
        P = len(r["prompt_ids"])
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        for L in a.layers:
            h = hs[L][0].float().cpu().numpy()
            for (lo, hi), clo in zip(r["step_spans"], r["content_lo"]):
                acc[L].add(h[P + clo:P + hi])
        if (n + 1) % 50 == 0:
            print(f"[t2-ref] {n+1}/{len(ref)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    res = {L: acc[L].finalize() for L in a.layers}
    torch.save({"layers": a.layers, "max_offset": a.max_offset, "min_count": a.min_count,
                "n_traces": len(ref),
                "means": {L: {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
                              for k, v in res[L].items()} for L in a.layers}}, out)
    c = res[a.layers[0]]["counts"]
    print(f"[t2-ref] done: {len(ref)} traces, {res[a.layers[0]]['n_final']} step lines, "
          f"offsets with >= {a.min_count} refs: {int((c >= a.min_count).sum())}", flush=True)


def run_patch(a) -> None:
    import torch
    out = a.out_dir / f"track2.shard{a.shard_idx:02d}.jsonl"
    if out.exists() and not a.force:
        sys.exit(f"[t2] refusing to overwrite {out}")
    doc = json.loads(a.fork_split.read_text())
    forks = [(i, f) for i, f in enumerate(doc["forks"]) if i % a.num_shards == a.shard_idx]
    if a.limit:
        forks = forks[:a.limit]
    ref = torch.load(a.ref_means, weights_only=False)
    missing = [L for L in a.layers if L not in ref["means"]]
    if missing:
        sys.exit(f"[t2] ref_means lacks layers {missing}")
    model, device = load_model(a)
    t0 = time.perf_counter()
    with open(out, "w", encoding="utf-8") as fh:
        for n, (fi, f) in enumerate(forks):
            P = len(f["prompt_ids"])
            s = f["step"]
            lo, hi = P + f["content_lo"][s], P + f["step_spans"][s][1]
            start = hi                                    # first continuation token
            seg = {"cont": (start, P + len(f["reply_ids"])),
                   "ans": (P + f["answer_span"][0], P + len(f["reply_ids"]))}
            if s + 1 < len(f["step_spans"]):
                nlo, nhi = f["step_spans"][s + 1]
                seg["next"] = (P + nlo, P + nhi)
            ids = torch.tensor([f["prompt_ids"] + f["reply_ids"]], device=device)
            clean, idx_check = clean_states(model, ids, a.layers, lo, hi)
            lp_clean, _ = patched_logprobs(model, ids, start)
            lp_clean2, _ = patched_logprobs(model, ids, start)
            base = lp_clean.float().cpu().numpy()
            row = {"fork_id": f["fork_id"], "fork_index": fi, "validity": f["validity"],
                   "position": f["position"], "step": s, "n_steps": f["n_steps"],
                   "span": [lo, hi], "n_span": hi - lo, "segments": seg,
                   "n_cont_tokens": seg["cont"][1] - seg["cont"][0],
                   "clean": segment_sums(base, start, seg),
                   "clean2_minus_clean": segment_sums(lp_clean2.float().cpu().numpy() - base, start, seg),
                   "index_check_max_abs": idx_check, "layers": {},
                   "lp_clean": base.round(5).tolist()}
            gen = torch.Generator().manual_seed(a.seed * 1009 + fi)
            for L in a.layers:
                st = clean[L]
                patches = {"self": st,
                           "mean": torch.from_numpy(mean_baseline(
                               {k: (v.numpy() if hasattr(v, "numpy") else v)
                                for k, v in ref["means"][L].items()}, hi - lo)).to(device),
                           "rand": norm_matched_random(st, gen)}
                lay = {}
                for cond, states in patches.items():
                    lp, land = patched_logprobs(model, ids, start, L, lo, hi, states)
                    lpn = lp.float().cpu().numpy()
                    lay[cond] = {"sum": segment_sums(lpn, start, seg),
                                 "delta": segment_sums(lpn - base, start, seg),
                                 "landing_max_abs": land,
                                 "lp": lpn.round(5).tolist(),
                                 "drop_xb": necessity_from_lp(base, lpn, start, seg)}
                # necessity = drop in log-prob of the true continuation, without the
                # boundary token the patched position predicts itself (pilot finding)
                xb = lay["mean"]["drop_xb"]
                lay["nec_sum"] = xb["cont"]
                lay["nec_tok"] = xb["cont"] / max(xb["n_cont_xb"], 1)
                lay["nec_next"] = xb.get("next", float("nan"))
                lay["nec_ans"] = xb["ans"]
                lay["nec_boundary"] = xb["boundary_drop"]
                lay["nec_sum_incl_boundary"] = -lay["mean"]["delta"]["cont"]
                row["layers"][str(L)] = lay
            fh.write(json.dumps(row) + "\n"); fh.flush()
            msg = " ".join(f"L{L}:nec={row['layers'][str(L)]['nec_sum']:+.2f}"
                           f"/self={row['layers'][str(L)]['self']['delta']['cont']:+.3f}"
                           for L in a.layers)
            print(f"[t2] {n+1}/{len(forks)} {f['fork_id']} {f['validity']}/{f['position']} "
                  f"clean={row['clean']['cont']:.2f} {msg} ({time.perf_counter()-t0:.0f}s)", flush=True)
    (a.out_dir / f"track2.shard{a.shard_idx:02d}.manifest.json").write_text(json.dumps({
        "experiment": "factorized_step_labels_v1", "track": 2, "shard_idx": a.shard_idx,
        "num_shards": a.num_shards, "n_forks": len(forks), "args": {k: str(v) for k, v in vars(a).items()},
        "fork_split_sha256": __import__("hashlib").sha256(a.fork_split.read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "seconds": time.perf_counter() - t0, "created": datetime.now(timezone.utc).isoformat()}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["ref", "patch"], required=True)
    p.add_argument("--fork_split", type=Path)
    p.add_argument("--reference_set", type=Path)
    p.add_argument("--ref_means", type=Path)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[1, 9, 18, 27, 35])
    p.add_argument("--max_offset", type=int, default=128)
    p.add_argument("--min_count", type=int, default=20)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.mode == "ref":
        if a.reference_set is None:
            sys.exit("--reference_set required")
        run_ref(a)
    else:
        if a.fork_split is None or a.ref_means is None:
            sys.exit("--fork_split and --ref_means required")
        run_patch(a)


if __name__ == "__main__":
    main()
