"""parametric_retrieval_component_v1: same-fact patching decomposed by depth
and by component (residual vs attention-output vs MLP-output).

Extends Experiment C (prga_expC_patch.py). C patched the WHOLE residual state
at three hand-picked layers and found a fact-specific rescue (matched donor
flips ~45% of failures). This asks two follow-ups in one grid:

  depth      sweep every decoder layer, not just {24,26,28}
  component  at each layer patch the residual (full), the attention
             contribution (attn), or the MLP contribution (mlp)

Same-fact "matched" donor is the causal claim; the six-way control battery
(noop / mismatched_type / mismatched_rand / random_noise / reverse) is reused
from Experiment C so the numbers stay comparable.

Layer axis L = decoder layer index (0 .. num_layers-1):
  full : edit resid_post of layer L  (ResidualEdit at hs_idx L+1, HSStore)
  attn : edit self_attn output of layer L        (ComponentStore attn_L)
  mlp  : edit mlp output of layer L               (ComponentStore mlp_L)
All three intervene at the SAME layer L on different quantities.

NOTE: attn and mlp patches at layer L still propagate downstream, so their
effects do NOT sum to the full-residual effect. This is component patching for
localization, not an additive decomposition.

Phases:
  --phase baseline   unedited scores for every prompt instance -> baseline.pq
  --phase val        matched + noop, all (mode, layer) cells, alpha=1
  --select           best layer per mode on val (fact-bootstrap d_margin)
  --phase test       full control battery at each mode's selected layer
  --phase capture    readout vectors for the spatial figure (prgc_viz.py)
  --analyze          depth_curve.csv (val) + results.csv (test)

Sharding: baseline/capture shard by instance; val/test shard by (mode, layer)
cell so each shard's forward batches are homogeneous.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.parametric_retrieval import prga_common as C  # noqa: E402
from scripts.parametric_retrieval.prga_expC_patch import (  # noqa: E402
    build_group_table,
)
from src.analysis.parametric_retrieval import grade_answer  # noqa: E402
from src.analysis.parametric_retrieval_causal import (  # noqa: E402
    assign_patch_donors,
    budget_pairs,
    fact_bootstrap_ci,
)

MODES = ["full", "attn", "mlp"]
ALPHA = 1.0
POSITION = "final_prompt_token"
VAL_CONDITIONS = ["matched", "noop"]
TEST_CONDITIONS = ["noop", "matched", "mismatched_type", "mismatched_rand",
                   "random_noise", "reverse"]
CAPTURE_SPECS = [  # (mode, condition) rows written for the figure
    ("none", "fail"), ("none", "success"),
    ("full", "matched"), ("full", "mismatched_type"), ("full", "random_noise"),
    ("attn", "matched"), ("mlp", "matched"),
]


# --------------------------------------------------------------------------- #
# vector / edit plumbing
# --------------------------------------------------------------------------- #

def get_vec(hstore, cstore, instance, mode, layer):
    if mode == "full":
        return hstore.vec(instance, POSITION, layer + 1)
    return cstore.vec(instance, mode, layer)


def make_edit(model, mode, layer):
    if mode == "full":
        return C.ResidualEdit(model, layer + 1, "patch")
    return C.ComponentEdit(model, layer, mode)


def donor_for(condition, r):
    return {"matched": r.donor_matched, "noop": r.donor_noop,
            "mismatched_type": r.donor_mismatched_type,
            "mismatched_rand": r.donor_mismatched_rand,
            "random_noise": r.donor_matched,
            "reverse": r.donor_reverse}[condition]


def prompt_for(condition, r):
    # reverse patches the failed state INTO the successful prompt
    return r.donor_instance_id if condition == "reverse" \
        else r.recipient_instance_id


# --------------------------------------------------------------------------- #
# shared context
# --------------------------------------------------------------------------- #

def load_pairs(args, out_dir, hstore):
    groups = build_group_table(out_dir, hstore.meta)
    pairs = pd.read_parquet(out_dir / "pairs.parquet")
    pairs["fact_id"] = pairs.fact_id.astype(str)
    pairs = pairs[pairs.split == args.split_of_phase].reset_index(drop=True)
    budget = args.val_budget if args.split_of_phase == "val" else args.test_budget
    pairs = budget_pairs(pairs, budget, seed=args.seed)
    return assign_patch_donors(pairs, groups, seed=args.seed)


def load_lookup(out_dir, tok, needed_instances):
    meta = pd.read_parquet(out_dir / "metadata.parquet")
    meta["fact_id"] = meta.fact_id.astype(str)
    sub = meta[meta.instance_id.isin(needed_instances)]
    prompt_ids = C.render_prompt_ids(tok, sub)
    cands = C.load_candidate_table(out_dir, tok)
    gold_of = {(r.fact_id, r.direction): r.gold_answer
               for r in meta.drop_duplicates(["fact_id", "direction"])
               .itertuples()}
    inst_group = {r.instance_id: (r.fact_id, r.direction)
                  for r in meta.itertuples()}
    return prompt_ids, cands, gold_of, inst_group


def score_batch(model, tok, device, prompt_ids, vecs, edit, alpha,
                golds, distrs, cand_rows, max_new_tokens):
    """Run generate + gold-score + distractor-score for one homogeneous
    (mode, layer, alpha) batch. Returns list of metric dicts."""
    gen = C.generate_with_edit(model, tok, device, prompt_ids,
                               edit if alpha > 0 else None, vecs, alpha,
                               max_new_tokens)
    e2 = edit if alpha > 0 else None
    sc_g = C.score_with_edit(model, tok, device, prompt_ids, golds, e2, vecs,
                             alpha, cand_rows)
    e3 = edit if alpha > 0 else None
    sc_d = C.score_with_edit(model, tok, device, prompt_ids, distrs, e3, vecs,
                             alpha)
    out = []
    for m in range(len(prompt_ids)):
        first = gen[m].strip().splitlines()[0] if gen[m].strip() else ""
        out.append({"logp_gold": sc_g[m]["logp_answer"],
                    "logp_distr": sc_d[m]["logp_answer"],
                    "margin": sc_g[m]["logp_answer"] - sc_d[m]["logp_answer"],
                    "entropy": sc_g[m]["entropy"],
                    "gold_rank": sc_g[m]["gold_rank"],
                    "exact_match": bool(grade_answer(first, golds[m])[0]),
                    "gen_text": gen[m]})
    return out


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #

def run_baseline(args, out_dir, model, tok, device, hstore):
    seen = set()
    for split in ("val", "test"):
        args.split_of_phase = split
        pr = load_pairs(args, out_dir, hstore)
        for r in pr.itertuples():
            for i in (r.recipient_instance_id, r.donor_instance_id):
                seen.add((i, r.fact_id, r.direction))
    tasks = sorted(seen)
    tasks = tasks[args.shard_idx::args.num_shards]
    prompt_ids, cands, gold_of, _ = load_lookup(
        out_dir, tok, {t[0] for t in tasks})
    rows = []
    B = args.batch_size
    for s in range(0, len(tasks), B):
        chunk = tasks[s:s + B]
        p_ids = [prompt_ids[i] for i, _, _ in chunk]
        golds = [gold_of[(f, d)] for _, f, d in chunk]
        distrs = [cands[(f, d)]["negatives"][0]
                  if cands[(f, d)]["negatives"] else "unknown"
                  for _, f, d in chunk]
        cand_rows = [cands[(f, d)]["first_ids"] for _, f, d in chunk]
        vecs = [None] * len(chunk)
        res = score_batch(model, tok, device, p_ids, vecs, None, 0.0,
                          golds, distrs, cand_rows, args.max_new_tokens)
        for (i, f, d), m in zip(chunk, res):
            rows.append({"prompt_instance": i, "fact_id": f, "direction": d,
                         **{k: m[k] for k in ("margin", "logp_gold",
                                              "entropy", "gold_rank",
                                              "exact_match")}})
        if (s // B) % 10 == 0:
            print(f"[baseline] {min(s + B, len(tasks))}/{len(tasks)}",
                  flush=True)
    _write(args, "baseline", rows)


# --------------------------------------------------------------------------- #
# val / test patch grid
# --------------------------------------------------------------------------- #

def run_grid(args, out_dir, model, tok, device, hstore, cstore):
    args.split_of_phase = args.phase
    pr = load_pairs(args, out_dir, hstore)
    if args.phase == "val":
        cells = [(mode, L) for mode in MODES for L in args.layers]
        conditions = VAL_CONDITIONS
    else:
        sel = json.loads((out_dir / "expE" / "selection.json").read_text())
        cells = [(mode, sel[mode]["layer"]) for mode in MODES]
        conditions = TEST_CONDITIONS
    cells = cells[args.shard_idx::args.num_shards]
    if not cells:
        _write(args, args.phase, [])
        return

    need = set(pr.recipient_instance_id) | set(pr.donor_instance_id) \
        | {d for c in TEST_CONDITIONS for d in
           _donors_col(pr, c) if d is not None}
    prompt_ids, cands, gold_of, inst_group = load_lookup(out_dir, tok, need)
    rng = np.random.default_rng(args.seed + args.shard_idx)
    rows = []
    for (mode, L) in cells:
        edit = make_edit(model, mode, L)
        tasks = [(cond, r) for r in pr.itertuples() for cond in conditions
                 if donor_for(cond, r) is not None]
        B = args.batch_size
        for s in range(0, len(tasks), B):
            chunk = tasks[s:s + B]
            p_ids, vecs, golds, distrs, cand_rows, meta_rows = \
                [], [], [], [], [], []
            for cond, r in chunk:
                pi = prompt_for(cond, r)
                di = donor_for(cond, r)
                key = (r.fact_id, r.direction)
                d_vec = get_vec(hstore, cstore, di, mode, L)
                if cond == "random_noise":
                    h_vec = get_vec(hstore, cstore, pi, mode, L)
                    nrm = np.linalg.norm(d_vec.astype(np.float32)
                                         - h_vec.astype(np.float32))
                    noise = rng.standard_normal(len(h_vec)).astype(np.float32)
                    noise *= nrm / max(np.linalg.norm(noise), 1e-9)
                    v = h_vec.astype(np.float32) + noise
                else:
                    v = d_vec.astype(np.float32)
                p_ids.append(prompt_ids[pi])
                vecs.append(v)
                golds.append(gold_of[key])
                distrs.append(cands[key]["negatives"][0]
                              if cands[key]["negatives"] else "unknown")
                cand_rows.append(cands[key]["first_ids"])
                meta_rows.append({"pair_id": r.pair_id, "fact_id": r.fact_id,
                                  "direction": r.direction, "mode": mode,
                                  "layer": L, "condition": cond,
                                  "prompt_instance": pi, "donor_instance": di})
            res = score_batch(model, tok, device, p_ids, vecs, edit, ALPHA,
                             golds, distrs, cand_rows, args.max_new_tokens)
            for mr, m, (cond, r) in zip(meta_rows, res, chunk):
                copied = False
                if cond in ("mismatched_type", "mismatched_rand"):
                    dg = gold_of[inst_group[mr["donor_instance"]]]
                    copied = bool(grade_answer(m["gen_text"], dg)[0])
                rows.append({**mr,
                             **{k: m[k] for k in ("margin", "logp_gold",
                                                  "entropy", "gold_rank",
                                                  "exact_match")},
                             "copied_donor": copied})
        print(f"[{args.phase}] cell ({mode},L{L}) done "
              f"({len(tasks)} runs)", flush=True)
    _write(args, args.phase, rows)


def _donors_col(pr, cond):
    return [donor_for(cond, r) for r in pr.itertuples()]


# --------------------------------------------------------------------------- #
# capture (for the spatial figure)
# --------------------------------------------------------------------------- #

def run_capture(args, out_dir, model, tok, device, hstore, cstore):
    sel = json.loads((out_dir / "expE" / "selection.json").read_text())
    readout = int(model.config.num_hidden_layers)  # final residual layer
    args.split_of_phase = "test"
    pr = load_pairs(args, out_dir, hstore)
    pr = budget_pairs(pr, args.capture_pairs, seed=args.seed)
    pr = pr.iloc[args.shard_idx::args.num_shards].reset_index(drop=True)
    need = set(pr.recipient_instance_id) | set(pr.donor_instance_id) \
        | {d for c in ("mismatched_type",) for d in _donors_col(pr, c)
           if d is not None}
    prompt_ids, cands, gold_of, _ = load_lookup(out_dir, tok, need)
    rng = np.random.default_rng(args.seed + args.shard_idx)
    rows = []
    B = max(1, args.batch_size // 2)
    for mode, cond in CAPTURE_SPECS:
        L = None if mode == "none" else sel[mode]["layer"]
        edit = None if mode == "none" else make_edit(model, mode, L)
        recs = list(pr.itertuples())
        for s in range(0, len(recs), B):
            chunk = recs[s:s + B]
            p_ids, vecs, keep = [], [], []
            for r in chunk:
                if cond == "success":
                    pi, di = r.donor_instance_id, None
                elif cond == "fail":
                    pi, di = r.recipient_instance_id, None
                elif cond == "mismatched_type":
                    pi, di = r.recipient_instance_id, r.donor_mismatched_type
                else:  # matched / random_noise
                    pi, di = r.recipient_instance_id, r.donor_matched
                if mode != "none" and di is None:
                    continue
                if mode == "none":
                    v = None
                elif cond == "random_noise":
                    h = get_vec(hstore, cstore, pi, mode, L).astype(np.float32)
                    d = get_vec(hstore, cstore, di, mode, L).astype(np.float32)
                    noise = rng.standard_normal(len(h)).astype(np.float32)
                    noise *= np.linalg.norm(d - h) / max(
                        np.linalg.norm(noise), 1e-9)
                    v = h + noise
                else:
                    v = get_vec(hstore, cstore, di, mode, L).astype(np.float32)
                p_ids.append(prompt_ids[pi])
                vecs.append(v)
                keep.append(r)
            if not p_ids:
                continue
            z = C.capture_readout(model, tok, device, p_ids,
                                  edit if mode != "none" else None,
                                  vecs, ALPHA, readout)
            for r, zi in zip(keep, z):
                rows.append({"pair_id": r.pair_id, "fact_id": r.fact_id,
                             "direction": r.direction, "mode": mode,
                             "condition": cond, "readout_layer": readout,
                             "z": zi.astype(np.float16).tolist()})
        print(f"[capture] {mode}/{cond} done", flush=True)
    _write(args, "capture", rows)


# --------------------------------------------------------------------------- #
# io + merge
# --------------------------------------------------------------------------- #

def _write(args, phase, rows):
    exp = args.out_dir / "expE"
    exp.mkdir(exist_ok=True)
    suf = "" if args.num_shards == 1 else f"_shard{args.shard_idx:02d}"
    p = exp / f"{phase}{suf}.parquet"
    if p.exists() and not args.force:
        sys.exit(f"refusing to overwrite {p}; pass --force")
    pd.DataFrame(rows).to_parquet(p, index=False)
    print(f"[{phase}] wrote {p} ({len(rows)} rows)", flush=True)


def merge(out_dir, phase, num_shards):
    exp = out_dir / "expE"
    parts = []
    for s in range(num_shards):
        suf = "" if num_shards == 1 else f"_shard{s:02d}"
        p = exp / f"{phase}{suf}.parquet"
        if not p.exists():
            sys.exit(f"missing shard output {p}")
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(exp / f"{phase}.parquet", index=False)
    return df


# --------------------------------------------------------------------------- #
# select + analyze
# --------------------------------------------------------------------------- #

def _deltas(df, base):
    b = base.set_index("prompt_instance")
    d = df.copy()
    for c in ("margin", "logp_gold", "entropy", "gold_rank"):
        d[f"d_{c}"] = d[c].to_numpy() - b.loc[d.prompt_instance, c].to_numpy()
    d["base_exact"] = b.loc[d.prompt_instance, "exact_match"].to_numpy()
    return d


def select(args):
    exp = args.out_dir / "expE"
    val = merge(args.out_dir, "val", args.num_shards)
    base = pd.read_parquet(exp / "baseline.parquet")
    d = _deltas(val[val.condition == "matched"], base)
    table, sel = [], {}
    for mode in MODES:
        best, best_val = None, -np.inf
        for L, g in d[d["mode"] == mode].groupby("layer"):
            mean, lo, hi = fact_bootstrap_ci(g.d_margin, g.fact_id)
            table.append({"mode": mode, "layer": int(L), "d_margin": mean,
                          "lo": lo, "hi": hi, "n": len(g)})
            if mean > best_val:
                best_val, best = mean, int(L)
        sel[mode] = {"layer": best, "val_d_margin": best_val}
    pd.DataFrame(table).sort_values(["mode", "layer"]).to_csv(
        exp / "depth_curve.csv", index=False)
    (exp / "selection.json").write_text(json.dumps(sel, indent=2))
    print("[select] " + json.dumps(sel), flush=True)


def analyze(args):
    exp = args.out_dir / "expE"
    base = pd.read_parquet(exp / "baseline.parquet")
    test = merge(args.out_dir, "test", args.num_shards)
    d = _deltas(test, base)
    out = []
    for (mode, cond), g in d.groupby(["mode", "condition"]):
        row = {"mode": mode, "condition": cond, "layer": int(g.layer.iloc[0]),
               "n": len(g), "n_facts": g.fact_id.nunique()}
        for met in ("d_margin", "d_logp_gold", "d_entropy"):
            mean, lo, hi = fact_bootstrap_ci(g[met], g.fact_id)
            row[met], row[f"{met}_lo"], row[f"{met}_hi"] = mean, lo, hi
        if cond == "reverse":
            row["exact_flip_down"] = float((g.base_exact & ~g.exact_match).mean())
        else:
            row["exact_match_rate"] = float(g.exact_match.mean())
            row["exact_flip_up"] = float((~g.base_exact & g.exact_match).mean())
        row["copied_donor_rate"] = float(g.copied_donor.mean()) \
            if "copied_donor" in g else 0.0
        out.append(row)
    res = pd.DataFrame(out).sort_values(["mode", "condition"])
    res.to_csv(exp / "results.csv", index=False)
    print(res.to_string(index=False), flush=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--model_name_or_path", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--phase",
                    choices=["baseline", "val", "test", "capture"],
                    default=None)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--merge_only", action="store_true")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="decoder layers to sweep (default: all)")
    ap.add_argument("--val_budget", type=int, default=400)
    ap.add_argument("--test_budget", type=int, default=800)
    ap.add_argument("--capture_pairs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.select:
        select(args)
        return
    if args.analyze:
        analyze(args)
        return
    if args.merge_only:
        merge(args.out_dir, args.phase, args.num_shards)
        return

    hstore = C.HSStore(args.out_dir)
    cstore = C.ComponentStore(args.out_dir) \
        if (args.out_dir / "component_states_v1" / "comp_meta.parquet").exists() \
        else None
    if args.layers is None:
        import json as _json
        man = args.out_dir / "component_manifest.json"
        if man.exists():
            n = _json.loads(man.read_text())["n_layers"]
        elif cstore is not None:
            n = 1 + max(int(f.stem.split("_L")[1])
                        for f in (args.out_dir / "component_states_v1").glob(
                            "attn_L*.safetensors"))
        else:
            n = 28
        args.layers = list(range(n))

    model, tok, device = C.load_model_and_tok(
        args.model_name_or_path, args.local_files_only)

    if args.phase == "baseline":
        run_baseline(args, args.out_dir, model, tok, device, hstore)
    elif args.phase == "capture":
        run_capture(args, args.out_dir, model, tok, device, hstore, cstore)
    elif args.phase in ("val", "test"):
        run_grid(args, args.out_dir, model, tok, device, hstore, cstore)
    else:
        sys.exit("pass --phase, --select, --analyze, or --merge_only")


if __name__ == "__main__":
    main()
