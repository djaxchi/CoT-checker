"""parametric_retrieval_access_v1 Experiment D: fact-independent access
subspace (estimate on train facts, tune alpha on val, test once).

Directions are estimated at CELLS = (hs_idx, final_prompt_token) from
TRAIN-fact mixed groups only, on confound-residualized states (template,
seed, direction, gbc_bin, category, prompt length regressed out before the
success-fail difference is taken):

  mean_diff / svd1 / svd4_proj   from paired diffs (see causal module)
  lda                            regularized LDA success-vs-fail direction
  raw_mean                       mean diff WITHOUT residualization (ablation)
  relcond                        per category x answer-type mean diff,
                                 fallback to global (relation-conditioned)
  random_0..2                    norm-matched random controls

Steering: h' = h + alpha * edit_norm * v applied at the final prompt token
and every subsequent generated token at the cell's layer. Failed direct
paraphrases of held-out facts only; the model never sees donor content.

Metrics per run: teacher-forced logP(gold), logP(reference distractor),
margin, decision entropy, gold candidate rank, greedy exact match.

Phases:
  --estimate     write expD/directions.npz (train facts only)
  --phase val    sweep direction x alpha on val failed instances (budgeted)
  --select       best (cell, direction, alpha) on val -> expD/selection.json
  --phase test   frozen config + 2 random controls, all test failed instances
  --analyze      fact-bootstrap stats -> expD/results.csv
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
from src.analysis.parametric_retrieval import grade_answer  # noqa: E402
from src.analysis.parametric_retrieval_causal import (  # noqa: E402
    confound_features,
    estimate_directions,
    fact_bootstrap_ci,
    lda_direction,
    paired_diffs,
    residualize,
)

CELLS = [27, 28]  # hs_idx at final_prompt_token
POSITION = "final_prompt_token"
GRID_ALPHA = [0.5, 1.0, 2.0, 4.0]


def load_cell_rows(store: C.HSStore, out_dir: Path,
                   hs_idx: int) -> tuple[np.ndarray, pd.DataFrame]:
    meta = store.meta
    mask = (meta.position_name == POSITION).to_numpy()
    rows = meta[mask].reset_index(drop=True)
    H = store.layer(hs_idx)[mask].astype(np.float32)
    inst = pd.read_parquet(out_dir / "metadata.parquet")
    inst["fact_id"] = inst.fact_id.astype(str)
    extra = inst.set_index("instance_id")[
        ["gbc_bin", "category", "gold_answer"]]
    rows = rows.join(extra, on="instance_id")
    return H, rows


def estimate(args) -> None:
    exp_dir = args.out_dir / "expD"
    exp_dir.mkdir(exist_ok=True)
    store = C.HSStore(args.out_dir)
    groups = pd.read_parquet(args.out_dir / "group_outcomes.parquet")
    groups["fact_id"] = groups.fact_id.astype(str)
    mixed_train = groups[groups.is_mixed & (groups.split == "train")]
    keys = set(zip(mixed_train.fact_id, mixed_train.direction))
    out = {}
    meta_all = pd.read_parquet(args.out_dir / "metadata.parquet")
    meta_all["fact_id"] = meta_all.fact_id.astype(str)
    atype = {(r.fact_id, r.direction):
             (r.object_type if r.direction == "direct" else r.subject_type)
             for r in meta_all.drop_duplicates(["fact_id", "direction"])
             .itertuples()}
    for k in CELLS:
        H, rows = load_cell_rows(store, args.out_dir, k)
        sel = np.array([(f, d) in keys for f, d
                        in zip(rows.fact_id, rows.direction)])
        sel &= rows.is_correct.notna().to_numpy()
        Hk, rk = H[sel], rows[sel].reset_index(drop=True)
        F = confound_features(rk)
        Hr = residualize(Hk, F)
        diffs, _ = paired_diffs(Hr, rk)
        diffs_raw, _ = paired_diffs(Hk, rk)
        dirs = estimate_directions(diffs, seed=args.seed, n_random=3)
        dirs["raw_mean"] = (diffs_raw.mean(axis=0)
                            / max(np.linalg.norm(diffs_raw.mean(axis=0)),
                                  1e-12))
        dirs["lda"] = lda_direction(Hr, rk.is_correct.to_numpy(bool))
        # relation-conditioned: per category x answer_type mean, min 8 groups
        rk = rk.assign(atype=[atype[(f, d)] for f, d
                              in zip(rk.fact_id, rk.direction)])
        rel = {}
        for (cat, at), g in rk.groupby(["category", "atype"]):
            dsub, _ = paired_diffs(Hr[g.index], g.reset_index(drop=True))
            if len(dsub) >= 8:
                m = dsub.mean(axis=0)
                rel[f"{cat}|{at}"] = m / max(np.linalg.norm(m), 1e-12)
        edit_norm = float(np.linalg.norm(diffs_raw, axis=1).mean())
        out[k] = {"dirs": dirs, "rel": rel, "edit_norm": edit_norm,
                  "n_groups": len(diffs)}
        print(f"[expD] cell hs{k}: {len(diffs)} train groups, "
              f"edit_norm {edit_norm:.1f}, {len(rel)} relcond cells",
              flush=True)
    payload = {}
    for k, v in out.items():
        for name, vec in v["dirs"].items():
            payload[f"hs{k}::{name}"] = vec.astype(np.float32)
        for name, vec in v["rel"].items():
            payload[f"hs{k}::relcond::{name}"] = vec.astype(np.float32)
        payload[f"hs{k}::edit_norm"] = np.array([v["edit_norm"]],
                                                dtype=np.float32)
    np.savez(exp_dir / "directions.npz", **payload)
    (exp_dir / "estimate_manifest.json").write_text(json.dumps(
        {"cells": CELLS, "n_groups": {k: out[k]["n_groups"] for k in out},
         "edit_norms": {k: out[k]["edit_norm"] for k in out},
         "seed": args.seed}, indent=2))


def load_directions(exp_dir: Path):
    z = np.load(exp_dir / "directions.npz")
    dirs: dict[int, dict[str, np.ndarray]] = {}
    rel: dict[int, dict[str, np.ndarray]] = {}
    norms: dict[int, float] = {}
    for key in z.files:
        parts = key.split("::")
        k = int(parts[0][2:])
        if parts[1] == "edit_norm":
            norms[k] = float(z[key][0])
        elif parts[1] == "relcond":
            rel.setdefault(k, {})[parts[2]] = z[key]
        else:
            dirs.setdefault(k, {})[parts[1]] = z[key]
    return dirs, rel, norms


def target_instances(args, split: str) -> pd.DataFrame:
    """Failed direct paraphrases of mixed groups in the split (these have
    stored states and a real chance of rescue)."""
    grading = pd.read_json(args.out_dir / "grading.jsonl", lines=True)
    grading["fact_id"] = grading.fact_id.astype(str)
    groups = pd.read_parquet(args.out_dir / "group_outcomes.parquet")
    groups["fact_id"] = groups.fact_id.astype(str)
    mixed = groups[groups.is_mixed & (groups.split == split)]
    keys = set(zip(mixed.fact_id, mixed.direction))
    d = grading[(grading.prompt_mode == "direct") & ~grading.is_correct
                & (grading.split == split)]
    d = d[[(f, dd) in keys for f, dd in zip(d.fact_id, d.direction)]]
    return d.reset_index(drop=True)


def run_phase(args) -> None:
    exp_dir = args.out_dir / "expD"
    suf = "" if args.num_shards == 1 else f"_shard{args.shard_idx:02d}"
    out_path = exp_dir / f"runs_{args.phase}{suf}.parquet"
    if out_path.exists() and not args.force:
        sys.exit(f"refusing to overwrite {out_path}; pass --force")
    dirs, rel, norms = load_directions(exp_dir)

    targets = target_instances(args, args.phase)
    rng = np.random.default_rng(args.seed)
    if args.phase == "val" and targets.fact_id.nunique() > 0:
        facts = sorted(set(zip(targets.fact_id, targets.direction)))
        order = rng.permutation(len(facts))
        keep, budgeted = set(), 0
        for i in order:
            keep.add(facts[i])
            budgeted += (targets.fact_id.eq(facts[i][0])
                         & targets.direction.eq(facts[i][1])).sum()
            if budgeted >= args.val_budget:
                break
        targets = targets[[(f, d) in keep for f, d in
                           zip(targets.fact_id, targets.direction)]]
    targets = targets.reset_index(drop=True)

    if args.phase == "test":
        sel = json.loads((exp_dir / "selection.json").read_text())
        combos = [(sel["hs_idx"], sel["direction_name"], sel["alpha"]),
                  (sel["hs_idx"], "random_0", sel["alpha"]),
                  (sel["hs_idx"], "random_1", sel["alpha"])]
    else:
        names = ["mean_diff", "svd1", "svd4_proj", "lda", "raw_mean",
                 "relcond", "random_0", "random_1", "random_2"]
        combos = [(k, n, a) for k in CELLS for n in names
                  for a in GRID_ALPHA]

    runs = []
    for t in targets.itertuples():
        runs.append({"instance_id": t.instance_id, "fact_id": t.fact_id,
                     "direction": t.direction, "hs_idx": CELLS[0],
                     "direction_name": "baseline", "alpha": 0.0})
        for (k, n, a) in combos:
            runs.append({"instance_id": t.instance_id, "fact_id": t.fact_id,
                         "direction": t.direction, "hs_idx": k,
                         "direction_name": n, "alpha": a})
    runs = pd.DataFrame(runs).iloc[args.shard_idx::args.num_shards
                                   ].reset_index(drop=True)
    print(f"[expD] phase={args.phase} shard {args.shard_idx}/"
          f"{args.num_shards}: {len(runs)} runs on {len(targets)} failed "
          f"instances", flush=True)

    model, tok, device = C.load_model_and_tok(
        args.model_name_or_path, args.local_files_only)
    meta = pd.read_parquet(args.out_dir / "metadata.parquet")
    meta["fact_id"] = meta.fact_id.astype(str)
    need = set(runs.instance_id)
    prompt_ids = C.render_prompt_ids(tok, meta[meta.instance_id.isin(need)])
    cands = C.load_candidate_table(args.out_dir, tok)
    gold_of = {(r.fact_id, r.direction): r.gold_answer
               for r in meta.drop_duplicates(["fact_id", "direction"])
               .itertuples()}
    atype = {(r.fact_id, r.direction):
             (r.object_type if r.direction == "direct" else r.subject_type)
             for r in meta.drop_duplicates(["fact_id", "direction"])
             .itertuples()}
    cat_of = {(r.fact_id, r.direction): r.category
              for r in meta.drop_duplicates(["fact_id", "direction"])
              .itertuples()}

    def steer_vec(r) -> np.ndarray | None:
        if r.direction_name == "baseline":
            return None
        k = r.hs_idx
        if r.direction_name == "relcond":
            key = f"{cat_of[(r.fact_id, r.direction)]}|" \
                  f"{atype[(r.fact_id, r.direction)]}"
            v = rel.get(k, {}).get(key, dirs[k]["mean_diff"])
        else:
            v = dirs[k][r.direction_name]
        return (r.alpha * norms[k] * v).astype(np.float32)

    rows = []
    B = args.batch_size
    for start in range(0, len(runs), B):
        chunk = runs.iloc[start:start + B]
        by_cfg: dict[int, list[int]] = {}
        for j, r in enumerate(chunk.itertuples()):
            by_cfg.setdefault(int(r.hs_idx) if r.direction_name != "baseline"
                              else -1, []).append(j)
        for k, idxs in by_cfg.items():
            sub = chunk.iloc[idxs]
            p_ids = [prompt_ids[r.instance_id] for r in sub.itertuples()]
            vecs = [steer_vec(r) for r in sub.itertuples()]
            golds = [gold_of[(r.fact_id, r.direction)]
                     for r in sub.itertuples()]
            distrs = [cands[(r.fact_id, r.direction)]["negatives"][0]
                      if cands[(r.fact_id, r.direction)]["negatives"]
                      else "unknown" for r in sub.itertuples()]
            cand_rows = [cands[(r.fact_id, r.direction)]["first_ids"]
                         for r in sub.itertuples()]
            edit = C.ResidualEdit(model, k, "steer") if k > 0 else None
            gen = C.generate_with_edit(model, tok, device, p_ids, edit,
                                       vecs, 1.0, args.max_new_tokens)
            e2 = C.ResidualEdit(model, k, "steer") if k > 0 else None
            sc_g = C.score_with_edit(model, tok, device, p_ids, golds, e2,
                                     vecs, 1.0, cand_rows)
            e3 = C.ResidualEdit(model, k, "steer") if k > 0 else None
            sc_d = C.score_with_edit(model, tok, device, p_ids, distrs, e3,
                                     vecs, 1.0)
            for m, j in enumerate(idxs):
                r = chunk.iloc[j]
                em = grade_answer(gen[m].strip().splitlines()[0]
                                  if gen[m].strip() else "", golds[m])[0]
                rows.append({**r.to_dict(),
                             "logp_gold": sc_g[m]["logp_answer"],
                             "logp_distr": sc_d[m]["logp_answer"],
                             "margin": sc_g[m]["logp_answer"]
                             - sc_d[m]["logp_answer"],
                             "entropy": sc_g[m]["entropy"],
                             "gold_rank": sc_g[m]["gold_rank"],
                             "exact_match": bool(em),
                             "gen_text": gen[m]})
        if (start // B) % 10 == 0:
            print(f"[expD] {min(start + B, len(runs))}/{len(runs)}",
                  flush=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[expD] wrote {out_path} ({len(rows)} rows)", flush=True)


def merge(args, phase: str) -> pd.DataFrame:
    exp_dir = args.out_dir / "expD"
    parts = []
    for s in range(args.num_shards):
        suf = "" if args.num_shards == 1 else f"_shard{s:02d}"
        p = exp_dir / f"runs_{phase}{suf}.parquet"
        if not p.exists():
            sys.exit(f"missing shard output {p}")
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(exp_dir / f"runs_{phase}.parquet", index=False)
    return df


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.direction_name == "baseline"].set_index("instance_id")
    rest = df[df.direction_name != "baseline"].copy()
    for c in ["margin", "logp_gold", "entropy", "gold_rank"]:
        rest[f"d_{c}"] = (rest[c].to_numpy()
                          - base.loc[rest.instance_id, c].to_numpy())
    return rest


def select(args) -> None:
    exp_dir = args.out_dir / "expD"
    d = add_deltas(merge(args, "val"))
    table = []
    for (k, n, a), g in d.groupby(["hs_idx", "direction_name", "alpha"]):
        mean, lo, hi = fact_bootstrap_ci(g.d_logp_gold, g.fact_id)
        table.append({"hs_idx": int(k), "direction_name": n,
                      "alpha": float(a), "d_logp_gold": mean,
                      "lo": lo, "hi": hi,
                      "exact_match": float(g.exact_match.mean()),
                      "d_margin": float(g.d_margin.mean()), "n": len(g)})
    tab = pd.DataFrame(table)
    tab.to_csv(exp_dir / "val_grid.csv", index=False)
    real = tab[~tab.direction_name.str.startswith("random")]
    best = real.sort_values("d_logp_gold", ascending=False).iloc[0]
    rand = tab[tab.direction_name.str.startswith("random")
               & (tab.hs_idx == best.hs_idx) & (tab.alpha == best.alpha)]
    beats_random = bool(best.lo > rand.d_logp_gold.max()) if len(rand) \
        else False
    (exp_dir / "selection.json").write_text(json.dumps(
        {"hs_idx": int(best.hs_idx), "direction_name": best.direction_name,
         "alpha": float(best.alpha), "val_d_logp_gold": float(best.d_logp_gold),
         "val_ci": [float(best.lo), float(best.hi)],
         "beats_random_on_val": beats_random,
         "best_random_val": float(rand.d_logp_gold.max()) if len(rand)
         else None}, indent=2))
    print(f"[expD] selected {best.direction_name} hs{int(best.hs_idx)} "
          f"alpha={best.alpha} (val d_logp_gold {best.d_logp_gold:.3f}, "
          f"beats_random={beats_random})", flush=True)


def analyze(args) -> None:
    exp_dir = args.out_dir / "expD"
    p = exp_dir / "runs_test.parquet"
    df = pd.read_parquet(p) if p.exists() else merge(args, "test")
    d = add_deltas(df)
    out = []
    for n, g in d.groupby("direction_name"):
        row = {"direction_name": n, "n": len(g),
               "n_facts": g.fact_id.nunique(),
               "exact_match": float(g.exact_match.mean())}
        for met in ["d_logp_gold", "d_margin", "d_entropy", "d_gold_rank"]:
            mean, lo, hi = fact_bootstrap_ci(g[met], g.fact_id)
            row[met], row[f"{met}_lo"], row[f"{met}_hi"] = mean, lo, hi
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(exp_dir / "results.csv", index=False)
    print(res.to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--model_name_or_path", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--phase", choices=["val", "test"], default=None)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--val_budget", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.estimate:
        estimate(args)
    elif args.select:
        select(args)
    elif args.analyze:
        analyze(args)
    elif args.phase:
        run_phase(args)
    else:
        sys.exit("pass --estimate, --phase val|test, --select, or --analyze")


if __name__ == "__main__":
    main()
