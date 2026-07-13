"""parametric_retrieval_access_v1 Experiment C: same-fact direct activation
patching (sharded TamIA worker + select/analyze phases).

For each mixed-outcome pair, transplant the donor's stored residual state at
the final prompt token into the recipient's forward pass at layer hs_idx,
h' = h + alpha (h_donor - h), and measure against matched controls:

  baseline         recipient prompt, no edit (one row per unique prompt)
  noop             patch the recipient's own stored state (hook sanity)
  matched          same-fact successful donor        <- the causal claim
  mismatched_type  successful donor, other fact, same answer type+category
  mismatched_rand  successful donor, other fact, popularity-matched
  random_noise     gaussian edit, norm-matched to the matched edit
  reverse          failed state patched INTO the successful prompt (necessity)

Metrics per run: teacher-forced logP(gold), logP(reference distractor),
gold-minus-distractor margin, decision-token entropy, gold first-token rank
among the candidate set, greedy exact match, donor-answer copying.

Phases:
  --phase val   sweep hs_idx x alpha grid on val pairs (budgeted)
  --select      pick best (hs_idx, alpha) on val, write expC/selection.json
  --phase test  frozen config on test pairs
  --analyze     paired fact-bootstrap stats vs controls -> expC/*.csv

  python scripts/parametric_retrieval/prga_expC_patch.py --phase val \
      --shard_idx $i --num_shards 4 --local_files_only
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
    assign_patch_donors,
    budget_pairs,
    fact_bootstrap_ci,
)

GRID_HS = [24, 26, 28]
GRID_ALPHA = [0.5, 1.0]
POSITION = "final_prompt_token"


def build_group_table(out_dir: Path, hs_meta: pd.DataFrame) -> pd.DataFrame:
    """Per fact x direction: answer_type/category/gbc_bin + one successful
    donor-pool instance that HAS stored vectors."""
    meta = pd.read_parquet(out_dir / "metadata.parquet")
    grading = pd.read_json(out_dir / "grading.jsonl", lines=True)
    grading["fact_id"] = grading.fact_id.astype(str)
    extracted = set(hs_meta[hs_meta.position_name == POSITION].instance_id)
    d = grading[(grading.prompt_mode == "direct") & grading.is_correct
                & grading.instance_id.isin(extracted)]
    pool = (d.sort_values("instance_id")
            .groupby(["fact_id", "direction"]).instance_id.first()
            .rename("donor_pool_instance_id").reset_index())
    g = meta.drop_duplicates(["fact_id", "direction"])[
        ["fact_id", "direction", "category", "gbc_bin",
         "subject_type", "object_type", "gold_answer"]].copy()
    g["fact_id"] = g.fact_id.astype(str)
    g["answer_type"] = np.where(g.direction == "direct", g.object_type,
                                g.subject_type)
    return g.merge(pool, on=["fact_id", "direction"], how="left")


def enumerate_runs(pairs: pd.DataFrame, grid_hs, grid_alpha) -> pd.DataFrame:
    """One row per executed forward-set. Baselines and noop are grid
    independent and emitted once."""
    runs = []
    seen_baseline = set()

    def baseline(prompt_iid, fact_id, direction, pair_id):
        if prompt_iid in seen_baseline:
            return
        seen_baseline.add(prompt_iid)
        runs.append({"pair_id": pair_id, "condition": "baseline",
                     "hs_idx": grid_hs[0], "alpha": 0.0,
                     "prompt_instance": prompt_iid, "donor_instance": None,
                     "fact_id": fact_id, "direction": direction})

    for r in pairs.itertuples():
        baseline(r.recipient_instance_id, r.fact_id, r.direction, r.pair_id)
        baseline(r.donor_instance_id, r.fact_id, r.direction, r.pair_id)
        runs.append({"pair_id": r.pair_id, "condition": "noop",
                     "hs_idx": grid_hs[0], "alpha": grid_alpha[-1],
                     "prompt_instance": r.recipient_instance_id,
                     "donor_instance": r.recipient_instance_id,
                     "fact_id": r.fact_id, "direction": r.direction})
        for k in grid_hs:
            for a in grid_alpha:
                base = {"pair_id": r.pair_id, "hs_idx": k, "alpha": a,
                        "fact_id": r.fact_id, "direction": r.direction}
                runs.append({**base, "condition": "matched",
                             "prompt_instance": r.recipient_instance_id,
                             "donor_instance": r.donor_matched})
                if r.donor_mismatched_type is not None:
                    runs.append({**base, "condition": "mismatched_type",
                                 "prompt_instance": r.recipient_instance_id,
                                 "donor_instance": r.donor_mismatched_type})
                if r.donor_mismatched_rand is not None:
                    runs.append({**base, "condition": "mismatched_rand",
                                 "prompt_instance": r.recipient_instance_id,
                                 "donor_instance": r.donor_mismatched_rand})
                runs.append({**base, "condition": "random_noise",
                             "prompt_instance": r.recipient_instance_id,
                             "donor_instance": r.donor_matched})
                runs.append({**base, "condition": "reverse",
                             "prompt_instance": r.donor_instance_id,
                             "donor_instance": r.recipient_instance_id})
    return pd.DataFrame(runs)


def run_phase(args) -> None:
    out_dir = args.out_dir
    exp_dir = out_dir / "expC"
    exp_dir.mkdir(exist_ok=True)
    suf = "" if args.num_shards == 1 else f"_shard{args.shard_idx:02d}"
    out_path = exp_dir / f"runs_{args.phase}{suf}.parquet"
    if out_path.exists() and not args.force:
        sys.exit(f"refusing to overwrite {out_path}; pass --force")

    store = C.HSStore(out_dir)
    groups = build_group_table(out_dir, store.meta)
    pairs = pd.read_parquet(out_dir / "pairs.parquet")
    pairs["fact_id"] = pairs.fact_id.astype(str)
    pairs = pairs[pairs.split == args.phase].reset_index(drop=True)
    budget = args.val_budget if args.phase == "val" else args.test_budget
    pairs = budget_pairs(pairs, budget, seed=args.seed)
    pairs = assign_patch_donors(pairs, groups, seed=args.seed)

    if args.phase == "test":
        sel = json.loads((exp_dir / "selection.json").read_text())
        grid_hs, grid_alpha = [sel["hs_idx"]], [sel["alpha"]]
    else:
        grid_hs, grid_alpha = GRID_HS, GRID_ALPHA
    runs = enumerate_runs(pairs, grid_hs, grid_alpha)
    runs = runs.iloc[args.shard_idx::args.num_shards].reset_index(drop=True)
    print(f"[expC] phase={args.phase} shard {args.shard_idx}/"
          f"{args.num_shards}: {len(runs)} runs "
          f"({len(pairs)} pairs)", flush=True)

    model, tok, device = C.load_model_and_tok(
        args.model_name_or_path, args.local_files_only)
    meta = pd.read_parquet(out_dir / "metadata.parquet")
    meta["fact_id"] = meta.fact_id.astype(str)
    need = set(runs.prompt_instance)
    prompt_ids = C.render_prompt_ids(tok, meta[meta.instance_id.isin(need)])
    cands = C.load_candidate_table(out_dir, tok)
    gold_of = {(r.fact_id, r.direction): r.gold_answer
               for r in meta.drop_duplicates(["fact_id", "direction"])
               .itertuples()}
    donor_gold = {}
    inst_group = {r.instance_id: (r.fact_id, r.direction)
                  for r in meta.itertuples()}
    rng = np.random.default_rng(args.seed + args.shard_idx)

    import torch
    rows = []
    B = args.batch_size
    for start in range(0, len(runs), B):
        chunk = runs.iloc[start:start + B]
        p_ids, vecs, golds, distrs, cand_rows, alphas_k = [], [], [], [], [], []
        for r in chunk.itertuples():
            key = (r.fact_id, r.direction)
            p_ids.append(prompt_ids[r.prompt_instance])
            golds.append(gold_of[key])
            distrs.append(cands[key]["negatives"][0]
                          if cands[key]["negatives"] else "unknown")
            cand_rows.append(cands[key]["first_ids"])
            if r.condition == "baseline":
                vecs.append(None)
            elif r.condition == "random_noise":
                d_vec = store.vec(r.donor_instance, POSITION, r.hs_idx)
                h_vec = store.vec(r.prompt_instance, POSITION, r.hs_idx)
                edit = np.linalg.norm(
                    d_vec.astype(np.float32) - h_vec.astype(np.float32))
                noise = rng.standard_normal(len(h_vec)).astype(np.float32)
                noise *= edit / max(np.linalg.norm(noise), 1e-9)
                vecs.append(h_vec.astype(np.float32) + noise)
            else:
                vecs.append(store.vec(r.donor_instance, POSITION,
                                      r.hs_idx).astype(np.float32))
        # group chunk rows by (hs_idx, alpha) for the hook
        by_cfg: dict[tuple, list[int]] = {}
        for j, r in enumerate(chunk.itertuples()):
            by_cfg.setdefault((r.hs_idx, float(r.alpha)), []).append(j)
        for (k, a), idxs in by_cfg.items():
            sub_p = [p_ids[j] for j in idxs]
            sub_v = [vecs[j] for j in idxs]
            edit = C.ResidualEdit(model, k, "patch") if a > 0 else None
            gen = C.generate_with_edit(model, tok, device, sub_p, edit,
                                       sub_v, a, args.max_new_tokens)
            edit2 = C.ResidualEdit(model, k, "patch") if a > 0 else None
            sc_g = C.score_with_edit(model, tok, device, sub_p,
                                     [golds[j] for j in idxs], edit2,
                                     sub_v, a, [cand_rows[j] for j in idxs])
            edit3 = C.ResidualEdit(model, k, "patch") if a > 0 else None
            sc_d = C.score_with_edit(model, tok, device, sub_p,
                                     [distrs[j] for j in idxs], edit3,
                                     sub_v, a)
            for m, j in enumerate(idxs):
                r = chunk.iloc[j]
                em = grade_answer(gen[m].strip().splitlines()[0]
                                  if gen[m].strip() else "", golds[j])[0]
                copied = False
                if r.condition in ("mismatched_type", "mismatched_rand"):
                    dg = donor_gold.get(r.donor_instance)
                    if dg is None:
                        dg = gold_of[inst_group[r.donor_instance]]
                        donor_gold[r.donor_instance] = dg
                    copied = grade_answer(gen[m], dg)[0]
                rows.append({**r.to_dict(),
                             "logp_gold": sc_g[m]["logp_answer"],
                             "logp_distr": sc_d[m]["logp_answer"],
                             "margin": sc_g[m]["logp_answer"]
                             - sc_d[m]["logp_answer"],
                             "entropy": sc_g[m]["entropy"],
                             "gold_rank": sc_g[m]["gold_rank"],
                             "exact_match": bool(em),
                             "copied_donor": bool(copied),
                             "gen_text": gen[m]})
        if (start // B) % 10 == 0:
            print(f"[expC] {min(start + B, len(runs))}/{len(runs)} runs",
                  flush=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[expC] wrote {out_path} ({len(rows)} rows)", flush=True)


def merge(args, phase: str) -> pd.DataFrame:
    exp_dir = args.out_dir / "expC"
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


def deltas_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.condition == "baseline"].set_index("prompt_instance")
    rest = df[df.condition != "baseline"].copy()
    for c in ["margin", "logp_gold", "entropy", "gold_rank"]:
        rest[f"d_{c}"] = (rest[c].to_numpy()
                          - base.loc[rest.prompt_instance, c].to_numpy())
    rest["base_exact"] = base.loc[rest.prompt_instance,
                                  "exact_match"].to_numpy()
    return rest


def select(args) -> None:
    exp_dir = args.out_dir / "expC"
    df = merge(args, "val")
    d = deltas_vs_baseline(df)
    m = d[d.condition == "matched"]
    best, best_val = None, -np.inf
    table = []
    for (k, a), g in m.groupby(["hs_idx", "alpha"]):
        mean, lo, hi = fact_bootstrap_ci(g.d_margin, g.fact_id)
        table.append({"hs_idx": int(k), "alpha": float(a),
                      "d_margin": mean, "lo": lo, "hi": hi, "n": len(g)})
        if mean > best_val:
            best_val, best = mean, (int(k), float(a))
    pd.DataFrame(table).to_csv(exp_dir / "val_grid.csv", index=False)
    (exp_dir / "selection.json").write_text(json.dumps(
        {"hs_idx": best[0], "alpha": best[1],
         "criterion": "max fact-bootstrap mean d_margin, matched condition",
         "val_d_margin": best_val}, indent=2))
    print(f"[expC] selected hs_idx={best[0]} alpha={best[1]} "
          f"(val d_margin {best_val:.3f})", flush=True)


def analyze(args) -> None:
    exp_dir = args.out_dir / "expC"
    out = []
    for phase in ["val", "test"]:
        p = exp_dir / f"runs_{phase}.parquet"
        if not p.exists():
            df = merge(args, phase)
        else:
            df = pd.read_parquet(p)
        d = deltas_vs_baseline(df)
        if phase == "val":
            sel = json.loads((exp_dir / "selection.json").read_text())
            d = d[(d.hs_idx == sel["hs_idx"]) & ((d.alpha == sel["alpha"])
                                                 | (d.condition == "noop"))]
        for cond, g in d.groupby("condition"):
            row = {"phase": phase, "condition": cond, "n": len(g),
                   "n_facts": g.fact_id.nunique()}
            for met in ["d_margin", "d_logp_gold", "d_entropy"]:
                mean, lo, hi = fact_bootstrap_ci(g[met], g.fact_id)
                row[met], row[f"{met}_lo"], row[f"{met}_hi"] = mean, lo, hi
            if cond == "reverse":
                row["exact_flip_down"] = float(
                    (g.base_exact & ~g.exact_match).mean())
            else:
                row["exact_match_rate"] = float(g.exact_match.mean())
                row["exact_flip_up"] = float(
                    (~g.base_exact & g.exact_match).mean())
            row["copied_donor_rate"] = float(g.copied_donor.mean())
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
    ap.add_argument("--phase", choices=["val", "test"], default=None)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--val_budget", type=int, default=600)
    ap.add_argument("--test_budget", type=int, default=1200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.select:
        select(args)
    elif args.analyze:
        analyze(args)
    elif args.phase:
        run_phase(args)
    else:
        sys.exit("pass --phase val|test, --select, or --analyze")


if __name__ == "__main__":
    main()
