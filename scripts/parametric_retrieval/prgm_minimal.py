"""parametric_retrieval_minimal_v1: what minimal part of the same-fact edit
actually flips the answer.

Given the winning full-residual layer (from expE) and MLP layer, and the edit
Delta = state_donor - state_recip at the final prompt token, this dissects the
edit at three granularities:

  coord   residual space at the full-selected layer. Inject only the top-k
          coordinates of Delta by |Delta| (vs random-k), and inject only its
          projection onto the top-r shared subspace of train-fact deltas
          (vs r). Answers: sparse specific coordinates, or a distributed /
          low-rank shared direction?

  neuron  MLP space at the mlp-selected layer. The MLP output is linear in the
          intermediate activation g (mlp_out = W_down @ g), so swapping neuron
          i from recipient to donor adds (dg_i) * W_down[:, i]. Inject only the
          top-k neurons ranked by gradient attribution
          (dlogP(gold)/dg_i * dg_i), by magnitude (|dg_i|*||W_down[:,i]||), or
          random. Answers: which MLP neurons carry the flip, and how few.

  greedy  on pairs the full edit flips, greedily forward-select neurons from
          the top attribution pool until the answer flips: the true minimal set
          size, and whether the same neurons recur across facts.

Flip criterion (cheap, no generation): the gold answer's first token becomes
rank 1 among the 32-candidate set at the decision token (gold_rank == 1), given
the recipient failed at baseline. Calibrated against the k=all reference.

Phases: --phase run (sharded by pair) -> --analyze (curves + greedy + neuron
recurrence). Layer axis and baseline are read from expE.
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
from src.analysis.parametric_retrieval_causal import (  # noqa: E402
    assign_patch_donors,
    budget_pairs,
    fact_bootstrap_ci,
)

COORD_KS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3584]
SUBSPACE_RS = [1, 2, 4, 8, 16, 32, 64]
NEURON_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 18944]
GREEDY_POOL = 32
GREEDY_MAX_STEPS = 12
TOP_SAVE = 32          # per-pair top attribution neurons saved for recurrence
POSITION = "final_prompt_token"


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested)
# --------------------------------------------------------------------------- #

def topk_mask_vec(delta: np.ndarray, k: int) -> np.ndarray:
    """Zero all but the k largest-|delta| coordinates."""
    v = np.zeros_like(delta)
    if k >= len(delta):
        return delta.copy()
    idx = np.argpartition(np.abs(delta), -k)[-k:]
    v[idx] = delta[idx]
    return v


def randk_mask_vec(delta: np.ndarray, k: int, rng) -> np.ndarray:
    v = np.zeros_like(delta)
    if k >= len(delta):
        return delta.copy()
    idx = rng.choice(len(delta), size=k, replace=False)
    v[idx] = delta[idx]
    return v


def subspace_vec(delta: np.ndarray, basis: np.ndarray, r: int) -> np.ndarray:
    """Project delta onto the top-r rows of basis (r x d, orthonormal)."""
    U = basis[:r]
    return U.T @ (U @ delta)


def build_subspace_basis(deltas: np.ndarray, r: int = 64) -> np.ndarray:
    """Top-r right-singular vectors of the raw (uncentered) train delta matrix
    (g x d): the shared directions the same-fact edits live in."""
    _, _, vt = np.linalg.svd(deltas, full_matrices=False)
    return vt[:r]


def rank_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores (descending)."""
    k = min(k, len(scores))
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


# --------------------------------------------------------------------------- #
# gradient attribution
# --------------------------------------------------------------------------- #

def grad_resid_batch(model, tok, device, prompt_ids, golds, hs_idx):
    """d logP(gold) / d(resid_post at hs_idx, final prompt token) per sample,
    via one forward+backward. Returns (n, hidden) float32."""
    import torch
    import torch.nn.functional as tF
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ans = [tok(" " + str(a), add_special_tokens=False)["input_ids"]
           for a in golds]
    seqs = [p + a for p, a in zip(prompt_ids, ans)]
    mx = max(len(s) for s in seqs)
    inp = torch.tensor([s + [pad] * (mx - len(s)) for s in seqs],
                       dtype=torch.long, device=device)
    att = torch.tensor([[1] * len(s) + [0] * (mx - len(s)) for s in seqs],
                       dtype=torch.long, device=device)
    cap = {}

    def fhook(module, a, output):
        hh = output[0] if isinstance(output, tuple) else output
        hh.retain_grad()
        cap["h"] = hh
        return output

    handle = model.model.layers[hs_idx - 1].register_forward_hook(fhook)
    model.zero_grad(set_to_none=True)
    logits = model(inp, attention_mask=att, use_cache=False).logits
    loss = 0.0
    for b, (p, a) in enumerate(zip(prompt_ids, ans)):
        lp = tF.log_softmax(logits[b].float(), dim=-1)
        for j, t in enumerate(a):
            loss = loss + lp[len(p) - 1 + j, t]
    loss.backward()
    g = cap["h"].grad.float()
    handle.remove()
    out = np.stack([g[b, len(p) - 1].cpu().numpy()
                    for b, p in enumerate(prompt_ids)])
    model.zero_grad(set_to_none=True)
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def load_ctx(out_dir, model, tok):
    sel = json.loads((out_dir / "expE" / "selection.json").read_text())
    full_layer = int(sel["full"]["layer"])
    mlp_layer = int(sel["mlp"]["layer"])
    hstore = C.HSStore(out_dir)
    nstore = C.NeuronStore(out_dir)
    import torch
    Wd = model.model.layers[mlp_layer].mlp.down_proj.weight.detach()
    Wd = Wd.to(torch.float32)                       # (hidden, inter)
    col_norms = Wd.norm(dim=0).cpu().numpy()        # (inter,) residual impact
    return sel, full_layer, mlp_layer, hstore, nstore, Wd, col_norms


def train_subspace(out_dir, hstore, full_layer, seed):
    pairs = pd.read_parquet(out_dir / "pairs.parquet")
    pairs["fact_id"] = pairs.fact_id.astype(str)
    tr = pairs[pairs.split == "train"]
    tr = budget_pairs(tr, 1500, seed=seed)
    deltas = []
    for r in tr.itertuples():
        hd = hstore.vec(r.donor_instance_id, POSITION, full_layer + 1)
        hr = hstore.vec(r.recipient_instance_id, POSITION, full_layer + 1)
        if hd is not None and hr is not None:
            deltas.append((hd.astype(np.float32) - hr.astype(np.float32)))
    return build_subspace_basis(np.stack(deltas), r=max(SUBSPACE_RS))


def run(args):
    import torch
    model, tok, device = C.load_model_and_tok(
        args.model_name_or_path, args.local_files_only)
    out_dir = args.out_dir
    sel, full_layer, mlp_layer, hstore, nstore, Wd, col_norms = \
        load_ctx(out_dir, model, tok)
    basis = train_subspace(out_dir, hstore, full_layer, args.seed)

    groups = build_group_table(out_dir, hstore.meta)
    pairs = pd.read_parquet(out_dir / "pairs.parquet")
    pairs["fact_id"] = pairs.fact_id.astype(str)
    pairs = pairs[pairs.split == "test"].reset_index(drop=True)
    pairs = budget_pairs(pairs, args.test_budget, seed=args.seed)
    pairs = assign_patch_donors(pairs, groups, seed=args.seed)
    pairs = pairs.iloc[args.shard_idx::args.num_shards].reset_index(drop=True)

    base = pd.read_parquet(out_dir / "expE" / "baseline.parquet")
    base_rank = dict(zip(base.prompt_instance, base.gold_rank))

    meta = pd.read_parquet(out_dir / "metadata.parquet")
    meta["fact_id"] = meta.fact_id.astype(str)
    need = set(pairs.recipient_instance_id)
    prompt_ids = C.render_prompt_ids(tok, meta[meta.instance_id.isin(need)])
    cands = C.load_candidate_table(out_dir, tok)
    gold_of = {(r.fact_id, r.direction): r.gold_answer
               for r in meta.drop_duplicates(["fact_id", "direction"])
               .itertuples()}
    rng = np.random.default_rng(args.seed + args.shard_idx)

    full_hs, mlp_hs = full_layer + 1, mlp_layer + 1
    Wd_np = Wd.cpu().numpy()                        # (hidden, inter), reused
    rows, greedy_rows = [], []
    B = args.pair_batch
    for s in range(0, len(pairs), B):
        chunk = pairs.iloc[s:s + B]
        recips = [r.recipient_instance_id for r in chunk.itertuples()]
        keys = [(r.fact_id, r.direction) for r in chunk.itertuples()]
        p_ids = [prompt_ids[i] for i in recips]
        golds = [gold_of[k] for k in keys]
        cand_rows = [cands[k]["first_ids"] for k in keys]
        # gradient attribution wrt the mlp-layer resid_post
        gres = grad_resid_batch(model, tok, device, p_ids, golds, mlp_hs)

        for bi, r in enumerate(chunk.itertuples()):
            hd = hstore.vec(r.donor_instance_id, POSITION, full_hs)
            hr = hstore.vec(r.recipient_instance_id, POSITION, full_hs)
            gd = nstore.vec(r.donor_instance_id, mlp_layer)
            gr = nstore.vec(r.recipient_instance_id, mlp_layer)
            if hd is None or hr is None or gd is None or gr is None:
                continue
            d_res = hd.astype(np.float32) - hr.astype(np.float32)
            dg = gd.astype(np.float32) - gr.astype(np.float32)
            grw = gres[bi] @ Wd_np                          # (inter,) resid-grad align
            attr = grw * dg                                 # attribution
            mag = np.abs(dg) * col_norms
            base_r = base_rank.get(r.recipient_instance_id, 99)

            configs = []   # (task, ranking, k, hs_idx, v_np)
            for k in COORD_KS:
                configs.append(("coord", "topk_mag", k, full_hs,
                                topk_mask_vec(d_res, k)))
                configs.append(("coord", "random", k, full_hs,
                                randk_mask_vec(d_res, k, rng)))
            for rr in SUBSPACE_RS:
                configs.append(("coord", "subspace", rr, full_hs,
                                subspace_vec(d_res, basis, rr)))
            # neuron reconstructions
            dg_t = torch.as_tensor(dg, device=device)

            def recon(S_idx):
                st = torch.as_tensor(np.asarray(S_idx), device=device,
                                     dtype=torch.long)
                return (Wd[:, st] @ dg_t[st]).cpu().numpy()

            for k in NEURON_KS:
                for ranking, sc in (("attr", attr), ("magnitude", mag)):
                    configs.append(("neuron", ranking, k, mlp_hs,
                                    recon(rank_indices(sc, k))))
                Sr = rng.choice(len(dg), size=min(k, len(dg)), replace=False)
                configs.append(("neuron", "random", k, mlp_hs, recon(Sr)))

            # score all configs, grouped by hs_idx
            for hs_idx in (full_hs, mlp_hs):
                grp = [c for c in configs if c[3] == hs_idx]
                if not grp:
                    continue
                edit = C.ResidualEdit(model, hs_idx, "add")
                sc = C.score_with_edit(
                    model, tok, device, [p_ids[bi]] * len(grp),
                    [golds[bi]] * len(grp), edit,
                    [c[4] for c in grp], 1.0,
                    [cand_rows[bi]] * len(grp))
                for c, m in zip(grp, sc):
                    rows.append({"pair_id": r.pair_id, "fact_id": r.fact_id,
                                 "task": c[0], "ranking": c[1], "k": c[2],
                                 "gold_rank": m["gold_rank"],
                                 "logp_gold": m["logp_answer"],
                                 "base_rank": int(base_r)})
            # save top attribution neurons for recurrence
            top = rank_indices(attr, TOP_SAVE)
            rows.append({"pair_id": r.pair_id, "fact_id": r.fact_id,
                         "task": "top_neurons", "ranking": "attr", "k": TOP_SAVE,
                         "gold_rank": -1, "logp_gold": 0.0,
                         "base_rank": int(base_r),
                         "neuron_ids": [int(x) for x in top]})

            # greedy minimal set on pairs the full edit flips
            full_v = (Wd @ dg_t).cpu().numpy()
            full_sc = C.score_with_edit(
                model, tok, device, [p_ids[bi]], [golds[bi]],
                C.ResidualEdit(model, mlp_hs, "add"), [full_v], 1.0,
                [cand_rows[bi]])[0]
            if base_r > 1 and full_sc["gold_rank"] == 1 \
                    and len(greedy_rows) < args.greedy_per_shard:
                mink = greedy_minimal(model, tok, device, p_ids[bi],
                                      golds[bi], cand_rows[bi], attr, dg_t, Wd,
                                      mlp_hs)
                greedy_rows.append({"pair_id": r.pair_id,
                                    "fact_id": r.fact_id, "min_k": mink,
                                    "pool": GREEDY_POOL})
        print(f"[run] {min(s + B, len(pairs))}/{len(pairs)} pairs", flush=True)

    _write(out_dir, "run", args, rows)
    _write(out_dir, "greedy", args, greedy_rows)


def greedy_minimal(model, tok, device, p_ids, gold, cand_row, attr, dg_t, Wd,
                   mlp_hs):
    """Forward-select neurons from the top-GREEDY_POOL attribution pool until
    gold_rank == 1. Returns the minimal set size, or -1 if not reached."""
    pool = rank_indices(attr, GREEDY_POOL).tolist()
    chosen: list[int] = []
    for _ in range(min(GREEDY_MAX_STEPS, len(pool))):
        cands = [n for n in pool if n not in chosen]
        vs = []
        for n in cands:
            S = chosen + [n]
            import torch
            vs.append((Wd[:, torch.tensor(S, device=device)]
                       @ dg_t[torch.tensor(S, device=device)]).cpu().numpy())
        edit = C.ResidualEdit(model, mlp_hs, "add")
        sc = C.score_with_edit(model, tok, device, [p_ids] * len(cands),
                               [gold] * len(cands), edit, vs, 1.0,
                               [cand_row] * len(cands))
        ranks = [m["gold_rank"] for m in sc]
        best = int(np.argmin(ranks))
        chosen.append(cands[best])
        if ranks[best] == 1:
            return len(chosen)
    return -1


def _write(out_dir, name, args, rows):
    exp = out_dir / "expF"
    exp.mkdir(exist_ok=True)
    suf = "" if args.num_shards == 1 else f"_shard{args.shard_idx:02d}"
    p = exp / f"{name}{suf}.parquet"
    if p.exists() and not args.force:
        sys.exit(f"refusing to overwrite {p}; pass --force")
    pd.DataFrame(rows).to_parquet(p, index=False)
    print(f"[{name}] wrote {p} ({len(rows)} rows)", flush=True)


def merge(out_dir, name, num_shards):
    exp = out_dir / "expF"
    parts = []
    for s in range(num_shards):
        suf = "" if num_shards == 1 else f"_shard{s:02d}"
        p = exp / f"{name}{suf}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.to_parquet(exp / f"{name}.parquet", index=False)
    return df


def analyze(args):
    exp = args.out_dir / "expF"
    df = merge(args.out_dir, "run", args.num_shards)
    greedy = merge(args.out_dir, "greedy", args.num_shards)

    sweeps = df[df.task.isin(["coord", "neuron"])].copy()
    sweeps["recovered"] = (sweeps.gold_rank == 1) & (sweeps.base_rank > 1)
    denom = sweeps[sweeps.base_rank > 1]
    curve = []
    for (task, ranking, k), g in denom.groupby(["task", "ranking", "k"]):
        mean, lo, hi = fact_bootstrap_ci(g.recovered.astype(float), g.fact_id)
        curve.append({"task": task, "ranking": ranking, "k": int(k),
                      "recovery": mean, "lo": lo, "hi": hi,
                      "n": len(g), "n_facts": g.fact_id.nunique()})
    pd.DataFrame(curve).sort_values(["task", "ranking", "k"]).to_csv(
        exp / "curves.csv", index=False)

    # neuron recurrence across facts
    tn = df[df.task == "top_neurons"]
    from collections import Counter
    cnt = Counter()
    for ids in tn.neuron_ids.dropna():
        cnt.update(int(x) for x in ids)
    rec = pd.DataFrame(sorted(cnt.items(), key=lambda x: -x[1]),
                       columns=["neuron_id", "n_pairs_in_top"])
    rec["frac_pairs"] = rec.n_pairs_in_top / max(len(tn), 1)
    rec.to_csv(exp / "neuron_recurrence.csv", index=False)

    if len(greedy):
        flipped = greedy[greedy.min_k > 0]
        summary = {"n_greedy": int(len(greedy)),
                   "n_reached_flip": int(len(flipped)),
                   "median_min_k": float(flipped.min_k.median())
                   if len(flipped) else None,
                   "mean_min_k": float(flipped.min_k.mean())
                   if len(flipped) else None,
                   "pool": GREEDY_POOL}
        (exp / "greedy_summary.json").write_text(json.dumps(summary, indent=2))
        print("[greedy] " + json.dumps(summary), flush=True)

    print(pd.DataFrame(curve).sort_values(["task", "ranking", "k"])
          .to_string(index=False), flush=True)
    print(f"[recurrence] top neuron in {rec.frac_pairs.iloc[0]:.1%} of pairs "
          f"(id {int(rec.neuron_id.iloc[0])})" if len(rec) else "no neurons",
          flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--model_name_or_path", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--phase", choices=["run"], default=None)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--test_budget", type=int, default=800)
    ap.add_argument("--pair_batch", type=int, default=8)
    ap.add_argument("--greedy_per_shard", type=int, default=20)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.analyze:
        analyze(args)
    elif args.phase == "run":
        run(args)
    else:
        sys.exit("pass --phase run or --analyze")


if __name__ == "__main__":
    main()
