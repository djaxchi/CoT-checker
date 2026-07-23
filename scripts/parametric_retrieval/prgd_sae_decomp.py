"""parametric_retrieval_sae_decomp: offline SAE-feature decomposition of the
flip, the cheap go/no-go check before committing to SAE feature steering.

prgm found the same-fact flip is a sparse set of RAW MLP neurons; prgs found
those raw neurons are polysemantic (steering them off-context does nothing).
The question this answers, without any model forward or generation: does the
flip concentrate on a few INTERPRETABLE SAE features?

We reuse the public BatchTopK SAE `andyrdt/saes-qwen2.5-7b-instruct` at
resid_post_layer_27 (= hidden_states index 28), which is exactly where the
layer-27 MLP output lands. For each test pair we encode the donor (success) and
recipient (fail) residual at the final prompt token through the SAE and
decompose the difference:

  f_donor = SAE.encode(h_donor) ;  f_recip = SAE.encode(h_recip)
  df = f_donor - f_recip

and ask (1) reconstruction FVU on our states (does the SAE even apply here?),
(2) how concentrated df is (how few features capture most of |df|), (3) which
"donor-added" features recur across facts (shared vs fact-specific). A diffuse,
non-recurring, or high-FVU result says the SAE basis will not help and we stop;
a concentrated/recurring result justifies the full feature-steering build.

Reuses HSStore (hs_idx 28) + the BatchTopKSAE loader from public_sae. GPU
optional (encode is a matmul). Outputs under expH/.
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
from scripts.public_sae.encode_public_sae import BatchTopKSAE  # noqa: E402
from src.analysis.parametric_retrieval_causal import (  # noqa: E402
    assign_patch_donors,
    budget_pairs,
)

HS_IDX = 28            # resid_post_layer_27, matches the L28 andyrdt SAE
POSITION = "final_prompt_token"
TOPK_GRID = [1, 2, 4, 8, 16, 32, 64]


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested)
# --------------------------------------------------------------------------- #

def frac_in_topk(vals: np.ndarray, k: int) -> float:
    """Fraction of total |vals| L1 mass held by the k largest-|.| entries."""
    a = np.abs(vals)
    tot = a.sum()
    if tot <= 0:
        return 0.0
    if k >= len(a):
        return 1.0
    return float(np.sort(a)[-k:].sum() / tot)


def n_to_capture(vals: np.ndarray, frac: float = 0.9) -> int:
    """How many largest-|.| entries are needed to reach `frac` of L1 mass."""
    a = np.sort(np.abs(vals))[::-1]
    tot = a.sum()
    if tot <= 0:
        return 0
    c = np.cumsum(a) / tot
    return int(np.searchsorted(c, frac) + 1)


def donor_added(f_donor: np.ndarray, f_recip: np.ndarray, top: int = 8):
    """Feature ids most increased from recipient to donor (the features the
    successful retrieval turned on), by df magnitude among df>0."""
    df = f_donor - f_recip
    pos = np.where(df > 0)[0]
    if len(pos) == 0:
        return []
    order = pos[np.argsort(df[pos])[::-1]]
    return order[:top].tolist()


# --------------------------------------------------------------------------- #
# run (encode + decompose)
# --------------------------------------------------------------------------- #

def run(args):
    from collections import Counter

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_pt = args.sae_root / "resid_post_layer_27" / f"trainer_{args.trainer}" \
        / "ae.pt"
    if not ae_pt.exists():
        sys.exit(f"[decomp] FATAL missing SAE {ae_pt} (download it first)")
    ae = BatchTopKSAE.from_ae_pt(ae_pt, device)

    out_dir = args.out_dir
    hstore = C.HSStore(out_dir)
    if ae.activation_dim != hstore.layer(HS_IDX).shape[1]:
        sys.exit(f"[decomp] dim mismatch SAE {ae.activation_dim} vs "
                 f"resid {hstore.layer(HS_IDX).shape[1]}")

    groups = build_group_table(out_dir, hstore.meta)
    pairs = pd.read_parquet(out_dir / "pairs.parquet")
    pairs["fact_id"] = pairs.fact_id.astype(str)
    pairs = pairs[pairs.split == "test"].reset_index(drop=True)
    pairs = budget_pairs(pairs, args.n_pairs, seed=args.seed)
    pairs = assign_patch_donors(pairs, groups, seed=args.seed)

    # gather + encode the unique instances we need
    insts = sorted(set(pairs.recipient_instance_id) | set(pairs.donor_instance_id))
    H = np.stack([hstore.vec(i, POSITION, HS_IDX).astype(np.float32)
                  for i in insts])
    row = {i: r for r, i in enumerate(insts)}
    F = np.zeros((len(insts), ae.dict_size), dtype=np.float32)
    sse = sst = 0.0
    hbar = H.mean(0)
    for s in range(0, len(insts), args.batch_size):
        xb = torch.tensor(H[s:s + args.batch_size], device=device)
        z = ae.encode(xb)
        xhat = ae.decode(z)
        F[s:s + args.batch_size] = z.cpu().numpy()
        sse += float(((xb - xhat) ** 2).sum())
        sst += float(((xb - torch.tensor(hbar, device=device)) ** 2).sum())
    fvu = sse / sst if sst > 0 else float("nan")
    print(f"[decomp] encoded {len(insts)} instances  FVU={fvu:.3f}  "
          f"active/row median={np.median((F > 0).sum(1)):.0f}", flush=True)

    rows, added_counter = [], Counter()
    null_rng = np.random.default_rng(args.seed)
    donor_rows = list(pairs.donor_instance_id)
    for r in pairs.itertuples():
        fd = F[row[r.donor_instance_id]]
        fr = F[row[r.recipient_instance_id]]
        df = fd - fr
        # matched-fact concentration
        rowd = {"pair_id": r.pair_id, "fact_id": r.fact_id,
                "n_change": int((df != 0).sum()),
                "n_cap90": n_to_capture(df, 0.9),
                "n_donor_active": int((fd > 0).sum()),
                "n_recip_active": int((fr > 0).sum())}
        for k in TOPK_GRID:
            rowd[f"cap_top{k}"] = frac_in_topk(df, k)
        # null: same recipient, a random OTHER-pair donor (surface/other-fact)
        rnd_donor = donor_rows[int(null_rng.integers(len(donor_rows)))]
        df_null = F[row[rnd_donor]] - fr
        rowd["n_cap90_null"] = n_to_capture(df_null, 0.9)
        rows.append(rowd)
        for fid in donor_added(fd, fr, top=8):
            added_counter[int(fid)] += 1

    exp = out_dir / "expH"
    exp.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(exp / "decomp.parquet", index=False)

    n_pairs = len(rows)
    rec = pd.DataFrame(sorted(added_counter.items(), key=lambda x: -x[1]),
                       columns=["feature_id", "n_pairs_added"])
    rec["frac_pairs"] = rec.n_pairs_added / max(n_pairs, 1)
    rec.to_csv(exp / "feature_recurrence.csv", index=False)

    d = pd.DataFrame(rows)
    curve = {k: float(d[f"cap_top{k}"].mean()) for k in TOPK_GRID}
    summary = {
        "sae_id": f"resid_post_layer_27/trainer_{args.trainer}",
        "hs_idx": HS_IDX, "n_pairs": n_pairs, "recon_fvu": round(fvu, 4),
        "fvu_gate": "OK" if fvu < 0.5 else ("HIGH" if fvu < 0.9 else "SUSPECT"),
        "median_n_change": float(d.n_change.median()),
        "median_n_cap90": float(d.n_cap90.median()),
        "median_n_cap90_null": float(d.n_cap90_null.median()),
        "concentration_curve": curve,
        "top_added_features": rec.head(15).to_dict("records"),
        "n_distinct_added": int(len(rec)),
        "frac_added_in_gt20pct": float((rec.frac_pairs > 0.2).mean()) if len(rec) else 0.0,
    }
    (exp / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_access_v1"))
    ap.add_argument("--sae_root", type=Path, required=True,
                    help="local andyrdt/saes-qwen2.5-7b-instruct dir")
    ap.add_argument("--trainer", type=int, default=1)
    ap.add_argument("--n_pairs", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
