"""parametric_retrieval_geometry_v0 stage 3: unsupervised geometry summaries.

No probe, no train/val/test, no gradient descent. For every analysis cell
(hs_idx x prompt_mode x position, sentence_end excluded from per-cell stats)
over the QA instances (completion controls excluded from all headline stats):

  centroid_distances.csv          class-pair centroid euclidean + cosine, and
                                  separation = euclid / pooled within-dispersion
  within_class_dispersion.csv     per-class mean distance to own centroid
  between_within_ratio.csv        Calinski-Harabasz pseudo-F (B/W) with an
                                  empirical label-shuffle p-value (default 1000)
  pairwise_cosine_stats.csv       same-class vs different-class pairwise cosine
  pca_coordinates.parquet         2D PCA per cell (fit on QA, controls
                                  transformed into the same plane)
  trajectory_to_centroids.csv     per CoT row: distance to the direct_retrieval
                                  and non_retrieved centroids (centroids from
                                  final_prompt_token states, both cot-mode and
                                  direct-mode sources), sentence_end included
  trajectory_summary.csv          mean +- sem per class x trajectory step

  python scripts/parametric_retrieval/prg_geometry.py \
      --out_dir runs/parametric_retrieval_geometry_v0
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.parametric_retrieval import (  # noqa: E402
    RETRIEVAL_CLASSES,
    block_idx,
)

CELL_POSITIONS = {
    "direct": ["final_prompt_token", "first_generated_token",
               "final_answer_token"],
    "cot": ["final_prompt_token", "first_generated_token",
            "token_before_final_answer", "first_final_answer_token",
            "final_answer_token"],
}
TRAJ_STEPS = ["final_prompt_token", "first_generated_token",
              "se_q1", "se_q2", "se_q3", "se_q4",
              "token_before_final_answer", "first_final_answer_token",
              "final_answer_token"]


def load_layer(hs_dir: Path, k: int) -> np.ndarray:
    from safetensors.numpy import load_file
    return load_file(hs_dir / f"layer_{k:02d}.safetensors")["h"]


def ch_ratio(Xc: np.ndarray, y: np.ndarray, classes: list[str]) -> float:
    """Calinski-Harabasz pseudo-F on globally centered X."""
    n, _ = Xc.shape
    total = float((Xc ** 2).sum())
    between = 0.0
    for c in classes:
        m = y == c
        s = Xc[m].sum(axis=0)
        between += float(s @ s) / m.sum()
    within = total - between
    c_ = len(classes)
    return (between / (c_ - 1)) / (within / (n - c_))


def traj_step(row) -> str | None:
    p = row.position_name
    if p in ("final_prompt_token", "first_generated_token",
             "token_before_final_answer", "first_final_answer_token",
             "final_answer_token"):
        return p
    if p == "sentence_end":
        frac = (row.position_rank + 1) / (row.n_sentence_ends + 1)
        return f"se_q{min(3, int(frac * 4)) + 1}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    ap.add_argument("--hs_indices", type=int, nargs="+", default=None,
                    help="defaults to the layers present in hidden_states/")
    ap.add_argument("--n_shuffles", type=int, default=1000)
    ap.add_argument("--n_pairs", type=int, default=20000)
    ap.add_argument("--min_cell", type=int, default=50,
                    help="skip cells with fewer QA rows (smoke: lower this)")
    ap.add_argument("--min_centroid_n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hs_dir = args.out_dir / "hidden_states"
    geo_dir = args.out_dir / "geometry"
    geo_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.hs_indices is None:
        args.hs_indices = sorted(
            int(p.stem.split("_")[1]) for p in hs_dir.glob("layer_*.safetensors"))
    if not args.hs_indices:
        sys.exit(f"no merged layer_*.safetensors in {hs_dir}")

    meta = pd.read_parquet(hs_dir / "hs_meta.parquet")
    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    meta = meta.reset_index().rename(columns={"index": "row_pos"})
    meta = meta.merge(grading[["question_id", "retrieval_class"]],
                      on="question_id", how="left", validate="many_to_one")
    n_ends = (meta[meta.position_name == "sentence_end"]
              .groupby("question_id").size().rename("n_sentence_ends"))
    meta = meta.merge(n_ends, on="question_id", how="left")
    meta["n_sentence_ends"] = meta["n_sentence_ends"].fillna(0).astype(int)

    cent_rows, disp_rows, bw_rows, cos_rows = [], [], [], []
    pca_parts, traj_parts = [], []

    for k in args.hs_indices:
        H = load_layer(hs_dir, k).astype(np.float32)
        assert len(H) == len(meta), f"hs_meta/layer_{k} row mismatch"
        print(f"[geometry] hs_idx {k} (block {block_idx(k)}): "
              f"{H.shape[0]} rows", flush=True)

        for mode, positions in CELL_POSITIONS.items():
            for pos in positions:
                cell = meta[(meta.prompt_mode == mode)
                            & (meta.position_name == pos)]
                qa = cell[~cell.is_control
                          & cell.retrieval_class.isin(RETRIEVAL_CLASSES)]
                if len(qa) < args.min_cell:
                    continue
                X = H[qa.row_pos.to_numpy()]
                y = qa.retrieval_class.to_numpy()
                classes = [c for c in RETRIEVAL_CLASSES if (y == c).any()]
                base = {"hs_idx": k, "block_idx": block_idx(k),
                        "prompt_mode": mode, "position": pos}

                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=args.seed)
                pca.fit(X)
                coords = pca.transform(H[cell.row_pos.to_numpy()])
                pca_parts.append(pd.DataFrame({
                    "row_id": cell.row_id.to_numpy(),
                    "question_id": cell.question_id.to_numpy(),
                    "hs_idx": k, "block_idx": block_idx(k),
                    "prompt_mode": mode, "position": pos,
                    "x": coords[:, 0], "y": coords[:, 1],
                    "evr1": float(pca.explained_variance_ratio_[0]),
                    "evr2": float(pca.explained_variance_ratio_[1]),
                }))
                if len(classes) < 2:  # separation stats need >=2 classes
                    continue

                mu = {c: X[y == c].mean(axis=0) for c in classes}
                within = {c: float(np.linalg.norm(
                    X[y == c] - mu[c], axis=1).mean()) for c in classes}
                for c in classes:
                    disp_rows.append({**base, "retrieval_class": c,
                                      "n": int((y == c).sum()),
                                      "mean_dist_to_centroid": within[c]})
                for a, b in combinations(classes, 2):
                    d = float(np.linalg.norm(mu[a] - mu[b]))
                    cs = float(mu[a] @ mu[b] / (np.linalg.norm(mu[a])
                                                * np.linalg.norm(mu[b])))
                    pooled = 0.5 * (within[a] + within[b])
                    cent_rows.append({**base, "class_a": a, "class_b": b,
                                      "centroid_euclidean": d,
                                      "centroid_cosine_sim": cs,
                                      "separation": d / pooled})

                Xc = X - X.mean(axis=0, keepdims=True)
                obs = ch_ratio(Xc, y, classes)
                perm = np.empty(args.n_shuffles)
                for s in range(args.n_shuffles):
                    perm[s] = ch_ratio(Xc, rng.permutation(y), classes)
                pval = float((1 + (perm >= obs).sum()) / (args.n_shuffles + 1))
                bw_rows.append({**base, "n": len(qa),
                                "ch_between_within": obs,
                                "shuffle_mean": float(perm.mean()),
                                "shuffle_p": pval,
                                "n_shuffles": args.n_shuffles})

                U = X / np.linalg.norm(X, axis=1, keepdims=True)
                i = rng.integers(0, len(U), args.n_pairs * 2)
                j = rng.integers(0, len(U), args.n_pairs * 2)
                keep = i != j
                i, j = i[keep], j[keep]
                cos = (U[i] * U[j]).sum(axis=1)
                same = y[i] == y[j]
                for label, m in [("same_class", same), ("diff_class", ~same)]:
                    cos_rows.append({**base, "pair_type": label,
                                     "n_pairs": int(m.sum()),
                                     "cosine_mean": float(cos[m].mean()),
                                     "cosine_std": float(cos[m].std())})

        # ---- CoT trajectories vs final_prompt_token centroids -------------
        for source_mode in ["cot", "direct"]:
            anchor = meta[(meta.prompt_mode == source_mode)
                          & (meta.position_name == "final_prompt_token")
                          & ~meta.is_control]
            cents = {}
            for c in ["direct_retrieval", "non_retrieved"]:
                m = anchor.retrieval_class == c
                if m.sum() < args.min_centroid_n:
                    break
                cents[c] = H[anchor.row_pos.to_numpy()[m.to_numpy()]] \
                    .mean(axis=0)
            if len(cents) < 2:
                continue
            gap = float(np.linalg.norm(cents["direct_retrieval"]
                                       - cents["non_retrieved"]))
            cot = meta[(meta.prompt_mode == "cot") & ~meta.is_control].copy()
            Xt = H[cot.row_pos.to_numpy()]
            d_dr = np.linalg.norm(Xt - cents["direct_retrieval"], axis=1)
            d_nr = np.linalg.norm(Xt - cents["non_retrieved"], axis=1)
            cot["traj_step"] = cot.apply(traj_step, axis=1)
            traj_parts.append(pd.DataFrame({
                "hs_idx": k, "block_idx": block_idx(k),
                "centroid_source": f"{source_mode}_prompt",
                "question_id": cot.question_id.to_numpy(),
                "retrieval_class": cot.retrieval_class.to_numpy(),
                "position_name": cot.position_name.to_numpy(),
                "position_rank": cot.position_rank.to_numpy(),
                "traj_step": cot.traj_step.to_numpy(),
                "dist_direct_retrieval": d_dr / gap,
                "dist_non_retrieved": d_nr / gap,
                "toward_retrieval": (d_nr - d_dr) / gap,
            }))

    pd.DataFrame(cent_rows).to_csv(geo_dir / "centroid_distances.csv",
                                   index=False)
    pd.DataFrame(disp_rows).to_csv(geo_dir / "within_class_dispersion.csv",
                                   index=False)
    pd.DataFrame(bw_rows).to_csv(geo_dir / "between_within_ratio.csv",
                                 index=False)
    pd.DataFrame(cos_rows).to_csv(geo_dir / "pairwise_cosine_stats.csv",
                                  index=False)
    if not pca_parts:
        sys.exit("[geometry] no cells passed --min_cell; nothing to write")
    pd.concat(pca_parts, ignore_index=True).to_parquet(
        geo_dir / "pca_coordinates.parquet", index=False)

    if not traj_parts:
        print("[geometry] WARNING: no trajectory centroids "
              "(classes below --min_centroid_n); skipping trajectory outputs",
              flush=True)
        print(f"[geometry] wrote {len(bw_rows)} cells to {geo_dir}", flush=True)
        return
    traj = pd.concat(traj_parts, ignore_index=True)
    traj.to_csv(geo_dir / "trajectory_to_centroids.csv", index=False)
    summ = (traj.dropna(subset=["traj_step"])
            .groupby(["hs_idx", "block_idx", "centroid_source",
                      "retrieval_class", "traj_step"], observed=True)
            .agg(n=("toward_retrieval", "size"),
                 dist_direct_retrieval=("dist_direct_retrieval", "mean"),
                 dist_direct_retrieval_sem=("dist_direct_retrieval", "sem"),
                 dist_non_retrieved=("dist_non_retrieved", "mean"),
                 dist_non_retrieved_sem=("dist_non_retrieved", "sem"),
                 toward_retrieval=("toward_retrieval", "mean"),
                 toward_retrieval_sem=("toward_retrieval", "sem"))
            .reset_index())
    summ["traj_order"] = summ.traj_step.map(
        {s: i for i, s in enumerate(TRAJ_STEPS)})
    summ = summ.sort_values(["hs_idx", "centroid_source", "retrieval_class",
                             "traj_order"])
    summ.to_csv(geo_dir / "trajectory_summary.csv", index=False)

    print(f"[geometry] wrote {len(bw_rows)} cells "
          f"({len(cent_rows)} centroid pairs) to {geo_dir}", flush=True)


if __name__ == "__main__":
    main()
