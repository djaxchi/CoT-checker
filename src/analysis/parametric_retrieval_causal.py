"""Core logic for parametric_retrieval_access_v1 Experiments C and D
(same-fact activation patching and fact-independent access subspace).

Pure functions only (no torch / no model calls): patch-condition donor
assignment, confound residualization, access-direction estimation, and
paired fact-level bootstrap. GPU scripts consume these.

Conventions: a "cell" is (hs_idx, position_name). Donor/recipient vectors are
rows of the stage-3 extraction (hidden_states_v1). All estimation happens on
TRAIN facts only; alpha and cell selection on VAL; TEST is touched once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PATCH_CONDITIONS = ["noop", "matched", "mismatched_type", "mismatched_rand",
                    "random_noise", "reverse"]


# ---------------------------------------------------------------------------
# Experiment C: donor assignment
# ---------------------------------------------------------------------------

def assign_patch_donors(pairs: pd.DataFrame, groups: pd.DataFrame,
                        seed: int = 42) -> pd.DataFrame:
    """Per (success, fail) pair, assign donor instances for every condition.

    pairs: pair_id, fact_id, direction, donor_instance_id (success),
           recipient_instance_id (fail).
    groups: fact_id, direction, category, answer_type, gbc_bin plus one
            successful donor candidate instance per group
            (donor_pool_instance_id), i.e. groups that HAVE a successful
            paraphrase.

    Conditions:
      noop             recipient patched with its own state (sanity)
      matched          same-fact successful donor (the pair's donor)
      mismatched_type  successful donor from another fact, same direction +
                       answer_type + category when possible (answer content
                       control: same kind of answer, wrong fact)
      mismatched_rand  successful donor from another fact, same direction +
                       gbc_bin (popularity-matched, arbitrary type)
      random_noise     no donor; gaussian noise norm-matched to the edit
      reverse          the pair's failed state patched INTO the successful
                       prompt (necessity direction)
    Deterministic under seed. Returns pairs with donor_<condition> columns
    holding instance_ids (or None where a pool is empty).
    """
    rng = np.random.default_rng(seed)
    pool = groups.dropna(subset=["donor_pool_instance_id"])
    out = pairs.copy()
    mm_type, mm_rand = [], []
    for r in pairs.itertuples():
        others = pool[(pool.direction == r.direction)
                      & (pool.fact_id != r.fact_id)]
        g = groups[(groups.fact_id == r.fact_id)
                   & (groups.direction == r.direction)].iloc[0]
        tier = others[(others.answer_type == g.answer_type)
                      & (others.category == g.category)]
        if tier.empty:
            tier = others[others.answer_type == g.answer_type]
        if tier.empty:
            tier = others
        mm_type.append(
            None if tier.empty else
            tier.donor_pool_instance_id.iloc[
                int(rng.integers(len(tier)))])
        tier2 = others[others.gbc_bin == g.gbc_bin]
        if tier2.empty:
            tier2 = others
        mm_rand.append(
            None if tier2.empty else
            tier2.donor_pool_instance_id.iloc[
                int(rng.integers(len(tier2)))])
    out["donor_matched"] = out.donor_instance_id
    out["donor_noop"] = out.recipient_instance_id
    out["donor_mismatched_type"] = mm_type
    out["donor_mismatched_rand"] = mm_rand
    # reverse: donor = the failed state, recipient prompt = the success one
    out["donor_reverse"] = out.recipient_instance_id
    return out


def budget_pairs(pairs: pd.DataFrame, max_pairs: int,
                 seed: int = 42) -> pd.DataFrame:
    """Deterministic fact-stratified subsample: keep whole groups (all pairs
    of a fact x direction) until the budget is filled, favoring coverage of
    many facts over many pairs per fact."""
    if len(pairs) <= max_pairs:
        return pairs.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keys = pairs[["fact_id", "direction"]].drop_duplicates().reset_index(
        drop=True)
    order = rng.permutation(len(keys))
    keep, total = [], 0
    by_key = {k: g for k, g in pairs.groupby(["fact_id", "direction"])}
    for i in order:
        k = (keys.fact_id.iloc[i], keys.direction.iloc[i])
        g = by_key[k]
        take = g.iloc[: max(1, min(len(g), max_pairs - total))]
        keep.append(take)
        total += len(take)
        if total >= max_pairs:
            break
    return pd.concat(keep, ignore_index=True)


# ---------------------------------------------------------------------------
# Experiment D: residualization + direction estimation
# ---------------------------------------------------------------------------

def residualize(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """OLS-residualize rows of X (n, d) against features F (n, k), with an
    intercept added here. Returns X - F_aug @ beta."""
    F_aug = np.concatenate([np.ones((len(F), 1)), F], axis=1)
    beta, *_ = np.linalg.lstsq(F_aug, X, rcond=None)
    return X - F_aug @ beta


def confound_features(meta: pd.DataFrame) -> np.ndarray:
    """One-hot template_id, seed_variant, direction, gbc_bin, category +
    z-scored prompt_token_count. Purely surface/metadata confounds."""
    cols = []
    for c in ["template_id", "seed_variant", "direction", "gbc_bin",
              "category"]:
        cols.append(pd.get_dummies(meta[c].astype(str), prefix=c,
                                   dtype=float))
    ptc = meta.prompt_token_count.astype(float)
    ptc = (ptc - ptc.mean()) / max(ptc.std(), 1e-9)
    cols.append(ptc.rename("ptc").to_frame())
    return pd.concat(cols, axis=1).to_numpy(dtype=np.float64)


def paired_diffs(H: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray,
                                                             pd.DataFrame]:
    """Within-group mean(success) - mean(fail) difference vectors.

    meta needs fact_id, direction, is_correct aligned with H rows. Returns
    (diffs (g, d), group frame fact_id/direction) for groups having both
    outcomes."""
    rows, keys = [], []
    for (fid, dirn), g in meta.groupby(["fact_id", "direction"], sort=True):
        pos = g.index[g.is_correct.astype(bool)]
        neg = g.index[~g.is_correct.astype(bool)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        rows.append(H[pos].mean(axis=0) - H[neg].mean(axis=0))
        keys.append({"fact_id": fid, "direction": dirn})
    return np.asarray(rows), pd.DataFrame(keys)


def estimate_directions(diffs: np.ndarray, seed: int = 42,
                        n_random: int = 3) -> dict[str, np.ndarray]:
    """Access-direction candidates from train paired diffs (g, d).

    mean_diff   normalized mean difference (baseline)
    svd1        top right-singular vector of the RAW (uncentered) diff
                matrix D, sign-aligned with mean_diff (spec 14.1: rank-k
                SVD subspace of D, where the shared component lives;
                centering would remove exactly that component)
    svd4_proj   mean_diff projected onto the top-4 singular subspace of D
    random_k    norm-matched random directions (controls)
    All unit-normalized; the caller scales by alpha times the mean paired
    edit norm."""
    rng = np.random.default_rng(seed)
    mean = diffs.mean(axis=0)
    mean_u = mean / max(np.linalg.norm(mean), 1e-12)
    _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    svd1 = vt[0] * np.sign(vt[0] @ mean_u + 1e-12)
    basis = vt[:4]
    proj = basis.T @ (basis @ mean)
    proj += mean_u * 1e-12  # keep nonzero if orthogonal
    out = {"mean_diff": mean_u,
           "svd1": svd1 / np.linalg.norm(svd1),
           "svd4_proj": proj / max(np.linalg.norm(proj), 1e-12)}
    for i in range(n_random):
        v = rng.standard_normal(diffs.shape[1])
        out[f"random_{i}"] = v / np.linalg.norm(v)
    return out


def lda_direction(H: np.ndarray, y: np.ndarray,
                  shrinkage: float = 0.1) -> np.ndarray:
    """Regularized LDA direction w ~ (S + shrinkage*tr(S)/d I)^-1 (mu1-mu0),
    unit-normalized."""
    mu1, mu0 = H[y].mean(axis=0), H[~y].mean(axis=0)
    Xc = np.concatenate([H[y] - mu1, H[~y] - mu0], axis=0)
    S = (Xc.T @ Xc) / max(len(Xc) - 2, 1)
    d = S.shape[0]
    S_reg = S + np.eye(d) * shrinkage * np.trace(S) / d
    w = np.linalg.solve(S_reg, mu1 - mu0)
    return w / max(np.linalg.norm(w), 1e-12)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def fact_bootstrap_ci(values: pd.Series, facts: pd.Series,
                      n_boot: int = 2000, seed: int = 42,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean with fact-level bootstrap CI: resample facts with replacement,
    average the per-fact means. Returns (mean, lo, hi)."""
    df = pd.DataFrame({"v": values.to_numpy(), "f": facts.to_numpy()})
    per_fact = df.groupby("f").v.mean()
    rng = np.random.default_rng(seed)
    arr = per_fact.to_numpy()
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    return (float(per_fact.mean()),
            float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))
