"""Core logic for the attention_routing_v0 experiment.

Do correct and incorrect CoT steps route attention differently toward the
question, previous reasoning steps, and the current step itself? This module
holds the pure feature computation (numpy only, no torch / no model calls):
fork text assembly, char-offset token-to-region assignment, per-head
attention-routing features read at first/last/mean candidate token, and
paired (correct minus wrong) statistics with problem-level cluster bootstrap.

Text assembly matches s4_contrib_extract_forks.py exactly:
    text = "\n".join([question, *prefix_steps, candidate])
so token counts and boundaries are comparable with the hidden-state arm.

Region convention: a token belongs to the segment containing its FIRST
character. Qwen's tokenizer merges the separator newline into the preceding
token (".\n", "?\n"), so separators attach to the preceding segment and the
candidate's first token starts cleanly on candidate text.

Feature rows assume causally masked, row-normalized attention. Masses over
region masks are automatically causal because entries beyond the query
position are zero; check_causal / check_row_normalized exist for the stage-1
extraction gate.
"""

from __future__ import annotations

import re

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEP = "\n"

# coarse region ids per token
REGION_OTHER = 0        # special tokens / anything outside the text
REGION_QUESTION = 1
REGION_OLDER = 2        # prefix steps S_1..S_{k-2}
REGION_PREV1 = 3        # immediately previous step S_{k-1}
REGION_CANDIDATE = 4
N_REGIONS = 5

REGION_NAMES = ["other", "question", "older", "prev1", "candidate"]

# per-(layer, head) features, in array order
FEATURES = [
    "question_mass",   # total attention to question tokens
    "prev_all_mass",   # older + prev1
    "prev1_mass",      # immediately previous step
    "older_mass",      # steps before the previous one
    "self_mass",       # within the candidate step (causally limited)
    "other_mass",      # special / out-of-text tokens
    "entropy",         # Shannon entropy of the visible attention row (nats)
    "mean_distance",   # sum_j p_j * (t - j), in tokens
    "top5_mass",       # attention mass on the 5 largest entries
    "sink_mass",       # attention to token 0 (Qwen attention sink; this
                       # position is question text, so it OVERLAPS
                       # question_mass rather than adding to the partition)
]
N_FEATURES = len(FEATURES)

READS = ["first", "last", "mean"]  # candidate-token read positions
N_READS = len(READS)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_OPERATOR_CHARS = set("+-*/=^<>")


# ---------------------------------------------------------------------------
# Fork text and token regions
# ---------------------------------------------------------------------------

def build_fork_segments(
    question: str, prefix_steps: list[str], candidate: str
) -> tuple[str, list[tuple[str, int, int]]]:
    """Assemble the teacher-forced text and char segments.

    Returns (text, segments) where segments is a list of
    (name, char_start, char_end) covering question, each prefix step
    (step_1..step_{k-1}) and the candidate, in order. Separator chars sit in
    the gaps between segments.
    """
    parts = [question, *prefix_steps, candidate]
    names = (
        ["question"]
        + [f"step_{i + 1}" for i in range(len(prefix_steps))]
        + ["candidate"]
    )
    text = SEP.join(parts)
    segments = []
    pos = 0
    for name, part in zip(names, parts):
        segments.append((name, pos, pos + len(part)))
        pos += len(part) + len(SEP)
    return text, segments


def assign_token_regions(
    offsets: list[tuple[int, int]], segments: list[tuple[str, int, int]]
) -> np.ndarray:
    """Map each token to a coarse region id via its start char.

    A char belongs to the last segment whose start is <= the char (separator
    chars between segments therefore attach to the preceding segment).
    Tokens with an empty span (special tokens) map to REGION_OTHER.
    The last segment must be named "candidate".
    """
    if segments[-1][0] != "candidate":
        raise ValueError("last segment must be the candidate")
    n_steps = len(segments) - 2  # prefix steps between question and candidate
    seg_starts = np.array([s[1] for s in segments])

    def seg_to_region(seg_idx: int) -> int:
        if seg_idx == 0:
            return REGION_QUESTION
        if seg_idx == len(segments) - 1:
            return REGION_CANDIDATE
        # prefix step index 1..n_steps; the last one is prev1
        return REGION_PREV1 if seg_idx == n_steps else REGION_OLDER

    regions = np.full(len(offsets), REGION_OTHER, dtype=np.int64)
    for ti, (start, end) in enumerate(offsets):
        if end <= start:  # special token
            continue
        seg_idx = int(np.searchsorted(seg_starts, start, side="right")) - 1
        if seg_idx < 0:
            continue
        regions[ti] = seg_to_region(seg_idx)
    return regions


def candidate_token_span(regions: np.ndarray) -> tuple[int, int]:
    """Return [c0, c1) token indices of the candidate; must be a non-empty
    contiguous suffix-side block (no candidate token may precede a
    non-candidate token that follows it)."""
    idx = np.flatnonzero(regions == REGION_CANDIDATE)
    if idx.size == 0:
        raise ValueError("no candidate tokens")
    c0, c1 = int(idx[0]), int(idx[-1]) + 1
    if idx.size != c1 - c0:
        raise ValueError("candidate token span is not contiguous")
    return c0, c1


def region_token_counts(regions: np.ndarray) -> dict[str, int]:
    return {
        name: int((regions == rid).sum())
        for rid, name in enumerate(REGION_NAMES)
    }


# ---------------------------------------------------------------------------
# Attention features
# ---------------------------------------------------------------------------

def check_row_normalized(attn: np.ndarray, tol: float = 1e-2) -> float:
    """Max |row_sum - 1| over all rows; attn is (..., q_len, k_len)."""
    return float(np.abs(attn.sum(axis=-1) - 1.0).max())


def check_causal(attn: np.ndarray, c0: int, tol: float = 1e-4) -> float:
    """Max attention mass beyond the query position (should be ~0).

    attn is (n_heads, cand_len, key_len) for query tokens c0..c0+cand_len-1.
    """
    n_heads, cand_len, key_len = attn.shape
    q_abs = c0 + np.arange(cand_len)
    future = np.arange(key_len)[None, :] > q_abs[:, None]  # (cand_len, key_len)
    return float((attn * future[None]).max())


def attention_step_features(
    attn: np.ndarray,
    regions: np.ndarray,
    c0: int,
    top_k: int = 5,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute routing features for one layer.

    attn     : (n_heads, cand_len, key_len) float, candidate-row slice of a
               causally masked, row-normalized attention matrix.
    regions  : (key_len,) region ids from assign_token_regions.
    c0       : absolute token index of the first candidate token.

    Returns (n_heads, N_FEATURES, N_READS) float32. Region masses rely on
    causal masking (entries beyond the query position are zero), so self_mass
    only counts already-visible candidate tokens.
    """
    n_heads, cand_len, key_len = attn.shape
    if regions.shape[0] != key_len:
        raise ValueError("regions length != attention key length")
    attn = attn.astype(np.float32)

    onehot = np.zeros((key_len, N_REGIONS), dtype=np.float32)
    onehot[np.arange(key_len), regions] = 1.0
    masses = attn @ onehot  # (n_heads, cand_len, N_REGIONS)

    q_mass = masses[..., REGION_QUESTION]
    older = masses[..., REGION_OLDER]
    prev1 = masses[..., REGION_PREV1]
    self_mass = masses[..., REGION_CANDIDATE]
    other = masses[..., REGION_OTHER]

    p = np.clip(attn, eps, None)
    entropy = -(attn * np.log(p)).sum(axis=-1)  # zeros contribute 0

    q_abs = c0 + np.arange(cand_len, dtype=np.float32)
    j_idx = np.arange(key_len, dtype=np.float32)
    # rows sum to 1, so sum_j p_j (t - j) = t - sum_j p_j j
    mean_dist = q_abs[None, :] - attn @ j_idx

    k = min(top_k, key_len)
    top5 = np.partition(attn, key_len - k, axis=-1)[..., key_len - k:].sum(axis=-1)

    per_token = np.stack(
        [q_mass, older + prev1, prev1, older, self_mass, other,
         entropy, mean_dist, top5, attn[..., 0]],
        axis=-1,
    )  # (n_heads, cand_len, N_FEATURES)

    out = np.empty((n_heads, N_FEATURES, N_READS), dtype=np.float32)
    out[..., READS.index("first")] = per_token[:, 0]
    out[..., READS.index("last")] = per_token[:, -1]
    out[..., READS.index("mean")] = per_token.mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# Surface features
# ---------------------------------------------------------------------------

def count_numbers(text: str) -> int:
    return len(_NUMBER_RE.findall(text))


def count_operators(text: str) -> int:
    return sum(1 for ch in text if ch in _OPERATOR_CHARS)


# ---------------------------------------------------------------------------
# Derived ratios (analysis-time, computed from stored masses)
# ---------------------------------------------------------------------------

def grounding_ratio(prev_all: np.ndarray, question: np.ndarray,
                    eps: float = 1e-12) -> np.ndarray:
    """prev / (prev + question); higher = more grounded in the CoT."""
    return prev_all / np.clip(prev_all + question, eps, None)


def recency_ratio(prev1: np.ndarray, prev_all: np.ndarray,
                  eps: float = 1e-12) -> np.ndarray:
    """prev1 / prev_all; 1 = all previous-step attention on S_{k-1}."""
    return prev1 / np.clip(prev_all, eps, None)


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def paired_stats(
    delta: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Statistics for paired differences delta = f(correct) - f(wrong).

    groups gives the problem id of each pair. All inference is cluster-aware:
    the bootstrap resamples whole groups, and the sign test and Wilcoxon run
    on PROBLEM-LEVEL mean deltas (pairs from the same problem are never
    treated as independent). P(f+ > f-) is reported descriptively at both
    the fork level (p_gt) and the problem level (p_gt_group), with ties
    counted half.
    """
    from scipy import stats

    delta = np.asarray(delta, dtype=np.float64)
    groups = np.asarray(groups)
    if delta.shape != groups.shape:
        raise ValueError("delta and groups must have the same shape")
    n = delta.size
    uniq, inv = np.unique(groups, return_inverse=True)
    n_groups = len(uniq)
    rng = np.random.default_rng(seed)

    group_sums = np.zeros(n_groups)
    np.add.at(group_sums, inv, delta)
    group_counts = np.bincount(inv, minlength=n_groups).astype(np.float64)
    group_means = group_sums / group_counts

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        boot_means[b] = group_sums[pick].sum() / group_counts[pick].sum()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    sd = delta.std(ddof=1) if n > 1 else 0.0
    dz = float(delta.mean() / sd) if sd > 0 else 0.0

    nonzero = group_means[group_means != 0]
    if nonzero.size:
        p_sign = float(stats.binomtest(
            int((nonzero > 0).sum()), nonzero.size, 0.5).pvalue)
        p_wilcoxon = float(stats.wilcoxon(nonzero).pvalue)
    else:
        p_sign = p_wilcoxon = 1.0

    p_gt = float(((delta > 0).sum() + 0.5 * (delta == 0).sum()) / n)
    p_gt_group = float(((group_means > 0).sum()
                        + 0.5 * (group_means == 0).sum()) / n_groups)

    return {
        "n_pairs": float(n),
        "n_groups": float(n_groups),
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "dz": dz,
        "p_gt": p_gt,
        "p_gt_group": p_gt_group,
        "p_sign": p_sign,
        "p_wilcoxon": p_wilcoxon,
    }


def paired_regression(
    y: np.ndarray,
    covariates: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """OLS of paired deltas on paired covariate deltas with CR1
    cluster-robust standard errors (clusters = problems).

    y          : (n,) feature deltas f+ - f-
    covariates : (n, k) surface deltas (length, logprob, entropy, counts, ...)
    groups     : (n,) problem ids

    The intercept beta0 is the correctness-associated difference after
    accounting for the measured candidate differences. Returns beta0, its
    clustered SE, a normal-approximation p-value, and the R^2. Descriptive
    control, not a classifier.
    """
    from scipy import stats

    y = np.asarray(y, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    groups = np.asarray(groups)
    if covariates.ndim != 2 or y.shape[0] != covariates.shape[0] \
            or y.shape[0] != groups.shape[0]:
        raise ValueError("y, covariates, groups must share the first axis")

    n, k = covariates.shape
    X = np.column_stack([np.ones(n), covariates])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    uniq, inv = np.unique(groups, return_inverse=True)
    n_groups = len(uniq)
    # CR1 sandwich: sum over clusters of (X_g' u_g)(X_g' u_g)'
    xu = X * resid[:, None]
    cluster_scores = np.zeros((n_groups, X.shape[1]))
    np.add.at(cluster_scores, inv, xu)
    meat = cluster_scores.T @ cluster_scores
    dof_adj = (n_groups / (n_groups - 1)) * ((n - 1) / (n - X.shape[1]))
    cov = dof_adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))

    z0 = beta[0] / se[0] if se[0] > 0 else 0.0
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0

    return {
        "n_pairs": float(n),
        "n_groups": float(n_groups),
        "beta0": float(beta[0]),
        "se0": float(se[0]),
        "z0": float(z0),
        "p0": float(2 * stats.norm.sf(abs(z0))),
        "r2": float(r2),
        "raw_mean": float(y.mean()),
    }
