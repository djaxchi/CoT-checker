"""The activation-artifact audit that gates the Instruct arm (instruct_arm_v1 §4).

Instruction tuning is known to sharpen a few residual dimensions and to route
attention onto sink and template positions. A correctness probe fitted on such
states can score well by reading format rather than step content, so an
Instruct gain only counts if it survives four measurements, each computed on
the Base arm through the same code for a reference value:

1. Outlier-dimension mass: how much of each token's squared norm sits in its
   top-k coordinates, which fixed dimensions dominate across tokens, and how
   often a token is "massive" (norm far above the median).
2. Attention-sink and template mass: where the step tokens' attention goes at
   the read layer, split into token 0, template text, the problem, prior steps
   and the step itself.
3. Probe localisation: per-position occlusion of the trained probe. The
   step_tokens probe reads only the step's own tokens (the boundary row is
   skipped), so template positions are out of its reach by construction; what
   can still carry format is the first step token and any massive token, and
   that is what the occlusion shares are measured against.
4. Length and position residualisation: step AUROC after regressing the probe
   logit on log step length, step index and log context length.

Pure functions only; the job script does the I/O and the forward passes.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------


def auroc(y, s) -> float:
    """Mann-Whitney AUROC with average ranks for ties. NaN for one class."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pb_step_labels(meta: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Step labels for ProcessBench under the first-error convention.

    ProcessBench annotates the index of the first wrong step per trace (-1 when
    none). Steps before it are correct (0), the step itself is the error (1),
    and steps after it are left out: they are neither verified correct nor the
    error being located.
    """
    y = np.zeros(len(meta), dtype=np.int8)
    keep = np.ones(len(meta), dtype=bool)
    for k, m in enumerate(meta):
        fe, si = int(m["label"]), int(m["step_idx"])
        if fe >= 0 and si == fe:
            y[k] = 1
        elif fe >= 0 and si > fe:
            keep[k] = False
    return y, keep


# ---------------------------------------------------------------------------
# 1. outlier-dimension mass
# ---------------------------------------------------------------------------


def topk_norm_share(H: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Mean over tokens of the share of squared norm in the token's top-k dims."""
    sq = np.asarray(H, dtype=np.float64) ** 2
    tot = sq.sum(1)
    ok = tot > 0
    srt = -np.sort(-sq[ok], axis=1)
    cum = np.cumsum(srt, axis=1)
    out = {}
    for k in ks:
        out[f"top{k}"] = float(np.mean(cum[:, k - 1] / tot[ok]))
    return out


def dimension_concentration(H: np.ndarray, k: int = 10) -> dict:
    """Which fixed dimensions carry the most mean-square mass across tokens."""
    ms = np.mean(np.asarray(H, dtype=np.float64) ** 2, axis=0)
    top = np.argsort(-ms)[:k]
    return {"top_dims": [int(i) for i in top],
            "share": float(ms[top].sum() / ms.sum()),
            "uniform_share": k / len(ms)}


def massive_token_stats(norms: list[np.ndarray], factor: float = 5.0) -> dict:
    """Rate of tokens whose norm exceeds factor x the pooled median norm.

    `norms` holds one array per step, in step-token order, so the share of
    massive tokens that sit at the step's first position can be read off: that
    is where a template-like attractor would show up in a span-only store.
    """
    allv = np.concatenate(norms)
    thr = factor * float(np.median(allv))
    n_mass = 0; n_first = 0
    for v in norms:
        m = v > thr
        n_mass += int(m.sum())
        n_first += int(m[0]) if len(v) else 0
    return {"threshold": thr, "median_norm": float(np.median(allv)),
            "rate": n_mass / len(allv),
            "share_at_first": (n_first / n_mass) if n_mass else float("nan"),
            "first_token_rate": n_first / len(norms)}


# ---------------------------------------------------------------------------
# 2. attention-sink and template mass
# ---------------------------------------------------------------------------


def verifier_char_spans(problem: str, prefix: str) -> list[tuple[int, int, str]]:
    """Character spans of `verifier_prefix(problem, prefix)`, by category.

    Mirrors src/onpolicy/prompts.verifier_prefix piece by piece; a test checks
    that the spans concatenate back to that exact string.
    """
    pieces: list[tuple[str, str]] = [("Problem:\n", "template"), (problem, "problem")]
    if prefix:
        pieces += [("\n\nPrevious reasoning:\n", "template"), (prefix, "prior"),
                   ("\n\nCurrent step:\n", "template")]
    else:
        pieces += [("\n\nPrevious reasoning:\n\nCurrent step:\n", "template")]
    out = []; c = 0
    for text, cat in pieces:
        if text:
            out.append((c, c + len(text), cat))
            c += len(text)
    return out


def token_categories(offsets, spans, n_step: int) -> list[str]:
    """Category of every token: prefix tokens by the span holding their first
    character, token 0 split out as the sink, then the step's own tokens."""
    cats = []
    for i, (a, _b) in enumerate(offsets):
        if i == 0:
            cats.append("sink"); continue
        cat = "template"
        for s0, s1, c in spans:
            if s0 <= a < s1:
                cat = c; break
        cats.append(cat)
    return cats + ["step"] * n_step


def attention_mass(A: np.ndarray, cats: list[str]) -> dict:
    """Mean attention of step-token queries onto each key category.

    `A` is (T, T) attention already averaged over heads (rows = queries).
    """
    cats_a = np.asarray(cats)
    q = np.where(cats_a == "step")[0]
    rows = np.asarray(A, dtype=np.float64)[q]
    out = {}
    for c in ("sink", "template", "problem", "prior", "step"):
        k = cats_a == c
        out[c] = float(rows[:, k].sum(1).mean()) if k.any() else 0.0
    return out


# ---------------------------------------------------------------------------
# 3. probe localisation
# ---------------------------------------------------------------------------


def occlusion_summary(deltas: list[np.ndarray], norms: list[np.ndarray]) -> dict:
    """Where the probe's attribution sits, from per-position occlusion deltas.

    deltas[i][j] is the logit change when position j of step i is masked out;
    norms[i][j] is that token's residual norm. Shares are of total |delta| per
    step, averaged over steps, each next to the value a probe spreading its
    attribution evenly would give (1/L), so a share is only read as
    concentration when it clears that baseline.
    """
    first, first_u, maxn, rho = [], [], [], []
    for d, v in zip(deltas, norms):
        a = np.abs(np.asarray(d, dtype=np.float64))
        tot = a.sum()
        if tot <= 0 or len(a) == 0:
            continue
        L = len(a)
        first.append(a[0] / tot)
        first_u.append(1.0 / L)
        maxn.append(a[int(np.argmax(v))] / tot)
        if L >= 3:
            ra = np.argsort(np.argsort(a)); rv = np.argsort(np.argsort(v))
            if ra.std() > 0 and rv.std() > 0:
                rho.append(float(np.corrcoef(ra, rv)[0, 1]))
    return {"n_items": len(first),
            "share_first": float(np.mean(first)) if first else float("nan"),
            "share_first_uniform": float(np.mean(first_u)) if first else float("nan"),
            "share_maxnorm": float(np.mean(maxn)) if maxn else float("nan"),
            "spearman_attr_vs_norm": float(np.mean(rho)) if rho else float("nan")}


# ---------------------------------------------------------------------------
# 4. length and position residualisation
# ---------------------------------------------------------------------------


def _design(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    mu, sd = C.mean(0), C.std(0)
    sd[sd == 0] = 1.0
    return np.c_[np.ones(len(C)), (C - mu) / sd]


def _logistic_fit(X, y, iters: int = 200, l2: float = 1e-4) -> np.ndarray:
    """Newton's method; the design is a handful of columns, so this is exact
    enough and keeps the job free of an sklearn dependency."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + l2 * w
        Hs = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def residual_auroc(y, score, covariates, n_folds: int = 5, seed: int = 0) -> dict:
    """AUROC of the probe score raw, after removing what the covariates explain
    linearly, and of the covariates alone (held-out folds, logistic)."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    X = _design(covariates)
    beta, *_ = np.linalg.lstsq(X, s, rcond=None)
    resid = s - X @ beta
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(y))
    cov_pred = np.empty(len(y))
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        w = _logistic_fit(X[tr], y[tr])
        cov_pred[te] = X[te] @ w
    return {"raw": auroc(y, s), "residual": auroc(y, resid),
            "covariates_only": auroc(y, cov_pred), "n": int(len(y)),
            "r2_score_on_covariates": float(1 - resid.var() / s.var()) if s.var() > 0 else 0.0}
