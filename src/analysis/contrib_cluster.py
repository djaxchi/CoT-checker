"""S4 contrib-cluster core logic: prefix construction, step representations,
weak regex tagging, and cluster/tag statistics.

Exploratory experiment: do hidden-state-derived step representations organize
PRM800K reasoning steps into interpretable clusters? No correctness labels are
used anywhere in this module.

Representations (given the prefix hidden states h_0..h_T of one trajectory):

  state_i        = h_i
  qres_i         = h_i - h_0
  contribution_i = h_i - h_{i-1}

`contribution` is the closed form of the intended recursion

  c_1 = h_1 - h_0
  c_i = h_i - (h_0 + sum_{k<i} c_k)

which telescopes exactly: h_0 + sum_{k<=i} c_k == h_i for every i, hence
c_i == h_i - h_{i-1}. test_contrib_cluster.py pins this identity, and we
compute the closed form directly.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Prefix construction (spec: p_0 = q, p_i = q + "\n" + s_1 + "\n" + ... + s_i)
# ---------------------------------------------------------------------------


def build_prefixes(question: str, steps: list[str]) -> list[str]:
    """Return [p_0, p_1, ..., p_T]: the question, then cumulative step prefixes."""
    prefixes = [question]
    for i in range(1, len(steps) + 1):
        prefixes.append(question + "\n" + "\n".join(steps[:i]))
    return prefixes


def fit_steps_to_length(count_tokens, question: str, steps: list[str],
                        max_seq_len: int, min_steps: int = 2) -> list[str] | None:
    """Length fallback: drop trailing steps until the full prefix p_T fits.

    count_tokens maps a prefix string to its token count. Returns the kept
    steps (possibly all of them), or None only if even the first min_steps
    steps exceed max_seq_len. Trailing truncation keeps the trajectory
    continuous: h_0..h_K are still cumulative prefixes of the same reasoning.
    """
    kept = list(steps)
    while kept:
        if count_tokens(build_prefixes(question, kept)[-1]) <= max_seq_len:
            return kept
        if len(kept) <= min_steps:
            return None
        kept.pop()
    return None


# ---------------------------------------------------------------------------
# Step representations
# ---------------------------------------------------------------------------

REPR_NAMES = ("state", "qres", "contribution")


def compute_reprs(H: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all step representations for one trajectory.

    H is (T+1, d) float32: rows h_0 (question only) through h_T. Returns a dict
    of (T, d) arrays, one row per step i=1..T (nothing is emitted for p_0).
    """
    if H.ndim != 2 or H.shape[0] < 2:
        raise ValueError(f"H must be (T+1, d) with T >= 1, got {H.shape}")
    H = H.astype(np.float32, copy=False)
    return {
        "state": H[1:].copy(),
        "qres": H[1:] - H[0],
        "contribution": H[1:] - H[:-1],
    }


def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


# ---------------------------------------------------------------------------
# Weak regex tags (interpretation only — never used during clustering)
# ---------------------------------------------------------------------------

TAG_RULES: dict[str, list[str]] = {
    "VARIABLE_DEFINE": [
        r"\blet\s+[a-zA-Z]\b",
        r"\bdefine\s+[a-zA-Z]\b",
        r"\bset\s+[a-zA-Z]\s*=",
        r"\bsuppose\s+[a-zA-Z]\b",
    ],
    "ARITHMETIC_COMPUTE": [
        r"\d+\s*[\+\-\*/]\s*\d+",
        r"=\s*-?\d+(\.\d+)?",
        r"\bcompute\b",
        r"\bcalculate\b",
    ],
    "EQUATION_TRANSFORM": [
        r"\bexpand\b",
        r"\bfactor\b",
        r"\bsimplify\b",
        r"\brearrange\b",
        r"\bcombine like terms\b",
    ],
    "EQUATION_SOLVE": [
        r"\bsolve\b",
        r"\bisolate\b",
        r"\btherefore\s+[a-zA-Z]\s*=",
        r"\bso\s+[a-zA-Z]\s*=",
    ],
    "SUBSTITUTION": [
        r"\bsubstitute\b",
        r"\bplug\s+in\b",
        r"\breplace\b",
        r"\busing\s+.*=",
    ],
    "CASE_SPLIT": [
        r"\bcase\s+\d+\b",
        r"\bif\b",
        r"\botherwise\b",
        r"\bwhen\b",
    ],
    "INEQUALITY_BOUND": [
        r"[<>≤≥]",
        r"\\leq|\\geq",
        r"\bat least\b",
        r"\bat most\b",
        r"\bgreater than\b",
        r"\bless than\b",
    ],
    "GEOMETRY_REASONING": [
        r"\btriangle\b",
        r"\bcircle\b",
        r"\bangle\b",
        r"\bradius\b",
        r"\barea\b",
        r"\bperimeter\b",
    ],
    "COUNTING_COMBINATORICS": [
        r"\bways\b",
        r"\bpermutation\b",
        r"\bcombination\b",
        r"\bchoose\b",
        r"\barrangements?\b",
    ],
    "PROBABILITY": [
        r"\bprobability\b",
        r"\bexpected value\b",
        r"\bchance\b",
    ],
    "THEOREM_INVOKE": [
        r"\btheorem\b",
        r"\bpythagorean\b",
        r"\bAM-GM\b",
        r"\bmodulo\b",
        r"\bcongruence\b",
    ],
    "INTERMEDIATE_CONCLUSION": [
        r"\btherefore\b",
        r"\bthus\b",
        r"\bhence\b",
        r"\bso\b",
    ],
    "FINAL_ANSWER": [
        r"\bthe answer is\b",
        r"\bfinal answer\b",
        r"\\boxed",
    ],
    "TEXTUAL_EXPLANATION": [
        # catch-all for prose steps: sentence-like text with no math symbols
        r"^[^=<>\\\d]{40,}$",
    ],
}

TAG_NAMES = tuple(TAG_RULES)

_COMPILED: dict[str, list[re.Pattern]] = {
    tag: [re.compile(p, re.IGNORECASE) for p in pats] for tag, pats in TAG_RULES.items()
}


def tag_step(text: str) -> dict[str, bool]:
    """Multi-label weak tags for one step text."""
    return {
        tag: any(p.search(text) for p in pats) for tag, pats in _COMPILED.items()
    }


_MATH_OPS = re.compile(r"[\+\-\*/\^]")
_LATEX = re.compile(r"\\[a-zA-Z]+|\$")


def surface_features(text: str) -> dict[str, int | bool]:
    """Trivial surface features used as controls, never as clustering input."""
    return {
        "char_len": len(text),
        "n_digits": sum(ch.isdigit() for ch in text),
        "n_equals": text.count("="),
        "n_math_ops": len(_MATH_OPS.findall(text)),
        "contains_boxed": "\\boxed" in text,
        "contains_latex": bool(_LATEX.search(text)),
    }


def assign_top_tag(tag_matrix: np.ndarray, tag_names: tuple[str, ...] = TAG_NAMES) -> list[str]:
    """Single display tag per step: the *rarest* matching tag corpus-wide.

    Rare tags are the most informative (INTERMEDIATE_CONCLUSION's "so" matches
    half the corpus); steps with no match get "NONE".
    """
    counts = tag_matrix.sum(axis=0).astype(float)
    order = np.argsort(counts)  # rarest first
    out = []
    for row in tag_matrix:
        top = "NONE"
        for j in order:
            if row[j]:
                top = tag_names[j]
                break
        out.append(top)
    return out


# ---------------------------------------------------------------------------
# Cluster / tag statistics
# ---------------------------------------------------------------------------


def tag_enrichment(labels: np.ndarray, tag_matrix: np.ndarray,
                   tag_names: tuple[str, ...] = TAG_NAMES) -> list[dict]:
    """enrichment(c, t) = P(t | c) / P(t) for every cluster x tag.

    Noise (label -1) is included as its own row so it can be inspected, but
    callers should treat it separately. Returns one dict per (cluster, tag).
    """
    n = len(labels)
    base = tag_matrix.mean(axis=0)  # P(t)
    rows = []
    for c in sorted(set(labels.tolist())):
        m = labels == c
        size = int(m.sum())
        p_in = tag_matrix[m].mean(axis=0)  # P(t | c)
        for j, t in enumerate(tag_names):
            rows.append({
                "cluster": int(c),
                "cluster_size": size,
                "tag": t,
                "p_tag_given_cluster": float(p_in[j]),
                "p_tag": float(base[j]),
                "enrichment": float(p_in[j] / base[j]) if base[j] > 0 else float("nan"),
                "n_total": n,
            })
    return rows


def tag_entropy(labels: np.ndarray, top_tags: list[str]) -> dict[int, float]:
    """Shannon entropy (bits) of the top-tag distribution inside each cluster."""
    out: dict[int, float] = {}
    tags = np.asarray(top_tags)
    for c in sorted(set(labels.tolist())):
        counts = Counter(tags[labels == c].tolist())
        total = sum(counts.values())
        ent = -sum((v / total) * math.log2(v / total) for v in counts.values() if v)
        out[int(c)] = ent
    return out


def surface_eta_squared(labels: np.ndarray, values: np.ndarray) -> float:
    """Effect size (eta^2) of cluster membership on one surface feature.

    Between-cluster variance / total variance, noise points excluded. High
    values flag clusterings that are largely explained by a trivial feature.
    """
    m = labels >= 0
    if m.sum() < 2:
        return float("nan")
    lab, val = labels[m], values[m].astype(np.float64)
    total = ((val - val.mean()) ** 2).sum()
    if total <= 0:
        return float("nan")
    between = 0.0
    for c in set(lab.tolist()):
        v = val[lab == c]
        between += len(v) * (v.mean() - val.mean()) ** 2
    return float(between / total)
