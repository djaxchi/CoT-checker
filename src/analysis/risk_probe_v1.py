"""risk_probe_v1 core: folds, view rows, linear readouts and their null.

Plan: docs/risk_probe_v1_plan.md.

Two readouts of downstream risk from the verifier's L35 step states:

* arm A regresses a step's risk (or classifies `harmful`) from the step's row;
* arm B fits a value probe V, the policy's error rate at a cut point, on the
  row just ahead of the step (cut s) and the step's last row (cut s + 1); the
  step's predicted risk is V(after) - V(before).

The solvers are closed-form ridge (one eigendecomposition per training set, every
alpha and every target reuses it) and L2 logistic regression on binomial counts
(L-BFGS in torch, so GPU when there is one). Both standardize the features on the
training rows, as sklearn's StandardScaler + Ridge / LogisticRegression did in
risk_probe_v0, and use the same objectives: ||y - Xw - b||^2 + alpha ||w||^2 and
sum_i [n_i log(1 + e^z_i) - k_i z_i] + ||w||^2 / (2C).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

N_FOLDS = 5
# risk_probe_v0's grids (RidgeCV logspace(0, 6, 13), LogisticRegressionCV logspace(-4, 2, 13))
# extended by two decades on the regularized side, same half-decade spacing: at ~25k
# training steps the smoke run selected the most regularized point of v0's grids
ALPHAS = np.logspace(0, 8, 17)
CS = np.logspace(-6, 2, 17)


def fold_of(problem_key: str, n: int = N_FOLDS, salt: str = "") -> int:
    """Problem fold, fixed by sha1 so every script and job agrees on it."""
    return int(hashlib.sha1((salt + problem_key).encode()).hexdigest(), 16) % n


def step_rows(H: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(before, after, mean) of a stored view: the row ahead of the step, its last row, its mean.

    In a compacted `full` view (build_context_views.compact) the row ahead of the
    step is the last token of the blank-line separator after the previous step, or
    the last header token for step 0; either way it is stored at start - 1.
    """
    if start < 1 or end <= start:
        raise ValueError(f"bad step span {start}:{end}")
    H = np.asarray(H)
    return H[start - 1], H[end - 1], H[start:end].astype(np.float32).mean(0)


def group_permutation(keys: Sequence, rng: np.random.Generator) -> np.ndarray:
    """A permutation of range(len(keys)) that only swaps items with equal keys.

    Used for the null: traces are permuted within (fold, n_steps), so each trace
    keeps a whole label vector of the right length and no label leaves its fold.
    """
    keys = list(keys)
    perm = np.arange(len(keys))
    by: dict = {}
    for i, k in enumerate(keys):
        by.setdefault(k, []).append(i)
    for idx in by.values():
        if len(idx) > 1:
            perm[idx] = rng.permutation(idx)
    return perm


def residualize(y: np.ndarray, covs: Sequence[np.ndarray]) -> np.ndarray:
    """y minus its OLS fit on [1, covs] (risk_probe_v0's residual risk)."""
    C = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in covs])
    return y - C @ np.linalg.lstsq(C, y, rcond=None)[0]


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #

def _torch():
    import torch
    return torch


@dataclass
class Linear:
    """Standardize then w.x + b; `link` is 'identity' or 'logit'."""
    mu: object
    sd: object
    w: object
    b: object
    link: str = "identity"

    def predict(self, X):
        torch = _torch()
        Xt = torch.as_tensor(X, dtype=self.w.dtype, device=self.w.device)
        z = ((Xt - self.mu) / self.sd) @ self.w + self.b
        if self.link == "logit":
            z = torch.sigmoid(z)
        return z.detach().cpu().numpy().astype(np.float64)


def standardize(X):
    torch = _torch()
    mu = X.mean(0)
    sd = X.std(0, unbiased=False)
    sd = torch.where(sd > 1e-8, sd, torch.ones_like(sd))
    return (X - mu) / sd, mu, sd


class RidgeSolver:
    """Ridge for every alpha and every target from one eigendecomposition of X'X."""

    def __init__(self, X, std=None):
        torch = _torch()
        self.Z, self.mu, self.sd = std if std is not None else standardize(X)
        Zd = self.Z.double()                  # X'X and its eigenvectors in float64
        self.evals, self.V = torch.linalg.eigh(Zd.T @ Zd)
        del Zd

    def fit(self, y, alpha: float) -> Linear:
        torch = _torch()
        yt = torch.as_tensor(y, dtype=torch.float64, device=self.Z.device)
        ym = yt.mean(0)
        proj = self.V.T @ (self.Z.T.double() @ (yt - ym))
        w = self.V @ (proj / (self.evals + alpha))
        return Linear(self.mu, self.sd, w.to(self.Z.dtype), ym.to(self.Z.dtype))


def logistic_fit(X, k, n, C: float, max_iter: int = 200, tol: float = 1e-7, std=None) -> Linear:
    """L2 logistic regression on binomial counts (k successes of n) at strength C.

    Minimizes sum_i [n_i softplus(z_i) - k_i z_i] + ||w||^2 / (2C), z = w.x_std + b,
    sklearn's LogisticRegression objective with counts in place of 0/1 rows.
    `std` passes a precomputed (Z, mu, sd) for the same X.
    """
    torch = _torch()
    Z, mu, sd = std if std is not None else standardize(X)
    kt = torch.as_tensor(k, dtype=Z.dtype, device=Z.device)
    nt = torch.as_tensor(n, dtype=Z.dtype, device=Z.device)
    w = torch.zeros(Z.shape[1], dtype=Z.dtype, device=Z.device, requires_grad=True)
    p0 = (kt.sum() / nt.sum()).clamp(1e-6, 1 - 1e-6)
    b = torch.log(p0 / (1 - p0)).detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=max_iter, tolerance_grad=tol, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")
    scale = float(nt.sum())                  # rescaled objective: same minimizer, better conditioned

    def closure():
        opt.zero_grad()
        z = Z @ w + b
        loss = (nt * torch.nn.functional.softplus(z) - kt * z).sum() + (w * w).sum() / (2 * C)
        loss = loss / scale
        loss.backward()
        return loss
    opt.step(closure)
    return Linear(mu, sd, w.detach(), b.detach(), link="logit")


def binomial_deviance(p: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(k * np.log(p) + (n - k) * np.log(1 - p)).sum() / n.sum())


def auroc(score, y) -> float | None:
    s, y = np.asarray(score, float), np.asarray(y, int)
    if len(y) == 0 or y.min() == y.max():
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def spearman(x, y) -> float | None:
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return None
    r = spearmanr(x[ok], y[ok]).statistic
    return None if not np.isfinite(r) else float(r)


def inner_folds(groups: Sequence[str], n: int = 3) -> np.ndarray:
    """Inner CV folds by problem, salted so they cut across the outer folds."""
    return np.array([fold_of(g, n, salt="inner:") for g in groups])


def choose_alpha(X, y, groups, alphas=ALPHAS, n_inner: int = 3) -> float:
    """Alpha with the lowest held-out MSE over problem-grouped inner folds."""
    torch = _torch()
    f = inner_folds(groups, n_inner)
    yt = np.asarray(y, float)
    err = np.zeros(len(alphas))
    for k in range(n_inner):
        tr, te = np.where(f != k)[0], np.where(f == k)[0]
        if not len(te) or not len(tr):
            continue
        rs = RidgeSolver(X[torch.as_tensor(tr, device=X.device)])
        Xte = X[torch.as_tensor(te, device=X.device)]
        for j, a in enumerate(alphas):
            err[j] += ((rs.fit(yt[tr], a).predict(Xte) - yt[te]) ** 2).sum()
    return float(alphas[int(np.argmin(err))])


def choose_C(X, k, n, groups, metric: str, Cs=CS, n_inner: int = 3) -> float:
    """C with the best held-out score: 'auroc' (0/1 targets) or 'deviance' (counts)."""
    torch = _torch()
    f = inner_folds(groups, n_inner)
    k, n = np.asarray(k, float), np.asarray(n, float)
    score = np.zeros(len(Cs))
    for j_f in range(n_inner):
        tr, te = np.where(f != j_f)[0], np.where(f == j_f)[0]
        if not len(te) or not len(tr):
            continue
        Xtr = X[torch.as_tensor(tr, device=X.device)]
        Xte = X[torch.as_tensor(te, device=X.device)]
        std = standardize(Xtr)
        for j, C in enumerate(Cs):
            p = logistic_fit(Xtr, k[tr], n[tr], C, std=std).predict(Xte)
            if metric == "auroc":
                score[j] += -(auroc(p, k[te]) or 0.5) * len(te)
            else:
                score[j] += binomial_deviance(p, k[te], n[te]) * n[te].sum()
    return float(Cs[int(np.argmin(score))])
