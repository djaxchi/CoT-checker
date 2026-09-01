"""Scale-free geometry of one step's token cloud, shared by the screen and the grid.

Kept in one place because the two paths must compute identical features: the
screen samples the store into npz files and the grid derives vectors through
`derive_split`, and a representation that scores 0.7897 in one and something else
in the other would be impossible to attribute.

The 20 features carry no direction in model space, so the whole vector is
invariant to rotating the residual stream, and no entry is a token count. Length
is left out on purpose: it scores 0.7039 on ProcessBench by itself and would
swamp everything here, which is exactly what happened when it was included
(`geom` 0.7000 against `geom_nolen` 0.5182).
"""

from __future__ import annotations

import numpy as np

N_GEOM = 20

EPS = 1e-8


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def _stats(v: np.ndarray, qs=(10, 50, 90)) -> list[float]:
    if v.size == 0:
        return [0.0] * (2 + len(qs))
    return [float(v.mean()), float(v.std())] + [float(np.percentile(v, q)) for q in qs]


def geom_feats(span: np.ndarray, bnd: np.ndarray, with_len: bool) -> np.ndarray:
    """Content-free geometry of one step: angles, spreads and norms only.

    Nothing here is a direction in model space, so the whole vector is invariant
    to any rotation of the residual stream. If it separates correct from
    incorrect steps, the signal is in the SHAPE of the step's token cloud rather
    than in where it points.
    """
    t = span.shape[0]
    n = np.linalg.norm(span, axis=1) + EPS
    u = span / n[:, None]
    mean = span.mean(0)
    md = _unit(mean)
    bd = _unit(bnd)
    cos_to_mean = u @ md
    consec = (u[:-1] * u[1:]).sum(1) if t >= 2 else np.zeros(1, np.float32)

    f = []
    f += _stats(cos_to_mean)                 # how tightly the tokens cone
    f += _stats(consec, qs=(50,))            # how far each token turns from the last
    f += [float(md @ bd),                    # does the step point where the prefix did
          float(_unit(span[0]) @ bd),
          float(_unit(span[-1]) @ bd),
          float(_unit(span[0]) @ _unit(span[-1]))]
    # ||mean|| / mean||token|| is 1 when every token points the same way and falls
    # toward 0 as they spread: the cone tightness the conicity work measured,
    # expressed without needing a metric on the space
    f += [float(np.linalg.norm(mean) / (n.mean() + EPS))]
    f += _stats(np.log(n), qs=(10, 90))      # norm distribution, scale free in log
    f += [float(np.log(np.linalg.norm(mean) + EPS)),
          float(np.log(np.linalg.norm(bnd) + EPS)),
          float(np.log((np.linalg.norm(mean) + EPS) / (np.linalg.norm(bnd) + EPS)))]
    if with_len:
        f += [float(np.log(max(t, 1)))]
    return np.asarray(f, dtype=np.float32)
