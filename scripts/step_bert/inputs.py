"""Shared input reduction for step_bert_probe_v0.

Each 4096-d token part (b_i and m_i for `state`, c_i for `contribution`) is
projected onto its top-512 PCA directions (fit on train-split steps only, not
whitened) before entering the encoder. This cuts the trainable input projection
from 8192 x 256 to 1024 x 256 and the per-epoch cost by about 8x, keeping about
90% of each part's variance (measured on the 1k subset: m 0.901, b 0.907, c 0.863
at 512). The fit is cached per (tokens, n_in) so pretraining and fine-tuning use
the identical projection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

CACHE_DIR = Path(__file__).resolve().parents[2] / "data/tamia/step_bert_probe_v0"


def parts_of(z, tokens: str) -> list[np.ndarray]:
    if tokens == "state":
        return [z["b"], z["m"]]
    if tokens == "contribution":
        return [(z["m"].astype(np.float32) - z["b"].astype(np.float32))]
    raise ValueError(tokens)


def train_rows(z, split_path: Path) -> np.ndarray:
    split = json.loads(Path(split_path).read_text())
    return np.flatnonzero(np.isin(z["trace_id"], split["train"]))


def fit_or_load(z, tokens: str, split_path: Path, n_in: int = 512) -> list[tuple[np.ndarray, np.ndarray]]:
    path = CACHE_DIR / f"input_pca_{Path(split_path).stem}_{tokens}_{n_in}.npz"
    if path.exists():
        c = np.load(path)
        return [(c[f"mean{k}"], c[f"comp{k}"]) for k in range(int(c["n_parts"]))]
    rows = train_rows(z, split_path)
    fitted, save = [], {}
    for k, part in enumerate(parts_of(z, tokens)):
        pca = PCA(n_in, random_state=0, svd_solver="randomized").fit(part[rows].astype(np.float32))
        fitted.append((pca.mean_.astype(np.float32), pca.components_.astype(np.float32)))
        save[f"mean{k}"], save[f"comp{k}"] = fitted[-1]
        save[f"var_kept{k}"] = np.float32(pca.explained_variance_ratio_.sum())
    np.savez(path, n_parts=len(fitted), **save)
    return fitted


def reduce(z, tokens: str, proj: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """(N, n_parts * n_in) float16 encoder inputs."""
    outs = [((part.astype(np.float32) - mu) @ comp.T) for part, (mu, comp) in zip(parts_of(z, tokens), proj)]
    return np.concatenate(outs, 1).astype(np.float16)
