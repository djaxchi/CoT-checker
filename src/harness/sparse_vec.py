"""Sparse pooled SAE codes, densified per batch.

A Qwen-Scope code is 65,536 wide with 50 non-zeros per token. Pooling a step's
tokens keeps it sparse, so storing it dense would cost 67 GB for the training
split against about 1.5 GB in sparse form, and every epoch would read the dense
version off Lustre. Storing CSR and scattering into a dense batch tensor at
collate time gives the learner exactly the dense vector it expects while the
store and the I/O stay small. The same trick as the span loader, one axis over.

On-disk (npz per split):
    indptr  int64 (n+1,)   row k occupies [indptr[k], indptr[k+1])
    indices int32 (nnz,)   feature ids
    values  float16 (nnz,) activations
    y       int8 (n,)      labels
    shape   (n, d_sae)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def write_csr(path: str | Path, rows: list[tuple[np.ndarray, np.ndarray]],
              labels: np.ndarray, d_sae: int) -> dict:
    """Write pooled sparse codes. `rows` is [(indices, values)] per item."""
    lengths = np.array([len(i) for i, _ in rows], dtype=np.int64)
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = np.concatenate([i for i, _ in rows]).astype(np.int32) if rows else np.zeros(0, np.int32)
    values = np.concatenate([v for _, v in rows]).astype(np.float16) if rows else np.zeros(0, np.float16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, indptr=indptr, indices=indices, values=values,
             y=np.asarray(labels, dtype=np.int8), shape=np.array([len(rows), d_sae]))
    return {"items": len(rows), "nnz": int(lengths.sum()),
            "mean_nnz": float(lengths.mean()) if len(rows) else 0.0,
            "density": float(lengths.mean() / d_sae) if len(rows) else 0.0,
            "bytes": int(path.stat().st_size)}


class SparseVecSplit:
    """Reader that hands the learner dense batches from a sparse store."""

    def __init__(self, path: str | Path, device=None):
        z = np.load(Path(path))
        self.indptr = z["indptr"]
        self.indices = z["indices"]
        self.values = z["values"]
        self.y = z["y"].astype(np.float32)
        self.n, self.d = (int(x) for x in z["shape"])
        self.device = device or torch.device("cpu")

    def __len__(self) -> int:
        return self.n

    @property
    def mean_nnz(self) -> float:
        return float(np.diff(self.indptr).mean())

    def collate(self, idx):
        """(x (B, d_sae) float32, None, y (B,)) — the vector-learner contract."""
        idx = np.asarray(idx, dtype=np.int64)
        starts, ends = self.indptr[idx], self.indptr[idx + 1]
        counts = ends - starts
        total = int(counts.sum())
        # gather the flat CSR ranges for the whole batch in one pass
        pos = np.repeat(starts, counts) + (
            np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts))
        rows = np.repeat(np.arange(len(idx), dtype=np.int64), counts)
        x = torch.zeros((len(idx), self.d), dtype=torch.float32)
        x[torch.from_numpy(rows), torch.from_numpy(self.indices[pos].astype(np.int64))] = \
            torch.from_numpy(self.values[pos].astype(np.float32))
        return (x.to(self.device), None,
                torch.from_numpy(self.y[idx]).to(self.device))
