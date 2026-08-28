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

    def __init__(self, path: str | Path, device=None, stats: dict | None = None):
        self.stats = stats
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
        vals = self.values[pos].astype(np.float32)
        cols = self.indices[pos].astype(np.int64)
        if self.stats is not None:
            vals = vals / self.stats["std"][cols]
        x[torch.from_numpy(rows), torch.from_numpy(cols)] = torch.from_numpy(vals)
        return (x.to(self.device), None,
                torch.from_numpy(self.y[idx]).to(self.device))


# ---------------------------------------------------------------------------
# Per-token codes, for the sequence learners
# ---------------------------------------------------------------------------

class SparseTokenBatch:
    """One padded batch of per-token sparse codes, never densified to (B,T,d_sae).

    A step of 39 tokens at 50 non-zeros each is ~1,950 entries; the dense form
    would be 39 x 65,536. At batch 128 that is 4.3 GB a batch against 47 KB of
    actual data, so the learners read this instead. Everything they need is here
    in flat form:

        indices, values   (nnz,)      the active features and their activations
        offsets           (n_tok+1,)  token boundaries into indices/values
        tok_id            (nnz,)      which token each entry belongs to
        batch_id, pos     (n_tok,)    where each token sits in the padded (B,T)
    """

    def __init__(self, indices, values, offsets, tok_id, batch_id, pos, B, T, d):
        self.indices, self.values, self.offsets = indices, values, offsets
        self.tok_id, self.batch_id, self.pos = tok_id, batch_id, pos
        self.B, self.T, self.d = B, T, d

    @property
    def n_tokens(self) -> int:
        return len(self.batch_id)

    def token_mask(self) -> torch.Tensor:
        """(B, T) with 1 where a real token sits."""
        m = torch.zeros(self.B, self.T, device=self.batch_id.device)
        m[self.batch_id, self.pos] = 1.0
        return m

    def scatter_tokens(self, per_token: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        """(n_tok, k) -> (B, T, k) padded."""
        out = torch.full((self.B, self.T, per_token.shape[-1]), fill,
                         dtype=per_token.dtype, device=per_token.device)
        out[self.batch_id, self.pos] = per_token
        return out


def write_token_csr(path: str | Path, steps: list[list[tuple[np.ndarray, np.ndarray]]],
                    labels: np.ndarray, d_sae: int) -> dict:
    """Write per-token codes. `steps` is [[(indices, values)] per token] per step."""
    tok_counts = np.array([len(s) for s in steps], dtype=np.int64)
    step_ptr = np.zeros(len(steps) + 1, dtype=np.int64)
    np.cumsum(tok_counts, out=step_ptr[1:])
    flat = [t for s in steps for t in s]
    nnz = np.array([len(i) for i, _ in flat], dtype=np.int64)
    tok_ptr = np.zeros(len(flat) + 1, dtype=np.int64)
    np.cumsum(nnz, out=tok_ptr[1:])
    indices = np.concatenate([i for i, _ in flat]).astype(np.int32) if flat else np.zeros(0, np.int32)
    values = np.concatenate([v for _, v in flat]).astype(np.float16) if flat else np.zeros(0, np.float16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, step_ptr=step_ptr, tok_ptr=tok_ptr, indices=indices, values=values,
             y=np.asarray(labels, dtype=np.int8), shape=np.array([len(steps), d_sae]))
    return {"items": len(steps), "tokens": len(flat), "nnz": int(nnz.sum()),
            "mean_tokens": float(tok_counts.mean()) if len(steps) else 0.0,
            "mean_nnz": float(nnz.mean()) if len(flat) else 0.0,
            "bytes": int(path.stat().st_size)}


class SparseTokenSplit:
    """Reader that hands the sequence learners sparse token batches."""

    def __init__(self, path: str | Path, t_max: int = 512, device=None,
                 stats: dict | None = None):
        self.stats = stats
        z = np.load(Path(path))
        self.step_ptr = z["step_ptr"]
        self.tok_ptr = z["tok_ptr"]
        self.indices = z["indices"]
        self.values = z["values"]
        self.y = z["y"].astype(np.float32)
        self.n, self.d = (int(x) for x in z["shape"])
        self.t_max = t_max
        self.device = device or torch.device("cpu")
        self.lengths = np.minimum(np.diff(self.step_ptr), t_max)

    def __len__(self) -> int:
        return self.n

    def collate(self, idx):
        """(SparseTokenBatch, mask (B,T), y (B,))."""
        idx = np.asarray(idx, dtype=np.int64)
        # keep the LAST t_max tokens, matching the dense span loader
        ends = self.step_ptr[idx + 1]
        lens = self.lengths[idx]
        starts = ends - lens
        T = int(lens.max())

        tok_rows = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) \
            if len(idx) else np.zeros(0, np.int64)
        batch_id = np.repeat(np.arange(len(idx), dtype=np.int64), lens)
        pos = np.concatenate([np.arange(l) for l in lens]) if len(idx) else np.zeros(0, np.int64)

        counts = self.tok_ptr[tok_rows + 1] - self.tok_ptr[tok_rows]
        total = int(counts.sum())
        flat_pos = np.repeat(self.tok_ptr[tok_rows], counts) + (
            np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts))
        tok_id = np.repeat(np.arange(len(tok_rows), dtype=np.int64), counts)
        offsets = np.zeros(len(tok_rows) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        dev = self.device
        t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)  # noqa: E731
        batch = SparseTokenBatch(
            indices=t(self.indices[flat_pos], torch.long),
            values=t(self.values[flat_pos].astype(np.float32) /
                     (self.stats["std"][self.indices[flat_pos]] if self.stats
                      else 1.0), torch.float32),
            offsets=t(offsets[:-1], torch.long),
            tok_id=t(tok_id, torch.long),
            batch_id=t(batch_id, torch.long), pos=t(pos, torch.long),
            B=len(idx), T=T, d=self.d)
        return batch, batch.token_mask(), t(self.y[idx], torch.float32)
