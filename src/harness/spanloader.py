"""Feed step-token spans to the sequence learners without starving the GPU.

Grid B spent 6h43 on 12 runs where Grid A did 45 in 68 minutes, and the giveaway
was that three transformers spanning an 11.6x parameter range finished within two
seconds of each other: 722,817 params in 8729.5s, 2,628,609 in 8731.4s, 8,402,945
in 8730.4s. Identical wall time across that range means the H100s were idle
waiting for data, not computing.

Two things were costing the time, and this module fixes both.

*Per-item Python reads.* The old path called one memory-mapped slice per step and
assembled the padded batch in a Python loop, about 1,270 items/second. Here a
batch is built with a single vectorized gather: row indices for the whole batch
are computed at once and one fancy-index copy fills the padded tensor.

*Padding waste.* Spans run 3 to 1,077 tokens with a mean of 38.8, so a randomly
drawn batch is padded to whatever its longest member happens to be and most of
the tensor is padding the model still attends over. Length bucketing shuffles,
sorts within a pool, cuts batches, then shuffles the batch order, so batches are
internally uniform in length while the epoch stays randomized. Nothing here reads
labels, so the bucketing cannot leak them.

`preload` trades RAM for disk: the whole split is read once into one contiguous
float16 array, after which an epoch touches no disk at all.
"""

from __future__ import annotations

import numpy as np
import torch


def span_bounds(handles, t_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Absolute start row and length of every item's span, truncated to t_max.

    Truncation keeps the *last* t_max tokens, matching the previous loader: the
    end of a step is where its conclusion lands, so that is the end to keep.
    """
    starts = np.empty(len(handles), dtype=np.int64)
    lengths = np.empty(len(handles), dtype=np.int64)
    for k, (rs, li, step_start, _n_tokens, _y) in enumerate(handles):
        a = int(rs.offsets[li]) + int(step_start)
        b = int(rs.offsets[li + 1])
        if b <= a:                      # degenerate step: fall back to its last row
            a, b = b - 1, b
        if b - a > t_max:
            a = b - t_max
        starts[k] = a
        lengths[k] = b - a
    return starts, lengths


def batch_gather_indices(starts: np.ndarray, lengths: np.ndarray, T: int):
    """Flat source rows and destination slots for one padded batch.

    Returns (src, dest) such that `out.reshape(B*T, d)[dest] = source[src]`
    fills a (B, T, d) padded tensor in a single copy.
    """
    total = int(lengths.sum())
    item_id = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    starts_within = np.repeat(np.cumsum(lengths) - lengths, lengths)
    within = np.arange(total, dtype=np.int64) - starts_within
    src = np.repeat(starts, lengths) + within
    dest = item_id * T + within
    return src, dest


def length_bucketed_batches(
    lengths: np.ndarray, batch_size: int, rng: np.random.Generator,
    pool_factor: int = 64, shuffle: bool = True,
) -> list[np.ndarray]:
    """Index batches whose members have similar length, in randomized order.

    Shuffle, cut into pools of `pool_factor` batches, sort each pool by length,
    slice batches out of it, then shuffle the batches. Every index appears
    exactly once, so an epoch is still an epoch.
    """
    n = len(lengths)
    order = rng.permutation(n) if shuffle else np.arange(n)
    pool = max(1, pool_factor) * batch_size
    batches: list[np.ndarray] = []
    for i in range(0, n, pool):
        chunk = order[i:i + pool]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        for j in range(0, len(chunk), batch_size):
            batches.append(chunk[j:j + batch_size])
    if shuffle:
        rng.shuffle(batches)
    return batches


class SpanLoader:
    """Vectorized padded-batch builder over a step-span store."""

    def __init__(self, handles, t_max: int, device, preload: bool = False,
                 stats: dict | None = None):
        self.handles = handles
        self.t_max = t_max
        self.device = device
        self.starts, self.lengths = span_bounds(handles, t_max)
        self.labels = np.array([h[4] for h in handles], dtype=np.float32)
        self.d = int(handles[0][0].spec.dim) if handles else 0
        # Optional per-position rescaling, applied to every token row so the
        # sequence learners see numbers of the same size as the vector ones.
        self.stats = stats
        self._flat = None
        if preload:
            self._preload()

    # -- memory -----------------------------------------------------------
    def preload_bytes(self) -> int:
        return int(self.lengths.sum()) * self.d * 2

    def _preload(self) -> None:
        """Copy every span into one contiguous float16 array, once."""
        total = int(self.lengths.sum())
        flat = np.empty((total, self.d), dtype=np.float16)
        cur = 0
        for k, (rs, _li, _s, _n, _y) in enumerate(self.handles):
            L = int(self.lengths[k])
            a = int(self.starts[k])
            flat[cur:cur + L] = rs.h[a:a + L]
            cur += L
        # After preloading, spans are contiguous in the new array, so restate
        # the offsets against it and never touch the store again.
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)
        self._flat = flat

    # -- batching ---------------------------------------------------------
    def _rows(self, src: np.ndarray) -> np.ndarray:
        if self._flat is not None:
            return self._flat[src]
        # mmap path: one gather per shard keeps it to a handful of reads
        out = np.empty((len(src), self.d), dtype=np.float16)
        rs = self.handles[0][0]
        if all(h[0] is rs for h in self.handles):
            out[:] = rs.h[src]
            return out
        for i, row in enumerate(src):          # mixed shards: per-row fallback
            out[i] = self._row_slow(int(row))
        return out

    def _row_slow(self, row: int) -> np.ndarray:
        for rs, *_ in self.handles:
            if row < rs.h.shape[0]:
                return rs.h[row]
        raise IndexError(row)

    def collate(self, idx: np.ndarray):
        """(x (B,T,d) float32, mask (B,T) float32, y (B,)) on the target device."""
        idx = np.asarray(idx, dtype=np.int64)
        lens = self.lengths[idx]
        T = int(lens.max())
        src, dest = batch_gather_indices(self.starts[idx], lens, T)
        flat = np.zeros((len(idx) * T, self.d), dtype=np.float32)
        rows = self._rows(src)
        if self.stats is not None:
            rows = (rows.astype(np.float32) - self.stats["mean"]) / self.stats["std"] \
                if self.stats["center"] else rows.astype(np.float32) / self.stats["std"]
        flat[dest] = rows
        mask = np.zeros(len(idx) * T, dtype=np.float32)
        mask[dest] = 1.0
        x = torch.from_numpy(flat.reshape(len(idx), T, self.d)).to(self.device)
        m = torch.from_numpy(mask.reshape(len(idx), T)).to(self.device)
        y = torch.from_numpy(self.labels[idx]).to(self.device)
        return x, m, y

    def batches(self, batch_size: int, rng, bucketed: bool = True,
                subset: np.ndarray | None = None) -> list[np.ndarray]:
        if subset is None:
            subset = np.arange(len(self.handles), dtype=np.int64)
        if not bucketed:
            order = rng.permutation(len(subset))
            return [subset[order[i:i + batch_size]]
                    for i in range(0, len(subset), batch_size)]
        local = length_bucketed_batches(self.lengths[subset], batch_size, rng)
        return [subset[b] for b in local]

    def eval_batches(self, batch_size: int) -> list[np.ndarray]:
        """In-order batches for scoring, where output order must be preserved."""
        n = len(self.handles)
        return [np.arange(i, min(i + batch_size, n), dtype=np.int64)
                for i in range(0, n, batch_size)]

    def padding_waste(self, batches: list[np.ndarray]) -> float:
        """Fraction of the padded tensors that is padding, for the run log."""
        real = pad = 0
        for b in batches:
            lens = self.lengths[b]
            real += int(lens.sum())
            pad += int(len(b) * lens.max())
        return 1.0 - real / max(pad, 1)
