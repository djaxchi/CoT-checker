"""Fingerprint a stored split, so a leaderboard can prove its rows read one input.

A benchmark that varies the representation is only meaningful if every cell was
shown the same activations. That is easy to believe and easy to get wrong: a
re-derived cache, a half-written shard, a store rebuilt with different code, and
two rows of the table stop being comparable without anything looking wrong. The
v1 leaderboard had no way to detect that at all.

`split_fingerprint` returns a short digest of a split's *content and structure*:
the shard names, the feature dimension, the full lengths and label arrays, the
packed array's shape, and a deterministic sample of its rows. Every cell records
it, and the merge script refuses to build a table from cells whose fingerprints
disagree.

Honest about what it is: the row sample makes this cheap (well under a second on
a 137GiB split) but means it is a fingerprint, not a checksum. It will catch a
different store, a different split, a rebuild, a truncated or reordered shard,
and any change to a label or an item length. It is not designed to catch an
adversarially placed single-row edit between sampled rows; pass `full=True` for
a complete hash of the packed bytes when that matters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SAMPLE_ROWS = 4096
CHUNK = 1 << 22


def _shard_dirs(path: Path) -> list[Path]:
    shards = sorted(d for d in path.iterdir()
                    if d.is_dir() and d.name.startswith("shard_") and (d / "h.npy").exists())
    return shards or [path]


def _hash_shard(h: hashlib._Hash, shard: Path, full: bool) -> None:
    spec = json.loads((shard / "spec.json").read_text())
    lengths = np.load(shard / "lengths.npy")
    y = np.load(shard / "y.npy")
    arr = np.load(shard / "h.npy", mmap_mode="r")

    h.update(shard.name.encode())
    h.update(str(spec.get("dim")).encode())
    h.update(str(spec.get("kind")).encode())
    h.update(np.ascontiguousarray(lengths).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())

    n_rows = arr.shape[0]
    if full:
        for i in range(0, n_rows, CHUNK):
            h.update(np.ascontiguousarray(arr[i:i + CHUNK]).tobytes())
        return
    if n_rows == 0:
        return
    # Deterministic even spread, so the sample depends only on the data's shape,
    # never on a random seed or on when the fingerprint was taken.
    idx = np.unique(np.linspace(0, n_rows - 1, min(SAMPLE_ROWS, n_rows)).astype(np.int64))
    h.update(np.ascontiguousarray(arr[idx]).tobytes())


def split_fingerprint(path: str | Path, full: bool = False) -> str:
    """Digest of one stored split. Same content -> same string, on any machine."""
    path = Path(path)
    h = hashlib.sha256()
    h.update(b"repstore-fingerprint-v1-full" if full else b"repstore-fingerprint-v1")
    for shard in _shard_dirs(path):
        _hash_shard(h, shard, full)
    return h.hexdigest()[:32]
