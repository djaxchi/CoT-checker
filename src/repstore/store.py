"""Uniform on-disk store for step-level representations.

The store decouples *representations* from *learners*. Every representation of a
data split is written in one packed format regardless of whether it is a fixed
vector (dense last-token, delta) or a variable-length sequence (all token states
of a step, a step-state trajectory). A learner then consumes it however it likes:
a linear probe or MLP reduces each item to a vector (mean / max / last), a
transformer or LSTM consumes the padded sequence directly. Pooling therefore
becomes a learner-side ablation, not a storage decision.

On-disk layout, one directory per (representation, split):

    <root>/<rep_name>/<split>/
        h.npy        float16 (total_rows, d)   packed item vectors, row-concatenated
        lengths.npy  int32   (N,)              vectors per item (1 for kind=vector)
        y.npy        int8    (N,)              step label (1 = incorrect)
        meta.jsonl   N lines                   per-step meta (uid, trace id, step_idx, ...)
        spec.json                              RepSpec (see below)

Item k occupies rows [offset[k] : offset[k] + lengths[k]) of h, where
offset = cumsum([0] + lengths). A fixed-vector representation is the special case
lengths == 1 everywhere, so the existing dense {stem}_h.npy caches migrate in
with a lengths-of-ones array and a spec.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VECTOR = "vector"
TOKEN_SEQ = "token_seq"
STEP_SEQ = "step_seq"
KINDS = (VECTOR, TOKEN_SEQ, STEP_SEQ)


@dataclass
class RepSpec:
    """Metadata describing a representation (what it is + how it was made)."""

    name: str
    kind: str            # one of KINDS
    dim: int             # feature dimension d of each stored vector
    layer: int           # transformer layer the states were read from (-1 = last)
    backbone: str        # e.g. "Qwen2.5-7B"
    readout: str         # e.g. "last", "delta", "step_tokens", "step_states"
    source_split: str = ""
    # Which context the step was encoded under. "verifier" is the template the
    # whole off-policy grid used; "generation" reconstructs the context an
    # on-policy sampler actually ran under (src/onpolicy/prompts.py). Defaulted
    # so specs written before the on-policy arm read back unchanged.
    prompt_style: str = "verifier"
    reduce_default: str = "mean"   # default seq->vector reduction for vector learners
    git_commit: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "RepSpec":
        return cls(**json.loads(text))


def _offsets(lengths: np.ndarray) -> np.ndarray:
    off = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=off[1:])
    return off


def write_split(
    out_dir: str | Path,
    items: list[np.ndarray],
    labels: np.ndarray | list[int],
    meta: list[dict],
    spec: RepSpec,
) -> None:
    """Write one (representation, split) to the store.

    items: list of (L_k, d) arrays, one per step (L_k >= 1). For a fixed-vector
    representation each item is (1, d).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(items)
    if not (len(labels) == n == len(meta)):
        raise ValueError(f"length mismatch: items={n} labels={len(labels)} meta={len(meta)}")
    lengths = np.array([it.shape[0] for it in items], dtype=np.int32)
    if np.any(lengths < 1):
        raise ValueError("every item must have at least one row")
    dims = {it.shape[1] for it in items}
    if dims != {spec.dim}:
        raise ValueError(f"item dims {dims} do not all equal spec.dim={spec.dim}")
    if spec.kind == VECTOR and np.any(lengths != 1):
        raise ValueError("kind=vector requires every item length == 1")

    h = np.concatenate(items, axis=0).astype(np.float16)
    np.save(out / "h.npy", h)
    np.save(out / "lengths.npy", lengths)
    np.save(out / "y.npy", np.asarray(labels, dtype=np.int8))
    with (out / "meta.jsonl").open("w") as f:
        for row in meta:
            f.write(json.dumps(row) + "\n")
    (out / "spec.json").write_text(spec.to_json())


def write_vector_split(
    out_dir: str | Path,
    h: np.ndarray,
    labels: np.ndarray | list[int],
    meta: list[dict],
    spec: RepSpec,
) -> None:
    """Fast path for a fixed-vector representation: h is (N, d) directly.

    Migrates an existing dense {stem}_h.npy cache into the store without
    re-packing row by row.
    """
    if spec.kind != VECTOR:
        raise ValueError("write_vector_split requires spec.kind == 'vector'")
    if h.ndim != 2 or h.shape[1] != spec.dim:
        raise ValueError(f"h must be (N, {spec.dim}), got {h.shape}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = h.shape[0]
    if not (len(labels) == n == len(meta)):
        raise ValueError(f"length mismatch: h={n} labels={len(labels)} meta={len(meta)}")
    np.save(out / "h.npy", h.astype(np.float16))
    np.save(out / "lengths.npy", np.ones(n, dtype=np.int32))
    np.save(out / "y.npy", np.asarray(labels, dtype=np.int8))
    with (out / "meta.jsonl").open("w") as f:
        for row in meta:
            f.write(json.dumps(row) + "\n")
    (out / "spec.json").write_text(spec.to_json())


class RepSplit:
    """Reader for one (representation, split) in the store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.spec = RepSpec.from_json((self.path / "spec.json").read_text())
        self.h = np.load(self.path / "h.npy", mmap_mode="r")
        self.lengths = np.load(self.path / "lengths.npy")
        self.y = np.load(self.path / "y.npy")
        self.offsets = _offsets(self.lengths)
        self._meta_path = self.path / "meta.jsonl"

    def __len__(self) -> int:
        return len(self.lengths)

    @property
    def is_vector(self) -> bool:
        return bool(np.all(self.lengths == 1))

    def item(self, k: int) -> np.ndarray:
        """Return item k as a (L_k, d) float32 array."""
        a, b = int(self.offsets[k]), int(self.offsets[k + 1])
        return np.asarray(self.h[a:b], dtype=np.float32)

    def vectors(self, reduce: str = "mean") -> np.ndarray:
        """Reduce every item to a single vector -> (N, d) for linear/MLP learners.

        reduce: 'mean' | 'max' | 'last' | 'first'. For a vector representation all
        reductions are identity.
        """
        n, d = len(self), self.spec.dim
        out = np.empty((n, d), dtype=np.float32)
        for k in range(n):
            seg = self.item(k)
            if reduce == "mean":
                out[k] = seg.mean(0)
            elif reduce == "max":
                out[k] = seg.max(0)
            elif reduce == "last":
                out[k] = seg[-1]
            elif reduce == "first":
                out[k] = seg[0]
            else:
                raise ValueError(f"unknown reduce {reduce!r}")
        return out

    def meta(self) -> list[dict]:
        return [json.loads(x) for x in self._meta_path.read_text().splitlines() if x.strip()]


class ShardedRepSplit:
    """Read a (representation, split) written as shard_NN/ subdirs, presenting a
    single view in global_index order without a 1 TB merge-copy.

    Each shard is a standalone RepSplit; every meta row carries ``global_index``
    (deterministic order over the full pre-shard file). Items are indexed across
    shards in that global order, so this is row-for-row identical to a merged
    store while keeping each shard's h.npy memory-mapped in place.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        shard_dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and d.name.startswith("shard_") and (d / "h.npy").exists()
        )
        if not shard_dirs:
            raise FileNotFoundError(f"no shard_NN/ dirs under {self.root}")
        self.shards = [RepSplit(d) for d in shard_dirs]
        self.spec = self.shards[0].spec
        entries: list[tuple[int, int, int]] = []
        for si, sh in enumerate(self.shards):
            for li, m in enumerate(sh.meta()):
                entries.append((int(m["global_index"]), si, li))
        entries.sort()
        self._map = [(si, li) for _, si, li in entries]

    def __len__(self) -> int:
        return len(self._map)

    @property
    def is_vector(self) -> bool:
        return all(sh.is_vector for sh in self.shards)

    def item(self, k: int) -> np.ndarray:
        si, li = self._map[k]
        return self.shards[si].item(li)

    @property
    def y(self) -> np.ndarray:
        out = np.empty(len(self), dtype=np.int8)
        for k, (si, li) in enumerate(self._map):
            out[k] = self.shards[si].y[li]
        return out

    def vectors(self, reduce: str = "mean") -> np.ndarray:
        n, d = len(self), self.spec.dim
        out = np.empty((n, d), dtype=np.float32)
        # reduce per shard then reorder, to reuse RepSplit.vectors' vectorized paths
        per_shard = {si: None for si in range(len(self.shards))}
        for si, sh in enumerate(self.shards):
            per_shard[si] = sh.vectors(reduce)
        for k, (si, li) in enumerate(self._map):
            out[k] = per_shard[si][li]
        return out

    def meta(self) -> list[dict]:
        """Per-item meta in global_index order."""
        cache = {si: sh.meta() for si, sh in enumerate(self.shards)}
        return [cache[si][li] for (si, li) in self._map]

    def item_handle(self, k: int) -> tuple["RepSplit", int]:
        """(shard RepSplit, local index) for item k, for streaming token spans."""
        si, li = self._map[k]
        return self.shards[si], li
