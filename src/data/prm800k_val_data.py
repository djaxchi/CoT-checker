"""Loader for the PRM800K val_1k hidden-state encodings (7B, last layer/last token).

Companion to :mod:`src.data.processbench_probe_data`. Where that module loads the
out-of-distribution ProcessBench eval shards, this one loads the IN-distribution
PRM800K validation split the dense probe's threshold was selected on:

    runs/s1_model_size_dense/qwen2_5_7b/merged/
        val_1k_h.npy      (n, 3584) float16  last-layer/last-token of candidate_step
        val_1k_y.npy      (n,)      int       binary step-correct label (1 = correct)
        val_1k_meta.jsonl            uid, problem_id, solution_id, step_idx,
                                     completion_idx, label, rating, n_tokens, ...

Produced by encode_prm800k_hidden_states.py + merge_prm800k_encoded_shards.py
(stageA/stageB of the s1 model-size sweep), Qwen2.5-7B, hidden_size 3584 — the same
readout and probe space as the ProcessBench work, so the 7B linear_probe.pt scores
these vectors directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MERGED_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b/merged")
DEFAULT_STEM = "val_1k"


@dataclass
class Prm800kStepData:
    """One row per PRM800K validation step, arrays share axis 0 (N)."""

    hidden: np.ndarray          # (N, H) float32
    label: np.ndarray           # (N,) int   1 = correct step, 0 = incorrect
    rating: np.ndarray          # (N,) int   PRM800K human rating (-1 / 0 / +1)
    uid: np.ndarray             # (N,) str
    problem_id: np.ndarray      # (N,) str
    solution_id: np.ndarray     # (N,) str
    step_idx: np.ndarray        # (N,) int   position of the step within its solution
    completion_idx: np.ndarray  # (N,) int
    n_tokens: np.ndarray        # (N,) int   tokenized prompt+step length

    def __len__(self) -> int:
        return self.hidden.shape[0]


def load_prm800k_val(
    merged_dir: str | Path = DEFAULT_MERGED_DIR,
    stem: str = DEFAULT_STEM,
) -> Prm800kStepData:
    """Load the merged PRM800K val_1k hidden states + labels + meta into one table."""
    merged_dir = Path(merged_dir)
    hidden = np.load(merged_dir / f"{stem}_h.npy").astype(np.float32)
    label = np.load(merged_dir / f"{stem}_y.npy").astype(np.int64)

    meta_path = merged_dir / f"{stem}_meta.jsonl"
    meta = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]

    return _assemble(hidden, label, meta, stem)


def _assemble(hidden: np.ndarray, label: np.ndarray, meta: list[dict],
              stem: str) -> Prm800kStepData:
    n = hidden.shape[0]
    if len(meta) != n or label.shape[0] != n:
        raise ValueError(
            f"{stem}: row mismatch hidden={n} label={label.shape[0]} meta={len(meta)}"
        )

    def col(key, cast, default=None):
        return np.array([cast(m.get(key, default)) for m in meta])

    return Prm800kStepData(
        hidden=hidden,
        label=label,
        rating=col("rating", lambda v: int(v) if v is not None else 0, 0).astype(np.int64),
        uid=col("uid", str, ""),
        problem_id=col("problem_id", str, ""),
        solution_id=col("solution_id", str, ""),
        step_idx=col("step_idx", lambda v: int(v) if v is not None else -1, -1).astype(np.int64),
        completion_idx=col("completion_idx", lambda v: int(v) if v is not None else -1, -1).astype(np.int64),
        n_tokens=col("n_tokens", lambda v: int(v) if v is not None else -1, -1).astype(np.int64),
    )


def load_prm800k_multitoken(
    merged_dir: str | Path,
    stem: str,
    layer_idx: int,
    token: str = "last",
) -> Prm800kStepData:
    """Slice one (layer, token) plane out of the 4D multi-token/multi-layer encoding.

    Reads {stem}_h.npy of shape (n, L, T, H) plus {stem}_manifest.json (layer_indices,
    token_order), resolves the requested layer/token to their stored positions, and
    returns a :class:`Prm800kStepData` for that plane so the existing projection /
    separation scripts run unchanged. The big array is mmapped; only the slice is
    materialised.
    """
    merged_dir = Path(merged_dir)
    manifest = json.loads((merged_dir / f"{stem}_manifest.json").read_text())
    layers = list(manifest["layer_indices"])
    tokens = list(manifest["token_order"])
    if layer_idx not in layers:
        raise ValueError(f"layer {layer_idx} not in stored layers {layers}")
    if token not in tokens:
        raise ValueError(f"token {token!r} not in stored tokens {tokens}")
    lpos, tpos = layers.index(layer_idx), tokens.index(token)

    h = np.load(merged_dir / f"{stem}_h.npy", mmap_mode="r")
    if h.ndim != 4:
        raise ValueError(f"{stem}_h.npy is {h.ndim}D, expected 4D (n,L,T,H)")
    hidden = np.asarray(h[:, lpos, tpos, :]).astype(np.float32)
    label = np.load(merged_dir / f"{stem}_y.npy").astype(np.int64)
    meta = [json.loads(l) for l in
            (merged_dir / f"{stem}_meta.jsonl").read_text().splitlines() if l.strip()]
    return _assemble(hidden, label, meta, f"{stem}[L{layer_idx},{token}]")
