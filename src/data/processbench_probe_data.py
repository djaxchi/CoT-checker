"""Load the S1 7B DenseLinear ProcessBench eval run into one per-step table.

This is the substrate for the S3 failure-mode analysis: for every ProcessBench
step it joins, row-for-row,

  - the hidden state           ``pb_step_h.npy``        (the geometry)
  - the gold label + keys      ``pb_step_meta.jsonl``   (id, step_idx, label, ...)
  - the probe score            recomputed ``sigmoid(W.h + b)`` from ``linear_probe.pt``
  - the step / problem text    ``Qwen/ProcessBench`` (HF), joined on (id, step_idx)

so we can cluster on probe direction, probe separation, and embedding geometry,
and split where the probe detects the first error vs where it struggles.

ProcessBench label convention (verified): ``label`` is the 0-based index of the
first erroneous step, or ``-1`` if the whole trace is correct. ``step_idx`` indexes
the ``steps`` list directly, so ``is_first_error = (label != -1 and step_idx == label)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "SUBSETS",
    "DEFAULT_RUN_DIR",
    "ProbeStepData",
    "load_probe",
    "compute_scores",
    "load_subset",
    "load_all",
]

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
DEFAULT_RUN_DIR = Path("runs/s1_model_size_dense/qwen2_5_7b")


@dataclass
class ProbeStepData:
    """One row per ProcessBench step, all sources aligned.

    Arrays share axis 0 (N steps). Text lists are aligned too (empty string when
    not loaded or when a step was skipped at encode time).
    """

    subset: np.ndarray          # (N,) str   ProcessBench subset name
    trace_id: np.ndarray        # (N,) str   e.g. "gsm8k-5"
    step_idx: np.ndarray        # (N,) int   index into the trace's step list
    n_steps: np.ndarray         # (N,) int   number of steps in the trace
    gold_first_error: np.ndarray  # (N,) int  trace first-error idx (-1 if correct)
    is_first_error: np.ndarray  # (N,) bool  step_idx == gold_first_error
    skipped: np.ndarray         # (N,) bool  step skipped at encode (h is zero)
    hidden: np.ndarray          # (N, d) float32  last-token hidden state
    score: np.ndarray           # (N,) float32    probe P(step is first-error)
    pred_first_error: np.ndarray  # (N,) int   trace-level predicted first-error idx
    step_text: list[str]        # (N,) raw text of this step
    problem: list[str]          # (N,) problem statement for this trace

    def __len__(self) -> int:
        return self.hidden.shape[0]

    @property
    def dim(self) -> int:
        return self.hidden.shape[1]


def load_probe(run_dir: str | Path = DEFAULT_RUN_DIR) -> tuple[np.ndarray, float]:
    """Return the probe direction ``w`` (d,) and scalar bias ``b``."""
    sd = torch.load(Path(run_dir) / "linear_probe.pt", map_location="cpu")
    w = sd["fc.weight"].detach().cpu().numpy().astype(np.float32).reshape(-1)
    b = float(sd["fc.bias"].detach().cpu().numpy().reshape(-1)[0])
    return w, b


def compute_scores(hidden: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """``sigmoid(h . w + b)`` in float32, matching the eval scoring convention."""
    logits = hidden.astype(np.float32) @ w.astype(np.float32) + b
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_pb_text(subset: str) -> dict[str, dict]:
    """Map trace id -> {"problem": str, "steps": list[str]} from cached HF data."""
    from datasets import load_dataset

    ds = load_dataset("Qwen/ProcessBench", split=subset)
    return {r["id"]: {"problem": r["problem"], "steps": r["steps"]} for r in ds}


def load_subset(
    run_dir: str | Path = DEFAULT_RUN_DIR,
    subset: str = "gsm8k",
    *,
    with_text: bool = True,
    probe: tuple[np.ndarray, float] | None = None,
) -> ProbeStepData:
    """Load one subset's eval shard into a :class:`ProbeStepData`."""
    run_dir = Path(run_dir)
    shard = run_dir / "processbench_eval_shards" / subset
    hidden = np.load(shard / "pb_step_h.npy").astype(np.float32)
    meta = _read_jsonl(shard / "pb_step_meta.jsonl")
    if len(meta) != hidden.shape[0]:
        raise ValueError(
            f"{subset}: meta rows ({len(meta)}) != hidden rows ({hidden.shape[0]})"
        )

    w, b = probe if probe is not None else load_probe(run_dir)
    if w.shape[0] != hidden.shape[1]:
        raise ValueError(
            f"{subset}: probe dim ({w.shape[0]}) != hidden dim ({hidden.shape[1]})"
        )
    score = compute_scores(hidden, w, b)

    # Trace-level predicted first-error index (first step over threshold, else -1).
    pred_by_id = {
        r["id"]: int(r["prediction"])
        for r in _read_jsonl(shard / "predictions.jsonl")
    }

    n = len(meta)
    trace_id = np.array([m["id"] for m in meta])
    step_idx = np.array([int(m["step_idx"]) for m in meta], dtype=np.int64)
    n_steps = np.array([int(m["n_steps"]) for m in meta], dtype=np.int64)
    gold = np.array([int(m["label"]) for m in meta], dtype=np.int64)
    skipped = np.array([bool(m.get("skipped", False)) for m in meta])
    is_first_error = (gold != -1) & (step_idx == gold)
    pred_first_error = np.array(
        [pred_by_id.get(m["id"], -1) for m in meta], dtype=np.int64
    )

    step_text = [""] * n
    problem = [""] * n
    if with_text:
        text = _load_pb_text(subset)
        for i, m in enumerate(meta):
            row = text.get(m["id"])
            if row is None:
                continue
            problem[i] = row["problem"]
            k = int(m["step_idx"])
            if 0 <= k < len(row["steps"]):
                step_text[i] = row["steps"][k]

    return ProbeStepData(
        subset=np.array([subset] * n),
        trace_id=trace_id,
        step_idx=step_idx,
        n_steps=n_steps,
        gold_first_error=gold,
        is_first_error=is_first_error,
        skipped=skipped,
        hidden=hidden,
        score=score,
        pred_first_error=pred_first_error,
        step_text=step_text,
        problem=problem,
    )


def load_all(
    run_dir: str | Path = DEFAULT_RUN_DIR,
    subsets: tuple[str, ...] = SUBSETS,
    *,
    with_text: bool = True,
) -> ProbeStepData:
    """Concatenate every subset into one :class:`ProbeStepData`."""
    probe = load_probe(run_dir)
    parts = [
        load_subset(run_dir, s, with_text=with_text, probe=probe) for s in subsets
    ]
    return ProbeStepData(
        subset=np.concatenate([p.subset for p in parts]),
        trace_id=np.concatenate([p.trace_id for p in parts]),
        step_idx=np.concatenate([p.step_idx for p in parts]),
        n_steps=np.concatenate([p.n_steps for p in parts]),
        gold_first_error=np.concatenate([p.gold_first_error for p in parts]),
        is_first_error=np.concatenate([p.is_first_error for p in parts]),
        skipped=np.concatenate([p.skipped for p in parts]),
        hidden=np.concatenate([p.hidden for p in parts], axis=0),
        score=np.concatenate([p.score for p in parts]),
        pred_first_error=np.concatenate([p.pred_first_error for p in parts]),
        step_text=[t for p in parts for t in p.step_text],
        problem=[t for p in parts for t in p.problem],
    )
