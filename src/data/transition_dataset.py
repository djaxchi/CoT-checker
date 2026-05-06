"""TransitionDataset: serves (h_k, delta_h) pairs from a cached .npz file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["TransitionDataset"]


class TransitionDataset(Dataset):
    """Loads pre-cached transition pairs for PTB training.

    Expected .npz fields:
        h_k        float16  (N, d)
        delta_h    float16  (N, d)
        problem_id int32    (N,)
        step_idx   int8     (N,)

    Item dict:
        h_k        float32  (d,)
        delta_h    float32  (d,)
        step_idx   int
        problem_id int
    """

    def __init__(
        self,
        npz_path: str | Path,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        d = np.load(npz_path)
        self.h_k       = torch.from_numpy(d["h_k"].astype(np.float32))
        self.delta_h   = torch.from_numpy(d["delta_h"].astype(np.float32))
        self.step_idx  = d["step_idx"].astype(np.int64)
        self.problem_id = d["problem_id"].astype(np.int64)
        if dtype != torch.float32:
            self.h_k     = self.h_k.to(dtype)
            self.delta_h = self.delta_h.to(dtype)

    def __len__(self) -> int:
        return len(self.h_k)

    def __getitem__(self, idx: int) -> dict:
        return {
            "h_k":        self.h_k[idx],
            "delta_h":    self.delta_h[idx],
            "step_idx":   int(self.step_idx[idx]),
            "problem_id": int(self.problem_id[idx]),
        }
