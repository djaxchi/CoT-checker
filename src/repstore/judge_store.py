"""Second store written alongside a judge-style span store (judge_prompt_v1).

Under the judge prompt each item is problem + previous steps + current step +
a verdict question. The span store keeps the step's rows exactly as before; this
store keeps, per item, two rows:

    row 0  the pre-step boundary state (same row the span store starts with)
    row 1  the last token of the verdict question, where the model is about to
           answer Yes or No

with the span offsets (pre_step_boundary_idx 0, step_start_idx 1, n_tokens 2).
So every existing reader works on it unchanged: the `last_token` readout is the
verdict-token state, `step_delta` is verdict minus boundary. The zero-shot
answer logits ride along in the meta.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.repstore.store import TOKEN_SEQ, RepSpec


def answer_ids(tokenizer, answers: tuple[str, str]) -> tuple[int, int]:
    """Token ids of (Yes, No). Each must be one token or the logit is not a verdict."""
    ids = []
    for w in answers:
        t = tokenizer(w, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            raise ValueError(f"answer {w!r} tokenizes to {t}; need a single token")
        ids.append(t[0])
    return ids[0], ids[1]


class JudgeWriter:
    """Streams (boundary, verdict) row pairs into <root>/<stem>/shard_XX."""

    def __init__(self, root: Path, stem: str, shard_idx: int, n_items: int, d: int):
        self.dir = root / stem / f"shard_{shard_idx:02d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.h = np.lib.format.open_memmap(self.dir / "h.npy", mode="w+",
                                           dtype=np.float16, shape=(2 * n_items, d))
        self.y = np.zeros(n_items, dtype=np.int8)
        self.meta: list[dict] = []
        self.n = n_items

    def add(self, k: int, boundary: np.ndarray, verdict: np.ndarray, label: int,
            meta: dict, logit_yes: float, logit_no: float) -> None:
        self.h[2 * k] = boundary
        self.h[2 * k + 1] = verdict
        self.y[k] = label
        row = dict(meta)
        row.update({"n_tokens": 2, "step_start_idx": 1, "pre_step_boundary_idx": 0,
                    "judge_logit_yes": logit_yes, "judge_logit_no": logit_no})
        self.meta.append(row)

    def close(self, spec: RepSpec) -> None:
        if len(self.meta) != self.n:
            raise ValueError(f"{self.dir}: wrote {len(self.meta)} of {self.n} items")
        self.h.flush()
        np.save(self.dir / "lengths.npy", np.full(self.n, 2, dtype=np.int32))
        np.save(self.dir / "y.npy", self.y)
        with (self.dir / "meta.jsonl").open("w") as f:
            for m in self.meta:
                f.write(json.dumps(m) + "\n")
        (self.dir / "spec.json").write_text(spec.to_json())


def judge_spec(name: str, d: int, layer: int, backbone: str, stem: str) -> RepSpec:
    return RepSpec(name=name, kind=TOKEN_SEQ, dim=d, layer=layer, backbone=backbone,
                   readout="boundary_and_verdict_token", source_split=stem,
                   prompt_style="judge")
