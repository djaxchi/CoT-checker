"""JSONL dataset for SSAE phase-1 training and latent extraction.

Text format (per spec section 6):

    Problem:
    {problem}

    Previous reasoning:
    {prefix}

    <|step_sep|>
    {candidate_step}

Tokenization:
    encoder_input = ctx_ids + [sep_id] + step_ids + [eos_id]
    sep_pos = len(ctx_ids) + 1   (index of the first step token)
    val_len = total length before padding
    loss_mask[i] = 1 iff sep_pos <= i < val_len   (CE only on step + eos)

Strict no-truncation: if any tokenized example exceeds max_seq_len we raise a
DatasetLengthError listing offending uids. A length-audit entry point lives in
the length_audit() helper for pre-flight inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


STEP_SEP_TOKEN = "<|step_sep|>"


def add_step_sep_token(tokenizer) -> int:
    """Ensure <|step_sep|> exists as a special token. Returns its id."""
    if STEP_SEP_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [STEP_SEP_TOKEN]})
    return tokenizer.convert_tokens_to_ids(STEP_SEP_TOKEN)


class DatasetLengthError(RuntimeError):
    """Raised when at least one example exceeds max_seq_len."""


@dataclass
class SSAEExample:
    uid: str
    input_ids: torch.Tensor          # (L,)
    attention_mask: torch.Tensor     # (L,)
    loss_mask: torch.Tensor          # (L,)
    sep_pos: int
    val_len: int
    label: int                       # 0 viable / 1 non-viable; -1 if unknown
    meta: dict[str, Any]


def _format_context(problem: str, prefix: str) -> str:
    return (
        "Problem:\n"
        f"{problem}\n\n"
        "Previous reasoning:\n"
        f"{prefix}\n\n"
    )


def _format_step(step: str) -> str:
    return f"\n{step}"


def tokenize_row(row: dict, tokenizer, sep_token_id: int,
                 eos_token_id: int, max_seq_len: int) -> SSAEExample:
    """Tokenize one JSONL row into an SSAEExample. Raises on overflow."""
    problem = row.get("problem") or ""
    prefix = row.get("prefix") or ""
    step = row.get("candidate_step")
    if step is None:
        # Fall back to input_text where the row already contains the joined string.
        # The benchmark JSONL always carries candidate_step; this is a hard fail.
        raise KeyError(f"Row missing 'candidate_step' (uid={row.get('uid')})")

    ctx_text = _format_context(problem, prefix)
    step_text = _format_step(step)

    # No truncation anywhere.
    ctx_ids = tokenizer.encode(ctx_text, add_special_tokens=False)
    step_ids = tokenizer.encode(step_text, add_special_tokens=False)

    input_list = list(ctx_ids) + [sep_token_id] + list(step_ids) + [eos_token_id]
    val_len = len(input_list)
    if val_len > max_seq_len:
        uid = row.get("uid") or row.get("id") or "<unknown>"
        raise DatasetLengthError(
            f"Example uid={uid} has tokenized length {val_len} > max_seq_len={max_seq_len}. "
            "Truncation is disabled per spec. Re-curate the JSONL or raise --max_seq_len."
        )

    sep_pos = len(ctx_ids) + 1  # first step token

    input_ids = torch.tensor(input_list, dtype=torch.long)
    attention_mask = torch.ones(val_len, dtype=torch.long)
    loss_mask = torch.zeros(val_len, dtype=torch.long)
    loss_mask[sep_pos:val_len] = 1

    label_field = row.get("label")
    label = int(label_field) if label_field is not None else -1

    meta = {
        "uid": row.get("uid"),
        "id": row.get("id"),
        "problem_id": row.get("problem_id"),
        "solution_id": row.get("solution_id"),
        "step_idx": row.get("step_idx"),
        "completion_idx": row.get("completion_idx"),
        "n_steps": row.get("n_steps"),
        "label": label,
    }

    return SSAEExample(
        uid=str(row.get("uid") or row.get("id") or ""),
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        sep_pos=sep_pos,
        val_len=val_len,
        label=label,
        meta=meta,
    )


class SSAEJsonlDataset(Dataset):
    """Lazily reads a JSONL file and tokenizes on __getitem__.

    All length checks happen at __getitem__ time; call length_audit() before
    training to surface any overlong examples up-front.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        max_seq_len: int = 2048,
        limit: int | None = None,
    ) -> None:
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.sep_token_id = tokenizer.convert_tokens_to_ids(STEP_SEP_TOKEN)
        if self.sep_token_id is None or self.sep_token_id < 0:
            raise RuntimeError(
                f"Tokenizer does not contain {STEP_SEP_TOKEN}. "
                "Call add_step_sep_token(tokenizer) before constructing the dataset."
            )
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no eos_token_id.")
        self.eos_token_id = int(tokenizer.eos_token_id)

        self.rows: list[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
                if limit is not None and len(self.rows) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> SSAEExample:
        return tokenize_row(
            self.rows[idx],
            self.tokenizer,
            self.sep_token_id,
            self.eos_token_id,
            self.max_seq_len,
        )

    def length_audit(self, raise_on_violation: bool = True) -> dict:
        """Tokenize every row to verify no example exceeds max_seq_len.

        Returns a dict with counts and the first 20 offenders. If
        raise_on_violation=True and any violation is found, raises
        DatasetLengthError immediately.
        """
        violations: list[tuple[str, int]] = []
        lengths: list[int] = []
        for row in self.rows:
            try:
                ex = tokenize_row(
                    row, self.tokenizer, self.sep_token_id,
                    self.eos_token_id, max_seq_len=10**9,  # never raise here
                )
                lengths.append(ex.val_len)
                if ex.val_len > self.max_seq_len:
                    uid = str(row.get("uid") or row.get("id") or "<unknown>")
                    violations.append((uid, ex.val_len))
            except KeyError as e:
                violations.append((f"<schema:{e}>", -1))
        report = {
            "path": str(self.path),
            "n_rows": len(self.rows),
            "max_seq_len": self.max_seq_len,
            "n_violations": len(violations),
            "first_violations": violations[:20],
            "length_max": max(lengths) if lengths else 0,
            "length_p99": (
                sorted(lengths)[int(0.99 * (len(lengths) - 1))] if lengths else 0
            ),
        }
        if violations and raise_on_violation:
            raise DatasetLengthError(
                f"{len(violations)} example(s) exceed max_seq_len={self.max_seq_len} "
                f"in {self.path}. First offenders: {violations[:20]}"
            )
        return report


class SSAECollator:
    """Right-pads a batch of SSAEExample into model-ready tensors."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[SSAEExample]) -> dict[str, Any]:
        max_len = max(ex.val_len for ex in batch)
        b = len(batch)

        input_ids = torch.full((b, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        loss_mask = torch.zeros((b, max_len), dtype=torch.long)
        labels = torch.full((b,), -1, dtype=torch.long)
        sep_pos = torch.zeros(b, dtype=torch.long)
        val_len = torch.zeros(b, dtype=torch.long)
        metas: list[dict] = []

        for i, ex in enumerate(batch):
            L = ex.val_len
            input_ids[i, :L] = ex.input_ids
            attention_mask[i, :L] = ex.attention_mask
            loss_mask[i, :L] = ex.loss_mask
            labels[i] = ex.label
            sep_pos[i] = ex.sep_pos
            val_len[i] = ex.val_len
            metas.append(ex.meta)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "labels": labels,
            "sep_pos": sep_pos,
            "val_len": val_len,
            "meta": metas,
        }
