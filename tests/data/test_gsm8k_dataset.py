"""Unit tests for GSM8K dataset loading. No files or models required."""

import json
import tempfile
from pathlib import Path

import torch
import pytest

from src.data.gsm8k_dataset import (
    split_answer_into_steps,
    gsm8k_acc_judge,
    GSM8KStepDataset,
    GSM8KCollateFn,
    _load_step_pairs,
)


# ---------------------------------------------------------------------------
# split_answer_into_steps
# ---------------------------------------------------------------------------

def test_split_basic():
    answer = "First step\nSecond step\n#### 42"
    steps = split_answer_into_steps(answer)
    assert len(steps) == 2
    assert "First step" in steps[0]
    assert "Second" in steps[1]


def test_split_deduplicates_blank_lines():
    answer = "Step A\n\nStep B\n#### 5"
    steps = split_answer_into_steps(answer)
    assert all(s.strip() for s in steps)


# ---------------------------------------------------------------------------
# gsm8k_acc_judge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ground,pred,expected", [
    ("The answer is 42", "So the result is 42", 1),
    ("Total = 3.14", "About 3.14 dollars", 1),
    ("Result is 10", "Result is 11", 0),
    ("No numbers here", "Also none", 0),
    ("Answer: 0", "The sum is 0.0", 1),
])
def test_gsm8k_acc_judge(ground, pred, expected):
    assert gsm8k_acc_judge(ground, pred) == expected


# ---------------------------------------------------------------------------
# Dataset loading from a temp JSONL file
# ---------------------------------------------------------------------------

SAMPLE_JSONL = [
    {"question": "A store sells 7 apples for $21. How much per apple?",
     "answer": "Divide 21 by 7.\n21 / 7 = 3.\n#### 3"},
    {"question": "Tom has 5 cats and 3 dogs. How many pets?",
     "answer": "Add cats and dogs.\n5 + 3 = 8.\n#### 8"},
]


@pytest.fixture
def sample_jsonl(tmp_path):
    p = tmp_path / "sample.json"
    with open(p, "w") as f:
        for item in SAMPLE_JSONL:
            f.write(json.dumps(item) + "\n")
    return p


def test_load_step_pairs_count(sample_jsonl):
    pairs = _load_step_pairs(sample_jsonl)
    # Each problem has 2 non-empty steps → 4 total
    assert len(pairs) == 4


def test_load_step_pairs_context_accumulates(sample_jsonl):
    pairs = _load_step_pairs(sample_jsonl)
    # Second step of first problem should have first step in context
    second_step = pairs[1]
    assert "Divide" in second_step["context"]  # first step is in context


def test_dataset_len(sample_jsonl):
    from unittest.mock import MagicMock
    tok = MagicMock()
    tok.encode = lambda text, **kw: [1, 2, 3]
    tok.eos_token_id = 0
    ds = GSM8KStepDataset(sample_jsonl, tok)
    assert len(ds) == 4


def test_dataset_item_has_correct_keys(sample_jsonl):
    from unittest.mock import MagicMock
    tok = MagicMock()
    tok.encode = lambda text, **kw: [1, 2, 3]
    tok.eos_token_id = 0
    ds = GSM8KStepDataset(sample_jsonl, tok)
    item = ds[0]
    assert "context" in item
    assert "step" in item
    assert isinstance(item["context_ids"], torch.Tensor)
    assert isinstance(item["step_ids"], torch.Tensor)


# ---------------------------------------------------------------------------
# CollateFn
# ---------------------------------------------------------------------------

def test_collate_output_shapes(sample_jsonl):
    from unittest.mock import MagicMock
    tok = MagicMock()
    tok.encode = lambda text, **kw: [1, 2, 3, 4]
    tok.eos_token_id = 1
    ds = GSM8KStepDataset(sample_jsonl, tok)

    collate = GSM8KCollateFn(eos_token_id=1, pad_token_id=0, sep_token_id=99)
    batch = collate([ds[0], ds[1]])

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["hints_sep_ids"].shape[0] == 2


def test_collate_sep_token_present(sample_jsonl):
    from unittest.mock import MagicMock
    tok = MagicMock()
    tok.encode = lambda text, **kw: [10, 20]
    tok.eos_token_id = 1
    ds = GSM8KStepDataset(sample_jsonl, tok)
    collate = GSM8KCollateFn(eos_token_id=1, pad_token_id=0, sep_token_id=99)
    batch = collate([ds[0]])
    # sep token (99) should appear in the input
    assert (batch["input_ids"][0] == 99).any()
