"""Encoding directly in span mode must equal encoding in full then compacting.

The previous backbone's store was built full (984 GB) and then compacted; that
compaction was verified byte-identical over all 513,810 items before the master
was deleted. Encoding the new backbone skips the full store entirely, because
1.1 TB does not fit. That is only safe if the shortcut lands on exactly the same
bytes and the same offsets, which is what these tests pin, using a stub model so
no weights are needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_step_span_store import compact_shard  # noqa: E402
from encode_prm800k_token_store import encode_split  # noqa: E402

from src.repstore.store import RepSplit  # noqa: E402

D = 8


class StubTokenizer:
    """Deterministic character-length tokenizer; ids encode (text, position)."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False):
        base = 2 if add_special_tokens else 0
        n = base + max(1, len(text) // 3)
        return {"input_ids": [(abs(hash(text)) % 900) + i + 1 for i in range(n)]}


class StubModel:
    """Hidden state of a token is a fixed function of its id, so the same token
    always produces the same row and full-vs-span writes are directly comparable."""

    class _C:
        hidden_size = D

    config = _C()

    def __call__(self, inp, attention_mask=None, output_hidden_states=True, use_cache=False):
        b, t = inp.shape
        ids = inp.to(torch.float32).unsqueeze(-1)
        feat = torch.arange(D, dtype=torch.float32).view(1, 1, D)
        hs = torch.sin(ids * 0.001 + feat)
        return type("O", (), {"hidden_states": [hs]})()


def _rows(tmp_path: Path, n: int) -> Path:
    """A jsonl split whose prefixes are long relative to the candidate step."""
    path = tmp_path / "prm800k_test.jsonl"
    with path.open("w") as f:
        for k in range(n):
            f.write(json.dumps({
                "uid": f"u{k}", "problem_id": f"p{k}", "solution_id": f"s{k}",
                "step_idx": 1, "label": k % 2, "rating": 1,
                "problem": "a long problem statement " * (3 + k),
                "prefix": "an earlier step " * (2 + k),
                "candidate_step": "the candidate step here",
            }) + "\n")
    return path


def _encode(tmp_path, name, span_only):
    root = tmp_path / name
    encode_split(
        _rows(tmp_path, 6), root, "test_2k", StubTokenizer(), StubModel(),
        torch.device("cpu"), -1, 4096, 2, 0, 0, 1, "stub", None, span_only,
    )
    return root


def test_span_encode_matches_full_encode_then_compact(tmp_path):
    full = _encode(tmp_path, "full", span_only=False)
    span = _encode(tmp_path, "span", span_only=True)
    compact = tmp_path / "compacted"
    compact_shard(full / "test_2k" / "shard_00", compact / "shard_00", "step_spans")

    direct = RepSplit(span / "test_2k" / "shard_00")
    via_compaction = RepSplit(compact / "shard_00")

    assert len(direct) == len(via_compaction)
    np.testing.assert_array_equal(direct.lengths, via_compaction.lengths)
    np.testing.assert_array_equal(direct.y, via_compaction.y)
    np.testing.assert_array_equal(np.asarray(direct.h), np.asarray(via_compaction.h))


def test_span_encode_rewrites_offsets_the_same_way(tmp_path):
    span = _encode(tmp_path, "span2", span_only=True)
    full = _encode(tmp_path, "full2", span_only=False)
    compact = tmp_path / "compacted2"
    compact_shard(full / "test_2k" / "shard_00", compact / "shard_00", "step_spans")

    for a, b in zip(RepSplit(span / "test_2k" / "shard_00").meta(),
                    RepSplit(compact / "shard_00").meta()):
        for key in ("uid", "n_tokens", "step_start_idx", "pre_step_boundary_idx",
                    "orig_n_tokens", "orig_step_start_idx", "global_index", "label"):
            assert a[key] == b[key], key


def test_span_mode_is_much_smaller(tmp_path):
    full = RepSplit(_encode(tmp_path, "f3", span_only=False) / "test_2k" / "shard_00")
    span = RepSplit(_encode(tmp_path, "s3", span_only=True) / "test_2k" / "shard_00")
    assert span.h.shape[0] < full.h.shape[0] / 2
    assert span.spec.readout == "step_span_with_boundary"
    assert full.spec.readout == "token_all_last_layer"


def test_span_mode_refuses_an_empty_prefix(tmp_path):
    class NoPrefixTokenizer(StubTokenizer):
        def __call__(self, text, add_special_tokens=True, truncation=False):
            if add_special_tokens:
                return {"input_ids": []}
            return super().__call__(text, add_special_tokens=False)

    with pytest.raises(ValueError, match="step_start_idx"):
        encode_split(
            _rows(tmp_path, 2), tmp_path / "bad", "test_2k", NoPrefixTokenizer(),
            StubModel(), torch.device("cpu"), -1, 4096, 2, 0, 0, 1, "stub", None, True,
        )
