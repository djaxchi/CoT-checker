"""Length-sorted, token-budgeted batching must not change what the store holds.

The encoder writes items in batch order and readers order them by global_index,
so sorting by length is free only if meta, y, lengths and the rows stay aligned
through the reordering. A stub model whose rows depend on token ids alone makes
the sorted and unsorted stores directly comparable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "harness"))

from encode_processbench_token_store import encode_subset, plan_batches  # noqa: E402
from test_span_only_encode import StubModel, StubTokenizer  # noqa: E402

from src.repstore.store import ShardedRepSplit  # noqa: E402


def test_plan_batches_respects_both_caps():
    lens = [10, 10, 10, 50, 50, 200]
    assert [list(r) for r in plan_batches(lens, 8)] == [[0, 1, 2, 3, 4, 5]]
    got = [list(r) for r in plan_batches(lens, 8, max_batch_tokens=120)]
    assert got == [[0, 1, 2], [3, 4], [5]]
    assert sum(len(r) for r in plan_batches(lens, 2, 1000)) == len(lens)


def _write(tmp: Path) -> Path:
    traces = [{"id": f"t{i}", "problem": "p" * (5 + i), "label": -1,
               "steps": ["s" * (3 + 7 * ((i * 5 + k) % 4)) for k in range(1 + i % 4)]}
              for i in range(9)]
    f = tmp / "traces.jsonl"
    f.write_text("\n".join(json.dumps(t) for t in traces))
    return f


def test_sorted_store_matches_unsorted(tmp_path):
    raw = _write(tmp_path)
    for tag, kw in (("plain", {}), ("sorted", {"sort_by_length": True, "max_batch_tokens": 40})):
        encode_subset(raw, "sub", tmp_path / tag, StubTokenizer(), StubModel(), "cpu", 0,
                      4096, 3, 0, 0, 1, "stub", span_only=True, **kw)
    a = ShardedRepSplit(tmp_path / "plain" / "sub")
    b = ShardedRepSplit(tmp_path / "sorted" / "sub")
    assert len(a) == len(b)
    ma, mb = a.meta(), b.meta()
    for k in range(len(a)):
        assert {x: ma[k][x] for x in ("id", "step_idx", "n_tokens")} == \
               {x: mb[k][x] for x in ("id", "step_idx", "n_tokens")}
        assert np.array_equal(a.item(k), b.item(k))
    assert np.array_equal(a.y, b.y)
