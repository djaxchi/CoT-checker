"""Generation-state readouts must be the training readouts, row for row.

A step block here is [boundary; step tokens], the layout of a span-store item.
So on a store built by the real encoder (stub model), applying `readout` to each
item must reproduce `derive_split`, which is what produced the training vectors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "tests" / "harness"):
    sys.path.insert(0, str(p))

from derive_delta_from_token_store import derive_split  # noqa: E402
from encode_processbench_token_store import encode_subset  # noqa: E402
from test_span_only_encode import StubModel, StubTokenizer  # noqa: E402

from scripts.onpolicy.score_gen_states_multi import READOUT, readout, step_blocks  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("s")
    traces = [{"id": f"t{i}", "problem": "q" * (4 + i), "label": -1,
               "steps": ["w" * (4 + 5 * ((i + k) % 3)) for k in range(1 + i % 3)]}
              for i in range(7)]
    (tmp / "t.jsonl").write_text("\n".join(json.dumps(t) for t in traces))
    encode_subset(tmp / "t.jsonl", "sub", tmp / "rep", StubTokenizer(), StubModel(), "cpu",
                  0, 4096, 4, 0, 0, 1, "stub", span_only=True)
    return tmp / "rep" / "sub"


@pytest.mark.parametrize("rep", sorted(READOUT))
def test_readout_matches_training_derivation(store, rep):
    X, _, _ = derive_split(store, READOUT[rep], sort=True)
    view = ShardedRepSplit(store)
    for k in range(len(view)):
        item = np.asarray(view.item(k))
        assert np.allclose(readout(item, rep).astype(np.float32),
                           np.asarray(X[k], dtype=np.float32), atol=1e-3), (rep, k)


def test_step_blocks_start_at_the_boundary_token():
    h = np.arange(10, dtype=np.float16)[:, None]
    blocks = step_blocks(h, n_prompt=3, spans=[(0, 2), (2, 5)])
    assert [b[:, 0].tolist() for b in blocks] == [[2, 3, 4], [4, 5, 6, 7]]
