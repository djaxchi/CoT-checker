"""A pre-derived vector store must read back as the vectors the cell would derive.

judge_prompt_v1 keeps only derived vectors on shared storage (the span store does
not fit), and the cells read them with --prederived. These tests pin that the
shortcut is exact: same vectors, same labels, same ProcessBench order and meta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_prm800k_token_store import encode_split  # noqa: E402
from encode_processbench_token_store import encode_subset  # noqa: E402
from test_judge_encode import StubModel, StubTokenizer, _pb_file, _rows  # noqa: E402

from scripts.derive_vector_store import derive_shard  # noqa: E402
from scripts.train_rep_learner_cell import load_vectors  # noqa: E402

REPS = ("last_token", "step_mean", "boundary_stats")


@pytest.fixture()
def stores(tmp_path):
    spans, pb = tmp_path / "step_spans", tmp_path / "pb_step_spans"
    rows = _rows(tmp_path, 7)
    for shard in (0, 1):
        encode_split(rows, spans, "test_2k", StubTokenizer(), StubModel(),
                     torch.device("cpu"), -1, 4096, 2, 0, shard, 2, "stub", None, True,
                     "judge", tmp_path / "judge_token")
        encode_subset(_pb_file(tmp_path), "math", pb, StubTokenizer(), StubModel(),
                      torch.device("cpu"), -1, 4096, 2, 0, shard, 2, "stub", True,
                      "judge", True, 0, tmp_path / "pb_judge")
    vec, pbvec = tmp_path / "vec", tmp_path / "pbvec"
    for sd in sorted((spans / "test_2k").glob("shard_*")):
        for rep in REPS:
            derive_shard(sd, vec, rep)
    for sd in sorted((pb / "math").glob("shard_*")):
        for rep in REPS:
            derive_shard(sd, pbvec, rep)
    return spans, pb, vec, pbvec


@pytest.mark.parametrize("rep", REPS)
def test_prederived_equals_derived(stores, rep):
    spans, pb, vec, pbvec = stores
    X0, y0, _ = load_vectors(spans, "test_2k", rep, None, sort=False)
    X1, y1, _ = load_vectors(vec / rep, "test_2k", rep, None, sort=False, prederived=True)
    np.testing.assert_array_equal(np.asarray(X0), np.asarray(X1))
    np.testing.assert_array_equal(y0, y1)
    P0, _, m0 = load_vectors(pb, "math", rep, None, sort=True)
    P1, _, m1 = load_vectors(pbvec / rep, "math", rep, None, sort=True, prederived=True)
    np.testing.assert_array_equal(np.asarray(P0), np.asarray(P1))
    for a, b in zip(m0, m1):
        for k in ("id", "step_idx", "label", "n_steps", "global_index"):
            assert a[k] == b[k]


def test_vector_store_is_one_row_per_item(stores):
    _, _, vec, _ = stores
    from src.repstore.store import RepSplit
    rs = RepSplit(vec / "boundary_stats" / "test_2k" / "shard_00")
    assert rs.spec.kind == "vector" and rs.spec.prompt_style == "judge"
    assert (rs.lengths == 1).all() and rs.h.shape[1] == 6 * 8
