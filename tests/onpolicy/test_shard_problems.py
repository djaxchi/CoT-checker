"""Shards must partition the problem list.

Generation is single-device, so a node runs four of these at once. If the slices
overlapped, the same trajectory id would be written twice and the merge would
either refuse or double-count; if they left a gap, problems would silently vanish
from the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from scripts.generate_onpolicy_steps import shard_problems  # noqa: E402


def problems(n):
    return [{"fork_id": str(i)} for i in range(n)]


@pytest.mark.parametrize("n,k", [(100, 4), (7, 4), (3, 4), (1, 1), (13, 3)])
def test_shards_partition_exactly(n, k):
    got = [p for s in range(k) for p in shard_problems(problems(n), s, k)]
    ids = [p["fork_id"] for p in got]
    assert sorted(ids, key=int) == [str(i) for i in range(n)]
    assert len(ids) == len(set(ids))


def test_single_shard_is_the_whole_list():
    ps = problems(9)
    assert shard_problems(ps, 0, 1) == ps


def test_shards_are_balanced_within_one():
    sizes = [len(shard_problems(problems(101), s, 4)) for s in range(4)]
    assert max(sizes) - min(sizes) <= 1


def test_out_of_range_shard_is_refused():
    with pytest.raises(ValueError):
        shard_problems(problems(10), 4, 4)
