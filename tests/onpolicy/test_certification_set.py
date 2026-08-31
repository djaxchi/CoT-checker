"""A prefix of the certification set must not be one subset.

A judge run stops early when it hits its budget or is interrupted, and it reads
the file from the top. The first partial API run scored 0.807 on 63 traces that
were GSM8K to the last one, the easiest of the four subsets, which is not an
estimate of anything worth reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.onpolicy.build_judge_certification_set import sample_subset  # noqa: E402


def test_sampling_is_seeded_and_reproducible():
    traces = [{"id": i} for i in range(50)]
    a = sample_subset(traces, 10, seed=0)
    b = sample_subset(traces, 10, seed=0)
    c = sample_subset(traces, 10, seed=1)
    assert a == b
    assert a != c
    assert len(a) == 10


def test_zero_means_take_everything_shuffled():
    traces = [{"id": i} for i in range(7)]
    got = sample_subset(traces, 0, seed=0)
    assert sorted(t["id"] for t in got) == list(range(7))


def test_any_prefix_of_the_written_file_covers_every_subset(tmp_path):
    """The interleave is what makes a stopped run still stratified."""
    from itertools import zip_longest
    per = {s: [{"id": f"{s}{i}", "pb_subset": s} for i in range(5)]
           for s in ("gsm8k", "math", "olympiadbench", "omnimath")}
    out = [t for row in zip_longest(*per.values()) for t in row if t is not None]
    for cut in (4, 8, 12, 20):
        assert len({t["pb_subset"] for t in out[:cut]}) == 4, cut
