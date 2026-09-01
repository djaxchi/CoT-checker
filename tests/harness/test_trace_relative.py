"""Trace-relative representations and between-step dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from trace_relative_reps import (  # noqa: E402
    build_trace, collect, dynamics, position, trace_key,
)


def _states(n=5, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32), rng.normal(size=(n, d)).astype(np.float32)


def test_leave_one_out_mean_matches_the_slow_definition():
    """Computed in closed form for speed; it has to equal deleting the row."""
    s, b = _states(7)
    got = build_trace(s, b, np.ones(7), np.arange(7))["trace_centered_all"]
    for i in range(7):
        want = s[i] - np.delete(s, i, 0).mean(0)
        np.testing.assert_allclose(got[i], want, rtol=1e-4, atol=1e-4)


def test_causal_centering_never_reads_a_later_step():
    """Change only the LAST step. Every earlier row must be untouched, or the
    causal variant is quietly using the future and its deployment claim is void."""
    s, b = _states(6)
    a = build_trace(s, b, np.ones(6), np.arange(6))["trace_centered_causal"]
    s2 = s.copy()
    s2[-1] += 100.0
    c = build_trace(s2, b, np.ones(6), np.arange(6))["trace_centered_causal"]
    np.testing.assert_allclose(a[:-1], c[:-1], rtol=1e-5, atol=1e-5)


def test_the_first_step_is_centered_on_the_prefix_not_on_itself():
    """Step 0 has no predecessor inside the trace. Centering it on itself would
    emit a zero vector and throw the step away."""
    s, b = _states(4)
    got = build_trace(s, b, np.ones(4), np.arange(4))["trace_centered_causal"]
    np.testing.assert_allclose(got[0], s[0] - b[0], rtol=1e-5)
    assert np.linalg.norm(got[0]) > 1e-3


def test_dynamics_are_scale_free():
    """Every entry is an angle or a ratio against the trace's own median, so
    scaling the whole trace must not move any of them. A raw norm leaking in
    would reintroduce exactly the kind of magnitude cue we removed."""
    s, b = _states(6)
    a = dynamics(s, b[0])
    c = dynamics((7.0 * s).astype(np.float32), (7.0 * b[0]).astype(np.float32))
    np.testing.assert_allclose(a, c, atol=1e-4)


def test_dynamics_carry_no_token_count():
    """Step length scores 0.7039 here on its own. dynamics() is not given lengths
    at all, so no entry can encode one."""
    s, b = _states(6)
    assert dynamics(s, b[0]).shape == (6, 10)


def test_a_turning_trajectory_scores_lower_persistence_than_a_straight_one():
    d = 8
    straight = np.cumsum(np.tile(np.ones(d, np.float32), (6, 1)), 0).astype(np.float32)
    zig = straight.copy()
    zig[3] -= 6.0                                    # one sharp turn at step 3
    b = np.zeros(d, np.float32)
    assert dynamics(zig, b)[3, 2] < dynamics(straight, b)[3, 2]


def test_position_control_is_only_position():
    """Kept out of dyn and emitted separately: first errors sit later on average,
    so position is a shortcut of the same kind step length turned out to be."""
    got = position(np.arange(5, dtype=np.float32), 5)
    assert got.shape == (5, 2)
    assert got[0, 0] == 0.0 and got[-1, 0] == 1.0
    assert len(set(got[:, 1].tolist())) == 1          # trace length is constant


def test_trace_key_groups_prm800k_by_solution_and_processbench_by_id():
    assert trace_key({"solution_id": "s22990", "problem_id": "p1"}) == "s22990"
    assert trace_key({"id": "gsm8k-0"}) == "gsm8k-0"


def _store(path, traces, d=8, seed=0, stride_shards=4):
    """A store that reproduces the real one's awkward layout: consecutive steps
    of one trace land in DIFFERENT shards, which is why grouping cannot be done
    per shard."""
    from src.repstore import STEP_SEQ, RepSpec, write_split
    rng = np.random.default_rng(seed)
    rows = []
    for tid, n in traces:
        for si in range(n):
            rows.append((tid, si, n))
    shards = [[] for _ in range(stride_shards)]
    for gi, (tid, si, n) in enumerate(rows):
        shards[gi % stride_shards].append((gi, tid, si, n))
    for sh, items in enumerate(shards):
        if not items:
            continue
        arrs = [rng.normal(size=(3 + (si % 3), d)).astype(np.float32)
                for _, _, si, _ in items]
        meta = [{"solution_id": tid, "step_idx": si, "label": si % 2, "n_steps": n,
                 "global_index": gi, "pre_step_boundary_idx": 0, "step_start_idx": 1}
                for gi, tid, si, n in items]
        sd = path / f"shard_{sh:02d}"
        sd.mkdir(parents=True)
        write_split(sd, arrs, [m["label"] for m in meta], meta,
                    RepSpec(name="s", kind=STEP_SEQ, dim=d, layer=-1,
                            backbone="t", readout="r"))
    return path


def test_collect_reassembles_traces_that_are_split_across_shards(tmp_path):
    """ProcessBench global_index strides by four across shards, so one trace's
    steps live in four different files. Per-shard sampling would hand back
    fragments with their siblings missing."""
    p = _store(tmp_path / "s", [("t0", 5), ("t1", 4), ("t2", 6)])
    got, y, lens = collect(p, None, 0, pb=False)
    assert len(y) == 15
    assert got["trace_centered_causal"].shape == (15, 8)
    assert got["dyn"].shape == (15, 10)
    # every trace contributed a full run of consecutive positions
    assert sorted(got["pos"][:, 0].round(3).tolist()).count(0.0) == 3


def test_sampling_takes_whole_traces_not_loose_steps(tmp_path):
    p = _store(tmp_path / "s", [(f"t{i}", 4) for i in range(10)])
    _, y, _ = collect(p, 3, 0, pb=False)
    assert len(y) == 12, "a sampled trace lost steps"
