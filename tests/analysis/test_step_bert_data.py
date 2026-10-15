"""Tests for step_bert_probe_v0 data prep (pooling) and the G0 leakage gate."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "step_bert"))

from g0_leakage import neighbour_pairs, ridge_r2  # noqa: E402
from pool_step_tokens import pool_store, select_traces  # noqa: E402
from src.repstore.store import RepSpec, write_split  # noqa: E402

D = 8


def _write_sharded(root: Path, traces: dict[str, int], n_shards: int = 2, seed: int = 0):
    """Synthetic token_seq store; steps round-robin over shards like the real one."""
    rng = np.random.default_rng(seed)
    per = [([], [], []) for _ in range(n_shards)]
    truth = {}
    g = 0
    for tid, n_steps in traces.items():
        for s in range(n_steps):
            L = int(rng.integers(2, 6))
            item = rng.normal(size=(L, D)).astype(np.float16)
            truth[(tid, s)] = (item[0].astype(np.float32), item[1:].astype(np.float32).mean(0),
                               item[-1].astype(np.float32))
            meta = {"id": tid, "step_idx": s, "label": 1, "n_steps": n_steps,
                    "problem_id": "p_" + tid, "n_tokens": L, "step_start_idx": 1,
                    "pre_step_boundary_idx": 0, "global_index": g}
            sh = per[g % n_shards]
            sh[0].append(item); sh[1].append(int(s == 1)); sh[2].append(meta)
            g += 1
    spec = RepSpec(name="t", kind="token_seq", dim=D, layer=35, backbone="x",
                   readout="step_span_with_boundary")
    for i, (items, ys, metas) in enumerate(per):
        write_split(root / f"shard_{i:02d}", items, ys, metas, spec)
    return truth


def test_pool_matches_boundary_and_span_mean(tmp_path):
    truth = _write_sharded(tmp_path, {"a": 3, "b": 2})
    out = pool_store([tmp_path], None, seed=0)
    assert list(zip(out["trace_id"], out["step_idx"])) == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1)]
    for i, (t, s) in enumerate(zip(out["trace_id"], out["step_idx"])):
        b, m, last = truth[(str(t), int(s))]
        np.testing.assert_allclose(out["b"][i].astype(np.float32), b, atol=1e-3)
        np.testing.assert_allclose(out["m"][i].astype(np.float32), m, atol=2e-3)
        np.testing.assert_allclose(out["l"][i].astype(np.float32), last, atol=1e-3)
    assert out["y"].tolist() == [0, 1, 0, 0, 1]


def test_subset_keeps_whole_traces(tmp_path):
    _write_sharded(tmp_path, {f"t{i}": 4 for i in range(10)}, n_shards=3)
    out = pool_store([tmp_path], 4, seed=7)
    tids, counts = np.unique(out["trace_id"], return_counts=True)
    assert len(tids) == 4 and set(counts.tolist()) == {4}


def test_select_traces_deterministic():
    ids = [f"t{i}" for i in range(50)]
    assert select_traces(ids, 10, 1) == select_traces(list(reversed(ids)), 10, 1)
    assert select_traces(ids, None, 1) == set(ids)


def test_neighbour_pairs_stay_within_trace():
    tid = np.array(["a", "a", "a", "b", "b"])
    step = np.array([0, 1, 2, 0, 1])
    src, dst = neighbour_pairs(tid, step, offset=+1)
    assert list(zip(src.tolist(), dst.tolist())) == [(0, 1), (1, 2), (3, 4)]
    src, dst = neighbour_pairs(tid, step, offset=-1)
    assert list(zip(src.tolist(), dst.tolist())) == [(1, 0), (2, 1), (4, 3)]


def test_ridge_r2_detects_copy_and_noise():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(100), 5)
    x = rng.normal(size=(500, 16))
    y = x[:, :4] @ rng.normal(size=(4, 6))
    assert ridge_r2(x, y, groups) > 0.95
    assert ridge_r2(rng.normal(size=(500, 16)), y, groups) < 0.1


def test_split_by_canonical_text_and_pb_exclusion():
    from make_split import make_split
    tp = {"t1": "Find x.", "t2": "find   X.", "t3": "Other", "t4": "PB problem", "t5": "Third"}
    err = {"t1": True, "t2": False, "t3": True, "t4": True, "t5": False}
    s = make_split(tp, err, {"pb  PROBLEM"}, seed=0, fracs=(0.5, 0.25, 0.25))
    where = {t: k for k in ("train", "val", "test") for t in s[k]}
    assert "t4" not in where and s["audit"]["pb_overlap_traces"] == 1
    assert where["t1"] == where["t2"]                      # same canonical problem, same split
    assert sorted(where) == ["t1", "t2", "t3", "t5"]


def test_trace_runs_contiguous():
    from pretrain import trace_runs
    assert trace_runs(np.array(["a", "a", "b", "c", "c", "c"])) == {"a": (0, 2), "b": (2, 1), "c": (3, 3)}


def test_finetune_crop_windows():
    from finetune import crop
    assert crop(10, 4, "local", 2) == (4, 5)
    assert crop(10, 4, "causal", 2) == (2, 5)
    assert crop(10, 4, "causal", -1) == (0, 5)
    assert crop(10, 4, "bidir", 2) == (2, 7)
    assert crop(10, 9, "bidir", 3) == (6, 10)
    assert crop(10, 4, "bidir", -1) == (0, 10)


def test_f1_pb_exact_first_error():
    from finetune import f1_pb
    tid = np.array(["a", "a", "a", "b", "b"])
    step = np.array([0, 1, 2, 0, 1])
    fe = np.array([1, 1, 1, -1, -1])
    # a flagged first at its true first error (1), b never flagged -> perfect
    assert f1_pb(np.array([0.1, 0.9, 0.9, 0.2, 0.3]), tid, step, fe, 0.5) == 1.0
    # a flagged too early -> erroneous accuracy 0 -> F1 0
    assert f1_pb(np.array([0.9, 0.9, 0.9, 0.2, 0.3]), tid, step, fe, 0.5) == 0.0


def test_f1_pb_vectorized_matches_loop():
    from finetune import f1_pb, f1_pb_oracle
    rng = np.random.default_rng(0)
    tid, step, fe = [], [], []
    for i in range(60):
        n = int(rng.integers(1, 8)); f = int(rng.integers(-1, n))
        tid += [f"t{i:02d}"] * n; step += list(range(n)); fe += [f] * n
    tid, step, fe = np.array(tid), np.array(step), np.array(fe)
    p = rng.random(len(tid))

    def loop(th):
        pred, fes = {}, {}
        for pi, t, s, f in zip(p, tid, step, fe):
            fes[t] = f
            if pi >= th and (t not in pred or s < pred[t]):
                pred[t] = s
        a = np.mean([pred.get(t, -1) == f for t, f in fes.items() if f >= 0])
        b = np.mean([t not in pred for t, f in fes.items() if f < 0])
        return 2 * a * b / (a + b) if a + b else 0.0

    for th in (0.2, 0.5, 0.8):
        assert abs(f1_pb(p, tid, step, fe, th) - loop(th)) < 1e-12
    assert f1_pb_oracle(p, tid, step, fe) >= max(loop(t) for t in (0.2, 0.5, 0.8)) - 1e-9
