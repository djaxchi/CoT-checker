"""End-to-end smoke of one grid cell, plus the guards that keep the grid honest.

These run the real entry point on a tiny synthetic store, so they catch the
plumbing faults that only appear when a representation, a learner, and the
ProcessBench first-error scan are wired together.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.repstore import TOKEN_SEQ, RepSpec, write_split  # noqa: E402

SCRIPT = ROOT / "scripts" / "train_rep_learner_cell.py"
D = 4


def _write_prm(root: Path, stem: str, n: int, seed: int) -> None:
    """A split whose label is linearly readable from the step tokens."""
    rng = np.random.default_rng(seed)
    items, metas, labels = [], [], []
    for k in range(n):
        label = int(k % 2)
        n_step = int(rng.integers(2, 5))
        prefix = rng.normal(size=(2, D))
        span = rng.normal(size=(n_step, D)) + (2.0 if label else -2.0)
        items.append(np.concatenate([prefix, span]).astype(np.float32))
        metas.append({
            "uid": f"{stem}_{k}", "n_tokens": 2 + n_step, "step_start_idx": 2,
            "pre_step_boundary_idx": 1, "global_index": k, "label": label,
            "step_idx": 1,
        })
        labels.append(label)
    write_split(root / stem / "shard_00", items, labels, metas,
                RepSpec(name="step_spans", kind=TOKEN_SEQ, dim=D, layer=-1,
                        backbone="test", readout="step_span_with_boundary"))


def _write_pb(root: Path, subset: str, n_traces: int, seed: int) -> None:
    """Traces of 3 steps; half are fully correct (label -1), half fail at step 1."""
    rng = np.random.default_rng(seed)
    items, metas, labels = [], [], []
    gi = 0
    for t in range(n_traces):
        trace_label = -1 if t % 2 == 0 else 1
        for step in range(3):
            bad = trace_label != -1 and step >= trace_label
            n_step = int(rng.integers(2, 5))
            prefix = rng.normal(size=(2, D))
            span = rng.normal(size=(n_step, D)) + (2.0 if bad else -2.0)
            items.append(np.concatenate([prefix, span]).astype(np.float32))
            metas.append({
                "uid": f"{subset}_{t}_{step}", "n_tokens": 2 + n_step,
                "step_start_idx": 2, "pre_step_boundary_idx": 1, "global_index": gi,
                "id": f"{subset}::{t}", "step_idx": step, "label": trace_label,
                "n_steps": 3,
            })
            labels.append(int(bad))
            gi += 1
    write_split(root / subset / "shard_00", items, labels, metas,
                RepSpec(name="step_spans", kind=TOKEN_SEQ, dim=D, layer=-1,
                        backbone="test", readout="step_span_with_boundary"))


@pytest.fixture
def store(tmp_path):
    prm, pb = tmp_path / "prm", tmp_path / "pb"
    _write_prm(prm, "probe_train_full", 64, 0)
    _write_prm(prm, "val_5k", 32, 1)
    _write_prm(prm, "test_2k", 32, 2)
    _write_pb(pb, "gsm8k", 8, 3)
    return prm, pb


def _run(prm, pb, out, rep, learner, extra=()):
    cmd = [sys.executable, str(SCRIPT), "--rep", rep, "--learner", learner,
           "--prm_store", str(prm), "--pb_store", str(pb), "--out_dir", str(out),
           "--pb_subsets", "gsm8k", "--epochs", "2", "--patience", "1",
           "--batch_size", "16", "--lr_grid", "1e-2", "--wd_grid", "0.0",
           "--threshold_grid", "0.1", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


@pytest.mark.parametrize("rep,learner", [
    ("last_token", "linear"),
    ("step_stats", "mlp:h8"),
    ("step_tokens", "attn_query"),
])
def test_cell_runs_and_reports_both_domains(store, tmp_path, rep, learner):
    prm, pb = store
    out = tmp_path / f"out_{rep}_{learner.replace(':', '_')}"
    r = _run(prm, pb, out, rep, learner)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / "results.json").read_text())
    assert res["rep"] == rep and res["learner"] == learner
    assert res["n_params"] > 0
    assert 0.0 <= res["in_domain"]["auroc"] <= 1.0
    assert "gsm8k" in res["processbench"]
    assert (out / "pb_step_scores_gsm8k.jsonl").exists()


def test_full_train_is_the_default_and_is_recorded(store, tmp_path):
    """No cap unless one is asked for: the v1 leaderboard's sequence rows silently
    trained on 150k of 513,810 rows, which is exactly what this flag makes visible."""
    prm, pb = store
    out = tmp_path / "out_full"
    assert _run(prm, pb, out, "last_token", "linear").returncode == 0
    res = json.loads((out / "results.json").read_text())
    assert res["full_train"] is True
    assert res["n_train"] == res["n_train_available"] == 64

    out2 = tmp_path / "out_capped"
    assert _run(prm, pb, out2, "last_token", "linear",
                extra=("--train_cap", "16")).returncode == 0
    res2 = json.loads((out2 / "results.json").read_text())
    assert res2["full_train"] is False and res2["n_train"] == 16


def test_hyperparameter_grid_is_searched_and_the_winner_recorded(store, tmp_path):
    prm, pb = store
    out = tmp_path / "out_hp"
    cmd_extra = ("--lr_grid", "1e-2", "1e-4", "--wd_grid", "0.0", "0.01")
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--rep", "last_token", "--learner", "linear",
         "--prm_store", str(prm), "--pb_store", str(pb), "--out_dir", str(out),
         "--pb_subsets", "gsm8k", "--epochs", "2", "--patience", "1",
         "--batch_size", "16", "--threshold_grid", "0.1", *cmd_extra],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    hp = json.loads((out / "results.json").read_text())["hp"]
    assert len(hp["trials"]) == 4
    assert hp["selected"]["lr"] in (1e-2, 1e-4)


def test_incompatible_rep_learner_pairs_are_refused(store, tmp_path):
    prm, pb = store
    seq_on_vector = _run(prm, pb, tmp_path / "a", "last_token", "attn_query")
    assert seq_on_vector.returncode != 0
    assert "step_tokens" in seq_on_vector.stderr

    vector_on_seq = _run(prm, pb, tmp_path / "b", "step_tokens", "linear")
    assert vector_on_seq.returncode != 0
    assert "step_mean" in vector_on_seq.stderr


def test_vector_cache_is_reused_across_learners(store, tmp_path):
    """The derived vectors are shared by every learner on that representation,
    which is what keeps a 15-cell grid affordable."""
    prm, pb = store
    cache = tmp_path / "veccache"
    assert _run(prm, pb, tmp_path / "c1", "step_mean", "linear",
                extra=("--vec_cache_dir", str(cache))).returncode == 0
    written = sorted(p.name for p in cache.glob("step_mean__probe_train_full__*_h.npy"))
    assert len(written) == 1, written
    cached = cache / written[0]
    stamp = cached.stat().st_mtime_ns
    assert _run(prm, pb, tmp_path / "c2", "step_mean", "mlp:h8",
                extra=("--vec_cache_dir", str(cache))).returncode == 0
    assert cached.stat().st_mtime_ns == stamp


def test_hp_is_selected_once_per_cell_and_reused_across_seeds(store, tmp_path):
    """Re-searching per seed lets each seed pick the config suiting its own
    initialisation, which inflates the cell and shrinks the seed spread the three
    seeds exist to measure. So the search runs once and siblings reuse it."""
    prm, pb = store
    first = tmp_path / "seed42"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--rep", "last_token", "--learner", "linear",
         "--prm_store", str(prm), "--pb_store", str(pb), "--out_dir", str(first),
         "--pb_subsets", "gsm8k", "--epochs", "2", "--patience", "1",
         "--batch_size", "16", "--threshold_grid", "0.1", "--seed", "42",
         "--lr_grid", "1e-2", "1e-4", "--wd_grid", "0.0", "0.01"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    chosen = json.loads((first / "results.json").read_text())["hp"]["selected"]

    second = tmp_path / "seed43"
    r2 = _run(prm, pb, second, "last_token", "linear",
              extra=("--seed", "43", "--hp_from", str(first / "results.json")))
    assert r2.returncode == 0, r2.stderr[-3000:]
    hp = json.loads((second / "results.json").read_text())["hp"]
    assert hp["selected"] == chosen
    assert hp["reused_from"].endswith("results.json")
    assert "reused from" in r2.stdout


def test_hp_from_a_different_cell_is_refused(store, tmp_path):
    """Borrowing another cell's tuning would silently break the protocol."""
    prm, pb = store
    donor = tmp_path / "donor"
    assert _run(prm, pb, donor, "step_mean", "linear").returncode == 0
    r = _run(prm, pb, tmp_path / "recipient", "last_token", "linear",
             extra=("--hp_from", str(donor / "results.json")))
    assert r.returncode != 0
    assert "step_mean" in r.stderr and "one-config-per-cell" in r.stderr


def test_the_vector_cache_is_keyed_by_the_store_fingerprint(store, tmp_path):
    """A vector cell trains on the derived cache, not on the store. If the store
    changes, the cache must miss rather than be reused under a fingerprint the
    cell never actually read."""
    prm, pb = store
    cache = tmp_path / "fpcache"
    assert _run(prm, pb, tmp_path / "d1", "last_token", "linear",
                extra=("--vec_cache_dir", str(cache))).returncode == 0
    before = sorted(p.name for p in cache.glob("last_token__probe_train_full__*_h.npy"))
    assert len(before) == 1

    # Flip one training label: same shapes, different store, different fingerprint.
    y_path = prm / "probe_train_full" / "shard_00" / "y.npy"
    y = np.load(y_path)
    y[0] = 1 - y[0]
    np.save(y_path, y)

    assert _run(prm, pb, tmp_path / "d2", "last_token", "linear",
                extra=("--vec_cache_dir", str(cache))).returncode == 0
    after = sorted(p.name for p in cache.glob("last_token__probe_train_full__*_h.npy"))
    assert len(after) == 2, after      # the old cache was not silently reused

    fps = {json.loads((tmp_path / d / "results.json").read_text())
           ["inputs"]["prm/probe_train_full"] for d in ("d1", "d2")}
    assert len(fps) == 2
    for fp in fps:
        assert any(fp in name for name in after)
