"""Unit tests for the ProcessBench probe-data loader (synthetic, no HF/model)."""

from __future__ import annotations

import json

import numpy as np
import torch

from src.data.processbench_probe_data import (
    compute_scores,
    load_probe,
    load_subset,
)


def _write_fake_run(tmp_path, d=4):
    run = tmp_path / "run"
    shard = run / "processbench_eval_shards" / "gsm8k"
    shard.mkdir(parents=True)

    # Two traces: t0 has first error at step 1; t1 is fully correct (label -1).
    meta = [
        {"id": "gsm8k-0", "step_idx": 0, "label": 1, "n_steps": 2, "skipped": False},
        {"id": "gsm8k-0", "step_idx": 1, "label": 1, "n_steps": 2, "skipped": False},
        {"id": "gsm8k-1", "step_idx": 0, "label": -1, "n_steps": 1, "skipped": False},
    ]
    (shard / "pb_step_meta.jsonl").write_text(
        "\n".join(json.dumps(m) for m in meta) + "\n"
    )

    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((len(meta), d)).astype(np.float16)
    np.save(shard / "pb_step_h.npy", hidden)

    (shard / "predictions.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"id": "gsm8k-0", "label": 1, "prediction": 1, "threshold": 0.5},
                {"id": "gsm8k-1", "label": -1, "prediction": -1, "threshold": 0.5},
            ]
        )
        + "\n"
    )

    w = torch.randn(1, d)
    b = torch.randn(1)
    torch.save({"fc.weight": w, "fc.bias": b}, run / "linear_probe.pt")
    return run, hidden.astype(np.float32), w.numpy().reshape(-1), float(b)


def test_alignment_labels_and_scores(tmp_path):
    run, hidden, w, b = _write_fake_run(tmp_path)
    data = load_subset(run, "gsm8k", with_text=False)

    assert len(data) == 3
    assert data.dim == 4
    assert list(data.trace_id) == ["gsm8k-0", "gsm8k-0", "gsm8k-1"]
    assert list(data.step_idx) == [0, 1, 0]

    # is_first_error only at (t0, step 1); correct trace has no first error.
    assert list(data.is_first_error) == [False, True, False]
    assert list(data.gold_first_error) == [1, 1, -1]
    assert list(data.pred_first_error) == [1, 1, -1]

    # Scores equal the explicit sigmoid(W.h + b).
    expected = compute_scores(hidden, w, b)
    np.testing.assert_allclose(data.score, expected, rtol=1e-5, atol=1e-6)
    assert np.all((data.score >= 0) & (data.score <= 1))


def test_load_probe_roundtrip(tmp_path):
    run, _, w, b = _write_fake_run(tmp_path)
    w2, b2 = load_probe(run)
    np.testing.assert_allclose(w2, w, rtol=1e-6)
    assert abs(b2 - b) < 1e-6


def test_dim_mismatch_raises(tmp_path):
    run, _, _, _ = _write_fake_run(tmp_path)
    bad = {"fc.weight": torch.randn(1, 99), "fc.bias": torch.randn(1)}
    torch.save(bad, run / "linear_probe.pt")
    try:
        load_subset(run, "gsm8k", with_text=False)
    except ValueError as e:
        assert "probe dim" in str(e)
    else:
        raise AssertionError("expected ValueError on probe/hidden dim mismatch")
