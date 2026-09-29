"""Protect contrast direction and fail loudly on conflicting shard summaries."""
import json

import pytest

from scripts.analysis.plot_instruct_downstream import contrast, load_checked


def test_contrast_reverses_interval_endpoints_and_converts_to_points():
    gain, ci = contrast({"delta": -.02, "ci95": [-.03, -.01]})
    assert gain == pytest.approx(2)
    assert ci == pytest.approx([1, 3])


def test_load_checks_repeated_baselines_instead_of_overwriting(tmp_path):
    (tmp_path / "parts").mkdir()
    row = {"dataset": "math500", "n": 2, "rule": "oracle", "delta": -.02,
           "ci95": [-.03, -.01], "baseline_acc": .8, "rule_acc": .82, "n_problems": 500}
    summary = {"lift": {"math500|2|oracle": .02}, "ci95": {"math500|2|oracle": [.01, .03]},
               "accuracy": {"math500|2|oracle": .82}, "majority": {"math500|2": .8}}
    (tmp_path / "summary_t10.json").write_text(json.dumps(summary))
    (tmp_path / "parts/contrast_t10_0.json").write_text(json.dumps([row]))
    assert load_checked(tmp_path, "10") == summary
    row["n_problems"] = 499
    (tmp_path / "parts/contrast_t10_1.json").write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="Conflicting repeated contrast"):
        load_checked(tmp_path, "10")


def test_load_rejects_summary_with_wrong_gain_sign(tmp_path):
    (tmp_path / "parts").mkdir()
    row = {"dataset": "math500", "n": 2, "rule": "oracle", "delta": -.02,
           "ci95": [-.03, -.01], "baseline_acc": .8, "rule_acc": .82, "n_problems": 500}
    summary = {"lift": {"math500|2|oracle": -.02}, "ci95": {"math500|2|oracle": [.01, .03]},
               "accuracy": {"math500|2|oracle": .82}, "majority": {"math500|2": .8}}
    (tmp_path / "summary_t10.json").write_text(json.dumps(summary))
    (tmp_path / "parts/contrast_t10_0.json").write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="Summary differs"):
        load_checked(tmp_path, "10")
