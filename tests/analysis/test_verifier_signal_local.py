"""Reject incompatible local replays before interpreting the diagnostic."""

import json

import pytest

from scripts.analysis.verifier_signal_local import compare_scores, validate_checkpoint
from src.analysis.verifier_signal import sha256
from tests.harness.test_instruct_leaderboard import result


def test_score_agreement_and_failures():
    assert compare_scores({"a": 0.5}, {"a": 0.50001})["max_absolute_error"] < 1e-4
    for actual, reference in [({"a": 0.5}, {"b": 0.5}),
                              ({"a": 0.5}, {"a": 0.6}),
                              ({"a": float("nan")}, {"a": 0.5}),
                              ({"a": 1.1}, {"a": 1.1})]:
        with pytest.raises(ValueError):
            compare_scores(actual, reference)


def test_reference_is_bound_to_checkpoint_and_source_validation(tmp_path):
    data = result()
    data["in_domain"] = {"auroc": 0.9}
    (tmp_path / "results.json").write_text(json.dumps(data))
    (tmp_path / "model.pt").write_bytes(b"stub checkpoint")
    reference = {"rep": "last_token", "learner": "linear", "seed": 42,
                 "model_sha256": sha256(tmp_path / "model.pt"),
                 "results_sha256": sha256(tmp_path / "results.json"),
                 "dataset_sha256": "dataset", "store_fingerprint": "store",
                 "score_orientation": "P(incorrect)", "rescale": "none",
                 "source_test_auroc_recorded": 0.9, "source_test_auroc_reproduced": 0.9001}
    identity = ("last_token", "linear", 42)
    assert validate_checkpoint(tmp_path, reference, identity, "dataset", "store") == data
    for key, value in [("model_sha256", "wrong"), ("store_fingerprint", "wrong"),
                       ("seed", 43), ("source_test_auroc_reproduced", 0.8),
                       ("source_test_auroc_reproduced", float("nan"))]:
        with pytest.raises(ValueError):
            validate_checkpoint(tmp_path, dict(reference, **{key: value}), identity, "dataset", "store")
