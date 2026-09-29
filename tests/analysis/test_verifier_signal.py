"""Mathematical semantics, paired estimands, and frozen-input integration."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.verifier_signal import (
    build_dataset,
    family_metrics,
    summarize,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def config():
    return json.loads((ROOT / "experiments/verifier_signal_v1/config.json").read_text())


@pytest.fixture
def rows(config):
    return build_dataset(config)


def test_frozen_counts_partitions_and_determinism(rows, config):
    assert rows == build_dataset(config)
    assert len(rows) == 912
    assert len({r["family_id"] for r in rows}) == 72
    dev = {r["family_id"] for r in rows if r["partition"] == "dev"}
    test = {r["family_id"] for r in rows if r["partition"] == "test"}
    assert len(dev) == 18 and len(test) == 54 and not dev & test
    assert sum(r["role"] == "before" for r in rows) == 48


def test_inequality_semantics_by_boundary_points(rows):
    for r in rows:
        if r["domain"] != "inequality":
            continue
        a, b = r["witness"]["a"], r["witness"]["b"]
        sign = 1 if r["prefix_variant"] == 0 else -1
        # Independent numerical truth sets on either side and at the boundary.
        xs = [b - 0.5, b, b + 0.5]
        prefix_truth = [sign*a*x < sign*a*b for x in xs]
        claim_truth = [x < b if r["candidate_variant"] == 0 else x > b for x in xs]
        assert r["local_invalid"] == int(prefix_truth != claim_truth)


def test_repeated_candidate_and_opposite_validity(rows):
    for r in rows:
        if r["arm"] != "reversal" or r["prefix_variant"] != 0:
            continue
        opposite = next(q for q in rows if q["family_id"] == r["family_id"]
                        and q["prefix_variant"] == 1 and q["style"] == r["style"]
                        and q["candidate_variant"] == r["candidate_variant"])
        assert r["candidate_step"] == opposite["candidate_step"]
        assert r["local_invalid"] != opposite["local_invalid"]


def test_inheritance_distinguishes_local_trace_and_conclusion(rows):
    r = next(r for r in rows if r["arm"] == "inheritance" and r["role"] == "target"
             and r["prefix_variant"] == r["candidate_variant"] == 1)
    assert (r["local_invalid"], r["prefix_invalid"], r["conclusion_invalid"], r["trace_invalid"]) == (0, 1, 1, 1)
    r = next(r for r in rows if r["arm"] == "inheritance" and r["role"] == "target"
             and r["prefix_variant"] == 1 and r["candidate_variant"] == 0)
    assert (r["local_invalid"], r["prefix_invalid"], r["conclusion_invalid"], r["trace_invalid"]) == (1, 1, 0, 1)


@pytest.mark.parametrize("kind", ["label", "text", "missing", "duplicate", "partition"])
def test_corrupted_data_rejected(rows, config, kind):
    bad = copy.deepcopy(rows)
    if kind == "label":
        bad[0]["local_invalid"] ^= 1
    elif kind == "text":
        bad[0]["candidate_step"] += " extra"
    elif kind == "missing":
        bad.pop()
    elif kind == "duplicate":
        bad.append(bad[0])
    else:
        bad[0]["partition"] = "test"
    with pytest.raises(ValueError):
        validate_dataset(bad, config)


def test_oracle_local_detector_and_style_invariance(rows):
    scores = {r["uid"]: 0.1 + 0.8*r["local_invalid"] for r in rows}
    metrics = family_metrics(rows, scores)
    for r in metrics:
        assert r["metrics"]["local_contrast_mean"] == pytest.approx(0.8)
        assert r["metrics"]["both_preferences_correct"] == 1
        if r["arm"] == "inheritance":
            assert r["metrics"]["prefix_error_contrast"] == pytest.approx(0)
        if r["style"] != "plain":
            assert r["metrics"]["style_mean_absolute_shift"] == 0


def test_prefix_only_detector_is_not_misread_as_local_detector(rows):
    scores = {r["uid"]: 0.1 + 0.8*r["prefix_invalid"] for r in rows}
    for r in family_metrics(rows, scores):
        assert r["metrics"]["local_contrast_mean"] == 0
        assert r["metrics"]["both_preferences_correct"] == 0
        if r["arm"] == "inheritance":
            assert r["metrics"]["prefix_error_contrast"] == pytest.approx(0.8)


def test_answer_only_detector_fails_preference_reversal(rows):
    scores = {r["uid"]: 0.8 if r["candidate_variant"] == 1 else 0.2 for r in rows}
    assert all(m["metrics"]["local_contrast_mean"] == 0 for m in family_metrics(rows, scores))
    assert all(m["metrics"]["both_preferences_correct"] == 0 for m in family_metrics(rows, scores))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_bad_scores_rejected(rows, bad):
    scores = {r["uid"]: 0.5 for r in rows}
    scores[rows[0]["uid"]] = bad
    with pytest.raises(ValueError):
        family_metrics(rows, scores)


def test_missing_scores_rejected(rows):
    with pytest.raises(ValueError):
        family_metrics(rows, {r["uid"]: 0.5 for r in rows[:-1]})


def test_bootstrap_groups_whole_families_and_is_reproducible(rows):
    metrics = family_metrics(rows, {r["uid"]: 0.1+0.8*r["local_invalid"] for r in rows})
    first = summarize(metrics, 100, 42)
    assert first == summarize(metrics, 100, 42)
    for r in first:
        assert r["n_families"] in (3, 6, 9, 18)
        if r["metric"] == "local_contrast_mean":
            assert r["mean"] == pytest.approx(0.8)
            assert r["ci95"] == pytest.approx([0.8, 0.8])


def test_real_encoder_and_scoring_adapter_with_stubs(tmp_path, rows):
    from types import SimpleNamespace

    import torch

    from scripts.encode_prm800k_token_store import encode_split
    from scripts.onpolicy.score_cells_on_split import score_cell
    from src.harness.learners import build_learner

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) % 31 + 1 for c in text]}

    class Model:
        config = SimpleNamespace(hidden_size=4)

        def __call__(self, inputs, **kwargs):
            hidden = inputs.float().unsqueeze(-1).repeat(1, 1, 4) / 32
            return SimpleNamespace(hidden_states=(hidden, hidden))

    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows[:4]))
    for shard in (0, 1):
        encode_split(path, tmp_path / "store", "diagnostic", Tokenizer(), Model(),
                     torch.device("cpu"), 1, 2048, 2, 0, shard, 2, "stub", None, True)
    for rep, learner in [("last_token", "linear"), ("step_mean", "linear"),
                         ("step_tokens", "attn_query"),
                         ("step_tokens", "transformer:d8,l1,f16,h2")]:
        cell = tmp_path / f"cell_{rep}_{learner}"
        cell.mkdir()
        model = build_learner(learner, 4, t_max=512, dropout=0.1)
        torch.save(model.state_dict(), cell / "model.pt")
        result = {"rep": rep, "learner": learner, "dim": 4, "protocol": {"dropout": 0.1}}
        scores, labels, meta = score_cell(cell, result, tmp_path / "store/diagnostic",
                                          None, torch.device("cpu"), 2, 512)
        assert [m["uid"] for m in meta] == [r["uid"] for r in rows[:4]]
        assert labels.tolist() == [r["label"] for r in rows[:4]]
        assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()


@pytest.mark.parametrize("mixed_seeds", [False, True])
def test_bundle_and_complete_analysis_handoff(tmp_path, config, mixed_seeds):
    from types import SimpleNamespace

    from scripts.analysis.verifier_signal_experiment import analyze, build, load_bundle, seed_roster
    from scripts.validate_instruct_leaderboard import cell_tag
    from src.analysis.verifier_signal import sha256

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    if mixed_seeds:
        config["seed_overrides"] = [
            {"rep": "step_tokens", "learner": learner, "seeds": [42, 43]}
            for learner in ("attn_query", "transformer:d128,l1,f512,h4")]
    (bundle / "config.json").write_text(json.dumps(config))
    (bundle / "probes.cells").write_text((ROOT / "experiments/verifier_signal_v1/probes.cells").read_text())
    build(bundle)
    frozen, examples = load_bundle(bundle)
    with pytest.raises(ValueError, match="refusing"):
        build(bundle)
    out = tmp_path / "run"
    (out / "scores").mkdir(parents=True)
    for (rep, learner), seeds in seed_roster(bundle, config).items():
        for seed in seeds:
            cell = cell_tag(rep, learner, seed)
            value = {"rep": rep, "learner": learner, "seed": seed, "cell": cell,
                     "dataset_sha256": sha256(bundle / "examples.jsonl"),
                     "scores": {r["uid"]: 0.1+0.8*r["local_invalid"] for r in examples}}
            (out / "scores" / f"{cell}.json").write_text(json.dumps(value))
    frozen["bootstrap_replicates"] = 20
    args = SimpleNamespace(bundle=bundle, out=out)
    analyze(args, frozen, examples)
    result = json.loads((out / "analysis.json").read_text())
    assert len(result["per_cell"]) == (16 if mixed_seeds else 18)
    assert len(result["seed_averaged"]) == 6
    for cell in result["seed_averaged"]:
        assert tuple(cell["seeds"]) == seed_roster(bundle, config)[cell["rep"], cell["learner"]]
    assert (out / "summary.md").exists()
    next((out / "scores").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="roster incomplete"):
        analyze(args, frozen, examples)
    with (bundle / "examples.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(bundle)
