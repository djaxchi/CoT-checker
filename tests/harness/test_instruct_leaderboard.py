"""Prevent capped runs, protocol drift and incomplete grids from becoming ranks."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.validate_instruct_leaderboard import PROTOCOL, roster, validate_result


def result():
    return {"rep": "last_token", "learner": "linear", "seed": 42,
            "n_train": 513810, "full_train": True, "protocol": deepcopy(PROTOCOL),
            "hp": {"search_rows": 100000, "trials": [
                {"lr": lr, "weight_decay": wd}
                for lr in (1e-3, 3e-4, 1e-4) for wd in (0., .01)]},
            "processbench": {s: {} for s in ("gsm8k", "math", "olympiadbench", "omnimath")}}


def test_frozen_roster_contains_full_core_and_explicit_extension():
    pairs = roster(Path("experiments/instruct_leaderboard_v1/all.cells"))
    assert len(pairs) == 22
    assert sum(rep == "lengthfree_geom" for rep, _ in pairs) == 3
    assert ("step_tokens", "transformer:d512,l2,f2048,h8") in pairs


def test_reference_protocol_passes():
    validate_result(result(), "last_token", "linear", 42)


@pytest.mark.parametrize("field,value", [("n_train", 100000), ("full_train", False)])
def test_capped_fit_rejected(field, value):
    data = result()
    data[field] = value
    with pytest.raises(ValueError):
        validate_result(data, "last_token", "linear", 42)


def test_changed_scaling_rejected():
    data = result()
    data["protocol"]["rescale"] = "zscore"
    with pytest.raises(ValueError):
        validate_result(data, "last_token", "linear", 42)


def test_researching_later_seed_rejected():
    data = result()
    data["seed"] = 43
    with pytest.raises(ValueError):
        validate_result(data, "last_token", "linear", 43)


def test_partial_search_rejected():
    data = result()
    data["hp"]["trials"].pop()
    with pytest.raises(ValueError):
        validate_result(data, "last_token", "linear", 42)
