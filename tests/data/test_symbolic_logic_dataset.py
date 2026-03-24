"""Unit tests for symbolic logic dataset. No files or models required."""

import json
import random
import tempfile
from pathlib import Path

import pytest

from src.data.symbolic_logic_dataset import (
    PropLogicSolver,
    corrupt_step_logic,
    label_chain,
    _load_step_records,
)


# ---------------------------------------------------------------------------
# PropLogicSolver.from_question — rule / fact parsing
# ---------------------------------------------------------------------------


def test_parses_positive_rule():
    solver = PropLogicSolver.from_question("Every wumpus is a rompus.")
    assert "rompus" in solver.rules.get("wumpus", set())


def test_parses_property_rule():
    solver = PropLogicSolver.from_question("Every rompus is luminous.")
    assert "luminous" in solver.rules.get("rompus", set())


def test_parses_seed_fact():
    solver = PropLogicSolver.from_question("Rex is a wumpus.")
    assert "wumpus" in solver.known.get("Rex", set())


def test_parses_negation_rule():
    solver = PropLogicSolver.from_question("Every wumpus is not large.")
    assert "large" in solver.neg_rules.get("wumpus", set())


def test_parses_mixed_question():
    q = "Every wumpus is a rompus. Every rompus is luminous. Rex is a wumpus."
    solver = PropLogicSolver.from_question(q)
    assert "rompus" in solver.rules["wumpus"]
    assert "luminous" in solver.rules["rompus"]
    assert "wumpus" in solver.known["Rex"]


# ---------------------------------------------------------------------------
# PropLogicSolver.is_valid_step — step verification
# ---------------------------------------------------------------------------


def test_valid_step_direct_fact():
    solver = PropLogicSolver.from_question("Rex is a wumpus.")
    assert solver.is_valid_step("Rex is a wumpus.")


def test_valid_step_one_hop():
    solver = PropLogicSolver.from_question("Every wumpus is red. Rex is a wumpus.")
    assert solver.is_valid_step("Rex is red.")


def test_invalid_step_no_supporting_rule():
    solver = PropLogicSolver.from_question("Every wumpus is red. Rex is a fumpus.")
    # Rex is a fumpus, not a wumpus — can't derive "red"
    assert not solver.is_valid_step("Rex is red.")


def test_valid_step_requires_prior_step():
    # 2-hop: wumpus -> rompus -> luminous
    # Without applying the intermediate step, "Rex is luminous" should NOT be valid
    solver = PropLogicSolver.from_question(
        "Every wumpus is a rompus. Every rompus is luminous. Rex is a wumpus."
    )
    # Direct check: Rex is luminous should fail (no single-step rule wumpus->luminous)
    assert not solver.is_valid_step("Rex is luminous.")
    # After applying intermediate step:
    solver.apply_step("Rex is a rompus.")
    assert solver.is_valid_step("Rex is luminous.")


def test_valid_negation_step():
    solver = PropLogicSolver.from_question("Every wumpus is not large. Rex is a wumpus.")
    assert solver.is_valid_step("Rex is not large.")


def test_invalid_negation_step_wrong_entity():
    solver = PropLogicSolver.from_question("Every wumpus is not large. Rex is a fumpus.")
    assert not solver.is_valid_step("Rex is not large.")


def test_unparseable_step_is_valid():
    solver = PropLogicSolver.from_question("Rex is a wumpus.")
    # Non-standard step — no match, assumed valid
    assert solver.is_valid_step("We conclude the answer is True.")


# ---------------------------------------------------------------------------
# PropLogicSolver.all_reachable
# ---------------------------------------------------------------------------


def test_all_reachable_one_hop():
    solver = PropLogicSolver.from_question("Every wumpus is red. Rex is a wumpus.")
    reachable = solver.all_reachable("Rex")
    assert "wumpus" in reachable
    assert "red" in reachable


def test_all_reachable_three_hop():
    q = "Every wumpus is a rompus. Every rompus is a yumpus. Every yumpus is red. Rex is a wumpus."
    solver = PropLogicSolver.from_question(q)
    reachable = solver.all_reachable("Rex")
    assert "rompus" in reachable
    assert "yumpus" in reachable
    assert "red" in reachable


def test_all_reachable_irrelevant_entity():
    q = "Every wumpus is red. Rex is a wumpus. Polly is a fumpus."
    solver = PropLogicSolver.from_question(q)
    assert "red" not in solver.all_reachable("Polly")


# ---------------------------------------------------------------------------
# label_chain
# ---------------------------------------------------------------------------


def test_label_chain_all_correct():
    q = "Every wumpus is a rompus. Every rompus is luminous. Rex is a wumpus."
    cot = ["Rex is a wumpus.", "Rex is a rompus.", "Rex is luminous."]
    labels = label_chain(q, cot)
    assert labels == [1, 1, 1]


def test_label_chain_first_step_corrupt():
    q = "Every wumpus is a rompus. Every rompus is luminous. Rex is a wumpus."
    # Step 1 is wrong: Rex is a yumpus (not supported)
    cot = ["Rex is a yumpus.", "Rex is a rompus.", "Rex is luminous."]
    labels = label_chain(q, cot)
    assert labels[0] == 0


def test_label_chain_middle_step_corrupt():
    q = "Every wumpus is a rompus. Every rompus is luminous. Rex is a wumpus."
    # Step 2 is wrong: Rex is luminous without having derived Rex is a rompus
    cot = ["Rex is a wumpus.", "Rex is luminous.", "Rex is a rompus."]
    labels = label_chain(q, cot)
    # Step 1 valid, step 2 invalid (no rule wumpus->luminous directly)
    assert labels[0] == 1
    assert labels[1] == 0


def test_label_chain_three_hop():
    q = "Every wumpus is a rompus. Every rompus is a yumpus. Every yumpus is red. Rex is a wumpus."
    cot = ["Rex is a wumpus.", "Rex is a rompus.", "Rex is a yumpus.", "Rex is red."]
    labels = label_chain(q, cot)
    assert labels == [1, 1, 1, 1]


# ---------------------------------------------------------------------------
# corrupt_step_logic
# ---------------------------------------------------------------------------


def test_corrupt_changes_step():
    q = "Every wumpus is a rompus. Every rompus is luminous. Rex is a wumpus."
    solver = PropLogicSolver.from_question(q)
    solver.apply_step("Rex is a wumpus.")
    corrupted, was_corrupted = corrupt_step_logic("Rex is a rompus.", solver, rng=random.Random(42))
    if was_corrupted:
        assert corrupted != "Rex is a rompus."


def test_corrupt_produces_invalid_step():
    q = "Every wumpus is a rompus. Every rompus is luminous. Rex is a wumpus."
    solver = PropLogicSolver.from_question(q)
    solver.apply_step("Rex is a wumpus.")
    corrupted, was_corrupted = corrupt_step_logic("Rex is a rompus.", solver, rng=random.Random(0))
    if was_corrupted:
        # The corrupted step should NOT be valid under the solver
        assert not solver.is_valid_step(corrupted)


def test_corrupt_returns_unchanged_if_no_wrong_prop():
    # Only one rule conclusion — nothing else to substitute
    q = "Every wumpus is red. Rex is a wumpus."
    solver = PropLogicSolver.from_question(q)
    solver.apply_step("Rex is a wumpus.")
    corrupted, was_corrupted = corrupt_step_logic("Rex is red.", solver, rng=random.Random(0))
    # All conclusions are reachable, so nothing to corrupt
    assert was_corrupted is False
    assert corrupted == "Rex is red."


# ---------------------------------------------------------------------------
# _load_step_records — integration with sample file
# ---------------------------------------------------------------------------


SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "prontoqa_sample.jsonl"


@pytest.mark.skipif(not SAMPLE_PATH.exists(), reason="sample file not present")
def test_load_sample_file_produces_records():
    records = _load_step_records(SAMPLE_PATH)
    assert len(records) > 0
    for rec in records:
        assert "context" in rec
        assert "step" in rec
        assert rec["label"] in (0, 1)


@pytest.mark.skipif(not SAMPLE_PATH.exists(), reason="sample file not present")
def test_sample_file_correct_chains_all_valid():
    """All True-answer examples in the sample should have fully valid CoT labels."""
    records_by_problem: dict[str, list[int]] = {}
    for item in _iter_prontoqa_items(SAMPLE_PATH):
        if item["answer"] == "True":
            key = item["question"]
            labels = label_chain(item["question"], item["chain_of_thought"])
            records_by_problem[key] = labels

    for question, labels in records_by_problem.items():
        assert all(l == 1 for l in labels), (
            f"Expected all steps valid in True-answer chain.\n"
            f"Question: {question}\nLabels: {labels}"
        )


def _iter_prontoqa_items(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def test_load_step_records_from_tmpfile():
    """End-to-end: write a minimal JSONL and load step records."""
    data = [
        {
            "question": "Every wumpus is a rompus. Every rompus is luminous. Rex is a wumpus.",
            "query": "True or false: Rex is luminous.",
            "chain_of_thought": ["Rex is a wumpus.", "Rex is a rompus.", "Rex is luminous."],
            "answer": "True",
        }
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        tmp_path = f.name

    records = _load_step_records(tmp_path)
    assert len(records) == 3  # 3 steps
    assert records[0]["label"] == 1
    assert records[1]["label"] == 1
    assert records[2]["label"] == 1
