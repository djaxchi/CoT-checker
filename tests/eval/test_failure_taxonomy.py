"""Sanity tests for the failure-mode taxonomy."""

from src.eval.failure_taxonomy import FAILURE_MODES, TAXONOMY, taxonomy_prompt_block


def test_modes_unique_and_nonempty():
    assert len(FAILURE_MODES) == len(set(FAILURE_MODES))
    assert "other" in FAILURE_MODES  # catch-all must exist
    assert len(FAILURE_MODES) >= 9


def test_every_mode_has_name_and_definition():
    for slug in FAILURE_MODES:
        name, definition = TAXONOMY[slug]
        assert name and definition
        assert len(definition) > 20  # definitions are judge-facing, not stubs


def test_prompt_block_lists_all_modes():
    block = taxonomy_prompt_block()
    for slug in FAILURE_MODES:
        assert slug in block
