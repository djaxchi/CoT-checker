"""
Pass 1 screening: broad keyword scan across title + abstract excerpt.

A paper passes if any anchor term from any profile appears in the title,
OR if any anchor term appears in the first 600 chars of the abstract.

This is intentionally high-recall — Pass 2 handles precision.
False positives cost one cheap LLM call; false negatives are lost forever.
"""

import re
import logging
from pathlib import Path

import yaml

from models import Paper

logger = logging.getLogger(__name__)

ABSTRACT_SCAN_CHARS = 600  # how much of the abstract to check


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _term_in_text(term: str, text: str) -> bool:
    """Case-insensitive whole-word (or whole-phrase) match."""
    escaped = re.escape(term)
    if " " in term or "-" in term:
        # Multi-word / hyphenated phrases: literal match, no boundary needed
        pattern = escaped
    else:
        pattern = r"\b" + escaped + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def pass1_filter(paper: Paper, config: dict | None = None) -> bool:
    """
    Return True if any anchor term from any profile appears in:
      - the title, OR
      - the first ABSTRACT_SCAN_CHARS characters of the abstract.
    """
    if config is None:
        config = _load_config()

    title = paper.title
    abstract_head = (paper.abstract or "")[:ABSTRACT_SCAN_CHARS]

    for profile_name, profile in config["profiles"].items():
        for term in profile["anchor_terms"]:
            if _term_in_text(term, title) or _term_in_text(term, abstract_head):
                logger.debug(f"[pass1] hit on profile '{profile_name}' term '{term}': {paper.title!r}")
                return True

    return False
