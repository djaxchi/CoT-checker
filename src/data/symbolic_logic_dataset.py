"""ProntoQA-style symbolic logic dataset.

Each problem consists of:
  - A set of rules ("Every A is [a/an] B.") and an initial fact ("[Name] is [a/an] A.")
  - A query ("True or false: [Name] is B.")
  - A chain-of-thought: a sequence of inference steps, each applying one modus ponens rule

Each step is of the form "[Name] is [a/an] [property/category]."
The PropLogicSolver deterministically verifies whether each step validly follows from
prior steps and the base rules/facts in the question — no model or heuristic required.

Data format (JSONL, one problem per line):
    {
        "question": "Every wumpus is a rompus. Every rompus is luminous. Rex is a wumpus.",
        "query":    "True or false: Rex is luminous.",
        "chain_of_thought": ["Rex is a wumpus.", "Rex is a rompus.", "Rex is luminous."],
        "answer":   "True",
        "meta":     {"n_hops": 2}   # optional
    }
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# "Every wumpus is [a/an] rompus."  or  "Each wumpus is luminous."
_RULE_RE = re.compile(
    r"(?:Every|Each)\s+(\w+)\s+is\s+(?:not\s+)?(?:a\s+|an\s+)?(\w+)\s*\.",
    re.IGNORECASE,
)
# Capture negation separately
_NEG_RULE_RE = re.compile(
    r"(?:Every|Each)\s+(\w+)\s+is\s+(not)\s+(?:a\s+|an\s+)?(\w+)\s*\.",
    re.IGNORECASE,
)

# "Wumpuses are [a/an] rompuses."  — plural form used in ProntoQA HuggingFace dataset
_PLURAL_RULE_RE = re.compile(
    r"([A-Za-z]+)\s+are\s+(?:not\s+)?(?:a\s+|an\s+)?([A-Za-z]+)\s*\.",
    re.IGNORECASE,
)
_PLURAL_NEG_RULE_RE = re.compile(
    r"([A-Za-z]+)\s+are\s+(not)\s+(?:a\s+|an\s+)?([A-Za-z]+)\s*\.",
    re.IGNORECASE,
)


def _singularize(word: str) -> str:
    """Strip plural suffix from ProntoQA category names (e.g. jompuses → jompus)."""
    w = word.lower()
    if w.endswith("es"):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w

# "[Name] is [a/an] [property]."  — Name must start with capital letter
_FACT_RE = re.compile(
    r"([A-Z][a-z]*)\s+is\s+(?:not\s+)?(?:a\s+|an\s+)?(\w+)\s*\."
)
_NEG_FACT_RE = re.compile(
    r"([A-Z][a-z]*)\s+is\s+(not)\s+(?:a\s+|an\s+)?(\w+)\s*\."
)

# Step pattern: same as fact but also accepts "not"
_STEP_RE = re.compile(
    r"([A-Z][a-z]*)\s+is\s+(?:not\s+)?(?:a\s+|an\s+)?(\w+)\s*\."
)
_NEG_STEP_RE = re.compile(
    r"([A-Z][a-z]*)\s+is\s+(not)\s+(?:a\s+|an\s+)?(\w+)\s*\."
)


def _normalise_prop(prop: str) -> str:
    return prop.strip().lower()


# ---------------------------------------------------------------------------
# PropLogicSolver
# ---------------------------------------------------------------------------


@dataclass
class PropLogicSolver:
    """Deterministic solver for propositional modus-ponens reasoning chains.

    Tracks:
      - rules:     dict[str, set[str]]  — premise -> {conclusions}
      - neg_rules: dict[str, set[str]]  — premise -> {negated conclusions}
      - known:     dict[str, set[str]]  — entity  -> {positive properties}
      - neg_known: dict[str, set[str]]  — entity  -> {negated properties}
    """

    rules: dict[str, set[str]] = field(default_factory=dict)
    neg_rules: dict[str, set[str]] = field(default_factory=dict)
    known: dict[str, set[str]] = field(default_factory=dict)
    neg_known: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_question(cls, question: str) -> "PropLogicSolver":
        """Parse rules and seed facts from a ProntoQA question string.

        Handles both singular forms ("Every wumpus is a rompus.") and the plural
        forms used in the HuggingFace ProntoQA dataset ("Wumpuses are rompuses.").
        """
        solver = cls()

        # --- Singular forms: Every/Each X is [not] [a/an] Y ---
        for m in _NEG_RULE_RE.finditer(question):
            premise = _normalise_prop(m.group(1))
            conclusion = _normalise_prop(m.group(3))
            solver.neg_rules.setdefault(premise, set()).add(conclusion)

        for m in _RULE_RE.finditer(question):
            full = m.group(0)
            if re.search(r"\bis\s+not\b", full, re.IGNORECASE):
                continue
            premise = _normalise_prop(m.group(1))
            conclusion = _normalise_prop(m.group(2))
            solver.rules.setdefault(premise, set()).add(conclusion)

        # --- Plural forms: Xs are [not] [a/an] Ys ---
        for m in _PLURAL_NEG_RULE_RE.finditer(question):
            premise = _singularize(m.group(1))
            conclusion = _singularize(m.group(3))
            solver.neg_rules.setdefault(premise, set()).add(conclusion)

        for m in _PLURAL_RULE_RE.finditer(question):
            full = m.group(0)
            if re.search(r"\bare\s+not\b", full, re.IGNORECASE):
                continue
            premise = _singularize(m.group(1))
            conclusion = _singularize(m.group(2))
            solver.rules.setdefault(premise, set()).add(conclusion)

        # --- Seed facts: [Name] is [not] [a/an] X  (capitalised entity names) ---
        for m in _NEG_FACT_RE.finditer(question):
            entity = m.group(1)
            prop = _normalise_prop(m.group(3))
            solver.neg_known.setdefault(entity, set()).add(prop)

        for m in _FACT_RE.finditer(question):
            full = m.group(0)
            if re.search(r"\bis\s+not\b", full, re.IGNORECASE):
                continue
            entity = m.group(1)
            prop = _normalise_prop(m.group(2))
            solver.known.setdefault(entity, set()).add(prop)

        return solver

    def apply_step(self, step: str) -> None:
        """Record a step as newly derived, expanding the known set."""
        neg_m = _NEG_STEP_RE.match(step.strip())
        if neg_m:
            entity, prop = neg_m.group(1), _normalise_prop(neg_m.group(3))
            self.neg_known.setdefault(entity, set()).add(prop)
            return

        m = _STEP_RE.match(step.strip())
        if m:
            entity, prop = m.group(1), _normalise_prop(m.group(2))
            self.known.setdefault(entity, set()).add(prop)

    def is_valid_step(self, step: str) -> bool:
        """Return True iff the step validly follows from current known facts + rules.

        A step "[Name] is [P]." is valid if:
          1. (Name, P) is already directly known, OR
          2. There exists a rule Q -> P and (Name, Q) is known (one-step modus ponens).

        A step "[Name] is not [P]." is valid if:
          1. (Name, not-P) is already negatively known, OR
          2. There exists a negation rule Q -> not-P and (Name, Q) is known.

        Non-parseable steps are assumed valid (no evidence of error).
        """
        neg_m = _NEG_STEP_RE.match(step.strip())
        if neg_m:
            entity, prop = neg_m.group(1), _normalise_prop(neg_m.group(3))
            if prop in self.neg_known.get(entity, set()):
                return True
            entity_props = self.known.get(entity, set())
            for premise, neg_conclusions in self.neg_rules.items():
                if prop in neg_conclusions and premise in entity_props:
                    return True
            return False

        m = _STEP_RE.match(step.strip())
        if not m:
            return True  # non-parseable: no evidence of error

        entity, prop = m.group(1), _normalise_prop(m.group(2))

        # Already known directly
        if prop in self.known.get(entity, set()):
            return True

        # One-step modus ponens
        entity_props = self.known.get(entity, set())
        for premise, conclusions in self.rules.items():
            if prop in conclusions and premise in entity_props:
                return True

        return False

    def all_reachable(self, entity: str) -> set[str]:
        """Return all properties reachable for entity by exhaustive forward chaining."""
        props = set(self.known.get(entity, set()))
        changed = True
        while changed:
            changed = False
            for premise, conclusions in self.rules.items():
                if premise in props:
                    new = conclusions - props
                    if new:
                        props |= new
                        changed = True
        return props


# ---------------------------------------------------------------------------
# Step corruption
# ---------------------------------------------------------------------------


def corrupt_step_logic(
    step: str,
    solver: PropLogicSolver,
    rng: "random.Random | None" = None,
) -> tuple[str, bool]:
    """Introduce a logical error into a valid step.

    Replaces the inferred property with one that does NOT follow from current
    known facts. Picks a property from all known rule conclusions that is NOT
    reachable for this entity.

    Returns:
        (corrupted_step, was_corrupted)
        was_corrupted is False when no suitable wrong property could be found
        and the step is returned unchanged.
    """
    if rng is None:
        rng = random

    m = _STEP_RE.match(step.strip())
    if not m:
        return step, False

    entity, correct_prop = m.group(1), _normalise_prop(m.group(2))

    # Collect all properties mentioned anywhere in rules (as conclusions)
    all_conclusion_props: set[str] = set()
    for conclusions in solver.rules.values():
        all_conclusion_props |= conclusions
    for conclusions in solver.neg_rules.values():
        all_conclusion_props |= conclusions

    reachable = solver.all_reachable(entity)
    wrong_props = list(all_conclusion_props - reachable - {correct_prop})

    if not wrong_props:
        return step, False

    wrong_prop = rng.choice(wrong_props)
    # Preserve article if original used one
    article_match = re.search(r"\bis\s+(a |an )", step, re.IGNORECASE)
    if article_match:
        article = article_match.group(1)
        corrupted = f"{entity} is {article}{wrong_prop}."
    else:
        corrupted = f"{entity} is {wrong_prop}."

    return corrupted, True


# ---------------------------------------------------------------------------
# Label a full chain
# ---------------------------------------------------------------------------


def label_chain(question: str, chain_of_thought: list[str]) -> list[int]:
    """Return a per-step correctness label list (1=valid, 0=invalid).

    Verifies each step in sequence, accumulating derived facts as it goes
    so that later steps can depend on earlier valid ones.
    """
    solver = PropLogicSolver.from_question(question)
    labels: list[int] = []
    for step in chain_of_thought:
        label = int(solver.is_valid_step(step))
        labels.append(label)
        solver.apply_step(step)  # always record the step, even if wrong
    return labels


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_step_records(path: str | Path) -> list[dict]:
    """Expand each problem into one record per CoT step.

    Each record:
        context  — question + query + all prior steps (string)
        step     — current step text (string)
        label    — 1 if step is logically valid, 0 otherwise (int)
    """
    records = []
    for item in _iter_jsonl(path):
        question = item["question"]
        query = item.get("query", "")
        cot = item["chain_of_thought"]
        labels = label_chain(question, cot)

        for i, (step, label) in enumerate(zip(cot, labels)):
            prior = " ".join(cot[:i])
            context = f"{question} {query} {prior}".strip()
            records.append({"context": context, "step": step, "label": label})

    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ProntoQAStepDataset(Dataset):
    """One sample = one (context, step, label) triple from a ProntoQA JSONL file.

    Args:
        file_path: Path to a JSONL file in ProntoQA format.
        tokenizer:  HuggingFace tokenizer (must have sep_token / eos_token set).
        max_length: Maximum token length for context and step sequences.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.records = _load_step_records(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        context_ids = torch.tensor(
            self.tokenizer.encode(
                rec["context"], max_length=self.max_length, truncation=True
            ),
            dtype=torch.long,
        )
        step_ids = torch.tensor(
            self.tokenizer.encode(
                rec["step"], max_length=self.max_length, truncation=True
            ),
            dtype=torch.long,
        )
        return {
            "context": rec["context"],
            "step": rec["step"],
            "label": rec["label"],
            "context_ids": context_ids,
            "step_ids": step_ids,
        }
