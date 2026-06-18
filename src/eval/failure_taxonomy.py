"""Failure-mode taxonomy for step-level CoT errors (S3).

The taxonomy is the analytical lens from the research proposal (ROSCOE-inspired,
moved inside the model). It is *experimental*: categories may be merged or split
based on annotation quality and representation geometry. ``OTHER`` is the
catch-all for steps that don't fit, or where the labeler is unsure.

Each entry has a short, judge-facing definition with a concrete cue. Keep the
keys stable: they are the enum the LLM judge is constrained to, and the join key
for downstream analysis.
"""

from __future__ import annotations

__all__ = ["TAXONOMY", "FAILURE_MODES", "taxonomy_prompt_block"]

# slug -> (short name, judge-facing definition)
TAXONOMY: dict[str, tuple[str, str]] = {
    "arithmetic_error": (
        "Arithmetic error",
        "A wrong numerical computation: addition, subtraction, multiplication, "
        "division, or evaluation that gives the wrong number (e.g. 7 x 8 = 54).",
    ),
    "algebraic_transformation_error": (
        "Algebraic transformation error",
        "An invalid symbolic manipulation: bad equation rearrangement, "
        "distributing/factoring wrong, an illegal cancellation or substitution.",
    ),
    "variable_or_entity_binding_error": (
        "Variable or entity binding error",
        "Confusing which quantity, person, object, variable, or role a value "
        "refers to: plugging the wrong known into the right formula, swapping two "
        "entities, or mislabelling what was just computed.",
    ),
    "quantity_or_unit_mismatch": (
        "Quantity or unit mismatch",
        "Combining incompatible quantities or semantic types: adding a rate to a "
        "count, mixing units (minutes vs hours), or a dimensional inconsistency.",
    ),
    "unsupported_premise": (
        "Unsupported premise",
        "Introducing information not given in the problem or derivable from prior "
        "steps: an invented number, assumption, or fact pulled from nowhere.",
    ),
    "goal_drift": (
        "Goal drift",
        "Solving a related but wrong sub-problem while staying locally coherent: "
        "answering a different question than the one asked, or optimizing the "
        "wrong quantity.",
    ),
    "constraint_violation": (
        "Constraint violation",
        "Ignoring a stated or implied domain constraint: integer/positivity "
        "requirement, an inequality, a boundary or range condition, or a rule "
        "given in the problem.",
    ),
    "logical_inference_error": (
        "Logical inference error",
        "An invalid deductive step not reducible to arithmetic or algebra: a non "
        "sequitur, affirming the consequent, an unjustified 'therefore', or a "
        "wrong case split.",
    ),
    "post_hoc_reasoning": (
        "Post-hoc reasoning",
        "A plausible-sounding step that is not actually the causal basis for the "
        "result: hand-waving, restating the goal as if proven, or a justification "
        "bolted on after an unsupported jump.",
    ),
    "other": (
        "Other / unclear",
        "None of the above fit, the step mixes several modes inseparably, or there "
        "is not enough context to classify confidently.",
    ),
}

FAILURE_MODES: list[str] = list(TAXONOMY.keys())


def taxonomy_prompt_block() -> str:
    """Render the taxonomy as a numbered list for the judge's system prompt."""
    lines = []
    for slug, (name, definition) in TAXONOMY.items():
        lines.append(f"- {slug} ({name}): {definition}")
    return "\n".join(lines)
