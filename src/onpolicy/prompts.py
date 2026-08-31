"""The two contexts an on-policy step can be read under, in one place.

A step of on-policy text can be encoded two ways, and they are not the same
experiment.

`verifier_prefix` is the template the whole off-policy grid was encoded under
(`build_prompt_prefix` in scripts/encode_prm800k_hidden_states.py, reproduced
here only so both templates sit together). Encoding on-policy text under it
changes exactly one thing against the off-policy arm, the text distribution,
which is the controlled comparison the rank claim needs. What it is *not* is the
state the model held while writing: it is the model re-reading its own words
under a context it never had.

`generation_prefix` reconstructs the context the sampler actually ran on. During
generation the model saw `generation_prompt(problem)` followed by the tokens it
had emitted so far, and the step splitter rejoins steps with "\n\n", so the
context at step k is the prompt followed by steps[:k] joined and terminated the
same way. Running a forward pass over that string reproduces the generative
states of the step's tokens exactly (teacher forcing over the model's own text),
up to the one caveat that re-tokenizing a string can differ from the ids the
sampler emitted at a boundary; that is the same caveat the off-policy encoder
already lives with when it tokenizes prefix and step separately.

`generation_prompt` must stay byte-identical to what
scripts/generate_onpolicy_steps.py sends to `model.generate`, or the
reconstruction is silently wrong and nothing downstream would notice. That
script imports it from here, and a test pins the string.
"""

from __future__ import annotations

STYLES = ("verifier", "generation")


def generation_prompt(problem: str) -> str:
    """The sampling prompt. Byte-identical to what the generator sends."""
    return (f"Problem:\n{problem}\n\n"
            "Solve the problem step by step. Put each step on its own line, and write "
            "the final answer inside \\boxed{}.\n\nSolution:\n")


def generation_prefix(problem: str, prefix: str) -> str:
    """Context the model had at the start of a step, given the steps before it.

    `prefix` is the earlier steps already joined by "\\n\\n" (what the item
    builder stores). Empty prefix means the first step, whose context is the
    prompt alone.
    """
    if not prefix:
        return generation_prompt(problem)
    return f"{generation_prompt(problem)}{prefix}\n\n"


def verifier_prefix(problem: str, prefix: str) -> str:
    """The off-policy grid's encoding template."""
    prefix_section = f"Previous reasoning:\n{prefix}\n\n" if prefix else "Previous reasoning:\n\n"
    return f"Problem:\n{problem}\n\n{prefix_section}Current step:\n"


def build_prefix(style: str, problem: str, prefix: str) -> str:
    if style == "verifier":
        return verifier_prefix(problem, prefix)
    if style == "generation":
        return generation_prefix(problem, prefix)
    raise ValueError(f"unknown prompt style {style!r}; expected one of {STYLES}")
