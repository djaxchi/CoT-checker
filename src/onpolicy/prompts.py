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

import json
from pathlib import Path

STYLES = ("verifier", "generation", "judge")

# Prompt styles a *sampler* can have run under, as recorded per trajectory.
SAMPLER_STYLES = ("zero", "fewshot", "chat", "prm800k")


def generation_prompt(problem: str) -> str:
    """The sampling prompt. Byte-identical to what the generator sends."""
    return (f"Problem:\n{problem}\n\n"
            "Solve the problem step by step. Put each step on its own line, and write "
            "the final answer inside \\boxed{}.\n\nSolution:\n")


# The Instruct policy's prompt (instruct_arm_v1). Qwen's own math instruction,
# the one its model cards and ReProbe's Qwen3-8B numbers use.
CHAT_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def chat_prompt(problem: str) -> str:
    """Qwen3's chat template for one user turn, non-thinking mode, as a literal.

    Byte-identical to `apply_chat_template(..., add_generation_prompt=True,
    enable_thinking=False)` on Qwen/Qwen3-8B, and tokenises to the same ids.
    Written out rather than rendered so the confidence encoder can rebuild the
    context from a trajectory row without a tokenizer in hand. The empty think
    block is what non-thinking mode emits; leaving it out lets the model open
    one itself.
    """
    return (f"<|im_start|>user\n{problem}\n{CHAT_INSTRUCTION}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


# PRM800K-style steps from the Instruct policy (online_reject_v2). Left to itself,
# Qwen3-8B writes markdown: a quarter of its blank-line "steps" are headers or
# horizontal rules and a quarter bare display equations, where PRM800K steps are
# one or two plain sentences (median 29 tokens). The verifiers were trained on the
# latter, so the policy is shown three PRM800K solutions (phase2 train, every step
# rated +1, none from MATH-500) and asked to write that way.
PRM800K_SYSTEM = ("Solve the problem step by step. Write each step as one or two plain "
                  "sentences, with any math inline in $...$. Put a blank line between "
                  "steps. Do not use headings, lists, bold text or horizontal rules. "
                  "Finish with a step that states the final answer in \\boxed{}.")
PRM800K_EXEMPLARS = json.loads((Path(__file__).with_name("prm800k_exemplars.json")).read_text())


def prm800k_prompt(problem: str) -> str:
    """Qwen3's chat template for the system turn, three exemplar turns and the
    problem, non-thinking; a literal so contexts rebuild without a tokenizer."""
    s = f"<|im_start|>system\n{PRM800K_SYSTEM}<|im_end|>\n"
    for ex in PRM800K_EXEMPLARS:
        s += f"<|im_start|>user\n{ex['problem']}<|im_end|>\n"
        s += "<|im_start|>assistant\n" + "\n\n".join(ex["steps"]) + "<|im_end|>\n"
    return s + f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


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


# judge_prompt_v1: the backbone is told it is verifying the current step before
# it reads it, then asked for a verdict after it. The step's own token states can
# see the instruction but not the question (causal attention), so the step-span
# readouts measure what the instruction changes; the last token of JUDGE_SUFFIX,
# where the model is about to answer, is a separate readout and also gives the
# zero-shot P(No) / P(Yes) baseline. "Current step:\n" ends the prefix exactly as
# in the verifier template, so the step is tokenized the same way under both.
JUDGE_SYSTEM = ("You are a careful math verifier. You are given a problem, the previous "
                "steps of a solution, and the current step. Decide whether the current "
                "step is correct, given the problem and the previous steps.")
JUDGE_SUFFIX = ("\n\nIs the current step correct? Answer Yes or No.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
JUDGE_ANSWERS = ("Yes", "No")


def judge_prefix(problem: str, prefix: str) -> str:
    """Qwen3 chat template (system + user turn) up to the current step."""
    return (f"<|im_start|>system\n{JUDGE_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\nProblem:\n{problem}\n\n"
            f"Previous steps:\n{prefix if prefix else '(none)'}\n\nCurrent step:\n")


def build_prefix(style: str, problem: str, prefix: str) -> str:
    if style == "verifier":
        return verifier_prefix(problem, prefix)
    if style == "generation":
        return generation_prefix(problem, prefix)
    if style == "judge":
        return judge_prefix(problem, prefix)
    raise ValueError(f"unknown prompt style {style!r}; expected one of {STYLES}")


def build_suffix(style: str) -> str:
    """Text after the step. Only the judge style has one."""
    if style not in STYLES:
        raise ValueError(f"unknown prompt style {style!r}; expected one of {STYLES}")
    return JUDGE_SUFFIX if style == "judge" else ""


def context(style: str, problem: str, prefix: str = "", dataset: str = "",
            n_shot: int = 4) -> str:
    """The exact string the model had in front of it at the start of a step.

    Every consumer that reconstructs generative states has to agree with the
    sampler byte for byte, and there are now two sampler prompts rather than one.
    Hardcoding either is a silent failure: the forward pass still runs, the
    logprobs and step scores still look plausible, and they describe a context
    the model never saw. So the style travels on the trajectory row and every
    reconstruction dispatches through here.

    `style` is the sampler style recorded by scripts/generate_onpolicy_steps.py,
    not the encoder-side `verifier`/`generation` distinction above.
    """
    if style == "zero":
        return generation_prefix(problem, prefix)
    if style == "fewshot":
        from src.onpolicy.fewshot import fewshot_prompt
        base = fewshot_prompt(problem, dataset, n_shot)
        return base if not prefix else f"{base}{prefix}\n\n"
    if style == "chat":
        base = chat_prompt(problem)
        return base if not prefix else f"{base}{prefix}\n\n"
    if style == "prm800k":
        base = prm800k_prompt(problem)
        return base if not prefix else f"{base}{prefix}\n\n"
    raise ValueError(f"unknown sampler prompt style {style!r}; "
                     f"expected one of {SAMPLER_STYLES}")


def context_from_row(row: dict, prefix: str = "") -> str:
    """`context` driven by a trajectory row's own recorded fields.

    Defaults to "zero" so a row written before tts_roster_v1, which carries no
    style at all, reconstructs exactly as it always did.
    """
    return context(row.get("prompt_style", "zero"), row["problem"], prefix,
                   row.get("dataset", ""), int(row.get("n_shot", 4)))
