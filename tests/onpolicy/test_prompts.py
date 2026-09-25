"""The two encoding contexts, pinned.

`generation_prefix` exists to reproduce the context the sampler actually ran
under. If it drifts from the sampling prompt by a single character, the states we
call "on-policy" are states the model never held, every number downstream is
quietly wrong, and nothing in the pipeline would fail. So the drift is what these
tests check.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.encode_prm800k_hidden_states import build_prompt_prefix  # noqa: E402
from scripts.generate_onpolicy_steps import build_prompt, split_into_steps  # noqa: E402
from src.onpolicy.prompts import (  # noqa: E402
    build_prefix, generation_prefix, generation_prompt, verifier_prefix,
)

PROBLEM = "What is $2+2$?"


def test_generation_prompt_is_the_sampling_prompt():
    assert generation_prompt(PROBLEM) == build_prompt(PROBLEM)


def test_verifier_prefix_is_the_off_policy_template():
    for prefix in ("", "First step.", "First step.\n\nSecond step."):
        assert verifier_prefix(PROBLEM, prefix) == build_prompt_prefix(PROBLEM, prefix)


def test_generation_prefix_reconstructs_the_sampled_context():
    """prefix_k + steps[k] must rebuild the model's own text, prefix by prefix."""
    solution = "We add two and two.\n\nThat gives four.\n\nSo \\boxed{4}."
    steps = split_into_steps(solution)
    full = generation_prompt(PROBLEM) + solution
    for k, step in enumerate(steps):
        ctx = generation_prefix(PROBLEM, "\n\n".join(steps[:k]))
        assert full.startswith(ctx), k
        assert full[len(ctx):].startswith(step), k


def test_first_step_context_is_the_bare_prompt():
    assert generation_prefix(PROBLEM, "") == generation_prompt(PROBLEM)


def test_the_two_styles_are_different_and_dispatch_by_name():
    assert build_prefix("verifier", PROBLEM, "a") == verifier_prefix(PROBLEM, "a")
    assert build_prefix("generation", PROBLEM, "a") == generation_prefix(PROBLEM, "a")
    assert verifier_prefix(PROBLEM, "a") != generation_prefix(PROBLEM, "a")
    try:
        build_prefix("nope", PROBLEM, "a")
    except ValueError:
        return
    raise AssertionError("an unknown style must not silently pick a template")


# ---- chat: the Instruct policy's sampler prompt (instruct_arm_v1) ----------

def test_chat_prompt_is_qwen3_non_thinking_template():
    """Byte-identical to Qwen3-8B's apply_chat_template(enable_thinking=False).

    Pinned as a literal so the encoder can rebuild it without a tokenizer. The
    string and its tokenisation were checked against the real template on
    TamIA (transformers 5.6.2): same text, same ids.
    """
    from src.onpolicy.prompts import CHAT_INSTRUCTION, chat_prompt
    assert chat_prompt(PROBLEM) == (
        "<|im_start|>user\n" + PROBLEM + "\n" + CHAT_INSTRUCTION + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert build_prompt(PROBLEM, "chat") == chat_prompt(PROBLEM)


def test_chat_context_reconstructs_the_sampled_context():
    from src.onpolicy.prompts import chat_prompt, context
    solution = "We add two and two.\n\nThat gives four.\n\nSo \\boxed{4}."
    steps = split_into_steps(solution)
    full = chat_prompt(PROBLEM) + solution
    for k, step in enumerate(steps):
        ctx = context("chat", PROBLEM, "\n\n".join(steps[:k]))
        assert full.startswith(ctx), k
        assert full[len(ctx):].startswith(step), k


def test_chat_is_a_recorded_sampler_style():
    from src.onpolicy.prompts import SAMPLER_STYLES, context_from_row
    assert "chat" in SAMPLER_STYLES
    row = {"prompt_style": "chat", "problem": PROBLEM}
    assert context_from_row(row).startswith("<|im_start|>user\n")
