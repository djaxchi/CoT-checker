import numpy as np

from src.data.reprobe_steps import (LABEL_KEY, LABEL_KEY_THINK, _claim_steps, manager_traces, sentence_item,
                                    token_claim_steps)
from src.onpolicy.prompts import REPROBE_INSTRUCTION, reprobe_prompt_phi4, reprobe_prompt_think


class _Claim:
    def __init__(self, sentence, ids):
        self.sentence, self.claim_text, self.aligned_token_ids = sentence, sentence.strip(), ids


def test_claim_steps_finds_sentences_in_order():
    reply = "- Step 1: a\n- Step 2: b\n  - sub\n<Answer>: c"
    steps = _claim_steps(reply, [_Claim("- Step 1: a\n", [0]), _Claim("- Step 2: b\n  - sub\n", [1]),
                                 _Claim("<Answer>: c", [2])])
    assert [reply[s["start"]:s["end"]] for s in steps] == ["- Step 1: a\n", "- Step 2: b\n  - sub\n", "<Answer>: c"]
    assert _claim_steps(reply, [_Claim("- Step 2: b\n", [0]), _Claim("- Step 1: a\n", [1])]) is None


def test_manager_traces_uses_stored_claims_when_the_port_disagrees():
    reply = "- Step 1: a\n- Step 2: b\n  - sub\n"
    m = {"stats": {"input_texts": ["p"], "greedy_texts": [reply], "greedy_tokens": [[1, 2, 3]],
                   "claims": [[_Claim("- Step 1: a\n", [0]), _Claim("- Step 2: b\n  - sub\n", [1, 2])]]},
         "gen_metrics": {LABEL_KEY: [0.0, 1.0]},
         "estimations": {("claim", "probe"): [0.1, 0.9], ("claim", "short"): [0.5]}}
    # a decode that never reproduces the reply makes the port find no steps
    t = manager_traces(m, lambda ids: "", "x")[0]
    assert t["steps_from"] == "claims"
    assert t["labels"] == [0, 1] and t["scores"] == {"probe": [0.1, 0.9]}
    assert [s["aligned_token_ids"] for s in t["steps"]] == [[0], [1, 2]]


def test_reprobe_prompt_phi4_template():
    p = reprobe_prompt_phi4("Q?")
    assert p.startswith("<|im_start|>system<|im_sep|>You are a helpful AI assistant.<|im_end|>"
                        "<|im_start|>user<|im_sep|>You will be presented")
    assert p.endswith("Q?<|im_end|><|im_start|>assistant<|im_sep|>Reasoning Steps:\n")
    assert "Good (One step on a single line)" in p and "Good (One step" not in REPROBE_INSTRUCTION
    assert np.isclose(p.count("- For example:"), 2)


# native thinking (Qwen3-1.7B, Table 4): token i decodes to VOCAB[i]
VOCAB = ["<think>", "\nOk", ".", " So", " x", ".\n\n", "Then", " y", "..."]


def _dec(ids):
    return "".join(VOCAB[i] for i in ids)


def test_token_claim_steps_locates_respaced_sentences_by_tokens():
    toks = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    reply = _dec(toks)
    # the sentence splitter re-spaces "..." as ". . .": not a substring, found by its tokens
    cl = [_Claim("\nOk.", [1, 2]), _Claim(" So x.", [3, 4, 5]), _Claim("Then y. . .", [6, 7, 8])]
    st = token_claim_steps(reply, toks, _dec, cl)
    assert [reply[s["start"]:s["end"]] for s in st] == ["\nOk.", " So x.\n\n", "Then y..."]
    # tokens covering only a prefix of the reply are fine; disagreeing text is not
    assert token_claim_steps(reply + " more", toks, _dec, cl) is not None
    assert token_claim_steps("other", toks, _dec, cl) is None


def test_sentence_item_widens_to_token_boundaries():
    reply = "<think>\nOk. So x.\n\nThen"
    ends = [7, 10, 11, 14, 16, 19, 23]  # tokens: <think> | \nOk | . | So | x | .\n\n | Then
    # a step ending inside ".\n\n" is read after that token
    assert sentence_item(reply, {"start": 11, "end": 17}, ends) == ("<think>\nOk.", " So x.\n\n")
    assert sentence_item(reply, {"start": 7, "end": 11}, ends) == ("<think>", "\nOk.")


def test_manager_traces_thinking_labels_and_token_spans():
    toks = [0, 1, 2, 3, 4, 5]
    m = {"stats": {"input_texts": ["p"], "greedy_texts": [_dec(toks)], "greedy_tokens": [toks],
                   "claims": [[_Claim("\nOk.", [1, 2]), _Claim(" So x.", [3, 4, 5])]]},
         "gen_metrics": {LABEL_KEY_THINK: [0.0, float("nan")]},
         "estimations": {("claim", "probe"): [0.2, 0.7]}}
    t = manager_traces(m, _dec, "x", token_spans=True)[0]
    assert t["steps_from"] == "claims_tokens" and t["labels"] == [0, -1]
    assert [s["end"] for s in t["steps"]] == [len("<think>\nOk."), len(_dec(toks))]


def test_reprobe_prompt_think_is_fig9():
    assert reprobe_prompt_think("Q?") == ("<|im_start|>user\nAnswer the following question. Enclose your answer in "
                                          "<answer></answer>.\n<Question>: Q?<|im_end|>\n<|im_start|>assistant\n")
