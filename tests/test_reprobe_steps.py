import numpy as np

from src.data.reprobe_steps import LABEL_KEY, _claim_steps, manager_traces
from src.onpolicy.prompts import REPROBE_INSTRUCTION, reprobe_prompt_phi4


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
