"""Few-shot CoT prompts, and the stop delimiter that makes generation terminate.

The zero-shot prompt this project has used until now ends "Solution:\\n" and
shows the model nothing about what finishing looks like. A base model at T=1.0
therefore has no signal for when to stop, and 21.2% of the on-policy pool ran to
its 768-token cap (REPORT.md §20.16). Those truncated traces are correct 5.5% of
the time against 45.8% for the rest, the grader's fallbacks still parse an answer
from them, and they enter the vote, which handed every cheap baseline a
correctness signal that was really a budget signal.

Raising the cap alone is the expensive fix, because its cost falls entirely on
traces that never terminate. Few-shot prompting is the cheap one: four worked
solutions that end teach the pattern, and the delimiter that starts the next
problem is a stop string.

Two properties the exemplars must have, and both are load-bearing:

1. **Steps separated by blank lines.** `split_into_steps` segments on "\\n\\n",
   so the exemplars are what teach the model the step convention every per-step
   score in this project depends on.
2. **A final answer in \\boxed{}.** The grader's fallbacks can parse other
   shapes, which is exactly how truncated traces got into the vote, so the
   exemplars have to make the boxed form the obvious one.

Exemplars are fixed and hand-written rather than sampled from the train split,
so there is no path by which a test problem can reach the prompt.
"""

from __future__ import annotations

# The string that begins every problem. The model emits it when it thinks the
# current solution is over, which is what makes it usable as a stop string.
DELIMITER = "Problem:"
STOP_STRING = "\n\nProblem:"

_GSM8K = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the "
     "grove today. After they are done, there will be 21 trees. How many trees "
     "did the grove workers plant today?",
     "There are 15 trees originally.\n\n"
     "After planting there are 21 trees.\n\n"
     "So the workers planted 21 - 15 = 6 trees.\n\n"
     "The answer is \\boxed{6}."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
     "pieces do they have left in total?",
     "Leah had 32 chocolates and her sister had 42.\n\n"
     "Together they had 32 + 42 = 74 chocolates.\n\n"
     "After eating 35 they have 74 - 35 = 39 left.\n\n"
     "The answer is \\boxed{39}."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
     "lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops.\n\n"
     "He now has 12.\n\n"
     "So he gave away 20 - 12 = 8 lollipops.\n\n"
     "The answer is \\boxed{8}."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does "
     "she have left?",
     "Five bagels at $3 each cost 5 * 3 = 15 dollars.\n\n"
     "Olivia started with 23 dollars.\n\n"
     "She has 23 - 15 = 8 dollars left.\n\n"
     "The answer is \\boxed{8}."),
]

_MATH = [
    ("Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
     "The expression under the top radical needs $x - 2 \\ge 0$, so $x \\ge 2$.\n\n"
     "The bottom radical needs $5 - x > 0$, strictly, because it is a "
     "denominator, so $x < 5$.\n\n"
     "Both conditions together give $2 \\le x < 5$.\n\n"
     "The answer is \\boxed{[2,5)}."),
    ("If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find "
     "$\\det (\\mathbf{A} \\mathbf{B}).$",
     "The determinant of a product is the product of the determinants.\n\n"
     "So $\\det(\\mathbf{A}\\mathbf{B}) = \\det(\\mathbf{A})\\det(\\mathbf{B})$.\n\n"
     "That is $2 \\cdot 12 = 24$.\n\n"
     "The answer is \\boxed{24}."),
    ("Terrell usually lifts two 20-pound weights 12 times. If he uses two "
     "15-pound weights instead, how many times must Terrell lift them in order "
     "to lift the same total weight?",
     "Two 20-pound weights lifted 12 times is $2 \\cdot 20 \\cdot 12 = 480$ "
     "pounds.\n\n"
     "Two 15-pound weights lifted $n$ times is $2 \\cdot 15 \\cdot n = 30n$ "
     "pounds.\n\n"
     "Setting $30n = 480$ gives $n = 16$.\n\n"
     "The answer is \\boxed{16}."),
    ("What is the value of $\\sqrt{36+64}-\\sqrt{25-16}$?",
     "First $36 + 64 = 100$, and $\\sqrt{100} = 10$.\n\n"
     "Next $25 - 16 = 9$, and $\\sqrt{9} = 3$.\n\n"
     "So the expression is $10 - 3 = 7$.\n\n"
     "The answer is \\boxed{7}."),
]

EXEMPLARS = {"gsm8k": _GSM8K, "math": _MATH}


def fewshot_prompt(problem: str, dataset: str, n_shot: int = 4) -> str:
    """The 4-shot CoT prompt. Must stay byte-identical to what the sampler sends.

    The encoder rebuilds this string to reproduce the states the model held while
    writing, so any change here silently invalidates every per-step score taken
    under it. `n_shot` is a parameter only so the smoke can run cheaper; the
    production runs use 4, matching the Qwen3 technical report's setting for
    both GSM8K and MATH.
    """
    key = dataset.lower()
    if key not in EXEMPLARS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(EXEMPLARS)}")
    shots = EXEMPLARS[key][:n_shot]
    if len(shots) < n_shot:
        raise ValueError(f"{key} has {len(EXEMPLARS[key])} exemplars, asked for {n_shot}")
    blocks = [f"{DELIMITER}\n{q}\n\nSolution:\n{a}" for q, a in shots]
    blocks.append(f"{DELIMITER}\n{problem}\n\nSolution:\n")
    return "\n\n".join(blocks)


def truncate_at_delimiter(text: str) -> str:
    """Cut a completion at the point the model started the next problem.

    Early stopping on a batched generate only fires when every sequence in the
    batch has stopped, so a completion can carry the beginning of a hallucinated
    next problem even when stop strings are enabled. Recording that text would
    inflate the token count and corrupt the step split, so the cut is applied
    unconditionally and the pre-cut length is recorded separately.
    """
    idx = text.find(STOP_STRING)
    if idx >= 0:
        return text[:idx]
    # A completion can also open the delimiter at the very start of a line
    # without the leading blank line when the model runs the sections together.
    for marker in (f"\n{DELIMITER}", DELIMITER):
        idx = text.find(marker)
        if idx > 0:
            return text[:idx]
    return text
