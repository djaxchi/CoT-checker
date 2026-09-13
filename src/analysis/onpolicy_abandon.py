"""Killing a trajectory while it is being written, instead of ranking it afterwards.

Every downstream use of the verifier so far has read finished solutions: §20.2
reranks them, §20.12 breaks ties between them, §20.14 counts how many of them to
draw. §20.9 is the one online experiment and it *spends* tokens, branching five
ways per step, which damages the policy before the head sees anything.

There is a cheaper online move that was never tried. The saved per-step scores
are prefix-causal by construction: `src/onpolicy/prompts.py` builds each step's
context from the problem and the steps *before* it, and the generation-style
context reproduces the states the sampler actually held. So the score of step k
is available the moment step k is written, at no extra cost in that arm. A
trajectory whose prefix already looks wrong can be abandoned and the budget spent
on a fresh sample.

Nothing about the sampler changes, so §20.9's two failure modes do not apply: no
branching, so the policy is untouched, and no step is ever *selected*, so nothing
pushes toward non-committal continuations. The verifier only ever answers "is
this one still worth finishing".
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

# A candidate carries, per step, its prefix-causal suspicion and its token cost,
# plus what it turns into if it is allowed to finish.
#   step_scores  higher = more suspicious, score of step k given steps < k
#   step_tokens  generation cost of each step
#   answer       normalized final answer, None if it did not parse
#   correct      recorded final-answer outcome
#   final_score  worst-step suspicion over the whole trace, the §20.14 quantity


def walk(candidate: dict, tau: float, min_steps: int,
         max_steps: int | None = None) -> tuple[float, bool]:
    """Generate a candidate step by step, abandoning it if the prefix is condemned.

    Returns the tokens actually spent and whether it was allowed to finish. The
    last step is never an abandonment: by then the answer is already written and
    killing it would pay the whole cost for nothing.

    `max_steps` is a deadline, and it is what makes the difference. A rule that
    may kill at any depth mostly kills late, once the running maximum has had
    many steps to cross the threshold, and a late kill has already paid for most
    of the trajectory. Past the deadline the candidate is committed and finishes.
    """
    scores = candidate["step_scores"]
    tokens = candidate["step_tokens"]
    spent, running = 0.0, -np.inf
    for k, (s, tok) in enumerate(zip(scores, tokens), start=1):
        spent += tok
        running = max(running, s)
        if max_steps is not None and k > max_steps:
            return spent + sum(tokens[k:]), True
        if k >= min_steps and k < len(scores) and running > tau:
            return spent, False
    return spent, True


def run_problem(candidates: Sequence[dict], order: Sequence[int], tau: float,
                min_steps: int, n_completions: int,
                max_steps: int | None = None) -> dict:
    """Draw in `order`, abandoning as we go, until `n_completions` finish.

    If the pool runs out with nothing finished we still have to answer, so the
    last candidate drawn is carried to completion and paid for in full. Charging
    nothing and scoring it wrong would make an aggressive threshold look free.
    """
    spent, finished, forced = 0.0, [], False
    for pos, i in enumerate(order):
        cost, done = walk(candidates[i], tau, min_steps, max_steps)
        spent += cost
        if done:
            finished.append(i)
            if len(finished) >= n_completions:
                break
        elif pos == len(order) - 1 and not finished:
            spent += sum(candidates[i]["step_tokens"]) - cost
            finished.append(i)
            forced = True
    return {"tokens": spent, "finished": finished, "forced": forced,
            "n_drawn": pos + 1}


def decide(candidates: Sequence[dict], finished: Sequence[int]) -> dict:
    """Majority over what finished, with and without the verifier breaking ties."""
    groups: dict[object, list[int]] = defaultdict(list)
    for i in finished:
        if candidates[i]["answer"] is not None:
            groups[candidates[i]["answer"]].append(i)
    if not groups:
        return {"majority": 0.0, "majority+tie": 0.0}
    top = max(len(v) for v in groups.values())
    tied = [a for a, v in groups.items() if len(v) == top]
    by_verifier = min(tied, key=lambda a: min(candidates[i]["final_score"]
                                              for i in groups[a]))
    return {
        "majority": float(np.mean([bool(candidates[groups[a][0]]["correct"])
                                   for a in tied])),
        "majority+tie": float(bool(candidates[groups[by_verifier][0]]["correct"])),
    }


def policy_outcome(candidates: Sequence[dict], order: Sequence[int], tau: float,
                   min_steps: int, n_completions: int,
                   max_steps: int | None = None) -> dict:
    run = run_problem(candidates, order, tau, min_steps, n_completions, max_steps)
    return {**run, **decide(candidates, run["finished"])}
