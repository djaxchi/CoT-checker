"""Reconstruct continuous golden-path reasoning trajectories from raw PRM800K.

A trajectory is (question, [step_1 .. step_T]) where step_i is the completion
the PRM800K annotation session actually advanced with (human_completion if
present, else the chosen_completion index into completions). This mirrors the
prefix-advancement logic of build_prm800k_prestudy.build_candidates exactly,
but keeps the selected path itself instead of the candidate forks.

Correctness ratings are deliberately NOT read: sprint 4 is unsupervised.
"""

from __future__ import annotations

import hashlib
from typing import Any


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def reconstruct_trajectory(sample: dict, sample_idx: int, counters: dict,
                           with_ratings: bool = False) -> dict | None:
    """Return {'trajectory_id', 'question', 'steps'} or None (counting why).

    with_ratings=True adds a parallel 'ratings' list: the PRM800K rating of the
    chosen completion at each golden step (-1/0/1), or 'human' when the step is
    a human-written completion (those carry no rating). Ratings are for
    POST-HOC interpretation only; the sprint-4 pipeline never reads them.
    """
    if not isinstance(sample, dict):
        counters["malformed_samples"] += 1
        return None

    # Raw PRM800K nests problem under "question" and steps under "label";
    # some mirrors hoist them to the top level. Support both (as prestudy does).
    question = sample.get("question")
    if isinstance(question, dict):
        problem = question.get("problem")
    else:
        problem = sample.get("problem")

    label_field = sample.get("label")
    if isinstance(label_field, dict):
        steps = label_field.get("steps")
    else:
        steps = sample.get("steps")

    if not problem or not isinstance(problem, str):
        counters["missing_problem"] += 1
        return None
    if not isinstance(steps, list) or not steps:
        counters["missing_steps"] += 1
        return None

    chosen: list[str] = []
    ratings: list[object] = []
    broken = False
    for step in steps:
        if not isinstance(step, dict):
            broken = True
            break
        human = step.get("human_completion")
        chosen_idx = step.get("chosen_completion")
        completions = step.get("completions")
        text = None
        rating: object = None
        if human is not None:
            text = human.get("text") if isinstance(human, dict) else None
            rating = "human"
        elif chosen_idx is not None and isinstance(completions, list):
            if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(completions):
                sel = completions[chosen_idx]
                text = sel.get("text") if isinstance(sel, dict) else None
                rating = sel.get("rating") if isinstance(sel, dict) else None
        if not text or not isinstance(text, str) or not text.strip():
            # PRM800K sessions can end without a selected completion (e.g. the
            # annotator gave up); the golden path stops here. Steps so far are
            # still a continuous trajectory, so keep them.
            broken = True
            break
        chosen.append(text.strip())
        ratings.append(rating)

    if broken:
        counters["truncated_paths"] += 1
    if len(chosen) < 2:
        counters["too_few_steps"] += 1
        return None

    problem_id = sample.get("problem_id") or f"p{sample_idx}_{stable_hash(problem)}"
    solution_id = sample.get("solution_id") or f"s{sample_idx}"
    out = {
        "trajectory_id": f"{problem_id}::{solution_id}",
        "question": problem,
        "steps": chosen,
    }
    if with_ratings:
        out["ratings"] = ratings
    return out


def extract_fork_pairs(sample: dict, counters: dict) -> list[dict]:
    """Matched correct/incorrect fork pairs from one PRM800K session.

    A fork is a step whose unflagged completions contain BOTH a rating +1 and
    a rating -1 continuation of the same prefix. The correct side is the
    chosen completion when it is itself rated +1, otherwise the first +1
    sibling (annotators often advance through a 0-rated step even when a +1
    exists, and sessions can end at a step that still holds a +1/-1 pair).
    Returns dicts with the question, the golden prefix steps before the fork
    (possibly empty), 1-based step_index, the correct text, and ALL wrong
    texts (caller samples). The prefix advances exactly like
    reconstruct_trajectory; the fork check runs before the advancement check,
    so a session's final rated step still yields its fork.
    """
    question = sample.get("question")
    problem = question.get("problem") if isinstance(question, dict) else sample.get("problem")
    label_field = sample.get("label")
    steps = label_field.get("steps") if isinstance(label_field, dict) else sample.get("steps")
    if not problem or not isinstance(problem, str) or not isinstance(steps, list):
        counters["skipped_sessions"] += 1
        return []

    forks: list[dict] = []
    prefix: list[str] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            break
        human = step.get("human_completion")
        chosen_idx = step.get("chosen_completion")
        completions = step.get("completions")

        # ---- fork check (before advancement: last rated step still counts)
        if isinstance(completions, list):
            def _txt(c):
                return c.get("text").strip() if isinstance(c.get("text"), str) else ""
            rated = [c for c in completions
                     if isinstance(c, dict) and not c.get("flagged") and _txt(c)]
            pos = [c for c in rated if c.get("rating") == 1]
            neg = [c for c in rated if c.get("rating") == -1]
            if pos and neg:
                correct = None
                if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(completions):
                    sel = completions[chosen_idx]
                    if isinstance(sel, dict) and sel.get("rating") == 1:
                        correct = _txt(sel)
                if not correct:
                    correct = _txt(pos[0])
                wrongs = [_txt(c) for c in neg if _txt(c) != correct]
                if correct and wrongs:
                    counters["forks_found"] += 1
                    forks.append({
                        "question": problem,
                        "prefix_steps": list(prefix),
                        "step_index": i + 1,
                        "correct": correct,
                        "wrongs": wrongs,
                    })

        # ---- golden-path advancement (identical to reconstruct_trajectory)
        text = None
        if human is not None:
            text = human.get("text") if isinstance(human, dict) else None
        elif chosen_idx is not None and isinstance(completions, list):
            if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(completions):
                sel = completions[chosen_idx]
                text = sel.get("text") if isinstance(sel, dict) else None
        if not text or not isinstance(text, str) or not text.strip():
            break
        prefix.append(text.strip())
    return forks


def audit_trajectories(samples: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Reconstruct every sample's golden path and count what was dropped."""
    counters: dict[str, int] = {
        "malformed_samples": 0,
        "missing_problem": 0,
        "missing_steps": 0,
        "truncated_paths": 0,
        "too_few_steps": 0,
    }
    trajectories: list[dict] = []
    seen: set[str] = set()
    for idx, sample in enumerate(samples):
        traj = reconstruct_trajectory(sample, idx, counters)
        if traj is None:
            continue
        if traj["trajectory_id"] in seen:
            counters["duplicate_trajectory_id"] = counters.get("duplicate_trajectory_id", 0) + 1
            traj["trajectory_id"] += f"::dup{idx}"
        seen.add(traj["trajectory_id"])
        trajectories.append(traj)

    steps_per = [len(t["steps"]) for t in trajectories]
    audit = {
        "n_raw_examples": len(samples),
        "n_usable_trajectories": len(trajectories),
        "n_usable_steps": int(sum(steps_per)),
        "steps_per_trajectory": {
            "mean": float(sum(steps_per) / len(steps_per)) if steps_per else 0.0,
            "median": float(sorted(steps_per)[len(steps_per) // 2]) if steps_per else 0.0,
            "max": int(max(steps_per)) if steps_per else 0,
            "min": int(min(steps_per)) if steps_per else 0,
        },
        "dropped": counters,
    }
    return trajectories, audit
