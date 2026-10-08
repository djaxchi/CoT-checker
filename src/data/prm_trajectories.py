"""Full PRM800K phase-2 trajectories with per-step supervision masks.

bidirectional_token_probe_v1 (docs/bidirectional_token_probe_v1_plan.md, section 4).

A phase-2 record holds the ORIGINAL generated solution in
``question.pre_generated_steps`` even when the human labels stop at the first
error. The labels live in ``label.steps[k]``: ``completions[...]`` are the rated
candidates at position k, and the session advances with ``human_completion`` or
``completions[chosen_completion]``. A label at position k describes the original
step k only if (a) a rated completion's text equals ``pre_generated_steps[k]``
and (b) every earlier position advanced with the original text, so the label
was given on the original prefix. Anything else (a human repair, an alternative
chosen earlier) puts later labels on another branch, and those are masked.

Label convention (the dataset's): rating -1 -> y=1 (incorrect), ratings 0 and +1
-> y=0. Unknown, unrated, flagged, contradictory or off-branch positions get
``label_mask=False`` and ``y=-1``; they never enter a loss. Later unlabeled steps
stay in the trajectory as input context.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable

# Masked-label sentinel. Never a valid target.
Y_UNKNOWN = -1


def sha1(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def normalize_problem(text: str) -> str:
    """Canonical problem text: NFKC, lowercase, whitespace collapsed."""
    t = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", t).strip()


def problem_key(text: str) -> str:
    """Stable problem identity from normalized text (record IDs can hide duplicates)."""
    return "pk_" + sha1(normalize_problem(text))


def trajectory_key(problem: str, steps: list[str]) -> str:
    return "tk_" + sha1(normalize_problem(problem) + "\x1e" + "\x1f".join(steps))


def rating_to_y(rating: int) -> int:
    if rating == -1:
        return 1
    if rating in (0, 1):
        return 0
    raise ValueError(f"invalid rating {rating!r}")


def _advanced_text(step: dict) -> str | None:
    """Text the annotation session advanced with at this position, or None."""
    human = step.get("human_completion")
    if human is not None:
        return human.get("text") if isinstance(human, dict) else None
    idx = step.get("chosen_completion")
    comps = step.get("completions") or []
    if isinstance(idx, int) and 0 <= idx < len(comps) and isinstance(comps[idx], dict):
        return comps[idx].get("text")
    return None


def align_record(record: dict, record_id: str, counters: Counter) -> dict | None:
    """Align one raw phase-2 record to its original trajectory.

    Returns None (counting the reason) for malformed records, bad-problem
    records, or records without any trustworthy label.
    """
    q = record.get("question")
    lab = record.get("label")
    if not isinstance(q, dict) or not isinstance(lab, dict):
        counters["drop_malformed"] += 1
        return None
    problem = q.get("problem")
    pre = q.get("pre_generated_steps")
    steps = lab.get("steps")
    if not isinstance(problem, str) or not problem.strip():
        counters["drop_missing_problem"] += 1
        return None
    if not isinstance(pre, list) or not pre or not all(isinstance(s, str) and s.strip() for s in pre):
        counters["drop_missing_or_empty_pre_generated_steps"] += 1
        return None
    if not isinstance(steps, list):
        counters["drop_malformed"] += 1
        return None
    if lab.get("finish_reason") == "bad_problem":
        counters["drop_bad_problem"] += 1
        return None
    if len(steps) > len(pre):
        counters["drop_more_labels_than_steps"] += 1
        return None

    T = len(pre)
    ratings: list[int | None] = [None] * T
    yk: list[int] = [Y_UNKNOWN] * T
    status: list[str] = ["unlabeled"] * T
    on_branch = True
    for k, st in enumerate(steps):
        if not isinstance(st, dict):
            status[k] = "malformed"
            on_branch = False
            continue
        if not on_branch:
            status[k] = "off_branch"
            continue
        comps = st.get("completions") or []
        matched = [c for c in comps if isinstance(c, dict) and c.get("text") == pre[k]]
        if not matched:
            status[k] = "no_original_completion"
        else:
            flagged = any(bool(c.get("flagged")) for c in matched)
            rs = {c.get("rating") for c in matched if c.get("rating") is not None}
            ys = {rating_to_y(r) for r in rs if r in (-1, 0, 1)}
            if flagged:
                status[k] = "flagged"
            elif not rs:
                status[k] = "unrated"
            elif len(ys) > 1 or any(r not in (-1, 0, 1) for r in rs):
                status[k] = "contradictory"
            else:
                yk[k] = ys.pop()
                if len(rs) == 1:
                    ratings[k] = next(iter(rs))
                    status[k] = "labeled"
                else:  # {0, +1}: y=0 either way, original rating ambiguous
                    status[k] = "labeled_rating_ambiguous"
        # Did the session continue on the ORIGINAL step k? If not, later
        # positions were labeled on another branch.
        adv = _advanced_text(st)
        if adv is not None and adv != pre[k]:
            on_branch = False
            counters["branch_diverged"] += 1

    y = [Y_UNKNOWN] * T
    mask = [False] * T
    for k in range(T):
        if status[k] in ("labeled", "labeled_rating_ambiguous"):
            y[k] = yk[k]
            mask[k] = True
    for s in status:
        counters[f"step_status_{s}"] += 1
    if not any(mask):
        counters["drop_no_trustworthy_label"] += 1
        return None
    first_error = next((k for k in range(T) if mask[k] and y[k] == 1), None)
    return {
        "problem": problem,
        "steps": list(pre),
        "y": y,
        "label_mask": mask,
        "ratings": ratings,
        "step_status": status,
        "first_error": first_error,
        "finish_reason": lab.get("finish_reason"),
        "source_record_ids": [record_id],
    }


def merge_duplicates(aligned: Iterable[dict], counters: Counter) -> list[dict]:
    """Collapse records of the same (problem, trajectory); mask conflicting labels.

    Agreeing labels are kept; a position labeled y=1 by one annotation and y=0 by
    another has no explicit adjudication in the source, so it is masked.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for a in aligned:
        groups[trajectory_key(a["problem"], a["steps"])].append(a)
    out = []
    for tk, recs in groups.items():
        base = recs[0]
        T = len(base["steps"])
        if len(recs) > 1:
            counters["dedup_groups"] += 1
            counters["dedup_records_collapsed"] += len(recs) - 1
        y, mask, ratings, status = [], [], [], []
        for k in range(T):
            ys = {r["y"][k] for r in recs if r["label_mask"][k]}
            rs = sorted({r["ratings"][k] for r in recs if r["label_mask"][k]}, key=str)
            if len(ys) == 1:
                y.append(ys.pop())
                mask.append(True)
                ratings.append(rs[0] if len(rs) == 1 else None)
                status.append("labeled" if len(recs) == 1 else "labeled_merged")
            elif len(ys) > 1:
                counters["dedup_conflicting_positions"] += 1
                y.append(Y_UNKNOWN)
                mask.append(False)
                ratings.append(None)
                status.append("conflict_across_annotations")
            else:
                y.append(Y_UNKNOWN)
                mask.append(False)
                ratings.append(None)
                status.append(base["step_status"][k])
        if not any(mask):
            counters["drop_all_labels_conflicting"] += 1
            continue
        first_error = next((k for k in range(T) if mask[k] and y[k] == 1), None)
        out.append({
            "trace_id": tk,
            "problem_key": problem_key(base["problem"]),
            "problem": base["problem"],
            "steps": base["steps"],
            "y": y,
            "label_mask": mask,
            "ratings": ratings,
            "step_status": status,
            "first_error": first_error,
            "finish_reasons": sorted({str(r["finish_reason"]) for r in recs}),
            "source_record_ids": sorted(i for r in recs for i in r["source_record_ids"]),
            "n_annotations": len(recs),
        })
    out.sort(key=lambda r: r["trace_id"])
    return out


def assign_splits(traces: list[dict], seed: int = 1729,
                  frac: tuple[float, float, float] = (0.8, 0.1, 0.1),
                  exclude_problem_keys: set[str] | None = None) -> dict[str, str]:
    """Problem-disjoint train/dev/calib/test, stratified by 'has an erroneous trace'.

    Returns problem_key -> split. Every trace of a problem lands in one split.
    Validation (frac[1]) is divided into equal dev and calibration halves with
    the same stratification. Excluded problems map to 'excluded_pb_overlap'.
    """
    exclude_problem_keys = exclude_problem_keys or set()
    has_err: dict[str, bool] = defaultdict(bool)
    for t in traces:
        has_err[t["problem_key"]] |= t["first_error"] is not None
    assignment: dict[str, str] = {}
    for pk in exclude_problem_keys & set(has_err):
        assignment[pk] = "excluded_pb_overlap"
    rng = random.Random(seed)
    for stratum in (True, False):
        pks = sorted(pk for pk, e in has_err.items() if e == stratum and pk not in assignment)
        rng.shuffle(pks)
        n = len(pks)
        n_tr = round(frac[0] * n)
        n_va = round(frac[1] * n)
        val = pks[n_tr:n_tr + n_va]
        for pk in pks[:n_tr]:
            assignment[pk] = "train"
        half = len(val) // 2
        for i, pk in enumerate(val):
            assignment[pk] = "dev" if i < half else "calib"
        for pk in pks[n_tr + n_va:]:
            assignment[pk] = "test"
    return assignment


def processbench_trace(row: dict, subset: str) -> dict:
    """A ProcessBench trace in the same schema. Labels follow the first-error
    convention: before the error correct, the error incorrect, later unknown;
    an error-free trace (label -1) is correct throughout."""
    steps = list(row["steps"])
    fe = int(row["label"])
    T = len(steps)
    y, mask = [], []
    for k in range(T):
        if fe < 0 or k < fe:
            y.append(0); mask.append(True)
        elif k == fe:
            y.append(1); mask.append(True)
        else:
            y.append(Y_UNKNOWN); mask.append(False)
    return {
        "trace_id": f"pb_{subset}_{row['id']}",
        "problem_key": problem_key(row["problem"]),
        "problem": row["problem"],
        "steps": steps,
        "y": y,
        "label_mask": mask,
        "ratings": [None] * T,
        "step_status": ["labeled" if m else "post_first_error_unknown" for m in mask],
        "first_error": fe if fe >= 0 else None,
        "pb_label": fe,
        "pb_subset": subset,
        "split": f"pb_{subset}",
        "source_record_ids": [str(row["id"])],
    }


# Public model inputs vs evaluation metadata. Encoders/probes read only these.
MODEL_INPUT_FIELDS = ("trace_id", "problem", "steps")
