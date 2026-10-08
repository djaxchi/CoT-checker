"""ReProbe-released Qwen3-8B trajectories with all-step LLM-judge labels.

bidirectional_token_probe_v2 (docs/bidirectional_token_probe_v2_plan.md, section 4).

Source rows (HF `rediska0123/train_prm800k_Qwen3-8B_finished`, DeepSeek-R1 labels;
`JingweiNi/train_prm800k_Qwen3-8B_finished_self_annotate`, Qwen3-8B self labels):
`question`, `answer` (ground-truth solution), `reply` ("- Step k: ..." one step per
line), `claims` (per-step `sentence` + token alignment) and `verified`, where
1 = INCORRECT, 0 = correct, NaN = unlabeled (ReProbe utils/step_fact_check.py).

Steps are the claim sentences with the "- Step k:" marker stripped. Labels map
verified 1 -> y=1, 0 -> y=0; NaN, empty alignments and length mismatches are
masked. A final step cut mid-sentence by the generation length cap (reply has no
trailing newline and no answer line) is masked; the rest of an unfinished trace
is kept.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from src.data.prm_trajectories import Y_UNKNOWN, problem_key, trajectory_key

STEP_PREFIX = re.compile(r"^\s*-\s*Step\s+\d+\s*:\s*")
ANSWER_LINE = re.compile(r"^\s*<\s*Answer\s*>", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def strip_step(sentence: str) -> str:
    return STEP_PREFIX.sub("", sentence).strip()


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def parse_row(row: dict, row_id: str, counters: Counter) -> dict | None:
    """One released row -> trajectory dict (v1 schema) or None (counted)."""
    q, reply, claims, ver = row.get("question"), row.get("reply"), row.get("claims"), row.get("verified")
    if not isinstance(q, str) or not q.strip() or not isinstance(reply, str):
        counters["drop_malformed"] += 1
        return None
    claims = list(claims) if claims is not None else []
    ver = list(ver) if ver is not None else []
    if not claims:
        counters["drop_no_claims"] += 1
        return None
    steps = [strip_step(c["sentence"]) for c in claims]
    if any(not s for s in steps):
        counters["drop_empty_step"] += 1
        return None
    # every step must be found, in order, among the reply's non-empty lines
    lines = [_norm(strip_step(l)) for l in reply.split("\n") if l.strip()]
    j = 0
    for s in steps:
        ns = _norm(s)
        while j < len(lines) and not (lines[j] == ns or lines[j].startswith(ns) or ns.startswith(lines[j])):
            j += 1
        if j == len(lines):
            counters["drop_unaligned"] += 1
            return None
        j += 1
    T = len(steps)
    y, mask, status = [Y_UNKNOWN] * T, [False] * T, ["unlabeled"] * T
    if len(ver) != T:
        counters["traces_label_length_mismatch"] += 1
        status = ["label_length_mismatch"] * T
    else:
        for k, (c, v) in enumerate(zip(claims, ver)):
            if _isnan(v):
                status[k] = "nan"
            elif c.get("aligned_token_ids") is None or len(c.get("aligned_token_ids")) == 0:
                status[k] = "no_alignment"
            elif v in (0, 0.0, 1, 1.0):
                y[k], mask[k], status[k] = int(v), True, "labeled"
            else:
                status[k] = "invalid"
    finished = any(ANSWER_LINE.match(l) for l in reply.split("\n"))
    if not finished and not reply.endswith("\n") and mask[-1]:
        mask[-1], y[-1], status[-1] = False, Y_UNKNOWN, "truncated_final_step"
        counters["masked_truncated_final_step"] += 1
    for s in status:
        counters[f"step_status_{s}"] += 1
    if not any(mask):
        counters["drop_no_label"] += 1
        return None
    fe = next((k for k in range(T) if mask[k] and y[k] == 1), None)
    return {"problem": q, "steps": steps, "y": y, "label_mask": mask, "step_status": status,
            "first_error": fe, "finished": finished, "source_record_ids": [row_id],
            "gt_solution": row.get("answer")}


def merge_duplicates(rows: list[dict], counters: Counter) -> list[dict]:
    """Identical (problem, steps) collapse; agreeing labels kept, conflicts masked."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(trajectory_key(r["problem"], r["steps"]), []).append(r)
    out = []
    for tk, rs in sorted(groups.items()):
        b = rs[0]
        T = len(b["steps"])
        y, mask = [], []
        for k in range(T):
            ys = {r["y"][k] for r in rs if r["label_mask"][k]}
            if len(ys) == 1:
                y.append(ys.pop()); mask.append(True)
            else:
                if len(ys) > 1:
                    counters["dedup_conflicting_positions"] += 1
                y.append(Y_UNKNOWN); mask.append(False)
        if len(rs) > 1:
            counters["dedup_records_collapsed"] += len(rs) - 1
        if not any(mask):
            continue
        out.append({**b, "trace_id": "rp_" + tk[3:], "problem_key": problem_key(b["problem"]),
                    "y": y, "label_mask": mask,
                    "first_error": next((k for k in range(T) if mask[k] and y[k] == 1), None),
                    "source_record_ids": sorted(i for r in rs for i in r["source_record_ids"]),
                    "n_annotations": len(rs)})
    return out


def post_error_slices(y: list[int], mask: list[bool]) -> tuple[list[bool], list[bool]]:
    """(pre, post) masks over labeled steps: pre = labeled steps up to and
    including the first labeled error (all labeled steps if none); post = labeled
    steps strictly after it."""
    fe = next((k for k, (v, m) in enumerate(zip(y, mask)) if m and v == 1), None)
    pre = [bool(m) and (fe is None or k <= fe) for k, m in enumerate(mask)]
    post = [bool(m) and fe is not None and k > fe for k, m in enumerate(mask)]
    return pre, post
