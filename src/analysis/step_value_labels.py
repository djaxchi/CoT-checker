"""step_value_labels_v1 core: downstream risk of every step from shared cut-point values.

Plan: docs/step_value_labels_v1_plan.md.

A cut point c of a trace is the reply truncated before step line c (c = 0..n-1)
or before the answer line (c = n). Rollouts from cut c estimate the error rate
err(c) of the policy finishing from there. factorized_step_labels_v1 defined

    risk_diff(s) = err(A) - mean_j err(C_j)

where A continues after the real step s and C_j samples a step s from the same
prefix, then continues. A C_j rollout is a free rollout from cut s, and an A
rollout is a free rollout from cut s + 1, so in expectation

    risk(s) = err(s + 1) - err(s).

One set of k rollouts per cut labels every step of the trace, each cut shared by
two neighboring steps. This module holds the torch-free pieces: cut offsets,
per-step risk from per-cut errors, and the chunked multi-round sampler that the
labeler drives with a model-backed `gen` function.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from src.analysis.factorized_step_labels import STEP_LINE


def cut_offsets(step_spans: Sequence[Sequence[int]], answer_lo: int) -> list[int]:
    """Reply offsets of cuts 0..n: the start of each step line, then of the answer line.

    The step lines must tile the reply from 0 and end where the answer line starts,
    so cut s + 1 is exactly the end of step s.
    """
    spans = [tuple(s) for s in step_spans]
    if not spans or spans[0][0] != 0:
        raise ValueError("step lines must start at reply offset 0")
    for (_, hi), (lo, _) in zip(spans, spans[1:]):
        if hi != lo:
            raise ValueError("step lines must tile the reply")
    if spans[-1][1] != answer_lo:
        raise ValueError("the answer line must start where the last step line ends")
    return [lo for lo, _ in spans] + [answer_lo]


def parse_unfinished_row(input_ids: Sequence[int], prompt_ids: Sequence[int], claims: Sequence[dict],
                         decode: Callable[[list[int]], str], eos_id: int) -> tuple[dict | None, str]:
    """Complete step lines of a released row that hit ReProbe's 256-token cap.

    The counterpart of factorized_step_labels.parse_reprobe_row for traces without
    an answer line: the prompt ids must match, the reply must hold no EOS, and the
    step-line claims must tile the reply from 0, each decoding byte-identically to
    its sentence. Only lines ending in a newline count as complete; anything after
    the last complete line (a cut-off fragment) is dropped. The cut after the last
    complete line, `end`, plays the role of the answer line's start.
    """
    ids = [int(x) for x in input_ids]
    P = len(prompt_ids)
    if ids[:P] != [int(x) for x in prompt_ids]:
        return None, "prompt_mismatch"
    rep = ids[P:]
    if eos_id in rep:
        return None, "has_eos"
    spans, cidx, cursor = [], [], 0
    for k, c in enumerate(claims):
        al = c.get("aligned_token_ids")
        al = [] if al is None else [int(x) for x in al]
        sent = c.get("sentence") or ""
        if not STEP_LINE.match(sent) or not sent.endswith("\n"):
            break
        if not al or al != list(range(cursor, cursor + len(al))):
            return None, "claims_not_tiling"
        if decode(rep[al[0]:al[-1] + 1]) != sent:
            return None, "sentence_decode_mismatch"
        spans.append((al[0], al[-1] + 1)); cidx.append(k); cursor = al[-1] + 1
    if not spans:
        return None, "no_step_lines"
    return {"prompt_len": P, "reply_ids": rep[:cursor], "step_spans": spans, "step_claim_idx": cidx,
            "end": cursor, "n_dropped_tokens": len(rep) - cursor}, "ok"


def step_risks(err_by_cut: dict[int, Sequence[bool]], n_steps: int) -> list[dict]:
    """Per-step risk from per-cut rollout errors, for every step whose two cuts were sampled.

    risk = err(s + 1) - err(s). `se` is the plug-in binomial SE of that difference
    (0 when both cuts are unanimous, as the bootstrap SE of factorized_step_labels_v1
    was); `harmful` = risk > 2 se and se > 0, the v1 rule.
    """
    out = []
    for s in range(n_steps):
        if s not in err_by_cut or s + 1 not in err_by_cut:
            continue
        b, a = list(map(bool, err_by_cut[s])), list(map(bool, err_by_cut[s + 1]))
        if not a or not b:
            continue
        pb, pa = sum(b) / len(b), sum(a) / len(a)
        se = math.sqrt(pb * (1 - pb) / len(b) + pa * (1 - pa) / len(a))
        risk = pa - pb
        out.append({"step": s, "err_before": pb, "err_after": pa, "k_before": len(b), "k_after": len(a),
                    "risk": risk, "se": se, "harmful": bool(se > 0 and risk > 2 * se)})
    return out


def trim_after_eos(ids: list[int], eos_ids: set[int]) -> tuple[list[int], bool]:
    """Generated ids through the first EOS (kept); Qwen3's pad id is an EOS id."""
    for j, t in enumerate(ids):
        if t in eos_ids:
            return ids[:j + 1], True
    return ids, False


def sample_rounds(gen: Callable[[list[list[int]], int], list[list[int]]],
                  contexts: Sequence[list[int]], budgets: Sequence[int], eos_ids: set[int],
                  chunk: int, batch: int, max_batch_tokens: int | None = None) -> list[dict]:
    """Sample one continuation per context, in rounds of at most `chunk` new tokens.

    `gen(ctxs, max_new)` returns one list of new ids per context (trailing padding
    allowed). After each round, rows that ended (EOS) or used their budget are done;
    the rest go into the next round with their tokens so far appended to the
    context. Sampling is autoregressive on the same ids, so a split continuation has
    the distribution of an unsplit one, and one runaway row no longer holds a whole
    batch to the reply cap. Rows are batched by context length, at most `batch` rows
    and, when set, at most `max_batch_tokens` padded tokens (rows x (longest context
    + step)) per call. With many rows per call (several traces), each round's batches
    stay full while the rows still running shrink, which is where the speed comes from.

    Returns per row: ids (through the EOS if any), eos, truncated (no EOS within the
    budget).
    """
    n = len(contexts)
    out: list[list[int]] = [[] for _ in range(n)]
    eos = [False] * n
    done = [b <= 0 for b in budgets]
    while True:
        todo = sorted((i for i in range(n) if not done[i]), key=lambda i: len(contexts[i]) + len(out[i]))
        if not todo:
            break
        for group in _batches(todo, [len(contexts[i]) + len(out[i]) for i in todo], batch,
                              max_batch_tokens, chunk):
            room = {i: budgets[i] - len(out[i]) for i in group}
            step = min(chunk, max(room.values()))
            gens = gen([list(contexts[i]) + out[i] for i in group], step)
            for i, g in zip(group, gens):
                g, hit = trim_after_eos(list(g), eos_ids)
                if hit and len(g) <= room[i]:
                    out[i] += g; eos[i] = True; done[i] = True
                elif len(g) >= room[i]:
                    out[i] += g[:room[i]]; done[i] = True
                elif not g:
                    done[i] = True                  # nothing generated and no EOS: give up
                else:
                    out[i] += g
    return [{"ids": out[i], "eos": eos[i], "truncated": not eos[i]} for i in range(n)]


def _batches(order: list[int], lens: list[int], batch: int, max_tokens: int | None, chunk: int) -> list[list[int]]:
    """Consecutive rows of `order` (sorted by length) into batches under both caps."""
    out, cur, longest = [], [], 0
    for i, L in zip(order, lens):
        lg = max(longest, L)
        if cur and (len(cur) == batch or (max_tokens and (len(cur) + 1) * (lg + chunk) > max_tokens)):
            out.append(cur); cur, lg = [], L
        cur.append(i); longest = lg
    if cur:
        out.append(cur)
    return out
