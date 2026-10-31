"""ReProbe's step-level test sets, recovered from its released LM-Polygraph managers.

reprobe_comparison_v1 (docs/reprobe_comparison_v1_plan.md).

ReProbe (arXiv 2511.06209) evaluates on Qwen3-8B traces it sampled itself. The
released `ue_manager.pth` files hold, per trace, the prompt (`stats.input_texts`),
the sampled reply (`stats.greedy_texts`, `stats.greedy_tokens`) and, flattened over
all traces, the DeepSeek-R1 step labels (`gen_metrics[('claim',
'StepFactCheck_correctness')]`, 1 = incorrect, NaN = unlabeled) and every
estimator's step scores (`estimations[('claim', name)]`). Steps are not stored; the
per-trace split comes from re-running ReProbe's `StepsExtractor`
(`synthetic_dataset_generation/utils/steps_extractor.py`), ported here line for
line, and the step count must equal the label count.

The managers are torch zip archives whose pickle only references numpy, list,
defaultdict and codecs; `load_manager` unpickles them with an allowlist instead of
`torch.load`, so nothing else can execute.
"""

from __future__ import annotations

import codecs
import collections
import io
import pickle
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np

SKIP_STARTS = ("Reasoning Steps:", "SOLUTION:", "<start of response>", "<end of response>")
LABEL_KEY = ("claim", "StepFactCheck_correctness")
RP_AL_KEY = ("claim", "LuqClaimEstimatorDummy_claim")

try:
    from numpy._core.multiarray import scalar as _SCALAR
except ImportError:  # numpy < 2
    from numpy.core.multiarray import scalar as _SCALAR
_ALLOWED = {
    ("numpy", "dtype"): np.dtype,
    ("__builtin__", "list"): list,
    ("builtins", "list"): list,
    ("_codecs", "encode"): codecs.encode,
    ("collections", "defaultdict"): collections.defaultdict,
    ("numpy.core.multiarray", "scalar"): _SCALAR,
    ("numpy._core.multiarray", "scalar"): _SCALAR,
}


class Claim:
    """Inert stand-in for lm_polygraph's Claim (claim_text, sentence, aligned_token_ids);
    the later managers (Qwen3-32B, Phi-4) pickle steps as these objects."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


_ALLOWED[("lm_polygraph.stat_calculators.extract_claims", "Claim")] = Claim


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return _ALLOWED[(module, name)]
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError(f"blocked persistent id {pid!r}")


def load_manager(path: Path) -> dict:
    """An LM-Polygraph `ue_manager.pth` as plain Python and numpy objects."""
    z = zipfile.ZipFile(path)
    pk = [n for n in z.namelist() if n.endswith("data.pkl")]
    if len(pk) != 1:
        raise ValueError(f"{path}: expected one data.pkl, found {pk}")
    return _Unpickler(io.BytesIO(z.read(pk[0]))).load()


def _keep(claim_text: str, skip_starts) -> bool:
    s = claim_text.strip()
    return len(s) > 0 and not any(s.lower().startswith(b.lower()) for b in skip_starts)


def split_to_steps(text: str, tokens: list[int], decode: Callable[[list[int]], str],
                   seps: str = "\n", skip_starts=SKIP_STARTS) -> list[dict]:
    """ReProbe's `StepsExtractor.split_to_steps`, same loop and the same quirks.

    Returns [{"sentence", "start", "end", "aligned_token_ids"}], where start:end is the
    sentence's character span in `text` and aligned_token_ids index `tokens` (the
    reply's own tokens). Decoded prefixes are cached; the loop is otherwise unchanged.
    """
    if not decode(tokens).startswith(text):
        return []
    cache: dict[int, str] = {}

    def dec(n: int) -> str:
        if n not in cache:
            cache[n] = decode(tokens[:n])
        return cache[n]

    prev_token_i, token_i, prev_text_i = 0, 0, 0
    out = []
    for text_i in range(len(text)):
        if text[text_i] in seps and _keep(text[prev_text_i:text_i + 1], skip_starts):
            out.append({"sentence": text[prev_text_i:text_i + 1], "start": prev_text_i, "end": text_i + 1,
                        "aligned_token_ids": list(range(prev_token_i, min(token_i + 1, len(tokens) - 1)))})
        while token_i < len(tokens) and dec(token_i + 1) in text[:text_i + 1]:
            token_i += 1
        if text[text_i] in seps:
            prev_text_i = text_i + 1
            prev_token_i = token_i
    if _keep(text[prev_text_i:], skip_starts):
        out.append({"sentence": text[prev_text_i:], "start": prev_text_i, "end": len(text),
                    "aligned_token_ids": list(range(prev_token_i, min(token_i + 1, len(tokens) - 1)))})
    return out


def step_item(reply: str, step: dict) -> tuple[str, str]:
    """(prefix, step) for the read token: prefix + "\\n" + step == the reply up to the
    end of the step's line (its trailing newline dropped), so the read token reads the
    reply's own text, blank and skipped lines included."""
    body = reply[step["start"]:step["end"]]
    body = body[:-1] if body.endswith("\n") else body
    before = reply[:step["start"]]
    if before and not before.endswith("\n"):
        raise ValueError("a step must start at the beginning of a line")
    return before[:-1] if before else "", body


def _claim_steps(reply: str, claims: list) -> list[dict] | None:
    """The manager's own stored steps (lm_polygraph claims) as split_to_steps rows; None
    if a claim's sentence is not found in order in the reply."""
    out, pos = [], 0
    for c in claims:
        c = vars(c) if hasattr(c, "__dict__") else c
        i = reply.find(c["sentence"], pos)
        if i < 0:
            return None
        pos = i + len(c["sentence"])
        out.append({"sentence": c["sentence"], "start": i, "end": pos,
                    "aligned_token_ids": [int(t) for t in c["aligned_token_ids"]]})
    return out


def manager_traces(m: dict, decode: Callable[[list[int]], str], name: str) -> list[dict]:
    """One dict per trace: prompt, reply, reply tokens, steps, labels (1 incorrect,
    0 correct, -1 unlabeled) and every stored estimator's step scores.

    Steps come from the StepsExtractor port. When its step count disagrees with the
    labels and the manager stores its own steps (`stats.claims`, the later managers),
    those are used instead for every trace of the set; "steps_from" says which."""
    st = m["stats"]
    labels = np.asarray(m["gen_metrics"][LABEL_KEY], dtype=float)
    scores = {k[1]: np.asarray(v, dtype=float) for k, v in m["estimations"].items() if k[0] == "claim"}
    rows = list(zip(st["input_texts"], st["greedy_texts"], st["greedy_tokens"]))
    split = [split_to_steps(reply, [int(t) for t in toks], decode) for _, reply, toks in rows]
    source = "port"
    if sum(len(x) for x in split) != len(labels) and "claims" in st:
        split = [_claim_steps(reply, cl) for (_, reply, _), cl in zip(rows, st["claims"])]
        if any(x is None for x in split):
            raise ValueError(f"{name}: a stored claim is not found in its reply")
        source = "claims"
    out, off = [], 0
    for i, ((prompt, reply, toks), steps) in enumerate(zip(rows, split)):
        toks = [int(t) for t in toks]
        n = len(steps)
        y = labels[off:off + n]
        tgt = st["target_texts"][i] if "target_texts" in st else None
        out.append({
            "trace_id": f"{name}_{i:04d}", "set": name, "prompt_text": prompt, "reply": reply,
            "reply_ids": toks, "target": tgt if isinstance(tgt, str) else (None if tgt is None else str(tgt)),
            "steps": [{"sentence": s["sentence"], "start": s["start"], "end": s["end"],
                       "aligned_token_ids": s["aligned_token_ids"]} for s in steps],
            "labels": [-1 if np.isnan(v) else int(v) for v in y],
            "scores": {k: [float(x) for x in v[off:off + n]] for k, v in scores.items() if len(v) == len(labels)},
            "steps_from": source,
        })
        off += n
    if off != len(labels):
        raise ValueError(f"{name}: {off} extracted steps against {len(labels)} labels")
    return out
