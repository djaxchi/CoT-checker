"""Tests for the full-solution encoder boundaries and the lookahead ceiling probe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from encode_processbench_full_store import tokenize_solution  # noqa: E402
from analysis.lookahead_ceiling import build_trace_examples, cv_eval  # noqa: E402

from src.repstore import TOKEN_SEQ, RepSpec, ShardedRepSplit, write_split


class StubTok:
    """Whitespace tokenizer: each word -> a stable id, optional BOS."""

    def __call__(self, text, add_special_tokens=False, truncation=False):
        ids = [(abs(hash(w)) % 997) + 3 for w in text.split()]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


def test_tokenize_solution_boundaries_are_exact():
    tok = StubTok()
    steps = ["alpha beta", "gamma", "delta epsilon zeta"]
    ids, ss, se = tokenize_solution(tok, "the problem", steps)
    assert len(ss) == len(se) == 3
    for j, step in enumerate(steps):
        # the [start, end) slice must equal the step tokenized on its own
        assert ids[ss[j]:se[j]] == tok(step)["input_ids"]
        if j + 1 < len(steps):
            assert se[j] <= ss[j + 1]           # non-overlapping, ordered
        assert ss[j] - 1 >= 0                    # a real pre-step boundary exists


def _synth_store(tmp_path, d=6, n_traces=40):
    spec = RepSpec(name="pbfull", kind=TOKEN_SEQ, dim=d, layer=-1, backbone="t",
                   readout="full_solution_tokens")
    items, labels, meta = [], [], []
    rng = np.random.default_rng(0)
    for gi in range(n_traces):
        ns = rng.integers(3, 6)
        lens = rng.integers(2, 5, size=ns)
        # boundaries with a 1-token separator between steps
        ss, se, cur = [], [], 1  # leave row 0 as a prompt/boundary token
        for L in lens:
            ss.append(cur); se.append(cur + int(L)); cur += int(L) + 1
        h = rng.standard_normal((cur, d)).astype(np.float32)
        label = int(rng.integers(0, ns)) if gi % 2 == 0 else -1
        # plant future-only signal: after the first error, shift downstream rows
        if label != -1:
            h[se[label]:] += 2.0
        items.append(h)
        labels.append(label)
        meta.append({"id": f"t-{gi}", "pb_subset": "s", "label": label,
                     "n_steps": int(ns), "step_starts": ss, "step_ends": se,
                     "n_tokens": cur, "global_index": gi})
    write_split(tmp_path / "s" / "shard_00", items, labels, meta, spec)
    return ShardedRepSplit(tmp_path / "s")


def test_build_and_cv_runs(tmp_path):
    store = _synth_store(tmp_path)
    traces = build_trace_examples(store, W=1)
    keys = {"cur", "pc", "pcf", "pcd", "pcd2", "label"}
    assert traces and all(set(st) == keys for _, steps in traces for st in steps)
    # every representation must produce a valid AUROC in [0, 1]
    for key in ("cur", "pcf", "pcd", "pcd2"):
        auc, f1 = cv_eval(traces, key, k_folds=4)
        assert 0.0 <= auc <= 1.0 and 0.0 <= f1 <= 1.0
