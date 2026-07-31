"""Tests for the pcd encoder boundaries and the ProcessBench pcd derivation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from encode_prm800k_pcd import tokenize_pcd  # noqa: E402
from derive_pb_pcd_from_full_store import derive_subset  # noqa: E402

from src.repstore import TOKEN_SEQ, RepSpec, ShardedRepSplit, write_split


class StubTok:
    def __call__(self, text, add_special_tokens=False, truncation=False):
        ids = [(abs(hash(w)) % 997) + 3 for w in text.split()]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


def test_tokenize_pcd_boundaries():
    tok = StubTok()
    ids, cs, cl, ns, nl = tokenize_pcd(tok, "the problem", "prior step here",
                                       "current step", "next generated step")
    assert ids[cs:cs + cl] == tok("current step")["input_ids"]
    assert ids[ns:ns + nl] == tok("next generated step")["input_ids"]
    assert cs - 1 >= 0                      # a pre-current boundary exists
    assert ns >= cs + cl                    # next starts after current (+ separator)


def test_tokenize_pcd_empty_next():
    tok = StubTok()
    ids, cs, cl, ns, nl = tokenize_pcd(tok, "p", "", "only current", "")
    assert nl == 0 and cl > 0               # empty future is allowed, current is not


def _synth_full_store(tmp_path, d=6, n_traces=8):
    spec = RepSpec(name="pbfull", kind=TOKEN_SEQ, dim=d, layer=-1, backbone="t",
                   readout="full_solution_tokens")
    items, labels, meta = [], [], []
    rng = np.random.default_rng(0)
    for gi in range(n_traces):
        ns = 4
        ss, se, cur = [], [], 1
        for _ in range(ns):
            L = int(rng.integers(2, 4)); ss.append(cur); se.append(cur + L); cur += L + 1
        h = rng.standard_normal((cur, d)).astype(np.float32)
        items.append(h); labels.append(1)
        meta.append({"id": f"t-{gi}", "pb_subset": "s", "label": 1, "n_steps": ns,
                     "step_starts": ss, "step_ends": se, "n_tokens": cur, "global_index": gi})
    write_split(tmp_path / "s" / "shard_00", items, labels, meta, spec)
    return ShardedRepSplit(tmp_path / "s")


def test_derive_pb_pcd_shape_and_last_step_zero_delta(tmp_path):
    store = _synth_full_store(tmp_path)
    h, meta = derive_subset(store)
    d = store.spec.dim
    assert h.shape == (8 * 4, 3 * d)                 # 8 traces x 4 steps, 3*d
    # last step of each trace has no future -> final d-block (delta) is zero
    last = [i for i, m in enumerate(meta) if m["step_idx"] == m["n_steps"] - 1]
    assert last and np.allclose(h[last, 2 * d:3 * d], 0.0)
    # a non-last step should generally have a nonzero delta block
    nonlast = [i for i, m in enumerate(meta) if m["step_idx"] == 0]
    assert not np.allclose(h[nonlast, 2 * d:3 * d], 0.0)
