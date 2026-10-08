"""judge_prompt_v1 encoding: the verdict question must not leak into the step span.

Under the judge style the input is prefix + step + verdict question. The span
store must still hold exactly the boundary row plus the step's own rows (so every
step readout reads the step, not the question), and the judge store must hold
the boundary row and the last question token, with the Yes/No logits taken at
that same position. A stub model whose state is a function of the token id makes
the rows directly comparable across styles.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from encode_prm800k_token_store import encode_split, tokenize_with_offsets  # noqa: E402
from encode_processbench_token_store import encode_subset  # noqa: E402

from scripts.derive_delta_from_token_store import derive_split  # noqa: E402
from src.onpolicy.prompts import JUDGE_ANSWERS, JUDGE_SUFFIX, build_prefix  # noqa: E402
from src.repstore.judge_store import answer_ids  # noqa: E402
from src.repstore.store import RepSplit  # noqa: E402

D, V = 8, 4096


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False):
        base = 2 if add_special_tokens else 0
        n = base + max(1, len(text) // 3)
        return {"input_ids": [(sum(map(ord, text)) * 7 % 900) + i + 1 for i in range(n)]}


def stub_states(ids: torch.Tensor) -> torch.Tensor:
    feat = torch.arange(D, dtype=torch.float32).view(1, 1, D)
    return torch.sin(ids.to(torch.float32).unsqueeze(-1) * 0.001 + feat)


def stub_logits(ids: torch.Tensor) -> torch.Tensor:
    vocab = torch.arange(V, dtype=torch.float32).view(1, 1, V)
    return torch.cos(ids.to(torch.float32).unsqueeze(-1) * 0.01 + vocab * 0.1)


class StubModel:
    class _C:
        hidden_size = D

    config = _C()

    def __call__(self, inp, attention_mask=None, output_hidden_states=True, use_cache=False):
        return type("O", (), {"hidden_states": [stub_states(inp)],
                              "logits": stub_logits(inp)})()


def _rows(tmp_path: Path, n: int) -> Path:
    path = tmp_path / "prm800k_test.jsonl"
    with path.open("w") as f:
        for k in range(n):
            f.write(json.dumps({
                "uid": f"u{k}", "problem_id": f"p{k}", "solution_id": f"s{k}",
                "step_idx": 1, "label": k % 2, "rating": 1,
                "problem": "a long problem statement " * (3 + k),
                "prefix": "an earlier step " * (2 + k),
                "candidate_step": "the candidate step number " + "x" * k,
            }) + "\n")
    return path


def _encode(tmp_path, name, style):
    root = tmp_path / name / "step_spans"
    judge = tmp_path / name / "judge_token" if style == "judge" else None
    encode_split(_rows(tmp_path, 5), root, "test_2k", StubTokenizer(), StubModel(),
                 torch.device("cpu"), -1, 4096, 2, 0, 0, 1, "stub", None, True,
                 style, judge)
    return root, judge


def test_suffix_is_tokenized_after_the_step_and_not_counted_in_it():
    ex = json.loads(_rows_text())
    tok = StubTokenizer()
    ids_v, start_v, end_v = tokenize_with_offsets(tok, ex, 4096, "verifier")
    ids_j, start_j, end_j = tokenize_with_offsets(tok, ex, 4096, "judge")
    assert end_v == len(ids_v)
    assert end_j - start_j == end_v - start_v
    assert ids_j[start_j:end_j] == ids_v[start_v:end_v]
    assert ids_j[end_j:] == tok(JUDGE_SUFFIX, add_special_tokens=False)["input_ids"]


def _rows_text() -> str:
    return json.dumps({"problem": "p q r", "prefix": "earlier", "candidate_step": "the step"})


def test_judge_span_holds_only_the_step_rows(tmp_path):
    v, _ = _encode(tmp_path, "v", "verifier")
    j, _ = _encode(tmp_path, "j", "judge")
    sv, sj = RepSplit(v / "test_2k" / "shard_00"), RepSplit(j / "test_2k" / "shard_00")
    np.testing.assert_array_equal(sv.lengths, sj.lengths)
    for k in range(len(sv)):
        a = np.asarray(sv.h[sv.offsets[k] + 1:sv.offsets[k + 1]])
        b = np.asarray(sj.h[sj.offsets[k] + 1:sj.offsets[k + 1]])
        np.testing.assert_array_equal(a, b)   # step rows identical, boundary differs
    assert sj.spec.prompt_style == "judge"
    assert sv.spec.prompt_style == "verifier"


def test_judge_store_is_boundary_then_verdict_token(tmp_path):
    span, judge = _encode(tmp_path, "j2", "judge")
    sj = RepSplit(span / "test_2k" / "shard_00")
    jt = RepSplit(judge / "test_2k" / "shard_00")
    tok = StubTokenizer()
    yes, no = answer_ids(tok, JUDGE_ANSWERS)
    rows = [json.loads(l) for l in _rows(tmp_path, 5).read_text().splitlines()]
    np.testing.assert_array_equal(jt.lengths, np.full(5, 2))
    np.testing.assert_array_equal(jt.y, sj.y)
    for k, (m, ex) in enumerate(zip(jt.meta(), rows)):
        ids, _, _ = tokenize_with_offsets(tok, ex, 4096, "judge")
        last = torch.tensor([[ids[-1]]])
        np.testing.assert_array_equal(np.asarray(jt.h[2 * k]), np.asarray(sj.h[sj.offsets[k]]))
        np.testing.assert_allclose(np.asarray(jt.h[2 * k + 1], np.float32),
                                   stub_states(last)[0, 0].numpy(), atol=1e-3)
        lg = stub_logits(last)[0, 0]
        assert m["judge_logit_yes"] == pytest.approx(float(lg[yes]))
        assert m["judge_logit_no"] == pytest.approx(float(lg[no]))
        assert (m["n_tokens"], m["step_start_idx"], m["pre_step_boundary_idx"]) == (2, 1, 0)
        assert m["orig_n_tokens"] == len(ids)
    X, _, _ = derive_split(judge / "test_2k", "last", sort=True)
    np.testing.assert_array_equal(X[0], np.asarray(jt.h[1]))


def _pb_file(tmp_path: Path) -> Path:
    path = tmp_path / "pb.jsonl"
    steps = ["first step text", "second step " + "y" * 40, "third"]
    path.write_text(json.dumps({"id": "t0", "label": 1, "problem": "prob " * 4,
                                "steps": steps}) + "\n")
    return path


def test_pb_judge_keeps_the_verifier_item_set(tmp_path):
    """A cap that the judge prompt exceeds but the verifier prompt fits must not
    drop the step under judge, or the two stores would score different steps."""
    tok, raw = StubTokenizer(), _pb_file(tmp_path)
    tr = json.loads(raw.read_text())
    lens_v = [len(tok(build_prefix("verifier", tr["problem"], "\n\n".join(tr["steps"][:k])))["input_ids"])
              + len(tok(s, add_special_tokens=False)["input_ids"]) for k, s in enumerate(tr["steps"])]
    cap = max(lens_v)
    out = {}
    for style in ("verifier", "judge"):
        root = tmp_path / style / "pb_step_spans"
        judge = tmp_path / style / "pb_judge" if style == "judge" else None
        encode_subset(raw, "math", root, tok, StubModel(), torch.device("cpu"), -1, cap,
                      2, 0, 0, 1, "stub", True, style, False, 0, judge)
        out[style] = RepSplit(root / "math" / "shard_00")
    assert [m["step_idx"] for m in out["judge"].meta()] == [0, 1, 2]
    np.testing.assert_array_equal(out["judge"].lengths, out["verifier"].lengths)
    jt = RepSplit(tmp_path / "judge" / "pb_judge" / "math" / "shard_00")
    np.testing.assert_array_equal(jt.y, out["judge"].y)
    assert jt.y.tolist() == [0, 1, 0]
