"""context_ablation_v3 views: exact token identity and identical probe layout across contexts."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3-8B tokenizer not cached")


PROBLEM = "Compute $2+3\\cdot 4$."
STEPS = ["First multiply: $3\\cdot 4 = 12$.", "Then add: $2 + 12 = 14$.", "# Answer\n\n14"]


def test_full_view_is_prefix_of_trajectory_and_layouts_match(tok):
    from build_context_views import N_HEADER, Q_HEADER, Tok, compact, view_steps
    from encode_trajectory_token_store import tokenize_solution, tokenize_view
    tk = Tok(tok)
    full_ids, fss, fse = tokenize_solution(tok, PROBLEM, STEPS)
    for i in range(len(STEPS)):
        layouts = {}
        for ctx in ("full", "prev1", "q", "none"):
            header = N_HEADER if ctx == "none" else Q_HEADER.format(p=PROBLEM)
            vs = view_steps(ctx, STEPS, i)
            ids, ss, se = tk.view(header, vs)
            # the fast path equals the encoder's verifier
            assert (ids, ss, se) == tokenize_view(tok, {"header": header, "steps": vs})
            keep, n, st_s, st_e = compact(ss, se)
            stored = [t for a, b in keep for t in ids[a:b]]
            assert len(stored) == n
            # current step's tokens are the last rows, identical across contexts
            assert stored[st_s[0]:st_e[0]] == ids[ss[-1]:se[-1]]
            if ctx == "full":
                assert ids == full_ids[:fse[i]] and ss == fss[:i + 1]  # causal prefix of the trajectory
            layouts[ctx] = (st_s[0], st_e[0], n)
        assert layouts["full"] == layouts["prev1"] == layouts["q"]
        assert layouts["none"][1] - layouts["none"][0] == layouts["q"][1] - layouts["q"][0]


def test_view_steps():
    from build_context_views import view_steps
    s = ["a", "b", "c", "d"]
    assert view_steps("full", s, 2) == ["a", "b", "c"]
    assert view_steps("prev1", s, 2) == ["b", "c"] and view_steps("prev1", s, 0) == ["a"]
    assert view_steps("q", s, 3) == ["d"] == view_steps("none", s, 3)
