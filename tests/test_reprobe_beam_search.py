import numpy as np

import scripts.reprobe_cmp.beam_search as bs


def test_beam_keeps_the_lowest_scores_and_returns_the_best_finished(monkeypatch):
    calls = []

    def fake_sample(model, tok, contexts, n, max_new, dev):
        calls.append((len(contexts), n))
        root = contexts[0].endswith("Reasoning Steps:\n")
        out = []
        for c in contexts:
            for j in range(n):
                out.append((f"- Step 1: s{j}" if root else f"<Answer>: {j}", False))
        return out

    def scorer(problem, trajs):
        # lower is better: the number written in the last step, answers first
        return np.array([float(t[-1].split()[-1].lstrip("s")) / 100 for t in trajs])

    monkeypatch.setattr(bs, "sample_steps", fake_sample)
    res = bs.search(None, None, scorer, "Q?", B=2, N=3, max_steps=5, max_new=8, dev=None)
    assert calls[0] == (1, 6)            # the root draws B x N
    assert calls[1] == (2, 3)            # then N per kept beam
    assert res["finished"] and res["steps"][0] in ("- Step 1: s0", "- Step 1: s1")
    assert res["steps"][-1] == "<Answer>: 0"


def test_is_final():
    assert bs.is_final("<Answer>: 3", False)
    assert bs.is_final("- Step 2: x", True)
    assert not bs.is_final("- Step 2: x", False)
