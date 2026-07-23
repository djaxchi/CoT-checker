"""Tests for latent_memory_v0 oracle core: pure helpers + a tiny stub-LM grad/opt path."""

from __future__ import annotations

import torch

from src.analysis.latent_memory import (
    OracleResult,
    belief_masses,
    candidate_scores_grad,
    chunk_pool_states,
    donor_win,
    full_cot_context_ids,
    gold_margin_t,
    joint_candidate_texts,
    latent_context_ids,
    no_cot_context_ids,
    optimize_latent,
    random_like,
    recovery,
)
from src.analysis.transition_operator import SEP_TOKEN_ID


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_latent_context_ids_span_and_layout():
    q = [5, 6, 7]
    full, lo, hi = latent_context_ids(q, m=4, placeholder_id=0)
    assert full == [5, 6, 7, SEP_TOKEN_ID, 0, 0, 0, 0, SEP_TOKEN_ID]
    assert (lo, hi) == (4, 8)
    assert hi - lo == 4
    assert all(full[p] == 0 for p in range(lo, hi))


def test_full_and_no_cot_context():
    q = [1, 2]
    assert no_cot_context_ids(q) == [1, 2, SEP_TOKEN_ID]
    full = full_cot_context_ids(q, [[9], [8, 7]])
    assert full == [1, 2, SEP_TOKEN_ID, 9, SEP_TOKEN_ID, 8, 7, SEP_TOKEN_ID]


def test_chunk_pool_mean_and_max_shapes():
    t, d = 10, 6
    states = torch.arange(t * d, dtype=torch.float32).reshape(t, d)
    for mode in ("mean", "max"):
        pooled = chunk_pool_states(states, m=4, mode=mode)
        assert pooled.shape == (4, d)
    # mean of the first chunk (rows 0..1 for m=4, T=10 -> bounds 0,2,5,7,10)
    mean_pooled = chunk_pool_states(states, m=4, mode="mean")
    assert torch.allclose(mean_pooled[0], states[0:2].mean(0))
    max_pooled = chunk_pool_states(states, m=4, mode="max")
    assert torch.allclose(max_pooled[0], states[0:2].amax(0))


def test_chunk_pool_fewer_tokens_than_m():
    states = torch.randn(2, 5)
    pooled = chunk_pool_states(states, m=8, mode="mean")
    assert pooled.shape == (8, 5)
    assert torch.isfinite(pooled).all()


def test_recovery_formula_and_degenerate():
    assert abs(recovery(0.5, 0.0, 1.0) - 0.5) < 1e-9
    assert abs(recovery(0.9, 0.1, 0.9) - 1.0) < 1e-9
    import math
    assert math.isnan(recovery(0.5, 0.5, 0.5))


def test_joint_candidate_texts_layout():
    texts, ib, ia = joint_candidate_texts("5", "7", ["9", "5"], ["11", "7"], k=6)
    assert ib == 0 and ia == 1
    assert texts[0] == "5" and texts[1] == "7"
    assert "9" in texts and "11" in texts
    assert len(texts) == len(set(texts))  # deduped


def test_joint_candidate_texts_same_gold_flagged():
    texts, ib, ia = joint_candidate_texts("5", "5", ["9"], ["8"], k=6)
    assert ib == ia == 0  # unusable pair: golds collide


def test_belief_masses_and_donor_win():
    scores = [2.0, 0.0, -1.0]  # recipient gold strongly preferred
    mb, ma = belief_masses(scores, idx_b=0, idx_a=1)
    assert mb > ma
    assert not donor_win(ma, mb)
    assert donor_win(0.9, 0.1)


def test_random_like_matches_norm():
    s = torch.randn(3, 8) * 5.0
    r = random_like(s, seed=1)
    assert r.shape == s.shape
    assert torch.allclose(r.norm(dim=-1), s.norm(dim=-1), atol=1e-4)


def test_extract_ints_and_probe_target():
    from src.analysis.latent_memory import extract_ints, pick_probe_target
    assert extract_ints("we get 42 and -7 then 100") == ["42", "-7", "100"]
    steps = ["start with x", "compute 5 times 7", "so we have 350 here",
             "divide by 2 giving 175", "the answer is 88", "final: 88"]
    # question mentions 5 and 7; answer 88 -> excluded. 350 is largest qualifying.
    tgt = pick_probe_target("given 5 and 7", steps, "88")
    assert tgt is not None
    assert tgt[0] == "350"
    assert 1 <= tgt[1] <= len(steps) - 3


def test_probe_target_none_when_too_short():
    from src.analysis.latent_memory import pick_probe_target
    assert pick_probe_target("q", ["a", "b", "c"], "5") is None


def test_pick_probe_targets_multi_distinguishable():
    from src.analysis.latent_memory import pick_probe_targets
    steps = ["intro line", "the discriminant is 49 here", "next we halve it",
             "the midpoint value is 175 exactly", "so the answer is 88", "final 88"]
    tgts = pick_probe_targets("no numbers here", steps, "88", n_targets=3)
    golds = {t["gold"] for t in tgts}
    assert "49" in golds and "175" in golds       # both intermediates found
    assert len({t["step_idx"] for t in tgts}) == len(tgts)   # distinct steps
    for t in tgts:
        assert t["cue"] and any(c.isalpha() for c in t["cue"])  # non-empty textual cue
    # cue anchors to text before the number
    d = next(t for t in tgts if t["gold"] == "49")
    assert d["cue"].endswith("discriminant is")


def test_optimize_latent_multi_serves_two_targets():
    model = _fixture()
    q = [3, 4, 5, 6]
    lat_ctx, lo, hi = latent_context_ids(q, m=4, placeholder_id=0)

    def teacher_for(steps, suffix, cands):
        ctx = full_cot_context_ids(q, steps) + suffix
        with torch.no_grad():
            s = candidate_scores_grad(model, ctx, cands, pad_id=0, device="cpu",
                                      layer=2, lo=0, hi=0, states=torch.zeros(0, 8))
        return torch.softmax(s, dim=-1)

    t1 = {"suffix_ids": [7], "cand_ids": [[10, 11], [12], [13, 14]]}
    t2 = {"suffix_ids": [9], "cand_ids": [[15], [16, 17], [18]]}
    t1["teacher_belief"] = teacher_for([[20, 21], [22]], t1["suffix_ids"], t1["cand_ids"])
    t2["teacher_belief"] = teacher_for([[23, 24], [25]], t2["suffix_ids"], t2["cand_ids"])

    from src.analysis.latent_memory import optimize_latent_multi
    res = optimize_latent_multi(model, lat_ctx, lo, hi, layer=2,
                                targets=[t1, t2], init_states=torch.zeros(4, 8),
                                pad_id=0, device="cpu", epochs=120, lr=1e-1)
    assert res["loss_history"][-1] < res["loss_history"][0]
    assert len(res["margins"]) == 2
    # both targets' student beliefs approach their teachers
    for tgt, scores in zip([t1, t2], res["scores"]):
        sb = torch.softmax(torch.tensor(scores), dim=-1)
        assert float((sb - tgt["teacher_belief"]).abs().max()) < 0.2


def test_gold_margin_t_differentiable():
    s = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
    m = gold_margin_t(s)
    assert abs(float(m.detach()) - 1.0) < 1e-6
    m.backward()
    assert s.grad is not None


# ---------------------------------------------------------------------------
# Tiny stub LM exercising the real hook + grad + optimiser path
# ---------------------------------------------------------------------------

class _Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, hidden):
        # Causal cumulative-mean mixing so each position depends on all earlier ones
        # (a stand-in for attention): this lets a patched latent influence downstream
        # candidate positions. Right-padding is safe (pads sit after real tokens).
        cs = hidden.cumsum(1)
        idx = torch.arange(1, hidden.shape[1] + 1, device=hidden.device).float()
        cummean = cs / idx[None, :, None]
        # HF-style: return a tuple so make_span_patch_hook exercises the tuple branch.
        return (hidden + torch.tanh(self.lin(cummean)),)


class _Inner(torch.nn.Module):
    def __init__(self, vocab, d, n_layers):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, d)
        self.layers = torch.nn.ModuleList(_Block(d) for _ in range(n_layers))


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _TinyLM(torch.nn.Module):
    """Minimal causal-LM shape: .model.layers hookable, __call__ returns .logits."""

    def __init__(self, vocab=256, d=8, n_layers=3):  # > SEP id 198
        super().__init__()
        self.model = _Inner(vocab, d, n_layers)
        self.lm_head = torch.nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None, **kw):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        return _Out(self.lm_head(h))

    __call__ = forward


def _fixture():
    torch.manual_seed(0)
    model = _TinyLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def test_candidate_scores_grad_flows_to_latent():
    model = _fixture()
    q = [3, 4, 5]
    full, lo, hi = latent_context_ids(q, m=3, placeholder_id=0)
    ctx = full + [6]  # a stand-in elicitation suffix token
    cands = [[10, 11], [12], [13, 14, 15]]
    z = torch.zeros(hi - lo, 8, requires_grad=True)
    scores = candidate_scores_grad(model, ctx, cands, pad_id=0, device="cpu",
                                   layer=2, lo=lo, hi=hi, states=z)
    assert scores.shape == (3,)
    assert scores.requires_grad
    scores.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_candidate_scores_grad_layer0_embedding_injection():
    model = _fixture()
    q = [3, 4, 5]
    full, lo, hi = latent_context_ids(q, m=2, placeholder_id=0)
    ctx = full + [6]
    cands = [[10, 11], [12]]
    z = torch.zeros(hi - lo, 8, requires_grad=True)
    scores = candidate_scores_grad(model, ctx, cands, pad_id=0, device="cpu",
                                   layer=0, lo=lo, hi=hi, states=z)
    scores.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_optimize_latent_reduces_loss_and_moves_margin():
    model = _fixture()
    q = [3, 4, 5, 6]
    suffix = [7]
    cands = [[10, 11], [12], [13, 14]]  # gold = index 0

    # Teacher belief: a full-CoT-style context scored through the same frozen stub.
    teacher_ctx = full_cot_context_ids(q, [[20, 21], [22]]) + suffix
    with torch.no_grad():
        t_scores = candidate_scores_grad(
            model, teacher_ctx, cands, pad_id=0, device="cpu",
            layer=2, lo=0, hi=0, states=torch.zeros(0, 8))
    teacher_belief = torch.softmax(t_scores, dim=-1)

    full, lo, hi = latent_context_ids(q, m=4, placeholder_id=0)
    init = torch.zeros(hi - lo, 8)
    res = optimize_latent(model, full, lo, hi, layer=2, cand_ids_list=cands,
                          suffix_ids=suffix, teacher_belief=teacher_belief,
                          init_states=init, pad_id=0, device="cpu", epochs=80, lr=1e-1)
    assert isinstance(res, OracleResult)
    assert res.z.shape == (4, 8)
    assert res.loss_history[-1] < res.loss_history[0]
    # optimised student belief should approach the teacher belief
    student_belief = torch.softmax(torch.tensor(res.scores), dim=-1)
    assert float((student_belief - teacher_belief).abs().max()) < 0.15
