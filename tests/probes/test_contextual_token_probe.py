"""Mask contract and objective of the contextual token probe (plan sections 7-9, 13)."""

import numpy as np
import pytest
import torch

from src.probes.contextual_token_probe import (
    ARMS, PREFIX_OWNER, ROLE_BOUNDARY, ROLE_PREFIX, ROLE_TOKEN, ContextualTokenProbe,
    collate, gather_scores, target_weights, trace_rows, views_for, weighted_bce,
)

D = 16


def make_trace(seed=0, prefix=4, step_lens=(3, 2, 4, 3, 2), sep=1):
    rng = np.random.default_rng(seed)
    starts, ends = [], []
    pos = prefix
    for j, n in enumerate(step_lens):
        if j > 0:
            pos += sep
        starts.append(pos)
        pos += n
        ends.append(pos)
    H = rng.standard_normal((pos, D)).astype(np.float32)
    return {"H": H, "step_starts": starts, "step_ends": ends}


def model(seed=0):
    torch.manual_seed(seed)
    m = ContextualTokenProbe(d_in=D, d=32, layers=2, heads=4, ff=64, dropout=0.0)
    return m.eval()


def scores(m, traces, arm, targets=None):
    b = collate(traces, arm, targets)
    with torch.no_grad():
        out = gather_scores(b, m(b))
    return {(t, s): v for t, s, v in out}


def perturb_step(tr, j, scale=5.0, seed=99):
    """Replace every state owned by step j: its tokens and the separator/boundary
    source right before it (that state belongs to step j as its boundary)."""
    tr = {**tr, "H": tr["H"].copy()}
    s, e = tr["step_starts"][j], tr["step_ends"][j]
    rng = np.random.default_rng(seed)
    tr["H"][s - 1:e] = rng.standard_normal((e - s + 1, D)) * scale
    return tr


# ---------------------------------------------------------------- structure

def test_rows_boundary_ownership_and_positions():
    tr = make_trace()
    r = trace_rows(tr["step_starts"], tr["step_ends"])
    assert (r.owner[:4] == PREFIX_OWNER).all() and (r.role[:4] == ROLE_PREFIX).all()
    for j, (s, e) in enumerate(zip(tr["step_starts"], tr["step_ends"])):
        idx = np.nonzero(r.owner == j)[0]
        assert r.role[idx[0]] == ROLE_BOUNDARY
        assert r.src[idx[0]] == s - 1 and r.pos[idx[0]] == s - 1  # copied pre-step state, original position
        assert (r.role[idx[1:]] == ROLE_TOKEN).all()
        assert list(r.src[idx[1:]]) == list(range(s, e))
        assert r.q_pos[j] == e - 1
    # separators (between steps) are never token rows
    seps = {tr["step_ends"][j] for j in range(len(tr["step_ends"]) - 1)}
    tok_src = set(r.src[r.role == ROLE_TOKEN].tolist())
    assert not (seps & tok_src)


def test_future1_views_are_physically_cropped():
    tr = make_trace()
    r = trace_rows(tr["step_starts"], tr["step_ends"])
    views = views_for(r, 5, "future1", [0, 2, 4])
    assert [q for _, q in views] == [[0], [2], [4]]
    for (sel, (t,)) in views:
        assert r.owner[sel].max() == min(t + 1, 4)
    assert len(views_for(r, 5, "full", [0, 1, 2])) == 1


@pytest.mark.parametrize("arm", ARMS)
def test_scores_every_step_no_monotonicity(arm):
    m = model()
    tr = make_trace()
    sc = scores(m, [tr], arm)
    assert sorted(sc) == [(0, j) for j in range(5)]
    vals = [sc[(0, j)] for j in range(5)]
    assert all(np.isfinite(vals))
    # random weights: nothing forces a monotone sequence
    assert not (np.all(np.diff(vals) >= 0) or np.all(np.diff(vals) <= 0))


# ---------------------------------------------------------------- leakage

@pytest.mark.parametrize("arm", ["local", "causal"])
def test_later_step_cannot_change_causal_or_local(arm):
    m = model()
    tr = make_trace()
    base = scores(m, [tr], arm)
    for j in range(1, 5):
        pert = scores(m, [perturb_step(tr, j)], arm)
        for i in range(j):
            assert pert[(0, i)] == pytest.approx(base[(0, i)], abs=1e-6), (arm, i, j)


def test_prefix_rows_do_not_relay_future():
    """Perturb only the LAST step: if prefix rows could read it, layer-2 causal
    queries for step 0 would move."""
    m = model()
    tr = make_trace()
    base = scores(m, [tr], "causal")
    pert = scores(m, [perturb_step(tr, 4, scale=50.0)], "causal")
    assert pert[(0, 0)] == pytest.approx(base[(0, 0)], abs=1e-6)


def test_future1_ignores_beyond_next_step_across_two_layers():
    m = model()
    tr = make_trace()
    base = scores(m, [tr], "future1")
    for j in range(2, 5):
        pert = scores(m, [perturb_step(tr, j)], "future1")
        for i in range(j - 1):  # targets i with i + 1 < j
            assert pert[(0, i)] == pytest.approx(base[(0, i)], abs=1e-6), (i, j)


@pytest.mark.parametrize("arm", ["future1", "full"])
def test_next_step_reaches_treatment_arms(arm):
    m = model()
    tr = make_trace()
    base = scores(m, [tr], arm)
    pert = scores(m, [perturb_step(tr, 2)], arm)
    assert abs(pert[(0, 1)] - base[(0, 1)]) > 1e-4


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_sees_the_whole_current_step(arm):
    m = model()
    tr = make_trace()
    base = scores(m, [tr], arm)
    t2 = {**tr, "H": tr["H"].copy()}
    t2["H"][tr["step_ends"][2] - 1] = np.random.default_rng(5).standard_normal(D) * 5  # last token of step 2
    pert = scores(m, [t2], arm)
    assert abs(pert[(0, 2)] - base[(0, 2)]) > 1e-4


def test_local_ignores_earlier_steps_but_reads_prefix():
    m = model()
    tr = make_trace()
    base = scores(m, [tr], "local")
    pert = scores(m, [perturb_step(tr, 1)], "local")
    assert pert[(0, 3)] == pytest.approx(base[(0, 3)], abs=1e-6)
    t2 = {**tr, "H": tr["H"].copy()}
    t2["H"][0] = np.random.default_rng(6).standard_normal(D) * 5
    assert abs(scores(m, [t2], "local")[(0, 3)] - base[(0, 3)]) > 1e-4


# ---------------------------------------------------------------- batching

@pytest.mark.parametrize("arm", ARMS)
def test_padding_and_batch_composition_do_not_change_outputs(arm):
    m = model()
    a = make_trace(0)
    b = make_trace(1, prefix=7, step_lens=(5, 6, 2, 8, 3, 4, 2))
    c = make_trace(2, prefix=2, step_lens=(1, 1))
    alone = scores(m, [a], arm)
    mixed = scores(m, [c, b, a], arm)
    for (t, s), v in alone.items():
        assert mixed[(2, s)] == pytest.approx(v, abs=1e-5)


def test_extra_queries_do_not_change_other_queries():
    """Queries are never keys: scoring a subset gives the same values."""
    m = model()
    tr = make_trace()
    full = scores(m, [tr], "full")
    sub = scores(m, [tr], "full", targets=[[3]])
    assert sub[(0, 3)] == pytest.approx(full[(0, 3)], abs=1e-6)


# ---------------------------------------------------------------- objective

def test_chunked_loss_matches_per_trajectory_objective():
    torch.manual_seed(0)
    m = ContextualTokenProbe(d_in=D, d=32, layers=2, heads=4, ff=64, dropout=0.0)
    trs = [make_trace(0), make_trace(1, step_lens=(2, 2, 2)), make_trace(2, step_lens=(4, 1, 3, 2))]
    labels = [{0: 0, 1: 0, 2: 1}, {0: 1}, {0: 0, 1: 0, 2: 0, 3: 1}]  # unknown steps absent
    nlab = [len(l) for l in labels]
    w = target_weights(nlab, len(trs))

    # reference: mean over trajectories of mean BCE over labeled steps
    with torch.no_grad():
        ref = 0.0
        for ti, tr in enumerate(trs):
            b = collate([tr], "future1")
            out = {s: v for _, s, v in gather_scores(b, m(b))}
            l = [torch.nn.functional.binary_cross_entropy_with_logits(
                torch.tensor(out[s]), torch.tensor(float(y))) for s, y in labels[ti].items()]
            ref += float(torch.stack(l).mean()) / len(trs)

    # chunked: future1 views, split across two microbatches with only labeled targets
    total = 0.0
    with torch.no_grad():
        for chunk in ([0, 1], [2]):
            tr_c = [trs[i] for i in chunk]
            tg = [sorted(labels[i]) for i in chunk]
            b = collate(tr_c, "future1", tg)
            y = {(k, s): labels[i][s] for k, i in enumerate(chunk) for s in labels[i]}
            ww = {(k, s): w[i] for k, i in enumerate(chunk) for s in labels[i]}
            total += float(weighted_bce(b, m(b), y, ww))
    assert total == pytest.approx(ref, rel=1e-5)


def test_unknown_labels_contribute_zero_loss_and_no_input():
    torch.manual_seed(0)
    m = ContextualTokenProbe(d_in=D, d=32, layers=2, heads=4, ff=64, dropout=0.0)
    tr = make_trace()
    b = collate([tr], "full")
    logits = m(b)
    loss = weighted_bce(b, logits, {(0, 0): 0}, {(0, 0): 1.0})
    loss.backward()
    # gradient wrt logits of unknown steps is zero
    lg = logits.detach().requires_grad_(True)
    weighted_bce(b, lg, {(0, 0): 0}, {(0, 0): 1.0}).backward()
    g = lg.grad[0]
    assert g[0] != 0 and torch.all(g[1:] == 0)
    # the model has no label input at all
    import inspect
    assert list(inspect.signature(ContextualTokenProbe.forward).parameters) == ["self", "b"]


def test_synthetic_next_step_task_full_learns_causal_cannot():
    """Only the next step's content determines the label: full can learn it,
    causal stays near chance (fast CPU version of the smoke-run check)."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    def sample():
        tr = make_trace(int(rng.integers(1 << 30)), step_lens=(2, 2, 2, 2))
        signs = rng.choice([-1.0, 1.0], size=4)
        for j in range(4):
            s, e = tr["step_starts"][j], tr["step_ends"][j]
            tr["H"][s:e, 0] = signs[j] * 3.0
        lab = {j: int(signs[j + 1] > 0) for j in range(3)}  # label = sign of NEXT step
        return tr, lab

    data = [sample() for _ in range(256)]
    test = [sample() for _ in range(128)]
    res = {}
    for arm in ("causal", "full"):
        torch.manual_seed(0)
        m = ContextualTokenProbe(d_in=D, d=32, layers=2, heads=4, ff=64, dropout=0.0)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        for ep in range(25):
            for i in range(0, len(data), 32):
                chunk = data[i:i + 32]
                b = collate([c[0] for c in chunk], arm)
                w = target_weights([3] * len(chunk), len(chunk))
                y = {(k, s): v for k, (_, l) in enumerate(chunk) for s, v in l.items()}
                ww = {(k, s): w[k] for k, (_, l) in enumerate(chunk) for s in l}
                loss = weighted_bce(b, m(b), y, ww)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        b = collate([c[0] for c in test], arm)
        with torch.no_grad():
            out = gather_scores(b, m(b))
        acc = np.mean([(v > 0) == bool(test[t][1][s]) for t, s, v in out if s < 3])
        res[arm] = acc
    assert res["full"] > 0.95
    assert res["causal"] < 0.65
