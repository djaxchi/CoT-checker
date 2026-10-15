import torch

from src.probes.token_bert import TokenBert, token_mask


def _batch(steps=(3, 4, 2), seed=0, D=32):
    """One trace: step s has one boundary token then steps[s] - 1 span tokens."""
    g = torch.Generator().manual_seed(seed)
    sid = torch.cat([torch.full((n,), s) for s, n in enumerate(steps)])[None]
    bnd = torch.cat([torch.arange(n) == 0 for n in steps])[None]
    x = torch.randn(1, sid.shape[1], D, generator=g)
    return x, sid, bnd, torch.ones_like(sid, dtype=torch.bool)


def _logits(m, x, sid, bnd, valid, kind):
    z = m.encode(x, bnd, token_mask(sid, valid, kind))
    return m.readout(z, sid, bnd, valid, int(sid.max()) + 1)


def _model():
    torch.manual_seed(0)
    return TokenBert(in_dim=32, d=16, n_heads=2, ff=32, hidden=16, dropout=0.0).eval()


def test_local_ignores_other_steps():
    m = _model()
    x, sid, bnd, v = _batch()
    base = _logits(m, x, sid, bnd, v, "local")
    x2 = x.clone()
    x2[0, sid[0] != 1] += torch.randn_like(x2[0, sid[0] != 1])
    out = _logits(m, x2, sid, bnd, v, "local")
    assert torch.allclose(base[1], out[1], atol=1e-5)
    assert not torch.allclose(base[0], out[0], atol=1e-4)


def test_causal_ignores_later_tokens_but_sees_earlier():
    m = _model()
    x, sid, bnd, v = _batch()
    base = _logits(m, x, sid, bnd, v, "causal")
    x2 = x.clone()
    x2[0, sid[0] == 2] += torch.randn_like(x2[0, sid[0] == 2])
    out = _logits(m, x2, sid, bnd, v, "causal")
    assert torch.allclose(base[:2], out[:2], atol=1e-5)
    x3 = x.clone()
    x3[0, sid[0] == 0] += torch.randn_like(x3[0, sid[0] == 0])
    assert not torch.allclose(base[1], _logits(m, x3, sid, bnd, v, "causal")[1], atol=1e-4)


def test_all_sees_future():
    m = _model()
    x, sid, bnd, v = _batch()
    base = _logits(m, x, sid, bnd, v, "all")
    x2 = x.clone()
    x2[0, sid[0] == 2] += torch.randn_like(x2[0, sid[0] == 2])
    assert not torch.allclose(base[0], _logits(m, x2, sid, bnd, v, "all")[0], atol=1e-4)


def test_padding_is_invisible():
    m = _model()
    x, sid, bnd, v = _batch()
    T = x.shape[1]
    xp = torch.cat([x, torch.randn(1, 5, x.shape[2])], 1)
    sidp = torch.cat([sid, torch.full((1, 5), -1)], 1)
    bndp = torch.cat([bnd, torch.zeros(1, 5, dtype=torch.bool)], 1)
    vp = torch.cat([v, torch.zeros(1, 5, dtype=torch.bool)], 1)
    for kind in ("local", "causal", "all"):
        a = _logits(m, x, sid, bnd, v, kind)
        b = _logits(m, xp, sidp, bndp, vp, kind)
        assert torch.allclose(a, b, atol=1e-5), kind
    assert T == sid.shape[1]


def test_readout_global_ids_across_batch():
    m = _model()
    x, sid, bnd, v = _batch()
    single = _logits(m, x, sid, bnd, v, "causal")
    x2 = torch.cat([x, x], 0)
    sid2 = torch.cat([sid, sid + 3], 0)
    out = m.readout(m.encode(x2, torch.cat([bnd, bnd]), token_mask(sid2, torch.cat([v, v]), "causal")),
                    sid2, torch.cat([bnd, bnd]), torch.cat([v, v]), 6)
    assert torch.allclose(out[:3], single, atol=1e-5) and torch.allclose(out[3:], single, atol=1e-5)
