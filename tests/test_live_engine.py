from types import SimpleNamespace

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts.inline_fast import MARKER, PAD, READ, TEXT  # noqa: E402
from scripts.live_engine import Engine, RowCache, StepState, sample, visibility  # noqa: E402
from scripts.read_token_inline import StartMarker, uncached_reference  # noqa: E402
from scripts.read_token_probe import ReadToken  # noqa: E402

DEV = torch.device("cpu")


def tiny(kind):
    torch.manual_seed(0)
    kw = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    if kind == "qwen3":
        model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(head_dim=8, **kw))
    else:
        model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**kw))
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


def token_and_marker():
    torch.manual_seed(1)
    rt = ReadToken(1, 32, torch.randn(1, 32) * 0.5, n_layers=3)
    mk = StartMarker(32, torch.randn(1, 32) * 0.5, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, 0.5)
        mk.deep.normal_(0, 0.5)
    return rt, mk


def call(eng, c, slots):
    """slots: per row, a list of (kind, id, pos, qlim, qord); rows padded with PAD slots,
    columns n[b] onwards."""
    S = max(len(s) for s in slots)
    t = {k: torch.zeros(len(slots), S, dtype=torch.long) for k in ("ids", "kind", "pos", "cols", "qlim", "qord")}
    for b, row in enumerate(slots):
        t["cols"][b] = c.n[b] + torch.arange(S)
        for j, (kd, i, p, ql, qo) in enumerate(row):
            t["kind"][b, j], t["ids"][b, j], t["pos"][b, j], t["qlim"][b, j], t["qord"][b, j] = kd, i, p, ql, qo
    ncol = max(c.n) + S + 3
    h = eng.forward(c, ncol=ncol, **t)
    for b, row in enumerate(slots):
        c.n[b] += len(row)
    return h


@pytest.mark.parametrize("arch", ["qwen2", "qwen3"])
def test_prefill_then_decode_equals_the_plain_model(arch):
    model = tiny(arch)
    eng, c = Engine(model), RowCache(model, 1, 64, DEV)
    ids = [1, 5, 9, 3, 7, 2, 11, 4]
    call(eng, c, [[(TEXT, i, t, t + 1, 0) for t, i in enumerate(ids[:5])]])
    for t in range(5, len(ids)):
        h = call(eng, c, [[(TEXT, ids[t], t, t + 1, 0)]])
    ref = model.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1]
    assert torch.allclose(h[0, 0], ref, atol=1e-5), (h[0, 0] - ref).abs().max()


def live_reads(eng, c, ids, P, pos, spos, b=0, other=None):
    """Feed ids[P:] one per call after a prefill; at each read position the marker and the
    read ride on the call that feeds the separator. Returns the read states (K, d)."""
    out = []
    rows = [None, None] if other is not None else [None]
    call(eng, c, [[(TEXT, i, t, t + 1, 0) for t, i in enumerate(ids[:P])]] +
         ([[(TEXT, i, t, t + 1, 0) for t, i in enumerate(other[:2])]] if other is not None else []))
    k = 0
    for t in range(P, len(ids)):
        row = [(TEXT, ids[t], t, t + 1, 0)]
        if t + 1 in pos:
            row += [(MARKER, 0, spos[k], spos[k], k), (READ, 0, t + 1, t + 1, k)]
        slots = [row]
        if other is not None:
            j = 2 + t - P
            slots.append([(TEXT, other[j], j, j + 1, 0)] if j < len(other) else [(PAD, 0, 0, 0, 0)])
        h = call(eng, c, slots)
        if other is not None and j >= len(other):
            c.n[1] -= 1  # a pad slot holds no column
        if t + 1 in pos:
            out.append(h[0, 2])
            k += 1
    return torch.stack(out)


@pytest.mark.parametrize("arch", ["qwen2", "qwen3"])
def test_live_reads_equal_the_uncached_reference(arch):
    model = tiny(arch)
    rt, mk = token_and_marker()
    ids = [1, 5, 9, 3, 7, 2, 11, 4, 6, 8, 12, 13]
    P, pos = 2, [5, 9, 12]
    spos = [P] + pos[:-1]
    got = live_reads(Engine(model, rt, mk), RowCache(model, 1, 64, DEV), ids, P, pos, spos)
    ref = uncached_reference(model, rt, ids, pos, DEV, spos, mk, chain=True)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_a_second_row_and_pad_slots_change_nothing():
    model = tiny("qwen3")
    rt, mk = token_and_marker()
    ids = [1, 5, 9, 3, 7, 2, 11, 4, 6, 8, 12, 13]
    P, pos = 2, [5, 9, 12]
    spos = [P] + pos[:-1]
    eng = Engine(model, rt, mk)
    alone = live_reads(eng, RowCache(model, 1, 64, DEV), ids, P, pos, spos)
    c = RowCache(model, 2, 64, DEV)
    c.k[0][1].fill_(5.0)  # stale values in another row's columns must not leak in
    both = live_reads(eng, c, ids, P, pos, spos, other=[3, 3, 7, 1, 2])
    assert torch.allclose(alone, both, atol=1e-5)


def test_visibility_rules():
    # columns: text 0..3, marker of step 0 (ord 0), read of step 0, text 4
    ck = torch.tensor([[TEXT, TEXT, TEXT, TEXT, MARKER, READ, TEXT]])
    ctp = torch.tensor([[0, 1, 2, 3, 1, 4, 4]])
    cord = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
    kind = torch.tensor([[TEXT, MARKER, READ, PAD]])
    qlim = torch.tensor([[5, 4, 5, 0]])
    qord = torch.tensor([[0, 1, 1, 0]])
    cols = torch.tensor([[6, 7, 8, 9]])
    ck = torch.cat([ck, torch.tensor([[MARKER, READ, PAD]])], 1)
    ctp = torch.cat([ctp, torch.tensor([[4, 5, 0]])], 1)
    cord = torch.cat([cord, torch.tensor([[1, 1, 0]])], 1)
    v = visibility(ck, ctp, cord, kind, qlim, qord, cols)[0].int().tolist()
    assert v[0] == [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]  # text: text only
    assert v[1] == [1, 1, 1, 1, 0, 0, 0, 1, 0, 0]  # marker of step 1 (starts at 4): text < 4, itself
    assert v[2] == [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]  # read of step 1: text < 5, read 0, its marker, itself
    assert v[3] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # pad: itself


def test_gather_and_put_back_round_trip():
    model = tiny("qwen2")
    c = RowCache(model, 3, 16, DEV)
    for x in c.k + c.v:
        x.normal_()
    c.kind.random_(0, 4)
    before = [x.clone() for x in c.k]
    sub = c.gather([2, 0], 8)
    for x in sub.k:
        x.add_(1.0)
    c.put_back(sub, [2, 0], 8)
    assert torch.allclose(c.k[0][2, :, :8], before[0][2, :, :8] + 1)
    assert torch.equal(c.k[0][1], before[0][1]) and torch.equal(c.k[0][2, :, 8:], before[0][2, :, 8:])


def test_sample_follows_the_filtered_distribution():
    torch.manual_seed(0)
    logits = torch.randn(1, 40) * 2
    assert int(sample(logits, torch.tensor([0.3]), 1.0, 1, 1.0)[0]) == int(logits.argmax())
    u = (torch.arange(20000) + 0.5) / 20000
    toks = sample(logits.expand(20000, -1), u, 0.7, 10, 0.8)
    v, i = (logits[0] / 0.7).topk(10)
    p = torch.softmax(v, -1)
    keep = (p.cumsum(-1) - p) < 0.8
    want = torch.zeros(40)
    want[i[keep]] = p[keep] / p[keep].sum()
    got = torch.bincount(toks, minlength=40).float() / 20000
    assert torch.allclose(got, want, atol=1e-3)


def test_step_state_matches_the_blank_line_split():
    from scripts.generate_onpolicy_steps import split_into_steps
    text = "First step.\n\nSecond $x$\n \nThird.\n\n\n\nLast"
    st, n = StepState(), 0
    for piece in ["First", " step", ".\n\n", "Second", " $x$", "\n", " \n", "Third", ".\n\n", "\n\n", "Last"]:
        n += st.feed(piece)
    assert n == len(split_into_steps(text)) - 1 and st.content


class CharTok:
    """Longest-match tokenizer over a few strings, for the live loop."""
    special = {"<|im_end|>": 60, "<extra_0>": 61, "<|endoftext|>": 62}

    def __init__(self):
        self.vocab = {"\n\n": 30, "\n": 31, " ": 32, ".": 33}
        self.vocab.update({ch: 1 + i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")})
        self.inv = {v: k for k, v in {**self.vocab, **self.special}.items()}
        self.eos_token_id = 60

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for n in (2, 1):
                if text[i:i + n] in self.vocab:
                    ids.append(self.vocab[text[i:i + n]])
                    i += n
                    break
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.inv.get(i, "") for i in ids if not (skip_special_tokens and i in self.special.values()))

    def convert_tokens_to_ids(self, t):
        return self.special.get(t)


def live(model, rt, mk, prm=None):
    from scripts.analysis.live_compare import Live
    tok = CharTok()
    a = SimpleNamespace(max_new_tokens=100, temperature=1.0, top_k=5, top_p=0.9, col_step=16)
    c = RowCache(model, 2, 129, DEV)
    vocab = [tok.decode([i]) for i in range(64)]
    return Live(Engine(model, rt, mk), c, tok, vocab, None, prm, a), tok


def forced(tok, text, P=3):
    ids = tok(text)["input_ids"]
    return {"jid": 0, "fork_id": "x", "sample_idx": 0, "problem": "p", "gold": "1", "seed": 0,
            "prompt_ids": list(range(1, P + 1)), "prm_prefix": [1, 2], "forced": ids + [60]}


def test_live_read_loop_equals_the_uncached_reference():
    model = tiny("qwen3")
    rt, mk = token_and_marker()
    lv, tok = live(model, rt, mk)
    job = forced(tok, "ab c\n\nde.\n\nfg")
    out, st = lv.run("read", [job, dict(job, jid=1)], 2, float("inf"))
    full = job["prompt_ids"] + job["forced"][:-1] + [30]  # the last step read after a separator
    pos = [i + 1 for i, x in enumerate(full) if x == 30]
    spos = [3] + pos[:-1]
    ref = torch.sigmoid(rt.logit(uncached_reference(model, rt, full, pos, DEV, spos, mk, chain=True))).tolist()
    assert len(out) == 2 and st["reads"] == 6
    for r in out:
        assert np.allclose(r["scores"], ref, atol=1e-5) and r["solution"] == "ab c\n\nde.\n\nfg"
    # the stop rule abstains at the first step above tau
    tau = (ref[0] + ref[1]) / 2 if ref[1] > ref[0] else None
    if tau is not None:
        o, _ = lv.run("read_stop", [job], 1, tau)
        assert o[0]["end"] == "abstain" and o[0]["abstain_step"] == 1


def test_live_prm_loop_equals_one_full_forward():
    model, pm = tiny("qwen3"), tiny("qwen2")
    torch.manual_seed(2)
    head = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)).eval()
    lv0, tok = live(model, None, None)
    prm = (Engine(pm), RowCache(pm, 2, 129, DEV), head, tok)
    lv, _ = live(model, None, None, prm)
    job = forced(tok, "ab c\n\nde.\n\nfg")
    out, st = lv.run("prm", [job, dict(job, jid=1)], 2, float("inf"))
    ids = [1, 2] + tok("ab c")["input_ids"] + [61] + tok("de.")["input_ids"] + [61] + tok("fg")["input_ids"] + [61]
    h = pm.model(input_ids=torch.tensor([ids])).last_hidden_state[0]
    at = [i for i, x in enumerate(ids) if x == 61]
    ref = torch.softmax(head(h[at]), -1)[:, 0].tolist()
    for r in out:
        assert np.allclose(r["scores"], ref, atol=1e-5)
    assert st["prm_rows"] == 6  # three steps per job (the prefix is part of the prefill)


def test_prm_call_keeps_pad_slots_inside_the_cache():
    # a row near the top of its cache, batched with a longer chunk: its pad slots must not
    # land past the cache, and each row must score as it does alone
    model, pm = tiny("qwen3"), tiny("qwen2")
    torch.manual_seed(3)
    head = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)).eval()

    def fresh():
        pc = RowCache(pm, 2, 65, DEV)  # 64 usable columns
        g = torch.Generator().manual_seed(0)
        for x in pc.k + pc.v:
            x.copy_(torch.randn(x.shape, generator=g))
        pc.kind.fill_(3)  # TEXT
        pc.tpos.copy_(torch.arange(65)[None].expand(2, -1))
        pc.n[:] = [10, 61]
        return pc

    chunks = [[3, 4, 5, 6, 7], [8, 9]]  # row 1 fills columns 61-62
    lv, _ = live(model, None, None, (Engine(pm), fresh(), head, CharTok()))
    both = lv.prm_call([0, 1], chunks)
    for b in (0, 1):
        lv1, _ = live(model, None, None, (Engine(pm), fresh(), head, CharTok()))
        assert np.isclose(lv1.prm_call([b], [chunks[b]])[0], both[b], atol=1e-5)
    lv2, _ = live(model, None, None, (Engine(pm), fresh(), head, CharTok()))
    with pytest.raises(ValueError, match="PRM cache full"):
        lv2.prm_call([1], [[8, 9, 10, 11]])
