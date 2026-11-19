"""Other per-layer parameterisations of the inline read token (reviewer W1) and the
--exclude_problems training filter (reviewer Q1)."""

import json
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts.read_token_inline import (  # noqa: E402
    StartMarker, add_extras, inline_read, prefix_len, uncached_reference)
from scripts.read_token_probe import ReadToken  # noqa: E402
from tests.test_read_token_inline import CliTok, MSEQS, SEQS, tiny_model  # noqa: E402

DEV = torch.device("cpu")
D, L, KV, HD = 32, 3, 2, 8


def token(scale=False, kv_prefix=0, interv=None, deep=True):
    torch.manual_seed(1)
    rt = ReadToken(1, D, torch.randn(1, D) * 0.02, n_layers=L if deep else 0, interv=interv,
                   rank=2 if interv else 0, n_interv_layers=L if interv else 0,
                   lora_dims=(D, KV * HD) if interv else None)
    add_extras(rt, L, D, KV, HD, scale, kv_prefix)
    with torch.no_grad():
        if deep:
            rt.deep.normal_(0, 0.5)
        if scale:
            rt.scale.normal_(0, 0.5)
        if kv_prefix:
            rt.pk.normal_(0, 1.0)
            rt.pv.normal_(0, 1.0)
        if interv:
            rt.B_q.normal_(0, 0.3)
            rt.B_v.normal_(0, 0.3)
    return rt


def marker():
    mk = StartMarker(D, torch.randn(1, D) * 0.02, n_layers=L)
    with torch.no_grad():
        mk.deep.normal_(0, 0.5)
    return mk


@pytest.mark.parametrize("kw", [dict(scale=True), dict(kv_prefix=2), dict(interv="lora"),
                                dict(kv_prefix=2, deep=False), dict(interv="lora", deep=False)])
@pytest.mark.parametrize("with_marker", [False, True])
def test_inline_read_equals_the_uncached_forward(kw, with_marker):
    model, rt = tiny_model(), token(**kw)
    mk = marker() if with_marker else None
    seqs = MSEQS if with_marker else SEQS
    got = inline_read(model, rt, seqs, 0, DEV, grad=True, mk=mk, chain=True)
    k = 0
    for s in seqs:
        ref = uncached_reference(model, rt, s[0], s[1], DEV, s[2] if mk else None, mk, chain=True)
        assert torch.allclose(got[k:k + len(s[1])].detach(), ref, atol=2e-4)
        k += len(s[1])
    got.sum().backward()
    for q in [q for q in (rt.scale, rt.pk, rt.pv) if q is not None] + rt.interv_params():
        assert q.grad is not None and q.grad.abs().sum() > 0


def test_the_parameterisations_change_the_read():
    model = tiny_model()
    base = inline_read(model, token(), SEQS, 0, DEV, grad=False)
    for kw in (dict(scale=True), dict(kv_prefix=2), dict(interv="lora")):
        assert not torch.allclose(inline_read(model, token(**kw), SEQS, 0, DEV, grad=False), base, atol=1e-3)


def test_prefix_equals_an_extra_text_token_seen_by_the_reads_only():
    # set the prefix to the cached key/value of a real token x placed after the text: the reads
    # must then equal reads that also see x (an independent check of the prefix columns)
    model = tiny_model()
    ids, x = [1, 5, 9, 3, 7, 2], 11
    rt = token(kv_prefix=1, deep=False)
    out = model.model(input_ids=torch.tensor([ids + [x]]), use_cache=True)
    with torch.no_grad():
        for l, layer in enumerate(out.past_key_values.layers):
            rt.pk[l, :, 0] = layer.keys[0, :, -1]
            rt.pv[l, :, 0] = layer.values[0, :, -1]
    got = inline_read(model, rt, [(ids, [3, 6])], 0, DEV, grad=False)
    # reference: text + x + 2 read tokens, x visible to the reads but not part of their prefix
    T = len(ids) + 1
    e = model.get_input_embeddings()(torch.tensor([ids + [x]]))
    xs = torch.cat([e, rt.soft[None].expand(1, 2, -1)], 1)
    n = T + 2
    mask = torch.zeros(n, n, dtype=torch.bool)
    mask[:T, :T] = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for k, p in enumerate([3, 6]):
        mask[T + k, :p] = True
        mask[T + k, T - 1] = True
        mask[T + k, T + k] = True
    pos = list(range(T)) + [3, 6]
    with torch.no_grad():
        ref = model.model(inputs_embeds=xs, attention_mask=mask[None, None],
                          position_ids=torch.tensor([pos])).last_hidden_state[0, -2:]
    assert torch.allclose(got, ref, atol=2e-4)
    assert prefix_len(rt) == 1


def _data(tmp_path):
    rows = [{"uid": f"u{p}{k}", "problem": f"Q{p}", "prefix": "\n\n".join(f"s{p}{j}" for j in range(k)),
             "candidate_step": f"s{p}{k}", "label": (p + k) % 2} for p in range(5) for k in range(3)]
    data = tmp_path / "items.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return data


@pytest.mark.parametrize("extra", [["--scale"], ["--kv_prefix", "2"], ["--interv", "lora", "--rank", "1"]])
def test_train_then_encode_with_each_parameterisation(tmp_path, monkeypatch, extra):
    import numpy as np

    import scripts.read_token_inline as ri
    import scripts.read_token_store as rs
    monkeypatch.setattr(rs, "load_model", lambda name, dev: (tiny_model(), CliTok()))
    data, out = _data(tmp_path), tmp_path / "tok"
    deep = [] if extra[0] != "--scale" else ["--deep"]
    monkeypatch.setattr(sys, "argv", ["x", "train", "--init_text", "x", *deep, *extra, "--train", str(data),
                                      "--n_train", "0", "--max_tokens", "60", "--out", str(out)])
    ri.main()
    ck = torch.load(out / "read_token.pt", weights_only=False)
    if "--scale" in extra:
        assert ck["scale"].abs().sum() > 0
    if "--kv_prefix" in extra:
        assert ck["kv_prefix_k"].shape == (3, 2, 2, 8) and ck["kv_prefix_k"].abs().sum() > 0
    if "--interv" in extra:
        assert ck["interv"] == "lora" and ck["interv_state"]["B_q"].abs().sum() > 0
    monkeypatch.setattr(sys, "argv", ["x", "encode", "--out", str(out), "--passive_blocks", "",
                                      "--prm800k", f"val:{data}"])
    ri.main()
    z = np.load(out / "val.shard0.npz")
    assert z["kept"].all() and np.isfinite(z["head_p_correct"]).all()


def test_exclude_problems_drops_their_items(tmp_path, monkeypatch, capsys):
    import scripts.read_token_inline as ri
    import scripts.read_token_store as rs
    monkeypatch.setattr(rs, "load_model", lambda name, dev: (tiny_model(), CliTok()))
    data, out = _data(tmp_path), tmp_path / "tok"
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps(["  q1 ", "Q3", "never"]))   # matched after normalisation
    monkeypatch.setattr(sys, "argv", ["x", "train", "--init_text", "x", "--deep", "--train", str(data),
                                      "--n_train", "0", "--max_tokens", "60", "--exclude_problems", str(ex),
                                      "--out", str(out)])
    ri.main()
    log = capsys.readouterr().out
    assert "[exclude] 6 of 15 drawn items" in log and "[data] 9 items" in log
    assert torch.load(out / "read_token.pt", weights_only=False)["n_excluded"] == 6
