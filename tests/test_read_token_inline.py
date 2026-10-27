import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts.read_token_inline import (  # noqa: E402
    build_sequences, group_items, inline_read, seq_batches)
from scripts.read_token_probe import ReadToken, read_all_states  # noqa: E402


def tiny_model():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=128)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


DEV = torch.device("cpu")
# (text ids, read positions): two rows of different lengths and read counts
SEQS = [([1, 5, 9, 3, 7, 2, 11, 4, 6], [3, 6, 9]), ([4, 8, 13, 2], [4])]


def deep_token(scale=0.5):
    rt = ReadToken(1, 32, torch.randn(1, 32) * 0.02, n_layers=3)
    with torch.no_grad():
        rt.deep.normal_(0, scale)
    return rt


def test_inline_read_equals_a_read_token_appended_to_each_prefix():
    model, rt = tiny_model(), deep_token()
    got = inline_read(model, rt, SEQS, 0, DEV, grad=True)
    assert got.shape == (4, 32)
    k = 0
    for ids, pos in SEQS:
        for p in pos:
            ref = read_all_states(model, rt, [ids[:p]], 0, DEV, grad=False)[0, -1]
            assert torch.allclose(got[k].detach(), ref, atol=2e-4), (ids, p)
            k += 1
    got.sum().backward()
    assert rt.deep.grad.abs().sum() > 0 and rt.soft.grad.abs().sum() > 0


def test_passive_states_are_the_plain_text_states_at_the_separator():
    model, rt = tiny_model(), deep_token()
    cap = {1: None, "final": None}
    inline_read(model, rt, SEQS, 0, DEV, grad=False, passive=cap)
    k = 0
    for ids, pos in SEQS:
        out = model.model(input_ids=torch.tensor([ids]), output_hidden_states=True)
        for p in pos:
            assert torch.allclose(cap["final"][k], out.last_hidden_state[0, p - 1], atol=1e-5)
            assert torch.allclose(cap[1][k], out.hidden_states[2][0, p - 1], atol=1e-5)
            k += 1


def test_inline_read_leaves_no_hooks_and_rejects_bad_positions():
    model, rt = tiny_model(), deep_token()
    inline_read(model, rt, SEQS, 0, DEV, grad=False, passive={0: None})
    assert all(not layer._forward_hooks for layer in model.model.layers)
    with pytest.raises(ValueError):
        inline_read(model, rt, [([1, 2, 3], [4])], 0, DEV, grad=False)
    with pytest.raises(ValueError):
        inline_read(model, ReadToken(2, 32, torch.randn(2, 32)), SEQS, 0, DEV, grad=False)


def item(problem, prefix, step, label=0):
    return {"problem": problem, "prefix": prefix, "step": step, "label": label}


def test_group_items_chains_items_of_one_trajectory_only():
    items = [item("p", "", "a"), item("p", "a", "b"), item("p", "a\n\nb", "c"),
             item("p", "a", "x"),       # another branch of the same prefix
             item("q", "", "a"),         # same text, other problem
             item("p", "", "ab")]       # "a" is a string prefix of "ab" but not at a separator
    groups = group_items(items)
    members = sorted(sorted(g["members"]) for g in groups)
    assert members == [[0, 1, 2], [3], [4], [5]]
    by_text = {g["text"]: sorted(g["members"]) for g in groups}
    assert by_text["a\n\nb\n\nc"] == [0, 1, 2]
    assert all(sum(i in g["members"] for g in groups) == 1 for i in range(len(items)))


class CharTok:
    """One token per character; merge_sep glues each "\\n\\n" to the next character."""

    def __init__(self, merge_sep=False):
        self.merge_sep = merge_sep

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans, j = [], 0
        while j < len(text):
            if self.merge_sep and text.startswith("\n\n", j) and j + 2 < len(text):
                spans.append((j, j + 3))
                j += 3
            else:
                spans.append((j, j + 1))
                j += 1
        return {"input_ids": [ord(text[s]) % 64 for s, _ in spans], "offset_mapping": spans}


def test_build_sequences_reads_each_item_after_its_separator(monkeypatch):
    import scripts.read_token_inline as ri
    monkeypatch.setitem(ri.LAYOUTS, "chat", lambda problem: f"<{problem}>")
    items = [item("p", "", "ab"), item("p", "ab", "cd"), item("p", "ab\n\ncd", "e")]
    seqs, alone = build_sequences(CharTok(), items)
    assert alone == 0 and len(seqs) == 1
    s = seqs[0]
    text = "<p>ab\n\ncd\n\ne\n\n"
    assert len(s["ids"]) == len(text)
    assert s["members"] == [0, 1, 2]
    assert s["pos"] == [len("<p>ab\n\n"), len("<p>ab\n\ncd\n\n"), len(text)]


def test_build_sequences_reads_an_item_alone_without_a_token_boundary(monkeypatch):
    import scripts.read_token_inline as ri
    monkeypatch.setitem(ri.LAYOUTS, "chat", lambda problem: f"<{problem}>")
    items = [item("p", "", "ab"), item("p", "ab", "cd")]
    seqs, alone = build_sequences(CharTok(merge_sep=True), items)
    assert alone == 1
    by_member = {s["members"][0]: s for s in seqs}
    assert by_member[0]["pos"] == [len(by_member[0]["ids"])]   # own text, read at its end
    assert by_member[1]["members"] == [1]


def test_seq_batches_respect_token_and_item_caps():
    lengths, n_items = [10, 12, 30, 5, 8], [3, 1, 2, 4, 1]
    bl = seq_batches(lengths, n_items, max_tokens=40, max_items=5, shuffle_seed=None)
    assert sorted(i for b in bl for i in b) == list(range(5))
    for b in bl:
        assert len(b) == 1 or (len(b) * max(lengths[i] for i in b) <= 40 and sum(n_items[i] for i in b) <= 5)


class CliTok(CharTok):
    pad_token_id, eos_token_id = 0, 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return super().__call__(text, add_special_tokens, return_offsets_mapping)


def test_train_then_encode_end_to_end(tmp_path, monkeypatch):
    import json
    import sys

    import numpy as np

    import scripts.read_token_inline as ri
    import scripts.read_token_store as rs
    monkeypatch.setattr(rs, "load_model", lambda name, dev: (tiny_model(), CliTok()))
    rows = []
    for p in range(6):
        steps = [f"s{p}a", f"s{p}b", f"s{p}c"]
        for k in range(3):
            rows.append({"uid": f"u{p}{k}", "problem": f"q{p}", "prefix": "\n\n".join(steps[:k]),
                         "candidate_step": steps[k], "label": (p + k) % 2})
    data = tmp_path / "items.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "tok"
    monkeypatch.setattr(sys, "argv", ["x", "train", "--init_text", "x", "--deep", "--train", str(data),
                                      "--n_train", "0", "--max_tokens", "400", "--out", str(out)])
    ri.main()
    ck = __import__("torch").load(out / "read_token.pt", weights_only=False)
    assert ck["inline"] and ck["m"] == 1 and ck["deep"].abs().sum() > 0
    monkeypatch.setattr(sys, "argv", ["x", "encode", "--out", str(out), "--passive_blocks", "1",
                                      "--prm800k", f"val:{data}"])
    ri.main()
    z = np.load(out / "val.shard0.npz")
    assert z["kept"].all() and z["sep"].shape == (18, 1, 32)
    assert z["passive_block1"].shape == z["passive_final"].shape == (18, 1, 32)
    from scripts.analysis.backbone_geometry import load_split
    split, meta = load_split(out, "val")   # every array but `layers` is per item
    assert len(split["sep"]) == len(meta) == 18
    assert np.isfinite(z["head_p_correct"]).all()
