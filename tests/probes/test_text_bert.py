import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.probes.text_bert import TextStepClassifier, build_example, span_window

ROOT = Path(__file__).resolve().parents[2]


def test_examples_contain_exactly_the_visible_steps():
    prob, steps = [7, 8], [[10, 11], [20], [30, 31, 32]]
    for cond, t, want in (("local", 1, [1]), ("causal", 1, [0, 1]), ("bidir", 1, [0, 1, 2])):
        ids, reads = build_example(prob, steps, *span_window(3, t, cond), cls=1, sep=2)
        assert [r[0] for r in reads] == want
        present = {s for s in range(3) if steps[s][0] in ids}
        assert present == set(want), cond
        for k, a, e, sp in reads:
            assert ids[a:e] == steps[k] and ids[sp] == 2
        assert ids[:4] == [1, 7, 8, 2]


def _tiny_model(path: Path):
    transformers = pytest.importorskip("transformers")
    from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel
    try:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    except OSError:
        pytest.skip("bert-base-uncased tokenizer not cached")
    cfg = ModernBertConfig(vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, pad_token_id=tok.pad_token_id, bos_token_id=tok.cls_token_id,
                           eos_token_id=tok.sep_token_id, cls_token_id=tok.cls_token_id,
                           sep_token_id=tok.sep_token_id, global_attn_every_n_layers=2, local_attention=16)
    torch.manual_seed(0)
    ModernBertModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def test_readout_ignores_padding(tmp_path):
    from transformers import AutoModel
    m = TextStepClassifier(AutoModel.from_pretrained(_tiny_model(tmp_path / "m"))).eval()
    ids, reads = build_example([1000, 1001], [[2000, 2001], [2002]], 0, 2, 101, 102)
    rb = torch.tensor([0, 0]); ra = torch.tensor([r[1] for r in reads]); re = torch.tensor([r[2] for r in reads])
    rs = torch.tensor([r[3] for r in reads])
    a = m(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long), rb, ra, re, rs)
    padded = torch.tensor([ids + [0] * 5])
    att = torch.tensor([[1] * len(ids) + [0] * 5])
    b = m(padded, att, rb, ra, re, rs)
    assert torch.allclose(a, b, atol=1e-4)


def test_finetune_txt_end_to_end(tmp_path):
    sys.path.insert(0, str(ROOT / "tests" / "analysis"))
    import test_step_bert_v1 as T
    model = _tiny_model(tmp_path / "m")
    data, split = T._data(tmp_path)
    texts = tmp_path / "texts"
    texts.mkdir()
    for coll in ["onpolicy"] + [f"pb_{s}" for s in ("gsm8k", "math", "olympiadbench", "omnimath")]:
        z = np.load(data / coll / "steps.npz")
        tids, n = np.unique(z["trace_id"], return_counts=True)
        (texts / f"{coll}.json").write_text(json.dumps(
            {t: {"problem": f"solve {t}", "steps": [f"step {k} of {t} is x = {k}" for k in range(c)]}
             for t, c in zip(tids.tolist(), n.tolist())}))
    for cond in ("local", "causal", "bidir"):
        out = tmp_path / f"fit_{cond}"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/step_bert/finetune_txt.py"), "--data", str(data),
                            "--texts", str(texts), "--model", str(model), "--split", str(split),
                            "--condition", cond, "--out_dir", str(out), "--epochs", "1", "--budget", "4096"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-3000:]
        assert np.isfinite(np.load(out / "probs_onpolicy.npy")).all()
        assert "processbench_mean_f1_pb_calib20" in json.loads((out / "metrics.json").read_text())


def test_driver_runs_both_arms(tmp_path):
    sys.path.insert(0, str(ROOT / "tests" / "analysis"))
    import test_step_bert_v1 as T
    model = _tiny_model(tmp_path / "m")
    data, split = T._data(tmp_path)
    texts = tmp_path / "texts"
    texts.mkdir()
    for coll in ["onpolicy"] + [f"pb_{s}" for s in ("gsm8k", "math", "olympiadbench", "omnimath")]:
        z = np.load(data / coll / "steps.npz")
        tids, n = np.unique(z["trace_id"], return_counts=True)
        (texts / f"{coll}.json").write_text(json.dumps(
            {t: {"problem": "p", "steps": [f"s {k}" for k in range(c)]} for t, c in zip(tids.tolist(), n.tolist())}))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/step_bert/driver_v1.py"), "--data", str(data),
                        "--texts", str(texts), "--model", str(model), "--split", str(split), "--out_root", str(out),
                        "--gpus", "2", "--tok_args", "--epochs 1 --budget 2048",
                        "--txt_args", "--epochs 1 --budget 4096"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    for arm in ("tok", "txt"):
        s = json.loads((out / arm / "summary.json").read_text())
        assert "GO" in s and s["contrasts"]
