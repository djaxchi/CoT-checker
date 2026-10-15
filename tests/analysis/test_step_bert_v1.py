"""Tests for step_bert_probe_v1 data building and the token-level fit loop."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "step_bert"))

from build_text_table import check, onpolicy_texts  # noqa: E402
from build_token_store import build  # noqa: E402
import finetune_tok as FT  # noqa: E402
from src.probes.token_bert import TokenBert  # noqa: E402
from src.repstore.store import RepSpec, write_split  # noqa: E402

D = 8


def _store(root: Path, traces: dict[str, int], seed=0, n_shards=2):
    rng = np.random.default_rng(seed)
    per = [([], [], []) for _ in range(n_shards)]
    truth, g = {}, 0
    for tid, n in traces.items():
        fe = int(rng.integers(-1, n))
        for s in range(n):
            L = int(rng.integers(2, 6))
            item = rng.normal(size=(L, D)).astype(np.float16)
            truth[(tid, s)] = item
            meta = {"id": tid, "step_idx": s, "label": fe, "n_steps": n, "problem_id": "p_" + tid,
                    "n_tokens": L, "step_start_idx": 1, "pre_step_boundary_idx": 0, "global_index": g}
            sh = per[g % n_shards]
            sh[0].append(item); sh[1].append(int(fe >= 0 and s >= fe)); sh[2].append(meta)
            g += 1
    spec = RepSpec(name="t", kind="token_seq", dim=D, layer=35, backbone="x", readout="step_span_with_boundary")
    for i, (items, ys, metas) in enumerate(per):
        write_split(root / f"shard_{i:02d}", items, ys, metas, spec)
    return truth


def test_token_store_concatenates_in_trace_order(tmp_path):
    truth = _store(tmp_path / "s", {"b": 3, "a": 2})
    info = build([tmp_path / "s"], tmp_path / "out")
    z = np.load(tmp_path / "out" / "steps.npz")
    X = np.load(tmp_path / "out" / "X.npy")
    assert list(z["trace_id"]) == ["a", "a", "b", "b", "b"] and list(z["step_idx"]) == [0, 1, 0, 1, 2]
    for k, (t, s) in enumerate(zip(z["trace_id"], z["step_idx"])):
        np.testing.assert_array_equal(X[z["tok_off"][k]:z["tok_off"][k + 1]], truth[(t, int(s))])
    assert info["tokens"] == X.shape[0]


def test_text_table_alignment(tmp_path):
    f = tmp_path / "items.jsonl"
    rows = [{"fork_id": "t1", "step_idx": k, "candidate_step": f"s{k}", "problem": "P"} for k in (1, 0)]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    texts = onpolicy_texts([f])
    assert texts["t1"]["steps"] == ["s0", "s1"]
    meta = [{"id": "t1", "step_idx": k, "n_steps": 2, "n_tokens": 3, "step_start_idx": 1} for k in (0, 1)]
    assert check(texts, meta)["missing_text"] == 0
    assert check(texts, meta + [{"id": "t2", "step_idx": 0, "n_steps": 1}])["missing_text"] == 1


def _data(tmp_path):
    rng = np.random.default_rng(1)
    on = {f"t{i}": int(rng.integers(1, 6)) for i in range(40)}
    _store(tmp_path / "raw_on", on, seed=2)
    build([tmp_path / "raw_on"], tmp_path / "data" / "onpolicy")
    for s in ("gsm8k", "math", "olympiadbench", "omnimath"):
        _store(tmp_path / f"raw_{s}", {f"{s}-{i}": int(rng.integers(1, 5)) for i in range(60)}, seed=hash(s) % 100)
        build([tmp_path / f"raw_{s}"], tmp_path / "data" / f"pb_{s}")
    ids = sorted(on)
    split = {"train": ids[:24], "val": ids[24:32], "test": ids[32:]}
    (tmp_path / "split.json").write_text(json.dumps(split))
    return tmp_path / "data", tmp_path / "split.json"


def test_wide_crop_equals_full_mask(tmp_path):
    data, split = _data(tmp_path)
    steps, X = FT.load_tokens(data / "onpolicy", torch.device("cpu"))
    runs = list(FT.C.trace_runs(steps["trace_id"]).values())
    torch.manual_seed(0)
    m = TokenBert(in_dim=D, d=16, n_heads=2, ff=32, hidden=16, dropout=0.0)
    a = FT.predict(m, X, steps, runs, "bidir", -1, "cpu", 4096)
    b = FT.predict(m, X, steps, runs, "bidir", 50, "cpu", 4096)
    c = FT.predict(m, X, steps, runs, "bidir", 0, "cpu", 4096)
    loc = FT.predict(m, X, steps, runs, "local", -1, "cpu", 4096)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_allclose(c, loc, atol=1e-5)
    # batching must not change predictions
    np.testing.assert_allclose(a, FT.predict(m, X, steps, runs, "bidir", -1, "cpu", 64), atol=1e-5)


def test_finetune_tok_end_to_end(tmp_path):
    data, split = _data(tmp_path)
    for cond, W in (("causal", -1), ("bidir", 1)):
        out = tmp_path / f"fit_{cond}"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/step_bert/finetune_tok.py"), "--data", str(data),
                            "--split", str(split), "--condition", cond, "--window", str(W), "--out_dir", str(out),
                            "--epochs", "2", "--budget", "2048"], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        m = json.loads((out / "metrics.json").read_text())
        assert "processbench_mean_f1_pb_calib20" in m and np.isfinite(np.load(out / "probs_onpolicy.npy")).all()
