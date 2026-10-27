import json

import numpy as np

from scripts.analysis import inline_passive_lda as ipl


def write(enc, name, n, rng, pb=False):
    y = rng.integers(0, 2, n).astype(np.int8)
    sep = (rng.normal(size=(n, 1, 8)) + y[:, None, None]).astype(np.float16)
    np.savez(enc / f"{name}.shard0.npz", layers=np.array([-1]), y=y, kept=np.ones(n, bool),
             head_p_correct=np.full(n, 0.5, np.float32), sep=sep,
             passive_block24=rng.normal(size=(n, 1, 8)).astype(np.float16),
             passive_final=rng.normal(size=(n, 1, 8)).astype(np.float16))
    with open(enc / f"{name}.shard0.meta.jsonl", "w") as f:
        for i in range(n):
            m = {"uid": f"{name}{i}", "label": int(y[i])}
            if pb:
                m.update({"id": f"t{i // 4}", "step_idx": i % 4, "first_error": 1 if (i // 4) % 2 else -1})
            f.write(json.dumps(m) + "\n")


def test_lda_over_read_and_passive_rows(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for name in ("train", "val", "test"):
        write(tmp_path, name, 200, rng)
    for s in ipl.PB_SUBSETS:
        write(tmp_path, f"pb_{s}", 40, rng, pb=True)
    out = tmp_path / "lda.json"
    monkeypatch.setattr("sys.argv", ["x", "--enc", str(tmp_path), "--out", str(out)])
    ipl.main()
    r = json.loads(out.read_text())["reps"]
    assert set(r) == {"read", "passive_block24", "passive_final"}
    assert r["read"]["test_auroc"] > 0.8 > r["passive_final"]["test_auroc"]
