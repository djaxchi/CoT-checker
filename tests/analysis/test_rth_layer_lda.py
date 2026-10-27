import json

import numpy as np

from scripts.analysis import rth_layer_lda as m1


def _split(root, name, X, y, meta):
    part = root / name / "part_0"
    part.mkdir(parents=True)
    np.save(part / "read.npy", X.astype(np.float16))
    np.save(part / "y.npy", y.astype(np.int8))
    np.save(part / "index.npy", np.arange(len(y)))
    (part / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))


def test_signal_is_found_in_the_layer_that_carries_it(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    d = 8

    def make(y, bad):
        X = rng.normal(size=(len(y), 2, d))
        X[bad, 1, 0] += 2.0      # row 1 carries the label, row 0 does not
        return X

    for name, n in (("probe_train_full", 2000), ("val_5k", 600), ("test_2k", 600)):
        y = rng.integers(0, 2, n)
        _split(tmp_path, name, make(y, y == 1), y, [{"uid": f"{name}{i}"} for i in range(n)])
    for sub in m1.PB_SUBSETS:
        meta, bad = [], []
        for t in range(60):
            fe = 2 if t % 2 else -1
            for j in range(4):
                meta.append({"id": f"t{t}", "step_idx": j, "label": fe, "n_steps": 4})
                bad.append(fe != -1 and j >= fe)
        bad = np.array(bad)
        _split(tmp_path, sub, make(bad.astype(int), bad), bad.astype(int), meta)
    out = tmp_path / "m1.json"
    monkeypatch.setattr("sys.argv", ["x", "--bundle", str(tmp_path), "--rep", "read", "--out", str(out)])
    m1.main()
    r = json.loads(out.read_text())["layers"]
    assert abs(r["0"]["test_auroc"] - 0.5) < 0.08 and r["1"]["test_auroc"] > 0.8
    assert r["1"]["pb_auroc_mean"] > 0.8 and r["1"]["pb_calib20"] > r["0"]["pb_calib20"]
