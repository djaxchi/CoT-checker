import json

import numpy as np

from scripts.analysis import save_split_vectors as ssv
from src.repstore.store import TOKEN_SEQ, RepSpec, write_split


def test_last_rows_in_global_order(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    items = [rng.normal(size=(n, 4)).astype(np.float32) for n in (3, 1, 2)]
    meta = [{"uid": f"u{k}", "global_index": g, "pre_step_boundary_idx": 0} for k, g in enumerate((2, 0, 1))]
    root = tmp_path / "prm"
    write_split(root / "test_2k" / "shard_00", items, [0, 1, 0], meta,
                RepSpec(name="s", kind=TOKEN_SEQ, dim=4, layer=1, backbone="t", readout="t"))
    monkeypatch.setattr("sys.argv", ["x", "--out", str(tmp_path / "v"), f"a={root}"])
    ssv.main()
    h = np.load(tmp_path / "v" / "a" / "test_2k" / "h.npy")
    uids = [json.loads(l)["uid"] for l in (tmp_path / "v" / "a" / "test_2k" / "meta.jsonl").read_text().splitlines()]
    assert uids == ["u1", "u2", "u0"]
    assert np.allclose(h, np.stack([items[1][-1], items[2][-1], items[0][-1]]), atol=1e-2)
    assert list(np.load(tmp_path / "v" / "a" / "test_2k" / "y.npy")) == [1, 0, 0]
