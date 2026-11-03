import json
import sys

import numpy as np

from scripts.instep_v1 import passive_curve as pc
from tests.instep_v1.test_passive_fit import make, write_set


def test_curve_and_concat_rows(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    states, data = tmp_path / "states", tmp_path / "data"
    for d in (states, data / "rp", data / "rt"):
        d.mkdir(parents=True)
    keys = pc.CONCAT

    def fake(n):
        y, xs = make(n, rng)
        return y, {k: xs["passive_block7"] + rng.normal(size=xs["passive_block7"].shape) for k in keys}

    y, xs = fake(2000)
    write_set(states, "train40k", [f"tr{i}::0" for i in range(2000)], y, xs)
    yv, xv = fake(400)
    vu = [f"v{i}::0" for i in range(400)]
    write_set(states, "val", vu, np.zeros(400, int), xv)
    with open(data / "rp" / "val.traces.jsonl", "w") as f:
        for i in range(400):
            f.write(json.dumps({"trace_id": f"v{i}", "labels": {"ds": [int(yv[i])]}}) + "\n")
    from scripts.reprobe_cmp.analyze import TEST
    for s in TEST:
        ys, xt = fake(300)
        write_set(states, s, [f"{s}{i}::0" for i in range(300)], ys, xt)
    out = tmp_path / "curve.json"
    monkeypatch.setattr(sys, "argv", ["passive_curve.py", "--states", str(states), "--data", str(data),
                                      "--val_label", "ds", "--out", str(out), "--sizes", "200,2000"])
    pc.main()
    rows = json.loads(out.read_text())["rows"]
    assert [r["n_train"] for r in rows] == [200, 2000, 2000, 2000]  # 20000 capped at the 2000 rows
    assert rows[2]["keys"] == list(keys)
    assert rows[1]["val_ap"] > rows[0]["val_ap"]  # more training rows help
    # training states split in two parts in another dir: same rows, same fit
    other = tmp_path / "full"
    other.mkdir()
    write_set(other, "train_p0", [f"tr{i}::0" for i in range(1000)], y[:1000], {k: v[:1000] for k, v in xs.items()})
    write_set(other, "train_p1", [f"tr{i}::0" for i in range(1000, 2000)], y[1000:], {k: v[1000:] for k, v in xs.items()})
    out2 = tmp_path / "curve2.json"
    monkeypatch.setattr(sys, "argv", ["passive_curve.py", "--states", str(states), "--data", str(data),
                                      "--val_label", "ds", "--out", str(out2), "--sizes", "0",
                                      "--concat_sizes", "0", "--train_states", str(other),
                                      "--train_names", "train_p0,train_p1", "--scores_out", str(tmp_path / "sc")])
    pc.main()
    rows2 = json.loads(out2.read_text())["rows"]
    assert [r["n_train"] for r in rows2] == [2000, 2000]
    assert abs(rows2[0]["val_ap"] - rows[1]["val_ap"]) < 1e-6 and abs(rows2[1]["val_ap"] - rows[3]["val_ap"]) < 1e-6
    assert rows[3]["val_ap"] > rows[1]["val_ap"]  # five noisy copies of the signal beat one
    lines = (tmp_path / "sc" / "concat5" / "math.jsonl").read_text().splitlines()
    assert len(lines) == 300 and json.loads(lines[0])["uid"] == "math0::0"
    assert (tmp_path / "sc" / "final" / "val.jsonl").exists() and not (tmp_path / "sc" / "final" / "train40k.jsonl").exists()
