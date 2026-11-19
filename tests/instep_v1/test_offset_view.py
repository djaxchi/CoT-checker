import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scripts.instep_v1 import offset_view as OV  # noqa: E402
from scripts.reprobe_cmp.analyze import TEST  # noqa: E402


def _fixture(tmp_path, n=160, d=6, seed=0):
    """States whose label lies on dimension 0, a head reading that dimension, the largest variance
    on dimension 1 (so PC1 carries no label)."""
    rng = np.random.default_rng(seed)
    st, data = tmp_path / "st", tmp_path / "data"
    st.mkdir()
    (data / "rp").mkdir(parents=True)
    w = np.zeros(d, np.float32)
    w[0] = 2.0
    vt = []
    for s in ["val", *TEST]:
        y = (rng.random(n) < 0.3).astype(int)
        X = rng.normal(size=(n, d))
        X[:, 1] *= 6.0
        X[:, 0] += 2.5 * y
        lg = X @ w - 1.0
        uids = [f"{s}_t{i // 4}::{i % 4}" for i in range(n)]
        np.savez(st / f"{s}.shard0.npz", sep=X[:, None].astype(np.float16), y=np.array(y if s != "val" else 0 * y, np.int8),
                 kept=np.ones(n, bool), head_p_correct=1 / (1 + np.exp(lg)))
        with open(st / f"{s}.shard0.meta.jsonl", "w") as f:
            f.writelines(json.dumps({"uid": u}) + "\n" for u in uids)
        if s == "val":
            vt += [{"trace_id": f"val_t{i // 4}", "labels": {"ds": y[i:i + 4].tolist()}} for i in range(0, n, 4)]
    with open(data / "rp" / "val.traces.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in vt)
    ck = tmp_path / "tok.pt"
    torch.save({"head": {"weight": torch.from_numpy(w)[None], "bias": torch.tensor([-1.0])}}, ck)
    return st, data, ck


def test_view_recovers_logit_and_unsupervised_pcs(tmp_path):
    st, data, ck = _fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    r = OV.view("t", st, ck, data, 3, 20, out, "cpu")
    assert r["logit"]["mean_auroc"] > 0.9
    assert abs(r["pc_auroc"][0]) < 0.65  # PC1 = the high-variance, label-free dimension
    assert r["cos_w_pc1"] < 0.2
    rows = list(csv.DictReader(open(out / "t.csv")))
    assert {"set", "y", "logit", "pc1", "spc1", "orth"} <= set(rows[0])


def test_head_weight_rejects_other_heads(tmp_path):
    ck = tmp_path / "bad.pt"
    torch.save({"head": {"0.weight": torch.zeros(4, 6), "2.weight": torch.zeros(2, 4)}}, ck)
    with pytest.raises(SystemExit):
        OV.head_weight(ck)


def test_signed_auroc_sign():
    y = np.array([0, 0, 1, 1])
    assert OV.signed_auroc(y, np.array([0.0, 1, 2, 3])) == 1.0
    assert OV.signed_auroc(y, np.array([3.0, 2, 1, 0])) == -1.0


def test_fig_offset_view_orients_and_balances(tmp_path):
    from scripts.instep_v1 import fig_offset_view as F
    X = np.array([[-1.0, 2.0], [-2.0, 1.0], [1.0, -1.0], [2.0, -2.0]])
    y = np.array([1, 1, 0, 0])
    Z = F.orient(X, y)
    assert (Z[y == 1].mean(0) > 0).all()  # incorrect steps on the positive side of both axes
    idx = F.sample(np.array([0] * 50 + [1] * 5), 10)
    assert (np.array([0] * 50 + [1] * 5)[idx] == 1).sum() == 5 and len(idx) == 15
    st, data, ck = _fixture(tmp_path)
    out = tmp_path / "view"
    out.mkdir()
    import json as _j
    (out / "summary.json").write_text(_j.dumps({"t": OV.view("t", st, ck, data, 3, 20, out, "cpu")}))
    tex = F.figure(out, ["t"], 30)
    assert tex.count("\\begin{axis}") == 1 and "errcol" in tex and "ylabel={PC 2}" in tex
