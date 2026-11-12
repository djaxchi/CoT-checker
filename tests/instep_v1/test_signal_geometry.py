import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scripts.instep_v1 import signal_geometry as G  # noqa: E402
from scripts.instep_v1 import signal_states as SS  # noqa: E402


def _write(d, name, uids, y, arrays, kept=None):
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{name}.shard0.npz", y=np.array(y, np.int8),
             kept=np.ones(len(y), bool) if kept is None else kept, **arrays)
    with open(d / f"{name}.shard0.meta.jsonl", "w") as f:
        for u in uids:
            f.write(json.dumps({"uid": u}) + "\n")


def _fixture(tmp_path, n=240, d=12, seed=0):
    """Text states carry the label on a weak low-variance axis, the step token on a strong
    one; the extra source holds a PRM-like key aligned on the same uids."""
    rng = np.random.default_rng(seed)
    st, ex, data = tmp_path / "st", tmp_path / "ex", tmp_path / "data"
    (data / "rp").mkdir(parents=True)
    val_traces = []
    for s in [G.TRAIN, "val", *G.TEST]:
        y = (rng.random(n) < 0.3).astype(int)
        uids = [f"{s}_t{i // 4}::{i % 4}" for i in range(n)]
        base = rng.normal(size=(n, d))
        big = np.diag(np.linspace(5, 0.5, d))
        text = base @ big
        text[:, -1] += 0.6 * y  # weak, on the smallest-variance direction
        step = base @ big
        step[:, 0] += 8.0 * y  # strong, on the dominant direction
        kept = np.ones(n, bool)
        kept[0] = False
        _write(st, s, uids, y if s != "val" else np.zeros(n, int),
               {"passive_final": text[:, None].astype(np.float16), "sep": step[:, None].astype(np.float16)}, kept)
        _write(ex, s, uids, y, {"prm_final": step[:, None].astype(np.float16)})
        if s == "val":
            for i in range(0, n, 4):
                val_traces.append({"trace_id": f"val_t{i // 4}", "labels": {"ds": y[i:i + 4].tolist()}})
    with open(data / "rp" / "val.traces.jsonl", "w") as f:
        for t in val_traces:
            f.write(json.dumps(t) + "\n")
    return st, ex, data


def test_signal_geometry_separates_dominant_from_hidden_axis(tmp_path):
    st, ex, data = _fixture(tmp_path)
    D = G.Data(st, [ex], data, "ds")
    assert D.mask["math"][0] == False and int(D.mask["math"].sum()) == 239  # noqa: E712
    out = tmp_path / "out"
    out.mkdir()
    r_text = G.analyze_key(D, "passive_final", (1e-2, 1.0), 6, "cpu", True, 20, out)
    r_step = G.analyze_key(D, "sep", (1e-2, 1.0), 6, "cpu", True, 20, out)
    # the step token's label lives on its first principal direction; the text state's does not
    assert r_step["best_pc"] == 1 and r_step["best_pc_auroc"] > 0.8
    assert r_text["best_pc"] != 1
    assert r_step["axis_var_share"] > 2 * r_text["axis_var_share"]
    assert r_step["ap"]["all"] > r_text["ap"]["all"]
    pts = (out / "points_sep.csv").read_text().splitlines()
    assert pts[0] == "set,y,x,y2" and len(pts) > 1
    assert r_step["ceiling_ap"]["all"] is not None


def test_signal_geometry_rejects_misaligned_extra(tmp_path):
    st, ex, data = _fixture(tmp_path)
    (ex / "math.shard0.meta.jsonl").write_text(json.dumps({"uid": "other::0"}) + "\n")
    with pytest.raises(SystemExit):
        G.Data(st, [ex], data, "ds")


def test_depth_and_family():
    assert G.depth("passive_block35") == 1.0 and G.depth("prm_block13") == 0.5 and G.depth("sep") == 1.0
    assert [G.family(k) for k in ("passive_final", "read_block3", "sep", "prm_final", "reprobe_rep")] == \
        ["text", "instep", "instep", "prm", "reprobe"]


def test_assemble_keeps_order_and_marks_missing():
    rows = {"a::1": (1, 0.9, {"k": np.ones(3)}), "a::0": (0, 0.1, {"k": np.zeros(3)})}
    arr = SS.assemble(["a::0", "a::1", "b::0"], rows, ["k"])
    assert arr["kept"].tolist() == [True, True, False]
    assert arr["y"].tolist() == [0, 1, -1] and arr["k"].shape == (3, 1, 3)
    assert arr["k"][1, 0].tolist() == [1, 1, 1]


def test_gate_spearman(tmp_path):
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps({"uid": f"u{i}", "p": i / 10}) for i in range(5)))
    arr = {"kept": np.ones(5, bool), "p": np.arange(5, dtype=np.float32) / 10}
    g = SS.gate("s", [f"u{i}" for i in range(5)], arr, tmp_path)
    assert g["pass"] and g["n"] == 5
    arr["p"] = arr["p"][::-1].copy()
    assert not SS.gate("s", [f"u{i}" for i in range(5)], arr, tmp_path)["pass"]


def test_fig_signal_from_geometry(tmp_path):
    from scripts.instep_v1 import fig_signal as F
    st, ex, data = _fixture(tmp_path)
    D = G.Data(st, [ex], data, "ds")
    out = tmp_path / "sig"
    out.mkdir()
    keys = [G.analyze_key(D, k, (1e-2,), 4, "cpu", k in G.FINAL, 20, out) for k in ("passive_final", "sep", "prm_final")]
    sig = {"keys": keys}
    dep = F.depth_fig(sig)
    assert "prmcol" in dep and "(100.0," in dep  # PRM final state is the PRM curve's last point
    assert "instepdark, mark=none" not in dep  # sep is excluded from the step-token curve (no blocks here)
    view = F.view_fig(sig, out, 10)
    assert view.count("\\begin{axis}") == 3 and "Text token" in view and "InStep" in view
    assert "| sep |" in F.summary_md(sig)


def test_signal_geometry_rejects_a_key_in_two_dirs(tmp_path):
    st, ex, data = _fixture(tmp_path)
    with pytest.raises(SystemExit):
        G.Data(st, [ex, ex], data, "ds")


def test_signal_pca_removes_set_offsets_and_finds_the_class_axis():
    from scripts.instep_v1.signal_pca import dprime, project
    rng = np.random.default_rng(0)
    n, d = 400, 30
    sets = np.repeat(["a", "b"], n)
    y = np.tile(np.r_[np.zeros(n // 2), np.ones(n // 2)], 2).astype(int)
    X = rng.normal(size=(2 * n, d))
    X[:, :10] += 1.5 * y[:, None]  # the class direction, spread over 10 dims
    X[sets == "b", 1] += 20.0     # a task offset, larger than the class gap
    from scripts.reprobe_cmp.analyze import auroc
    for scale in ("dim", "none"):
        Z, var = project(X, sets, y, 3, scale)
        assert auroc(y, Z[:, 0]) > 0.95 and dprime(Z[:, 0], y) > 2
        assert var[0] > var[1]
