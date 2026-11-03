"""Tests for the InStep V1 final aggregation (scripts/instep_v1/paper_v1.py and helpers),
on synthetic score directories laid out like TamIA's."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.instep_v1 import paper_numbers as PN  # noqa: E402
from scripts.instep_v1 import paper_v1 as PV  # noqa: E402
from scripts.instep_v1 import pb_metrics as PBM  # noqa: E402

TEST = PV.TEST
PBS = PBM.PB


# ---------------------------------------------------------------------------- fixtures
def _wjl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _make_data(root: Path, rng: np.random.Generator, n_traces: int = 12) -> dict:
    """data/test, data/rp/val, data/pb with labels; returns {name: {uid: label}}."""
    labels = {}
    for s in TEST:
        rows, lab = [], {}
        for t in range(n_traces):
            n = int(rng.integers(2, 6))
            y = [int(rng.random() < 0.35) for _ in range(n)]
            y[0] = -1 if t == 0 else y[0]  # one unlabeled step
            tid = f"{s}_{t:03d}"
            for k, v in enumerate(y):
                if v >= 0:
                    lab[f"{tid}::{k}"] = v
            rnd = [float(rng.random()) for _ in y]
            rows.append({"trace_id": tid, "steps": ["x"] * n, "labels": y,
                         "scores": {"RandomBaselineClaim": rnd, "MaximumClaimProbability": rnd,
                                    "LuqClaimEstimatorDummy_claim": [v * 0.6 + r * 0.4 for v, r in zip(y, rnd)]}})
        _wjl(root / "data" / "test" / f"{s}.traces.jsonl", rows)
        labels[s] = lab
    rows, lab = [], {}
    for t in range(30):
        n = int(rng.integers(2, 6))
        y = [int(rng.random() < 0.3) for _ in range(n)]
        tid = f"val_{t:03d}"
        lab.update({f"{tid}::{k}": v for k, v in enumerate(y)})
        rows.append({"trace_id": tid, "steps": ["x"] * n, "labels": {"ds": y, "self": y}})
    _wjl(root / "data" / "rp" / "val.traces.jsonl", rows)
    labels["val"] = lab
    for sub in PBS:
        rows, lab = [], {}
        for t in range(25):
            n = int(rng.integers(2, 7))
            fe = -1 if rng.random() < 0.4 else int(rng.integers(0, n))
            tid = f"pb{sub}_{t:03d}"
            rows.append({"trace_id": tid, "steps": ["x"] * n, "first_error": fe})
            for k in range(n):
                lab[f"{tid}::{k}"] = int(fe >= 0 and k >= fe)
        _wjl(root / "data" / "pb" / f"{sub}.traces.jsonl", rows)
        labels[f"pb_{sub}"] = lab
    return labels


def _scores(lab: dict, rng, signal: float) -> dict:
    return {u: float(np.clip(signal * y + rng.normal(0, 0.3) + 0.2, 0.001, 0.999)) for u, y in lab.items()}


def _write_rp(d: Path, labels: dict, rng, signal: float, names) -> None:
    for n in names:
        _wjl(d / f"{n}.jsonl", [{"uid": u, "p": p} for u, p in _scores(labels[n], rng, signal).items()])


def _write_rt(d: Path, labels: dict, rng, signal: float, names) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        sc = _scores(labels[n], rng, signal)
        uids = list(sc)
        pc = np.array([1 - sc[u] for u in uids])
        np.savez(d / f"{n}.shard0.npz", head_p_correct=pc, y=np.zeros(len(uids)), kept=np.ones(len(uids)))
        _wjl(d / f"{n}.shard0.meta.jsonl", [{"uid": u} for u in uids])


@pytest.fixture()
def tree(tmp_path):
    rng = np.random.default_rng(0)
    v1, v2, out = tmp_path / "v1", tmp_path / "v2", tmp_path / "out"
    labels = _make_data(v1, rng)
    allsets = list(TEST) + ["val"] + [f"pb_{s}" for s in PBS]
    for i in range(2):  # InStep-DS: 2 of 3 seeds present
        _write_rt(out / "rt" / f"instep_ds_seed{i}", labels, rng, 0.6, allsets)
    for i in range(3):
        _write_rt(out / "rt" / f"instep_prm_seed{i}", labels, rng, 0.5, allsets)
        _write_rp(v1 / "runs" / "rp" / f"seed{i}" / "scores" / "rp_ds", labels, rng, 0.3, allsets)
        _write_rt(out / "rt" / f"emb_ds_seed{i}", labels, rng, 0.2, list(TEST) + ["val"])
    _write_rp(v1 / "runs" / "released" / "scores" / "rel_ds", labels, rng, 0.3, allsets)
    _write_rp(out / "scores" / "qwen3_8b" / "prm7b", labels, rng, 0.4, allsets)
    for i in range(2):  # seed-subdir layout
        _write_rp(out / "scores" / "qwen3_8b" / "maxprob" / f"seed{i}", labels, rng, 0.05, allsets)
    # passive misses math and val: partial arm
    _write_rp(out / "scores" / "qwen3_8b" / "passive", labels, rng, 0.25, [s for s in TEST if s != "math"])
    # an rp_ds seed with a broken set (missing uids) must be skipped, not crash
    bad = v1 / "runs" / "rp" / "seed2" / "scores" / "rp_ds" / "gsm8k.jsonl"
    bad.write_text("".join(bad.read_text().splitlines(keepends=True)[:3]))
    return {"v1": v1, "v2": v2, "out": out, "labels": labels}


# ---------------------------------------------------------------------------- paper numbers
def test_paper_tables_internally_consistent():
    for bb, t in PN.TABLES.items():
        for m, v in t["rows"].items():
            assert len(v) == 11, (bb, m)
            assert abs(np.mean(v[:8]) - v[10]) <= 0.0016, (bb, m)
            assert abs(np.mean(v[:3]) - v[8]) <= 0.0016, (bb, m)
            assert abs(np.mean(v[3:8]) - v[9]) <= 0.0016, (bb, m)


def test_paper_known_values():
    q = PN.per_set("qwen3_8b", PN.PRM7B)
    assert [q[s] for s in TEST] == [.531, .702, .310, .711, .757, .745, .334, .429]
    assert q["overall"] == .565
    assert PN.per_set("qwen3_8b", PN.RANDOM)["overall"] == .278
    assert PN.per_set("qwen3_8b", PN.RP_HS_SELF)["overall"] == .604
    assert PN.per_set("qwen3_8b", PN.RP_HS_DS)["overall"] == .582
    assert PN.per_set("phi4", PN.PRM800K)["overall"] == .474
    assert PN.per_set("phi4", PN.RP_AL_Q3)["overall"] == .448
    assert PN.per_set("qwen3_32b", PN.RP_HS_OSS)["overall"] == .585


def test_table5_column_fix():
    printed = PN.per_set("qwen3_32b", PN.RP_HS_OSS, corrected=False)
    fixed = PN.per_set("qwen3_32b", PN.RP_HS_OSS)
    assert printed["meetings"] == .734 and fixed["meetings"] == .698
    assert printed["strategyqa"] == .302 and fixed["strategyqa"] == .303
    assert fixed["gsm8k"] == printed["gsm8k"] == .535
    assert PN.per_set("phi4", PN.MAXPROB)["t9"] == pytest.approx(np.mean([.127, .084, .123, .252, .158]))
    assert PN.per_set("qwen3_8b", "nope") == {}


# ---------------------------------------------------------------------------- pb metrics
def test_fast_metrics_match_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = (rng.random(200) < 0.2).astype(int)
        s = np.round(rng.random(200) + 0.3 * y, 1)  # ties
        assert PBM.fast_auroc(y, s) == pytest.approx(roc_auc_score(y, s))
        assert PBM.fast_ap(y, s) == pytest.approx(average_precision_score(y, s))
    assert np.isnan(PBM.fast_auroc(np.zeros(5, int), np.arange(5.0)))


def test_pb_items_drop_steps_after_first_error(tmp_path):
    _wjl(tmp_path / "pb" / "gsm8k.traces.jsonl", [
        {"trace_id": "a", "steps": ["s"] * 5, "first_error": 2},
        {"trace_id": "b", "steps": ["s"] * 3, "first_error": -1}])
    it = PBM.load_items(tmp_path)["gsm8k"]
    assert it["uids"] == ["a::0", "a::1", "a::2", "b::0", "b::1", "b::2"]
    assert it["y"].tolist() == [0, 0, 1, 0, 0, 0]
    assert it["trace"].tolist() == [0, 0, 0, 1, 1, 1]


def test_pb_pair_bootstrap_sign():
    rng = np.random.default_rng(2)
    items = {}
    for sub in PBS:
        tr = np.repeat(np.arange(40), 4)
        y = np.zeros(len(tr), int)
        y[::4][:25] = 1
        items[sub] = {"uids": [str(i) for i in range(len(tr))], "y": y, "trace": tr}
    good = {s: [items[s]["y"] + rng.normal(0, 0.3, len(items[s]["y"]))] for s in PBS}
    bad = {s: [rng.random(len(items[s]["y"]))] for s in PBS}
    idx = PBM.boot_indices(items, 100)
    r = PBM.boot_pair(items, good, bad, idx)
    assert r["auroc_mean"]["delta"] > 0.2 and r["auroc_mean"]["ci95"][0] > 0
    assert r["ap_mean"]["share_le0"] == 0.0


# ---------------------------------------------------------------------------- helpers
def test_resolve_kinds(tree):
    k, d = PV.resolve(str(tree["out"] / "rt" / "instep_ds_seed0"))
    assert k == "rt" and len(d) == 1
    k, d = PV.resolve(str(tree["out"] / "scores" / "qwen3_8b" / "maxprob"))
    assert k == "rp" and [Path(x).name for x in d] == ["seed0", "seed1"]
    assert PV.resolve(str(tree["out"] / "nope")) == (None, [])


def test_parse_per_layer_formats():
    a = PV.parse_per_layer({"passive": {"3": 0.4, "7": {"val_ap": 0.5}}, "read": {"3": 0.45, "7": 0.6}})
    assert a["passive"] == [(3, 0.4), (7, 0.5)] and a["read"] == [(3, 0.45), (7, 0.6)]
    b = PV.parse_per_layer({"rows": [{"kind": "passive", "block": 3, "ap": 0.4},
                                     {"kind": "passive", "block": 7, "ap": 0.5},
                                     {"kind": "read", "block": 3, "ap": 0.6},
                                     {"kind": "read", "block": 7, "ap": 0.7}]})
    assert b["passive"] == [(3, 0.4), (7, 0.5)] and b["read"] == [(3, 0.6), (7, 0.7)]
    assert PV.parse_per_layer({"note": "nothing"}) == {}


def test_abstention_perfect_ranker():
    tests = {s: {"y": np.array([1, 0, 0, 0]), "trace": np.array([0, 0, 1, 1]), "uids": []} for s in TEST}
    arm = {"test": {s: [np.array([0.9, 0.1, 0.2, 0.1])] for s in TEST}}
    r = PV.abstention(arm, tests)
    assert r["n_traces"] == 16 and r["base_rate"] == 0.5
    assert r["risk_at"]["0.5"] == 0.0  # the 8 clean traces come first


# ---------------------------------------------------------------------------- end to end
def _run(tree, final: Path, extra=()):
    return PV.main(["--out_root", str(tree["out"]), "--v1", str(tree["v1"]), "--v2", str(tree["v2"]),
                    "--final", str(final), "--n_boot", "30", "--pb_boot", "30", "--workers", "1",
                    "--backbones", "qwen3_8b,phi4", "--no_bon", *extra])


def test_end_to_end_degrades_gracefully(tree, tmp_path):
    final = tmp_path / "final"
    assert _run(tree, final) == 0
    F = json.loads((final / "final.json").read_text())
    assert not F["errors"], F["errors"]
    R = F["backbones"]["qwen3_8b"]
    M = R["metrics"]
    for k in ("instep_ds", "instep_prm", "rp_ds", "rp_rel_ds", "prm7b", "maxprob", "passive", "emb_ds", "rp_al"):
        assert k in M, k
    assert M["instep_ds"]["math"]["n_seeds"] == 2
    assert M["rp_ds"]["gsm8k"]["n_seeds"] == 2 and M["rp_ds"]["math"]["n_seeds"] == 3
    assert M["maxprob"]["math"]["n_seeds"] == 2
    assert "math" not in M["passive"] and "overall" not in M["passive"] and "ood" in M["passive"]
    assert "val_f1" in M["instep_ds"]["overall"] and "val_f1" not in M["passive"].get("ood", {})
    # metrics agree with analyze.py's functions on seed 0
    pairs = {(p["a"], p["b"]): p for p in R["pairs"]}
    assert pairs[("instep_ds", "rp_ds")]["overall"]["ap"]["delta"] == pytest.approx(
        M["instep_ds"]["overall"]["ap"] - M["rp_ds"]["overall"]["ap"])
    assert "overall" not in pairs[("instep_ds", "passive")] and "ood" in pairs[("instep_ds", "passive")]
    assert [(p["a"], p["b"]) for p in R["ladder_pairs"]] == [("instep_ds", "emb_ds")]
    # ProcessBench
    assert set(R["pb"]) >= {"instep_ds", "instep_prm", "rp_ds", "prm7b", "maxprob"}
    assert "passive" not in R["pb"]
    pbp = {(p["a"], p["b"]) for p in R["pb_pairs"]}
    assert ("instep_prm", "prm7b") in pbp and ("instep_ds", "rp_ds") in pbp
    # consistency check ran on the stored columns
    assert any(c["method"] == PN.RP_AL_DS for c in R["consistency"])
    # missing items are listed, phi4 skipped
    miss = "\n".join(F["missing"])
    for w in ("ptrue: not found", "prm800k: not found", "phi4: data root", "rp_ds: seed2/scores/rp_ds gsm8k unusable",
              "ladder rung off_ds missing", "qwen3_8b/passive has no math scores"):
        assert w in miss, w
    md = (final / "results.md").read_text()
    assert "## What is missing" in md and "Tier B: ReProbe paper Table 1" in md
    assert "ProcessBench first-error AUROC" in md and "Ladder on Qwen3-8B" in md
    tex = (final / "tables.tex").read_text()
    assert tex.count(r"\begin{tabular}") == len(F["tables"])
    for f in ("fig2_pareto", "fig3_backbones", "fig6_abstention", "fig4_ladder"):
        assert (final / f"{f}.png").exists() and (final / f"{f}.pdf").exists(), f


def test_end_to_end_empty_tree(tmp_path):
    final = tmp_path / "final"
    rc = PV.main(["--out_root", str(tmp_path / "o"), "--v1", str(tmp_path / "x"), "--v2", str(tmp_path / "y"),
                  "--final", str(final), "--workers", "1", "--n_boot", "5", "--no_bon"])
    assert rc == 0
    F = json.loads((final / "final.json").read_text())
    assert F["backbones"] == {} and not F["errors"]
    assert "qwen3_8b: data root" in "\n".join(F["missing"])
    assert (final / "results.md").exists() and (final / "tables.tex").exists()


def test_override_and_workers(tree, tmp_path):
    """--override replaces an arm's dirs; the forked pool gives the same numbers as serial."""
    final = tmp_path / "f2"
    ov = f"qwen3_8b:ptrue={tree['out'] / 'scores' / 'qwen3_8b' / 'prm7b'}"
    assert _run(tree, final, ["--override", ov, "--no_figures", "--no_calib20"]) == 0
    F = json.loads((final / "final.json").read_text())
    assert "ptrue" in F["backbones"]["qwen3_8b"]["metrics"]
    final3 = tmp_path / "f3"
    PV.main(["--out_root", str(tree["out"]), "--v1", str(tree["v1"]), "--v2", str(tree["v2"]), "--final",
             str(final3), "--n_boot", "30", "--pb_boot", "30", "--workers", "3", "--backbones", "qwen3_8b",
             "--no_bon", "--no_figures", "--no_calib20", "--override", ov])
    G = json.loads((final3 / "final.json").read_text())
    a = F["backbones"]["qwen3_8b"]["pairs"][0]["overall"]["ap"]
    b = G["backbones"]["qwen3_8b"]["pairs"][0]["overall"]["ap"]
    assert a == b
