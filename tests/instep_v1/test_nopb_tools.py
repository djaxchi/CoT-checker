import json
import math

import numpy as np
import pytest

from scripts.analysis import instep_lr_select as SEL
from scripts.instep_v1 import nopb_tools as T
from tests.instep_v1.test_paper_v1 import tree  # noqa: F401  (fixture)

LOG = """[config] model=Qwen/Qwen3-8B ...
[train inline] step 50/200 loss lr0.003_wu0.05 0.6000 (10s; text 7s, reads 3s)
[train inline] step 100/200 loss lr0.003_wu0.05 0.5000 (20s; text 14s, reads 6s)
[train inline] step 150/200 loss lr0.003_wu0.05 0.4500 (30s; text 21s, reads 9s)
[train inline] step 200/200 loss lr0.003_wu0.05 0.4316 (40s; text 28s, reads 12s)
[encode] token ...
"""


def test_const_loss_is_binary_entropy():
    assert T.const_loss(0.5) == pytest.approx(math.log(2))
    assert T.const_loss(0.0) == 0.0


def test_parse_log_reads_trace_and_end():
    r = T.parse_log(LOG)
    assert r["steps"] == 200 and r["tag"] == "lr0.003_wu0.05" and r["dead_at"] is None
    assert r["trace"][0] == [50, 0.6] and r["final_loss"] == 0.4316 and r["finished"]


def test_parse_log_flags_a_dead_run():
    r = T.parse_log(LOG.replace("step 150/200 loss lr0.003_wu0.05 0.4500", "step 150/200 loss lr0.003_wu0.05 dead"))
    assert r["dead_at"] == 150


def test_filter_draws_matches_the_training_draw(tmp_path):
    rows = [{"uid": f"u{i}", "problem": "PB  Problem" if i % 5 == 0 else f"p{i}", "label": i % 2} for i in range(50)]
    lines = [json.dumps(r) for r in rows]
    uids, info = T.filter_draws(lines, {"pb problem"}, [0], 20)
    drawn = [rows[i] for i in T.draw_indices(50, 20, 0)]
    want = [r["uid"] for r in drawn if r["problem"] != "PB  Problem"]
    assert uids[0] == want  # normalised match (case, whitespace), order kept
    assert info[0]["drawn"] == 20 and info[0]["kept"] == len(want) and info[0]["excluded"] == 20 - len(want)
    y = np.array([r["label"] for r in drawn if r["problem"] != "PB  Problem"])
    assert info[0]["share_incorrect"] == pytest.approx(y.mean())


def _sweep(tmp_path, seed, auroc_shift, final=0.4316):
    log = T.parse_log(LOG.replace("0.4316", f"{final:.4f}"))
    counts = {"kept": 100, "share_incorrect": 0.3, "const_loss": T.const_loss(0.3)}
    val = {"val_auroc": 0.95 - auroc_shift, "val_ap": 0.8, "val_n": 10, "val_prev": 0.3}
    d = tmp_path / f"seed{seed}"
    d.mkdir()
    (d / "sweep.json").write_text(json.dumps(T.build_sweep(log, seed, counts, val, "tr", "va", 3e-3, 0.05)))
    return str(d / "sweep.json")


def test_reconstructed_sweeps_go_through_the_frozen_rule(tmp_path):
    files = [_sweep(tmp_path, s, d) for s, d in ((0, 0.0), (1, 0.06), (2, 0.0), (3, 0.01))]
    t = SEL.summarize(files, "auroc")["lr0.003_wu0.05"]
    assert t["seeds"][1]["flags"] and "below the other seeds' median" in t["seeds"][1]["flags"][0]
    assert not t["seeds"][0]["flags"] and not t["seeds"][3]["flags"]


def test_rule_flags_a_late_loss_above_the_constant(tmp_path):
    files = [_sweep(tmp_path, 0, 0.0, final=0.7), _sweep(tmp_path, 1, 0.0)]
    t = SEL.summarize(files, "auroc")["lr0.003_wu0.05"]
    assert any("above the constant" in f for f in t["seeds"][0]["flags"])  # 0.7 > H(0.3) = 0.611


def test_nopb_report_end_to_end(tree, tmp_path, monkeypatch):
    """paper_v1 with the no-overlap arms (override + extra pairs), pb_oracle, the stability rule on
    rebuilt sweeps, then nopb_report: every table is written and the non-InStep diff is empty."""
    import sys
    from scripts.instep_v1 import nopb_report as R
    from scripts.instep_v1 import paper_v1 as PV
    from scripts.instep_v1 import pb_oracle as PO
    common = ["--out_root", str(tree["out"]), "--v1", str(tree["v1"]), "--v2", str(tree["v2"]), "--n_boot", "20",
              "--pb_boot", "20", "--workers", "1", "--backbones", "qwen3_8b", "--no_bon", "--no_figures"]
    ref = tmp_path / "ref"
    assert PV.main(common + ["--final", str(ref)]) == 0
    rt = lambda s: str(tree["out"] / "rt" / f"instep_prm_seed{s}")  # noqa: E731
    seeds = [0, 1]
    ov = ["qwen3_8b:instep_prm_nopb=" + rt(0), "qwen3_8b:instep_prm_nopb_all=" + ",".join(rt(s) for s in seeds)]
    arms = ["instep_prm_nopb", "instep_prm_nopb_all"] + [f"instep_prm_nopb_seed{s}" for s in seeds]
    ov += [f"qwen3_8b:instep_prm_nopb_seed{s}={rt(s)}" for s in seeds]
    tp = [x for k in arms for x in (f"instep_ds-{k}", f"{k}-prm7b")]
    pp = [f"{k}-{c}" for k in arms for c in ("prm7b", "prm800k", "instep_ds", "instep_self", "rp_rel_ds")]
    fin = tmp_path / "nopb" / "final"
    assert PV.main(common + ["--final", str(fin), "--override", *ov, "--extra_pairs", *tp,
                             "--extra_pb_pairs", *pp]) == 0
    pa = [f"{k}=" + ",".join(str(d) for d in sorted((fin / "pb_scores" / "qwen3_8b" / k).glob("seed*")))
          for k in ["instep_prm"] + arms]
    orc = tmp_path / "nopb" / "pb" / "pb_oracle.json"
    monkeypatch.setattr(sys, "argv", ["pb_oracle", "--arm", *pa, "--out", str(orc)])
    PO.main()
    sw = tmp_path / "sweep"
    sw.mkdir()
    files = [_sweep(sw, s, d) for s, d in ((0, 0.0), (1, 0.08))]
    sel = {"tables": {"instep_prm_nopb": SEL.summarize(files, "auroc")}}
    (tmp_path / "sel.json").write_text(json.dumps(sel))
    filt = {"seeds": {str(s): {"drawn": 120, "excluded": 20, "kept": 100, "share_incorrect": 0.3,
                               "const_loss": T.const_loss(0.3)} for s in seeds}}
    (tmp_path / "filter.json").write_text(json.dumps(filt))
    logs = tmp_path / "logs"
    logs.mkdir()
    for s in seeds:
        (logs / f"instep_prm_nopb_seed{s}-1.log").write_text(
            "[data] 100 items -> 60 sequences, 400 read slots (0 read alone, 0 sequences over 3072 tokens dropped); "
            "200 optimizer steps at <= 56 items (RTH: 190 steps, 55.7 items each)\n")
    out, js = tmp_path / "report.md", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "nopb_report", "--selection", str(tmp_path / "sel.json"), "--sweeps", str(sw), "--filter",
        str(tmp_path / "filter.json"), "--logs", str(logs), "--final", str(fin / "final.json"), "--reference",
        str(ref / "final.json"), "--pb_oracle", str(orc), "--out", str(out), "--json", str(js), "--seeds", "0,1"])
    R.main()
    res = json.loads(js.read_text())
    assert res["kept"] == [0] and not res["stability"][0]["flags"] and res["stability"][1]["flags"]
    assert res["diffs_outside_instep_prm"] == []
    assert res["training"][0]["optimiser_steps"] == 200 and res["training"][0]["cap"] == 56
    md = out.read_text()
    for h in ("## Stability rule", "## Training data", "## ProcessBench (", "paired intervals", "eight test sets",
              "All equal."):
        assert h in md, h
    a = res["arms"]["instep_prm_nopb_all"]
    assert set(a["p05"]) >= {"gsm8k", "math", "mean"} and a["pb_pairs"]["prm7b"] is not None
