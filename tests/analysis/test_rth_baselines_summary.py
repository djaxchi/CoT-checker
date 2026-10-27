import json

import torch

from scripts.analysis import rth_baselines_summary as bs


def test_trained_params_match_the_plan_counts():
    d = 4096
    head = {"weight": torch.zeros(1, d), "bias": torch.zeros(1)}
    deep = torch.zeros(36, d)
    assert bs.n_trained({"head": head, "soft": torch.zeros(1, d), "deep": deep}) == 155_649  # RTH
    assert bs.n_trained({"head": head, "soft": torch.zeros(1, d)}) == 8_193  # BL1
    assert bs.n_trained({"head": head, "soft": torch.zeros(1, d), "deep": deep, "freeze_soft": True}) == 151_553
    lora = {"A_q": torch.zeros(36, 8, d), "B_q": torch.zeros(36, d, 8)}
    assert bs.n_trained({"head": head, "soft": torch.zeros(1, d), "interv_state": lora}) == 8_193 + 2 * 36 * 8 * d


def test_readings_thresholds():
    r = {t.split(":")[0]: met for t, met, _ in bs.readings(
        {"BL1": -0.03, "BL2": 0.005, "BL3": 0.015, "BL4": 0.01},
        {"BL1": (0.52, 0.46, 0.58), "BL4": (0.61, 0.55, 0.66)})}
    assert r["BL1 at least 0.02 below RTH"] and r["BL2 within 0.01 of RTH"]
    assert not r["BL3 within 0.01 of RTH"] and not r["BL3 at least 0.02 above RTH"]
    assert r["BL4 at least 0.02 above RTH, or any baseline >= 0.60 on the confidence-wrong pairs with the interval above 0.5"]
    r2 = bs.readings({"BL4": 0.01}, {"BL4": (0.61, 0.49, 0.70)})
    assert not r2[-1][1]


def test_end_to_end_picks_the_val_best_seed_for_pairs(tmp_path, monkeypatch):
    d = 8
    runs = {"RTH": bs.RTH, "BL1": ["b0", "b1", "b2"]}
    for k, rs in runs.items():
        for i, r in enumerate(rs):
            g = tmp_path / f"geometry_{r}"
            g.mkdir()
            g.joinpath("geometry.json").write_text(json.dumps([
                {"method": "prm_head", "test_auroc": 0.9, "pb_calib20": 0.66 - (k == "BL1") * 0.05 + i * 0.001},
                {"method": "lda", "test_auroc": 0.9, "pb_calib20": 0.6}]))
            (tmp_path / "tokens" / r).mkdir(parents=True)
            torch.save({"head": {"weight": torch.zeros(1, d), "bias": torch.zeros(1)}, "soft": torch.zeros(1, d)},
                       tmp_path / "tokens" / r / "read_token.pt")
    for i, r in enumerate(runs["BL1"]):
        (tmp_path / f"pairs_{r}.json").write_text(json.dumps(
            {"math500": {"wrong_conf_question_weighted": {"probe": [0.5 + i / 10, 0.45 + i / 10, 0.55 + i / 10]}}}))
    (tmp_path / "val.json").write_text(json.dumps({"val_auroc": {"b0": 0.80, "b1": 0.85, "b2": 0.82}}))
    (tmp_path / "boot.json").write_text(json.dumps([{"comparison": "BL1 - RTH", "delta": -0.05,
                                                     "ci95_normal": [-0.07, -0.03]}]))
    out = tmp_path / "summary.md"
    monkeypatch.setattr("sys.argv", ["x", "--root", str(tmp_path), "--bootstrap", str(tmp_path / "boot.json"),
                                     "--val", str(tmp_path / "val.json"), "--pairs_dir", str(tmp_path),
                                     "--out", str(out), "--group", "BL1=b0,b1,b2"])
    monkeypatch.setattr(bs, "GROUPS", {})
    bs.main()
    res = json.loads(out.with_suffix(".json").read_text())
    assert res["groups"]["BL1"]["pairs_pick"] == "b1"
    assert res["groups"]["BL1"]["params"] == 2 * d + 1
    assert "| BL1 | 17 |" in out.read_text()
    met = {x["reading"].split(":")[0]: x["met"] for x in res["readings"]}
    assert met["BL1 at least 0.02 below RTH"]
