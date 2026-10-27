import numpy as np
import torch

from scripts.analysis import rth_mech_confidence as m4
from src.onpolicy.prompts import verifier_prefix
from tests.analysis.test_rth_mech_attention import OffTok, tiny_model

ITEMS = [{"problem": "2+2?", "prefix": "a\n\nb", "step": "so 4"},
         {"problem": "x", "prefix": "", "step": "ok then"},
         {"problem": "long one", "prefix": "c", "step": "z"}]


def test_confidence_matches_an_unpadded_forward():
    tok, model = OffTok(), tiny_model()
    got = m4.confidence(model, tok, ITEMS, 0, torch.device("cpu"), max_tokens=10_000, chunk=2)
    for i, it in enumerate(ITEMS):
        pre = tok(verifier_prefix(it["problem"], it["prefix"]))["input_ids"]
        step = tok(it["step"])["input_ids"]
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([pre + step])).logits[0].float(), -1)
        tl = torch.stack([lp[len(pre) - 1 + j, t] for j, t in enumerate(step)])
        assert abs(got["logp_mean"][i] - float(tl.mean())) < 1e-4
        assert abs(got["logp_min"][i] - float(tl.min())) < 1e-4
        rows = lp[len(pre) - 1:len(pre) - 1 + len(step)]
        ent = float((-(rows.exp() * rows).sum(-1)).mean())
        assert abs(got["entropy_mean"][i] - ent) < 1e-4


def test_ridge_recovers_a_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 20)).astype(np.float32)
    beta = rng.normal(size=20)
    y = X @ beta + 3.0 + 0.01 * rng.normal(size=3000)
    b, icpt, r2 = m4.ridge_fit(X[:2000], y[:2000], X[2000:], y[2000:], torch.device("cpu"))
    assert r2 > 0.999 and np.allclose(b, beta, atol=0.01) and abs(icpt - 3.0) < 0.01


def test_heads_remove_the_confidence_direction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5000, 6))
    beta = np.array([1.0, 0.5, 0, 0, 0, 0])
    conf = X @ beta + 0.1 * rng.normal(size=5000)
    w = np.array([1.0, 0.0, 1.0, 0, 0, 0])
    H = m4.heads(w, beta, X, conf)
    c = beta / np.linalg.norm(beta)
    assert abs(H["w_proj"] @ c) < 1e-12
    assert abs(np.corrcoef(X @ H["w_decor"], conf)[0, 1]) < 1e-10
    assert np.array_equal(H["w"], w)


def test_pair_metrics_on_a_toy_pool():
    # one question, answers: two correct "1", two wrong "2" / "3"
    trajs = {f"t{k}": {"traj_uid": f"t{k}", "fork_id": "q", "gradeable": True, "correct": k < 2,
                       "pred": ["1", "1", "2", "3"][k]} for k in range(4)}
    conf = {u: {"mean_token_conf": v} for u, v in zip(trajs, (1.0, 0.0, 2.0, -1.0))}
    good = {"t0": 0.9, "t1": 0.8, "t2": 0.1, "t3": 0.2}
    bad = {u: -v for u, v in good.items()}
    res = m4.pair_metrics(trajs, conf, {"good": good, "bad": bad}, n_boot=50)
    assert res["n_pairs"] == 4 and res["n_questions"] == 1
    # confidence orientation: corr(conf, correct) < 0 here, so it is flipped; then it prefers
    # the wrong one on (t0, t3) and (t1, t3)
    assert res["n_conf_wrong"] == 2
    assert res["cells"]["good"]["all"][0] == 1.0 and res["cells"]["bad"]["all"][0] == 0.0
    assert res["cells"]["bad"]["conf_wrong_minus_good"][0] == -1.0


def _bundle_split(root, name, n, rng, d, pb):
    import json
    part = root / name / "part_0"
    part.mkdir(parents=True)
    if pb:
        meta = [{"id": f"t{i // 4}", "step_idx": i % 4, "label": (2 if (i // 4) % 2 else -1), "n_steps": 4}
                for i in range(n)]
        y = np.array([int(m["label"] >= 0 and m["step_idx"] >= m["label"]) for m in meta])
    else:
        y = rng.integers(0, 2, n)
        meta = [{"uid": f"{name}{i}"} for i in range(n)]
    conf = rng.normal(size=n)
    for rep, rows in (("read", 37), ("last_step", 37), ("prm", 29)):
        X = rng.normal(size=(n, rows, d))
        X[:, -1, 0] += 1.5 * y      # label direction
        X[:, -1, 1] += conf         # confidence direction
        X[:, -1, 2] += 1.0 * conf * (rep == "read")
        np.save(part / f"{rep}.npy", X.astype(np.float16))
    np.save(part / "y.npy", y.astype(np.int8))
    np.save(part / "index.npy", np.arange(n))
    Xr = np.load(part / "read.npy")[:, -1].astype(np.float64)
    w = np.zeros(d); w[0], w[2] = 1.0, 1.0
    np.save(part / "head_logit.npy", (Xr @ w + 0.1).astype(np.float32))
    (part / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    return conf


def test_analyze_end_to_end(tmp_path, monkeypatch):
    import json
    rng = np.random.default_rng(0)
    d = 6
    bundle, conf_dir = tmp_path / "bundle", tmp_path / "conf"
    conf_dir.mkdir()
    for name, n, pb in (("probe_train_full", 1600, False), ("val_5k", 400, False), ("test_2k", 400, False),
                        ("gsm8k", 240, True), ("math", 240, True), ("olympiadbench", 240, True),
                        ("omnimath", 240, True)):
        c = _bundle_split(bundle, name, n, rng, d, pb)
        np.savez(conf_dir / f"conf_{name}.npz", index=np.arange(n), y=np.zeros(n),
                 **{k: c for k in m4.CONF_KEYS})
    w = torch.zeros(1, d); w[0, 0], w[0, 2] = 1.0, 1.0
    ck = {"soft": torch.zeros(1, d), "head": {"weight": w, "bias": torch.tensor([0.1])}, "m": 1}
    (tmp_path / "tok").mkdir()
    torch.save(ck, tmp_path / "tok" / "read_token.pt")
    out = tmp_path / "m4.json"
    monkeypatch.setattr("sys.argv", ["x", "analyze", "--bundle", str(bundle), "--conf", str(conf_dir),
                                     "--ckpt", str(tmp_path / "tok" / "read_token.pt"),
                                     "--ckpt_out", str(tmp_path / "ck"), "--out", str(out)])
    m4.main()
    r = json.loads(out.read_text())["reps"]
    assert r["read"]["head_check_max_abs_logit_diff"]["test_2k"] < 1e-2
    assert r["read"]["conf_r2"]["test_2k"] > 0.3 and r["read"]["cos_w_c"] > 0.1
    h = r["read"]["heads"]
    assert h["w"]["corr_conf_test"] > 0.3 and abs(h["w_decor"]["corr_conf_test"]) < 0.15
    assert "pb_calib20" in r["prm"]["heads"]["w_proj"]
    ck2 = torch.load(tmp_path / "ck" / "tok_w_proj" / "read_token.pt", weights_only=False)
    assert ck2["head"]["weight"].shape == (1, d) and ck2["m4_head"] == "w_proj"
