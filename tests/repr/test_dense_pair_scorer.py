"""Tests for classifier-level fork supervision (dense_* methods).

These train ONLY the final linear scorer s(h)=w.h+b on PRM800K fork pairs over
frozen dense hidden states. No representation, no BCE. CPU-only, tiny tensors.

Score convention: score = sigmoid(s(h)) = P(non_viable), so the preferred
(label 0) sibling should get a LOW logit, the rejected (label 1) sibling HIGH.
"""

import json

import numpy as np
import torch

from scripts.train_easy_probe_method import (
    LinearProbe,
    load_fork_pairs,
    train_dense_pair_scorer,
)
from src.repr.objectives import (
    dense_absmargin_loss,
    dense_anchor_rank_loss,
    dense_rank_loss,
)

DEVICE = torch.device("cpu")
HID = 4


def _separable_items(seed=0):
    """9 rows = 3 forks of (anchor, pos, neg).

    neg points along +e0 (should score HIGH), pos along -e0 (LOW), anchor near pos.
    Rows are interleaved a,p,n,a,p,n,... so index resolution is non-trivial.
    """
    rng = np.random.default_rng(seed)
    rows = []
    anchor_idx, pos_idx, neg_idx = [], [], []
    for f in range(3):
        base = rng.standard_normal(HID).astype(np.float32) * 0.05
        anchor = base.copy()
        # pos departs the anchor on axis e1, neg on axis e0 -> the scorer can
        # push neg up (via e0) while keeping pos near the anchor (zero its e1).
        pos = base + np.array([0.0, 1.0, 0, 0], dtype=np.float32)
        neg = base + np.array([1.0, 0.0, 0, 0], dtype=np.float32)
        anchor_idx.append(len(rows))
        rows.append(anchor)
        pos_idx.append(len(rows))
        rows.append(pos)
        neg_idx.append(len(rows))
        rows.append(neg)
    items = np.stack(rows).astype(np.float32)
    return items, np.array(anchor_idx), np.array(pos_idx), np.array(neg_idx)


def _logits(scorer, h):
    with torch.no_grad():
        return scorer(torch.from_numpy(h).to(DEVICE)).cpu().numpy()


# --- loss-function unit checks (closed form on toy logits) -----------------

def test_dense_rank_loss_minimized_when_neg_above_pos():
    pos = torch.tensor([0.0, 0.0])
    neg_bad = torch.tensor([0.0, 0.0])     # no gap -> larger loss
    neg_good = torch.tensor([5.0, 5.0])    # big gap -> small loss
    assert dense_rank_loss(pos, neg_good) < dense_rank_loss(pos, neg_bad)


def test_dense_anchor_rank_penalizes_pos_drift_and_low_neg():
    anchor = torch.tensor([0.0, 0.0])
    pos_close = torch.tensor([0.0, 0.0])   # preserves anchor
    pos_far = torch.tensor([3.0, 3.0])     # drifts from anchor
    neg = torch.tensor([5.0, 5.0])
    assert dense_anchor_rank_loss(anchor, pos_close, neg) < dense_anchor_rank_loss(anchor, pos_far, neg)


def test_dense_absmargin_rewards_low_pos_high_neg():
    good = dense_absmargin_loss(torch.tensor([-2.0]), torch.tensor([2.0]))
    bad = dense_absmargin_loss(torch.tensor([2.0]), torch.tensor([-2.0]))
    assert good < bad


# --- 1. dense_rank loss decreases on synthetic pairs -----------------------

def test_dense_rank_loss_decreases():
    items, a, p, n = _separable_items()
    scorer, stats = train_dense_pair_scorer(
        items, a, p, n, method="dense_rank", hidden_dim=HID,
        epochs=80, batch_size=3, lr=5e-2, rank_margin=1.0,
        low_margin=1.0, high_margin=1.0, device=DEVICE, seed=0,
    )
    hist = stats["history"]
    assert hist[-1]["loss"] < hist[0]["loss"]
    assert stats["final_pair_accuracy"] == 1.0  # separable toy set


# --- 2. dense_anchor_rank: gradient to scorer + reduces pos-anchor drift ----

def test_dense_anchor_rank_reduces_pos_anchor_drift():
    items, a, p, n = _separable_items()
    scorer0 = LinearProbe(HID)
    w_before = scorer0.fc.weight.detach().clone()
    scorer, stats = train_dense_pair_scorer(
        items, a, p, n, method="dense_anchor_rank", hidden_dim=HID,
        epochs=120, batch_size=3, lr=5e-2, rank_margin=1.0,
        low_margin=1.0, high_margin=1.0, device=DEVICE, seed=0,
    )
    hist = stats["history"]
    # gradient actually moved the scorer
    assert not torch.allclose(w_before, scorer.fc.weight.detach())
    # anchor-relative drift on the preferred sibling shrinks over training
    assert hist[-1]["pos_anchor_drift"] < hist[0]["pos_anchor_drift"]
    assert "anchor_acc" in hist[-1]


# --- 3. dense_rank_absmargin pushes pos logits down, neg logits up ----------

def test_dense_rank_absmargin_pushes_logits_apart():
    items, a, p, n = _separable_items()
    scorer, stats = train_dense_pair_scorer(
        items, a, p, n, method="dense_rank_absmargin", hidden_dim=HID,
        epochs=150, batch_size=3, lr=5e-2, rank_margin=1.0,
        low_margin=1.0, high_margin=1.0, device=DEVICE, seed=0,
    )
    l_pos = _logits(scorer, items[p]).mean()
    l_neg = _logits(scorer, items[n]).mean()
    assert l_pos < 0.0 < l_neg
    assert l_pos < l_neg
    hist_last = stats["history"][-1]
    assert hist_last["pos_low_sat"] > 0.5 and hist_last["neg_high_sat"] > 0.5


# --- 4. pair indexing resolves to the correct encoded rows -----------------

def test_pair_indexing_resolves_interleaved_rows(tmp_path):
    # rows are interleaved a,p,n per fork; resolution must follow uids not order
    meta = tmp_path / "items_meta.jsonl"
    meta.write_text(
        "\n".join(
            json.dumps({"row": i, "item_uid": uid})
            for i, uid in enumerate(
                ["f0::anchor", "f0::positive::0", "f0::negative::0",
                 "f1::anchor", "f1::positive::0", "f1::negative::0"]
            )
        )
    )
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "\n".join([
            json.dumps({"anchor_uid": "f1::anchor", "positive_uid": "f1::positive::0",
                        "negative_uid": "f1::negative::0"}),
            json.dumps({"anchor_uid": "f0::anchor", "positive_uid": "f0::positive::0",
                        "negative_uid": "f0::negative::0"}),
        ])
    )
    a, p, n = load_fork_pairs(meta, pairs)
    assert a.tolist() == [3, 0]
    assert p.tolist() == [4, 1]
    assert n.tolist() == [5, 2]


# --- 5. end-to-end output schema (no representation.pt) --------------------

def _write_cache(cache_dir, stem, h, y):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{stem}_h.npy", h.astype(np.float32))
    np.save(cache_dir / f"{stem}_y.npy", y.astype(np.int32))
    (cache_dir / f"{stem}_meta.jsonl").write_text(
        "\n".join(json.dumps({"uid": f"{stem}-{i}"}) for i in range(len(y)))
    )


def test_dense_method_writes_expected_schema(tmp_path, monkeypatch):
    import scripts.train_easy_probe_method as M

    cache = tmp_path / "cache"
    rng = np.random.default_rng(0)
    _write_cache(cache, "tiny_train", rng.standard_normal((8, HID)),
                 np.array([0, 1] * 4))
    _write_cache(cache, "tiny_val", rng.standard_normal((6, HID)),
                 np.array([0, 1] * 3))

    items, a, p, n = _separable_items()
    enc = tmp_path / "enc"
    enc.mkdir()
    np.save(enc / "forks_train_items_h.npy", items)
    (enc / "forks_train_items_meta.jsonl").write_text(
        "\n".join(json.dumps({"row": i, "item_uid": f"u{i}"}) for i in range(len(items)))
    )
    pairs = tmp_path / "forks_train_pairs.jsonl"
    pairs.write_text(
        "\n".join(
            json.dumps({"anchor_uid": f"u{ai}", "positive_uid": f"u{pi}",
                        "negative_uid": f"u{ni}"})
            for ai, pi, ni in zip(a.tolist(), p.tolist(), n.tolist())
        )
    )

    # tiny ProcessBench: 2 traces x 2 steps
    pb_h = rng.standard_normal((4, HID)).astype(np.float32)
    np.save(tmp_path / "pb_h.npy", pb_h)
    pb_meta = tmp_path / "pb_meta.jsonl"
    pb_meta.write_text("\n".join([
        json.dumps({"id": "t0", "step_idx": 0, "label": 1, "n_steps": 2}),
        json.dumps({"id": "t0", "step_idx": 1, "label": 1, "n_steps": 2}),
        json.dumps({"id": "t1", "step_idx": 0, "label": -1, "n_steps": 2}),
        json.dumps({"id": "t1", "step_idx": 1, "label": -1, "n_steps": 2}),
    ]))

    out = tmp_path / "out"
    argv = [
        "prog", "--method", "dense_anchor_rank",
        "--cache_dir", str(cache),
        "--out_dir", str(out),
        "--probe_train_stem", "tiny_train", "--val_stem", "tiny_val",
        "--skip_size_asserts",
        "--fork_items_h", str(enc / "forks_train_items_h.npy"),
        "--fork_items_meta", str(enc / "forks_train_items_meta.jsonl"),
        "--fork_pairs", str(pairs),
        "--pair_epochs", "5", "--batch_size", "4", "--threshold_grid", "0.5",
        "--pb_h", str(tmp_path / "pb_h.npy"), "--pb_meta", str(pb_meta),
        "--pb_name", "gsm8k",
    ]
    monkeypatch.setattr(M.sys, "argv", argv)
    M.main()

    for name in [
        "config.yaml", "linear_probe.pt", "threshold.json", "train_metrics.json",
        "val_scores.npy", "eval_summary.json", "eval_metrics.json",
        "training_history.json",
        "pb_step_scores_gsm8k.jsonl", "pb_predictions_gsm8k.jsonl",
    ]:
        assert (out / name).exists(), f"missing output file: {name}"
    # No representation is trained for dense_* methods.
    assert not (out / "representation.pt").exists()

    tm = json.loads((out / "train_metrics.json").read_text())
    assert tm["method"] == "dense_anchor_rank"
    assert tm["final_reconstruction_mse"] is None
    assert tm["objective"] == "dense_anchor_rank"
    assert "pair_diagnostics" in tm and "pos_anchor_drift" in tm["pair_diagnostics"]
    assert tm["n_pairs"] == 3
    hist = json.loads((out / "training_history.json").read_text())
    assert len(hist) == 5 and hist[0]["epoch"] == 0
