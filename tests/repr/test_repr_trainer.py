"""Tests for the fork-based representation trainer paths in the easy-probe script.

CPU-only, tiny tensors, no model downloads.
"""

import json

import numpy as np
import torch

from scripts.train_easy_probe_method import (
    SAE,
    encode_with_sae,
    load_fork_pairs,
    train_repr_with_pairs,
)

DEVICE = torch.device("cpu")


def _toy_items(hidden_dim=6, n=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, hidden_dim)).astype(np.float32)


def test_load_fork_pairs_maps_uids_to_rows(tmp_path):
    meta = tmp_path / "items_meta.jsonl"
    meta.write_text(
        "\n".join(
            json.dumps({"row": i, "item_uid": uid})
            for i, uid in enumerate(["a0", "p0", "n0", "a1", "p1", "n1"])
        )
    )
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "\n".join([
            json.dumps({"anchor_uid": "a0", "positive_uid": "p0", "negative_uid": "n0"}),
            json.dumps({"anchor_uid": "a1", "positive_uid": "p1", "negative_uid": "n1"}),
        ])
    )
    a, p, n = load_fork_pairs(meta, pairs)
    assert a.tolist() == [0, 3]
    assert p.tolist() == [1, 4]
    assert n.tolist() == [2, 5]


def test_load_fork_pairs_unknown_uid_raises(tmp_path):
    meta = tmp_path / "m.jsonl"
    meta.write_text(json.dumps({"row": 0, "item_uid": "a0"}))
    pairs = tmp_path / "p.jsonl"
    pairs.write_text(json.dumps({"anchor_uid": "a0", "positive_uid": "MISSING", "negative_uid": "a0"}))
    try:
        load_fork_pairs(meta, pairs)
        assert False, "expected ValueError"
    except ValueError:
        pass


def _run(objective, **kw):
    hidden_dim = 6
    items = _toy_items(hidden_dim=hidden_dim, n=12)
    # 4 forks, each anchor/pos/neg drawn from distinct rows
    anchor_idx = np.array([0, 3, 6, 9])
    pos_idx = np.array([1, 4, 7, 10])
    neg_idx = np.array([2, 5, 8, 11])
    return train_repr_with_pairs(
        items_h=items, anchor_idx=anchor_idx, pos_idx=pos_idx, neg_idx=neg_idx,
        hidden_dim=hidden_dim, latent_dim=hidden_dim,
        epochs=kw.get("epochs", 30), batch_size=4, lr=1e-2,
        l1_weight=kw.get("l1_weight", 0.0),
        objective=objective, obj_weight=1.0,
        rank_kind=kw.get("rank_kind", "logistic"), rank_margin=1.0,
        triplet_metric=kw.get("triplet_metric", "l2"), triplet_margin=1.0,
        device=DEVICE, seed=0,
    )


def test_rank_objective_returns_encoder_and_decreases_loss():
    sae, stats = _run("rank")
    assert isinstance(sae, SAE)
    assert stats["objective"] == "rank"
    assert stats["n_pairs"] == 4
    # objective should be driven down to near-zero on a tiny separable toy set
    assert stats["final_objective_loss"] < 0.5


def test_triplet_objective_runs_and_encodes():
    sae, stats = _run("triplet", triplet_metric="l2")
    z = encode_with_sae(sae, _toy_items(n=5), batch_size=2, device=DEVICE)
    assert z.shape == (5, 6)
    assert (z >= 0).all()  # ReLU latents


def test_sae_path_keeps_l1_penalty_recorded():
    _, stats = _run("rank", l1_weight=1e-2)
    assert stats["final_l1_mean"] >= 0.0
    assert stats["final_aux_bce"] is None
