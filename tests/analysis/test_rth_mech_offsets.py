import numpy as np
import torch

from scripts.analysis import rth_mech_offsets as m2


def test_offset_table_and_norms():
    X = np.zeros((3, 2, 4), np.float16)
    X[:, 0, 0] = [1.0, 2.0, 3.0]          # median norm 2 at row 0
    X[:, 1, 1] = [4.0, 4.0, 4.0]          # median norm 4 at row 1
    med = m2.median_norms(X, chunk=2)
    assert med.tolist() == [2.0, 4.0]
    deep = np.array([[1.0, 0, 0, 0], [0, 0, 0, 0]])
    t = m2.offset_table(deep, med, med * 2, np.array([2.0, 0, 0, 0]))
    assert t[0]["rel_read"] == 0.5 and t[0]["rel_last"] == 0.25 and t[0]["cos_head"] == 1.0
    assert t[1]["norm"] == 0.0 and t[1]["cos_head"] == 0.0


def test_vocab_top_orders_tokens():
    W_U = torch.eye(5)
    V = torch.tensor([[0.0, 3.0, 1.0, -2.0, 0.5]])
    r = m2.vocab_top(V, W_U, lambda t: f"t{t}", 2)[0]
    assert [t for t, _ in r["top"]] == ["t1", "t2"] and [t for t, _ in r["bottom"]] == ["t3", "t0"]
