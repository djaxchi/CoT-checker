import numpy as np
import scipy.sparse as sp

from scripts.analysis.sae_feature_localization import cohen_d_dense, cohen_d_sparse, known_mask, pb_traces


def test_cohen_d_sparse_matches_dense():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 6)) * (rng.random((400, 6)) > 0.5)
    y = (rng.random(400) > 0.5).astype(np.int8)
    x[y == 1, 2] += 1.0
    ds = cohen_d_sparse(sp.csr_matrix(x.astype(np.float32)), y)
    dd = cohen_d_dense(x, y, chunk=97)
    assert np.allclose(ds, dd, atol=1e-4)
    assert np.argmax(np.abs(dd)) == 2 and dd[2] > 0


def test_pb_traces_first_error_and_order():
    # ProcessBench meta: `label` is the trace's first-error index on every row
    meta = [{"id": "a", "step_idx": 1, "label": 1}, {"id": "a", "step_idx": 0, "label": 1},
            {"id": "a", "step_idx": 2, "label": 1},
            {"id": "b", "step_idx": 0, "label": -1}, {"id": "b", "step_idx": 1, "label": -1}]
    traces = pb_traces(meta, np.array([0.9, 0.1, 0.5, 0.2, 0.3]))
    assert traces[0] == (1, [0.1, 0.9, 0.5])
    assert traces[1] == (-1, [0.2, 0.3])
    assert known_mask(meta).tolist() == [True, True, False, True, True]
