import json

import numpy as np

from scripts.analysis import plot_rep_separation as prs
from tests.analysis.test_prm_geometry import _run


def test_figures_and_summary_from_geometry_outputs(tmp_path, monkeypatch):
    metrics, arrays = _run()
    for bb in ("instruct", "rt", "prm", "rt_final", "drv"):
        out = tmp_path / "geo" / bb / "last_token"
        out.mkdir(parents=True)
        np.savez_compressed(out / "arrays.npz", **arrays)
        (out / "metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr("sys.argv", ["x", "--geo_dir", str(tmp_path / "geo"),
                                     "--out_dir", str(tmp_path / "fig")])
    prs.main()
    for f in ("views_test.png", "views_pb.png", "separation.png", "summary.md"):
        assert (tmp_path / "fig" / f).stat().st_size > 0
    au = prs.pb_auroc(arrays)
    assert au["mean"] > 0.75  # synthetic first-error steps are shifted along the LDA axis
    assert "| Qwen3-8B + read token <v> (L35) | 16 |" in (tmp_path / "fig" / "summary.md").read_text()
