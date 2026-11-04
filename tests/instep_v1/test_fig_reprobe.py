"""Smoke test: the ReProbe-table figures render from a synthetic per-set JSON."""

import json

from scripts.instep_v1 import fig_reprobe as F
from scripts.instep_v1.paper_numbers import TEST


def _arm(v: float) -> dict:
    d = {s: v for s in TEST}
    d["overall"] = v
    return d


def _ours() -> dict:
    return {"qwen3_8b": {"instep_ds": _arm(0.64), "instep_self": _arm(0.65), "passive_full": _arm(0.60)},
            "phi4": {"instep": _arm(0.41), "passive": _arm(0.37)},
            "qwen3_32b": {"instep_ep4": _arm(0.63), "passive": _arm(0.62)}}


def test_paper_figures_fit_text_width(tmp_path):
    from PIL import Image

    ours = _ours()
    finalj = {"backbones": {"qwen3_32b": {"metrics": {"prm7b_stored": {"overall": {"ap": 0.566}}}}}}
    paths = (F.fig_table1(ours, tmp_path, paper=True) + F.fig_cost(ours, tmp_path, paper=True)
             + F.fig_backbones(ours, finalj, tmp_path, paper=True))
    for q in paths:
        if q.endswith(".png"):
            w, _ = Image.open(q).size
            assert w / 220 <= F.PAPER_WIDTH + 0.05, q  # tight bbox stays within the text width


def test_figures_render(tmp_path):
    ours = {"qwen3_8b": {"instep_ds": _arm(0.64), "instep_self": _arm(0.65), "passive_full": _arm(0.60)},
            "phi4": {"instep": _arm(0.41), "passive": _arm(0.37)},
            "qwen3_32b": {"instep_ep4": _arm(0.63), "passive": _arm(0.62)}}
    p = tmp_path / "ours.json"
    p.write_text(json.dumps(ours))
    paths = F.fig_table1(ours, tmp_path) + F.fig_cost(ours, tmp_path) + F.fig_backbones(ours, None, tmp_path)
    assert len(paths) == 6
    assert all((tmp_path / q.split("/")[-1]).stat().st_size > 0 for q in paths)
