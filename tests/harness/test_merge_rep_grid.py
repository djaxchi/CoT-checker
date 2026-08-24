"""The leaderboard must aggregate over seeds and must never mix capped cells in."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.merge_rep_grid_leaderboard import pb_avg, summarise  # noqa: E402

SCRIPT = ROOT / "scripts" / "merge_rep_grid_leaderboard.py"
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def _cell(rep, learner, seed, auroc, f1, full=True, n_params=3585):
    return {
        "rep": rep, "learner": learner, "seed": seed, "dim": 3584,
        "n_params": n_params, "n_train": 513810 if full else 150000,
        "n_train_available": 513810, "full_train": full,
        "hp": {"selected": {"lr": 1e-3, "weight_decay": 0.0}},
        "in_domain": {"auroc": auroc},
        "processbench": {s: {"val_selected": {"F1_PB": f1},
                             "oracle_F1_PB": f1 + 0.05} for s in SUBSETS},
    }


def test_pb_avg_needs_every_subset():
    c = _cell("last_token", "linear", 42, 0.8, 0.4)
    assert pb_avg(c, "val") == 0.4
    del c["processbench"]["omnimath"]
    assert pb_avg(c, "val") is None


def test_seeds_are_aggregated_with_a_spread():
    cells = [_cell("last_token", "linear", s, a, 0.4)
             for s, a in zip((42, 43, 44), (0.80, 0.82, 0.84))]
    s = summarise(cells)[("last_token", "linear")]
    assert s["n_seeds"] == 3 and s["seeds"] == [42, 43, 44]
    mean, sd = s["auroc"]
    assert abs(mean - 0.82) < 1e-9
    assert sd > 0.0


def test_capped_cells_are_reported_separately(tmp_path):
    root = tmp_path / "runs"
    for i, c in enumerate([
        _cell("last_token", "linear", 42, 0.80, 0.40),
        _cell("last_token", "linear", 43, 0.81, 0.41),
        _cell("step_tokens", "transformer:d256,l2", 42, 0.87, 0.52,
              n_params=2_500_000),
        _cell("step_stats", "linear", 42, 0.86, 0.48, full=False),
    ]):
        d = root / f"cell_{i}"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps(c))

    out = tmp_path / "leaderboard.md"
    r = subprocess.run([sys.executable, str(SCRIPT), "--run_root", str(root),
                        "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "3 full-split cells" in text
    assert "Excluded: cells trained with a cap" in text
    # the capped cell must not appear in the main crossed table
    main_table = text.split("## Excluded")[0]
    assert "step_stats" not in main_table
    assert "±" in main_table            # the two-seed cell carries a spread
    assert "2,500,000" in text          # capacity view reports parameter counts
