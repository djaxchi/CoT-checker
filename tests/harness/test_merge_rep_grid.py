"""The leaderboard aggregates over seeds, never mixes capped cells in, and
refuses to render at all unless every cell provably read the same inputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pytest  # noqa: E402

from scripts.merge_rep_grid_leaderboard import (  # noqa: E402
    check_inputs, pb_avg, summarise,
)

SCRIPT = ROOT / "scripts" / "merge_rep_grid_leaderboard.py"
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


FP = {"prm/probe_train_full": "aaaa1111", "prm/val_5k": "bbbb2222",
      "pb/gsm8k": "cccc3333"}


def _cell(rep, learner, seed, auroc, f1, full=True, n_params=3585, inputs=None):
    return {
        "inputs": dict(FP if inputs is None else inputs),
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


def test_a_cell_that_read_a_different_store_is_a_hard_error():
    """Two rows trained on different activations are not a controlled comparison,
    so this fails the merge rather than adding a footnote."""
    other = dict(FP, **{"prm/probe_train_full": "deadbeef"})
    cells = [_cell("last_token", "linear", 42, 0.80, 0.40),
             _cell("step_mean", "linear", 42, 0.83, 0.44, inputs=other)]
    with pytest.raises(SystemExit, match="did not read the same inputs"):
        check_inputs(cells)


def test_a_cell_missing_its_fingerprint_is_a_hard_error():
    cells = [_cell("last_token", "linear", 42, 0.80, 0.40)]
    del cells[0]["inputs"]
    with pytest.raises(SystemExit, match="no input fingerprint"):
        check_inputs(cells)


def test_matching_fingerprints_pass_and_are_returned():
    cells = [_cell("last_token", "linear", s, 0.8, 0.4) for s in (42, 43)]
    assert check_inputs(cells) == FP


def test_the_fingerprints_are_published_in_the_table(tmp_path):
    root = tmp_path / "runs"
    for i, c in enumerate([_cell("last_token", "linear", s, 0.8, 0.4)
                           for s in (42, 43)]):
        d = root / f"cell_{i}"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps(c))
    out = tmp_path / "lb.md"
    r = subprocess.run([sys.executable, str(SCRIPT), "--run_root", str(root),
                        "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "aaaa1111" in text and "prm/probe_train_full" in text


def test_mixing_rescaled_and_unrescaled_cells_is_refused():
    """Rescaling changes the numbers entering every probe, so a table mixing the
    two is not one protocol -- the same reasoning as capped vs uncapped."""
    a = _cell("last_token", "linear", 42, 0.80, 0.40)
    b = _cell("step_mean", "linear", 42, 0.83, 0.44)
    a["protocol"] = {"rescale": "zscore"}
    b["protocol"] = {"rescale": "none"}
    with pytest.raises(SystemExit, match="different rescaling settings"):
        check_inputs([a, b])


def test_one_rescaling_setting_passes():
    cells = [_cell("last_token", "linear", s, 0.8, 0.4) for s in (42, 43)]
    for c in cells:
        c["protocol"] = {"rescale": "zscore"}
    assert check_inputs(cells) == FP
