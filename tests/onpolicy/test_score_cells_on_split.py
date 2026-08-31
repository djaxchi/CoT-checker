"""Scoring a trained cell out of band must reproduce what the cell itself wrote.

The T1 arm rests entirely on this: the on-policy numbers are only comparable to
the leaderboard if the verifier that produced them is the same function the
leaderboard measured. The cell scores ProcessBench at the end of training and
saves those scores; this script rebuilds the cell from disk and scores the same
split. The two files must agree exactly, including on the rescaled cells, whose
statistics are refit rather than loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tests.harness.test_rep_learner_cell import _write_pb, _write_prm  # noqa: E402

TRAIN = ROOT / "scripts" / "train_rep_learner_cell.py"
SCORE = ROOT / "scripts" / "onpolicy" / "score_cells_on_split.py"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    prm, pb = root / "prm", root / "pb"
    _write_prm(prm, "probe_train_full", 64, 0)
    _write_prm(prm, "val_5k", 32, 1)
    _write_prm(prm, "test_2k", 32, 2)
    _write_pb(pb, "gsm8k", 8, 3)
    return prm, pb


def train_cell(prm, pb, out, rep, learner, rescale):
    cmd = [sys.executable, str(TRAIN), "--rep", rep, "--learner", learner,
           "--prm_store", str(prm), "--pb_store", str(pb), "--out_dir", str(out),
           "--pb_subsets", "gsm8k", "--epochs", "2", "--patience", "1",
           "--batch_size", "16", "--lr_grid", "1e-2", "--wd_grid", "0.0",
           "--threshold_grid", "0.1", "--rescale", rescale]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


def score(cells, split_dir, name, prm, extra=()):
    cmd = [sys.executable, str(SCORE), "--cells", *[str(c) for c in cells],
           "--split_dir", str(split_dir), "--split_name", name,
           "--prm_store", str(prm), "--threshold_grid", "0.1", "--batch_size", "16",
           *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


@pytest.mark.parametrize("rep,learner,rescale", [
    ("last_token", "linear", "none"),
    ("step_stats", "mlp:h8", "zscore"),
    ("step_tokens", "attn_query", "zscore"),
])
def test_out_of_band_scores_match_the_cells_own(store, tmp_path, rep, learner, rescale):
    prm, pb = store
    cell = train_cell(prm, pb, tmp_path / "cell", rep, learner, rescale)
    r = score([cell], pb / "gsm8k", "rescored", prm)
    assert r.returncode == 0, r.stdout + r.stderr

    own = [json.loads(l) for l in (cell / "pb_step_scores_gsm8k.jsonl").read_text().splitlines()]
    new = [json.loads(l) for l in (cell / "pb_step_scores_rescored.jsonl").read_text().splitlines()]
    assert len(own) == len(new)
    for a, b in zip(own, new):
        assert a["id"] == b["id"]
        assert a["label"] == b["label"] and a["prediction"] == b["prediction"]
        assert a["scores"] == pytest.approx(b["scores"], abs=1e-6)


def test_a_different_store_is_refused_rather_than_silently_refit(store, tmp_path):
    """The statistics are refit, so the store they come from is checked."""
    prm, pb = store
    cell = train_cell(prm, pb, tmp_path / "cell_fp", "last_token", "linear", "zscore")
    other = tmp_path / "other"
    _write_prm(other, "probe_train_full", 64, 99)       # same shape, different data
    r = score([cell], pb / "gsm8k", "wrongstore", other)
    assert r.returncode != 0
    assert "fingerprints" in (r.stdout + r.stderr)


def test_summary_reports_rank_material_and_flags_the_f1_caveat(store, tmp_path):
    prm, pb = store
    cell = train_cell(prm, pb, tmp_path / "cell_sum", "last_token", "linear", "none")
    out = tmp_path / "summary.json"
    r = score([cell], pb / "gsm8k", "rescored", prm, extra=("--summary", str(out)))
    assert r.returncode == 0, r.stdout + r.stderr
    rows = json.loads(out.read_text())
    assert len(rows) == 1
    row = rows[0]
    for k in ("rep", "learner", "seed", "step_auroc", "F1_PB_at_val_threshold",
              "oracle_F1_PB", "split_fingerprint", "n_traces"):
        assert k in row
    assert "NOT comparable" in r.stdout


def test_a_cell_without_a_recorded_rescale_is_refused_not_guessed(store, tmp_path):
    """Two grids ran before protocol.rescale existed. Guessing it would rescale
    the scores by statistics the cell never saw and nothing would look wrong."""
    import json
    prm, pb = store
    cell = train_cell(prm, pb, tmp_path / "cell_norescale", "last_token", "linear", "zscore")
    res = json.loads((cell / "results.json").read_text())
    del res["protocol"]["rescale"]
    (cell / "results.json").write_text(json.dumps(res))
    r = score([cell], pb / "gsm8k", "norescale", prm)
    assert r.returncode != 0
    assert "backfill_rescale_field" in (r.stdout + r.stderr)
    r2 = score([cell], pb / "gsm8k", "stated", prm, extra=("--assume_rescale", "zscore"))
    assert r2.returncode == 0, r2.stdout + r2.stderr
    own = [json.loads(l) for l in (cell / "pb_step_scores_gsm8k.jsonl").read_text().splitlines()]
    new = [json.loads(l) for l in (cell / "pb_step_scores_stated.jsonl").read_text().splitlines()]
    for a, b in zip(own, new):
        assert a["scores"] == pytest.approx(b["scores"], abs=1e-6)
