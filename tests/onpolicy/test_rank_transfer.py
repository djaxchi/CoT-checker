"""The analysis has to be able to say "no".

A rank-transfer script that reports agreement whatever it is fed is worse than
none, so these tests build cases where the right answer is known by
construction: a ranking that is preserved, one that is reversed, a measurement
too noisy for any correlation to mean anything (which is what the reliability
ceiling exists to expose), and a correlation that stays high while the
load-bearing contrast underneath it has flipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.analysis.onpolicy_rank_transfer import (  # noqa: E402
    bootstrap_over_cells, kendall, length_matched_ids, per_cell_table,
    rankdata, reliability_ceiling, sign_test_p, spearman,
)

CELLS = [("last_token", "linear"), ("step_mean", "linear"), ("step_delta", "linear"),
         ("step_stats", "linear"), ("step_tokens", "attn_query")]
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def write_scores(path: Path, quality: float, n_traces: int, seed: int,
                 n_steps: int = 4) -> None:
    """Traces where the first-error step is separated by `quality`, so a cell's
    F1_PB at calib-20 is a monotone function of one knob."""
    rng = np.random.default_rng(seed)
    with path.open("w") as f:
        for t in range(n_traces):
            label = -1 if t % 2 == 0 else int(rng.integers(0, n_steps))
            scores = list(rng.uniform(0.05, 0.35, n_steps))
            if label >= 0:
                scores[label] = float(min(0.99, 0.35 + quality))
            f.write(json.dumps({"id": f"tr{t}", "label": label, "n_steps": n_steps,
                                "scores": [float(s) for s in scores],
                                "threshold": 0.5, "prediction": -1}) + "\n")


def make_grid(root: Path, off_q: dict, on_q: dict, seeds=(42, 43, 44),
              jitter: float = 0.0, jitter_seed: int = 0) -> Path:
    rng = np.random.default_rng(jitter_seed)
    for (rep, learner) in CELLS:
        for sd in seeds:
            d = root / f"{rep}__{learner}__seed{sd}"
            d.mkdir(parents=True)
            (d / "results.json").write_text(json.dumps(
                {"rep": rep, "learner": learner, "seed": sd, "dim": 8,
                 "protocol": {"rescale": "none", "dropout": 0.1},
                 "inputs": {"prm/probe_train_full": "x"}, "prm_store": "s",
                 "in_domain": {"val_threshold": 0.5}}))
            for sub in SUBSETS:
                write_scores(d / f"pb_step_scores_{sub}.jsonl",
                             off_q[(rep, learner)] + float(rng.normal(0, jitter)),
                             120, sd)
            write_scores(d / "pb_step_scores_onpolicy.jsonl",
                         on_q[(rep, learner)] + float(rng.normal(0, jitter)),
                         120, sd + 100)
    return root


def arms(root: Path, name: str = "onpolicy"):
    cells = per_cell_table(root, name, None, None)
    keys = sorted(cells)
    off = np.array([np.mean(cells[k]["off"]) for k in keys])
    on = np.array([np.mean(cells[k]["on"]) for k in keys])
    return cells, keys, off, on


LADDER = {c: 0.10 + 0.12 * i for i, c in enumerate(CELLS)}
REVERSED = {c: 0.10 + 0.12 * (len(CELLS) - 1 - i) for i, c in enumerate(CELLS)}


def test_a_preserved_ranking_reads_as_preserved(tmp_path):
    make_grid(tmp_path, LADDER, LADDER)
    _, _, off, on = arms(tmp_path)
    assert spearman(off, on) > 0.8
    assert kendall(off, on) > 0.6


def test_a_reversed_ranking_reads_as_reversed(tmp_path):
    make_grid(tmp_path, LADDER, REVERSED)
    _, _, off, on = arms(tmp_path)
    assert spearman(off, on) < -0.5


def test_the_ceiling_exposes_a_measurement_too_noisy_to_rank(tmp_path):
    """Cells that barely differ, with seed noise larger than the gaps. The
    correlation between arms is then meaningless, and the split-half ceiling is
    what says so."""
    flat = {c: 0.30 for c in CELLS}
    make_grid(tmp_path, flat, flat, jitter=0.25, jitter_seed=7)
    cells, _, _, _ = arms(tmp_path)
    ceiling = reliability_ceiling(cells, "off")["mean"]
    assert ceiling < 0.6, ceiling      # remeasuring the same arm already disagrees


def test_the_ceiling_is_high_when_cells_are_genuinely_separated(tmp_path):
    make_grid(tmp_path, LADDER, LADDER)
    cells, _, _, _ = arms(tmp_path)
    assert reliability_ceiling(cells, "off")["mean"] > 0.8


def test_a_flipped_contrast_can_hide_inside_a_high_correlation(tmp_path):
    """Swap one adjacent pair only. Spearman barely moves; the contrast is gone.
    This is why the contrasts are reported one at a time."""
    on_q = dict(LADDER)
    a, b = CELLS[0], CELLS[1]
    on_q[a], on_q[b] = LADDER[b], LADDER[a]
    make_grid(tmp_path, LADDER, on_q)
    cells, _, off, on = arms(tmp_path)
    assert spearman(off, on) > 0.6                       # still looks fine
    gap_off = np.mean(cells[b]["off"]) - np.mean(cells[a]["off"])
    gap_on = np.mean(cells[b]["on"]) - np.mean(cells[a]["on"])
    assert gap_off > 0 and gap_on < 0                    # but the pair has flipped


def test_bootstrap_over_cells_widens_when_there_are_few_cells(tmp_path):
    make_grid(tmp_path, LADDER, LADDER)
    _, _, off, on = arms(tmp_path)
    ci = bootstrap_over_cells(off, on, 500, 0)["ci95"]
    assert ci[0] <= spearman(off, on) <= ci[1] + 1e-9
    assert ci[1] - ci[0] > 0.0


def test_sign_test_is_the_exact_binomial():
    assert sign_test_p(4, 4) == 2 * (1 / 16)
    assert sign_test_p(2, 4) == 1.0
    assert sign_test_p(0, 4) == 2 * (1 / 16)


def test_rankdata_averages_ties():
    assert list(rankdata(np.array([1.0, 1.0, 2.0]))) == [1.5, 1.5, 3.0]


def test_length_matching_keeps_the_overlap_and_notices_when_there_is_none():
    on = {f"a{i}": float(i) for i in range(20)}          # 0..19
    off = {f"b{i}": float(i + 10) for i in range(20)}    # 10..29
    keep_on, keep_off, info = length_matched_ids(on, off)
    assert not info.get("degenerate")
    assert all(info["window"][0] <= on[k] <= info["window"][1] for k in keep_on)
    disjoint_off = {f"b{i}": float(i + 500) for i in range(20)}
    _, _, info2 = length_matched_ids(on, disjoint_off)
    assert info2.get("degenerate") is True


def test_the_unlabelled_split_is_refused_rather_than_scored(tmp_path):
    """A placeholder label of -1 everywhere makes every trace look error-free, so
    F1_PB would come back as a number and mean nothing."""
    import json
    import pytest
    from scripts.analysis.onpolicy_rank_transfer import onpolicy_score
    d = tmp_path / "cell"
    d.mkdir()
    with (d / "pb_step_scores_unlab.jsonl").open("w") as f:
        for i in range(50):
            f.write(json.dumps({"id": f"t{i}", "label": -1, "n_steps": 3,
                                "scores": [0.1, 0.2, 0.3]}) + "\n")
    with pytest.raises(SystemExit) as e:
        onpolicy_score(d, "unlab", None)
    assert "onpolicy_downstream" in str(e.value)
