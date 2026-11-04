"""The LaTeX tables and pgfplots figure of the ReProbe comparison are generated from the numbers."""

from scripts.instep_v1 import paper_tex as T
from scripts.instep_v1.paper_numbers import TEST


def _arm(v: float) -> dict:
    d = {s: v for s in TEST}
    d["overall"] = v
    return d


OURS = {"qwen3_8b": {"instep_ds": _arm(0.64), "instep_self": _arm(0.95), "passive_full": _arm(0.60),
                     "passive_full_final": _arm(0.59), "step": _arm(0.58)},
        "phi4": {"instep": _arm(0.41), "step": _arm(0.37)},
        "qwen3_32b": {"instep_ep4": _arm(0.63), "step": _arm(0.62)}}


def test_marks_best_and_second():
    assert T._marks([0.5, 0.7, None, 0.6]) == [".500", "\\textbf{.700}", "--", "\\underline{.600}"]


def test_table1_has_every_paper_row_and_bolds_best_overall():
    tex = T.table_qwen3_8b(OURS)
    assert tex.count("\\\\") == 2 + 22 + 3 + 5  # header rows, paper rows, our rows, group titles
    instep_self = next(line for line in tex.splitlines() if "InStep} (self)" in line)
    assert instep_self.rstrip(" \\").endswith("\\textbf{.950}")


def test_backbone_tables_have_no_holes():
    finalj = {"backbones": {"qwen3_32b": {"metrics": {"prm7b_stored": {"overall": {"ap": 0.566}}}}}}
    phi, q32 = T.table_backbone(OURS, "phi4"), T.table_backbone(OURS, "qwen3_32b", finalj)
    assert "--" not in phi and "--" not in q32
    assert "$^\\dagger$" in q32 and "$^\\dagger$" not in phi
    assert "\\textbf{.410}" in phi  # InStep best on Phi-4


def test_figure_renders_one_labelled_point_per_verifier():
    fig = T.figure_cost(OURS)
    assert fig.count("\\begin{tikzpicture}") == 2 and "(3.072e+05,0.950)" in fig
    assert "InStep} (.950)" in fig and "STEP (.580)" in fig and "(2.098e+06,0.580)" in fig
    assert "const plot" not in fig and "legend" not in fig.split("\\hfill")[0]  # panel (a): no legend


TTS = {"bon_curve": {"sets": ["math"], "n": [1, 2, 10],
                     "curves": {"majority": [60.0, 61.0, 65.0], "InStep-self": [60.0, 64.0, 70.0]}},
       "table2_beam": {"rows": [
           {"row": "Random scores (same search)", "math": 60.0, "gsm8k": 80.0, "trips": None},
           {"row": "Qwen2.5-Math-PRM-7B", "math": None, "gsm8k": None},  # not run yet: no point
           {"row": "InStep, Self-anno", "math": 67.0, "gsm8k": 79.0}],
           "seconds": {"random": {"math": 10.0, "gsm8k": 4.0}, "rt_self": {"math": 12.0, "gsm8k": 6.0}}}}


def test_beam_points_average_accuracy_and_time_over_common_sets():
    sets, pts = T.beam_points(TTS["table2_beam"])
    assert sets == ["math", "gsm8k"]
    assert [(p[0], p[2], p[3]) for p in pts] == [("random scores", 7.0, 70.0), ("\\textbf{InStep}", 9.0, 73.0)]


def test_tts_figure_has_accuracy_and_cost_panels():
    fig = T.figure_tts(TTS)
    assert fig.count("\\begin{tikzpicture}") == 3
    assert "(10,70.00)" in fig and "Majority vote" in fig and "PRM-7B" not in fig
    # (b): InStep at N = 10 candidates x its per-candidate latency at 10 per call
    assert f"({10 * T.LATENCY['InStep'][1]:.1f},70.00)" in fig and "majority vote (no verifier)" in fig
    assert "(9.00,73.00)" in fig  # (c)


def test_add_step_takes_the_seed_mean_per_set():
    o = {"phi4": {}}
    T.add_step(o, "phi4", {"test_seed_mean": {"math": 0.4, "gsm8k": 0.2}})
    assert o["phi4"]["step"] == {"math": 0.4, "gsm8k": 0.2, "overall": 0.30000000000000004}


def test_add_q17_takes_the_instep_seed_mean_per_set():
    o = {}
    T.add_q17(o, {"table": {"InStep": {s: {"ap": 0.5, "n_seeds": 3} for s in TEST}}})
    assert o["qwen3_1_7b"]["instep"]["overall"] == 0.5 and len(o["qwen3_1_7b"]["instep"]) == 9


def test_size_figure_uses_printed_baselines_and_our_instep_per_policy():
    o = {k: dict(v) for k, v in OURS.items()}
    o["qwen3_1_7b"] = {"instep": _arm(0.50)}
    pts = {label: p for label, _, p in T.size_points(o)}
    # Table 4 / 1 / 5 as printed: Qwen2.5-Math-PRM-7B .421 / .565 / .613, ReProbe HS .447 / .604 / .585
    assert [round(y, 3) for _, y in pts["PRM-7B"]] == [0.421, 0.565, 0.613]
    assert [round(y, 3) for _, y in pts["ReProbe"]] == [0.447, 0.604, 0.585]
    assert [x for x, _ in pts["\\textbf{InStep}"]] == [1.7, 8.0, 32.0]
    assert [y for _, y in pts["\\textbf{InStep}"]] == [0.50, 0.95, 0.63]
    tex = T.figure_size(o)
    assert "mark=none" in tex and "xticklabels={1.7B,8B,32B}" in tex
    # the 32B PRM point can be replaced by the value recomputed from the released scores
    over = {label: p for label, _, p in T.size_points(o, {"qwen3_32b": 0.566})}
    assert [round(y, 3) for _, y in over["PRM-7B"]] == [0.421, 0.565, 0.566]
