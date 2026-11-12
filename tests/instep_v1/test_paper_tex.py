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
    # ReProbe's column groups and shaded group rows, so the two tables compare line by line
    assert "{Planning (OOD)}" in tex and "{QA (OOD)}" in tex and tex.count(T.GROUP_ROW) == 5


def test_one_column_table1_keeps_every_row_with_the_average_block_only():
    tex = T.table_qwen3_8b_avg(OURS)
    assert "Method & ID & OOD & All" in tex and "Planning" not in tex
    assert "MATH, GSM8K," in tex and "StrQA, SciQA" in tex  # the sets behind each average
    assert tex.count("\\\\") == 2 + 22 + 3 + 5 + 3  # header rows, paper rows, our rows, group titles, set-name line breaks
    instep_self = next(line for line in tex.splitlines() if "InStep} (self)" in line)
    assert instep_self.rstrip(" \\").endswith("& \\textbf{.950} & \\textbf{.950} & \\textbf{.950}")


def test_backbone_table_merges_both_generators():
    finalj = {"backbones": {"qwen3_32b": {"metrics": {"prm7b_stored": {"overall": {"ap": 0.566}}}}}}
    tex = T.table_backbones(OURS, finalj)
    assert "Method & Qwen3-32B & Phi-4" in tex  # no Qwen3-1.7B run in OURS: no column
    instep = next(line for line in tex.splitlines() if "InStep}" in line)
    assert instep.rstrip(" \\").endswith("\\textbf{.630} & \\textbf{.410}")  # best on both
    prm7b = next(line for line in tex.splitlines() if line.startswith("Qwen2.5-Math-PRM-7B"))
    assert "$^\\dagger$" in prm7b and tex.count("$^\\dagger$") == 1
    maxprob = next(line for line in tex.splitlines() if line.startswith("MaxProb"))
    assert maxprob.endswith("& -- & .149 \\\\")  # ReProbe reports no uncertainty scores on Qwen3-32B


def test_backbone_table_adds_qwen3_1_7b_with_printed_table4_rows():
    o = {k: dict(v) for k, v in OURS.items()}
    o["qwen3_1_7b"] = {"instep": _arm(0.50)}  # no STEP run on 1.7B
    tex = T.table_backbones(o)
    assert "Method & Qwen3-1.7B & Qwen3-32B & Phi-4" in tex
    prm7b = next(line for line in tex.splitlines() if line.startswith("Qwen2.5-Math-PRM-7B"))
    assert prm7b.startswith("Qwen2.5-Math-PRM-7B & .421 &")  # Table 4 as printed
    step = next(line for line in tex.splitlines() if line.startswith("STEP"))
    assert step.startswith("STEP & -- &")
    instep = next(line for line in tex.splitlines() if "InStep}" in line)
    assert "\\textbf{InStep} & \\textbf{.500}" in instep  # above ReProbe's .447


def test_figure_renders_one_labelled_point_per_verifier():
    fig = T.figure_cost(OURS)
    assert fig.count("\\begin{tikzpicture}") == 2 and "(3.072e+05,0.950)" in fig
    assert "InStep} (.950)" in fig and "STEP (.580)" in fig and "(2.098e+06,0.580)" in fig
    assert "const plot" not in fig and "legend" not in fig.split("\\vspace")[0]  # panel (a): no legend


def _row(section, row, **v):
    return {"section": section, "row": row, **v}


TTS = {"bon_curve": {"sets": ["math"], "n": [1, 2, 10],
                     "curves": {"majority": [60.0, 61.0, 65.0], "InStep-self": [60.0, 64.0, 70.0]}},
       "table3_bon": {"rows": [
           _row("Pass@N, majority voting", "Qwen3-8B pass@1 (lower bound)", math=50.0, ID=50.0, OOD=40.0, Overall=45.0),
           _row("Pass@N, majority voting", "Majority voting", math=52.0, ID=None, OOD=None, Overall=None),
           _row("PRMs", "Skywork-PRM-1.5B", math=None, ID=None, OOD=None, Overall=None),  # not released: dropped
           _row("PRMs", "Qwen2.5-Math-PRM-7B", math=60.0, ID=60.0, OOD=44.0, Overall=52.0),
           _row("ReProbes", "ReProbe, Hidden States, Self-anno", math=58.0, ID=58.0, OOD=45.0, Overall=51.0),
           _row("InStep (ours)", "InStep, Self-anno", math=59.0, ID=59.0, OOD=46.0, Overall=52.04)]},
       "table2_beam": {"rows": [
           _row("Our search", "Random scores (same search)", math=60.0, gsm8k=80.0, ID=70.0, OOD=50.0, Overall=58.0),
           _row("Our search", "InStep, Self-anno", math=67.0, gsm8k=79.0, ID=73.0, OOD=49.0, Overall=59.0)],
           "seconds": {"random": {"math": 10.0, "gsm8k": 4.0}, "rt_self": {"math": 12.0, "gsm8k": 6.0}}}}


def test_beam_points_average_accuracy_and_time_over_common_sets():
    sets, pts = T.beam_points(TTS["table2_beam"])
    assert sets == ["math", "gsm8k"]
    assert [(p[0], p[2], p[3]) for p in pts] == [("random scores", 7.0, 70.0), ("\\textbf{InStep}", 9.0, 73.0)]


def test_tts_figure_has_curve_and_eight_set_cost_panels():
    fig = T.figure_tts(TTS)
    assert fig.count("\\begin{tikzpicture}") == 2 and "fig:tts-beam" not in fig
    assert "(10,70.00)" in fig and "Majority vote" in fig
    # (b): the table's Overall column at N = 10 candidates x per-candidate latency at 10 per call
    assert f"({10 * T.LATENCY['InStep'][1]:.1f},52.04)" in fig and "InStep} (52.0)" in fig
    assert f"({10 * T.LATENCY['PRM'][1]:.1f},52.00)" in fig and "one candidate, no verifier (45.0)" in fig


def test_tts_table_puts_best_of_n_and_beam_side_by_side():
    tex = T.table_tts(TTS)
    assert "\\multicolumn{3}{c}{Best-of-$N$} & \\multicolumn{3}{c}{Beam search}" in tex
    instep = next(line for line in tex.splitlines() if "InStep} (self)" in line)
    assert instep.rstrip(" \\").endswith("& \\textbf{52.0} & \\textbf{73.0} & \\underline{49.0} & \\textbf{59.0}")
    prm = next(line for line in tex.splitlines() if line.startswith("Qwen2.5-Math-PRM-7B"))
    assert prm.endswith("& -- & -- & -- \\\\")  # not run in our beam search
    rand = next(line for line in tex.splitlines() if line.startswith("Random step scores"))
    assert rand.startswith("Random step scores & -- & -- & -- &")


def test_tts_set_table_drops_rows_without_values():
    tex = T.table_tts_sets(TTS["table3_bon"]["rows"], ["math"])
    assert "Skywork" not in tex and "Majority vote & 52.0" in tex and "No verifier" in tex


def _pb_final() -> dict:
    vals = {"maxprob": 0.70, "ptrue": 0.89, "prm800k": 0.92, "prm7b": 0.96, "rp_rel_self": 0.89,
            "rp_rel_ds": 0.91, "instep_self": 0.93, "instep_ds": 0.935, "instep_prm": 0.945}
    sub = ["gsm8k", "math", "olympiadbench", "omnimath"]
    pb = {k: {"auroc_mean": v, "ap_mean": v - 0.2, "per_subset": {s: {"auroc": v, "ap": v - 0.2} for s in sub}}
          for k, v in vals.items()}
    c20 = {k: {"calib20_mean": v - 0.3, "per_subset": {s: v - 0.3 for s in sub}} for k, v in vals.items()}
    return {"backbones": {"qwen3_8b": {"pb": pb, "calib20": c20}}}


def test_processbench_table_has_three_metrics_and_marks_the_prm_best():
    tex = T.table_processbench(_pb_final())
    assert "Method & AUROC & PR-AUC & F1" in tex and tex.count(T.GROUP_ROW) == 4
    prm = next(line for line in tex.splitlines() if line.startswith("Qwen2.5-Math-PRM-7B"))
    assert prm.endswith("& \\textbf{.960} & \\textbf{.760} & \\textbf{.660} \\\\")
    ours = next(line for line in tex.splitlines() if "InStep} (PRM800K)" in line)
    assert ours.startswith(T.OURS_ROW) and "\\underline{.945}" in ours


def test_processbench_set_table_has_twelve_columns():
    tex = T.table_processbench_sets(_pb_final())
    assert "\\multicolumn{3}{c}{Omni-MATH}" in tex and "\\cmidrule(lr){11-13}" in tex
    row = next(line for line in tex.splitlines() if line.startswith("MaxProb"))
    assert row.count("&") == 12


def test_ablation_table_counts_parameters_and_shades_instep():
    keys = ["ptrue", "passive", "emb_ds", "off_ds", "chain_ds", "instep_ds", "nooff_ds"]
    ap = dict(zip(keys, [0.50, 0.57, 0.60, 0.643, 0.644, 0.647, 0.585]))
    b = {"metrics": {k: {"overall": {"ap": v}} for k, v in ap.items()},
         "pb": {k: {"auroc_mean": v + 0.3} for k, v in ap.items()}}
    tex = T.table_ablation({"backbones": {"qwen3_8b": b}})
    instep = next(line for line in tex.splitlines() if "(InStep)" in line)
    assert instep.startswith(T.OURS_ROW) and "& 307K & \\textbf{.647} &" in instep
    assert "embedding only & 8.2K &" in tex and "without offsets & 12K &" in tex
    allmk = next(line for line in tex.splitlines() if line.startswith("InStep, every earlier marker"))
    assert allmk.endswith("& 307K & .640 & .935 \\\\")  # REPORT 21.44 values, not from the fixture


def test_layer_series_drops_the_final_norm_row_and_numbers_layers_from_one():
    pl = {"layers": [{"kind": "passive", "block": 3, "val_ap": 0.3}, {"kind": "read", "block": 3, "val_ap": 0.31},
                     {"kind": "passive", "block": "final", "val_ap": 0.5}, {"kind": "read", "block": 35, "val_ap": 0.6}]}
    s = T.layer_series(pl)
    assert s == {"passive": [(4, 0.3)], "read": [(4, 0.31), (36, 0.6)]}


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
    # grouped bars, not lines: the three benchmarks are not one controlled sweep
    assert "ybar" in tex and "symbolic x coords={1.7B,8B,32B}" in tex and "mark=none" not in tex
    # the 32B PRM point can be replaced by the value recomputed from the released scores
    over = {label: p for label, _, p in T.size_points(o, {"qwen3_32b": 0.566})}
    assert [round(y, 3) for _, y in over["PRM-7B"]] == [0.421, 0.565, 0.566]
