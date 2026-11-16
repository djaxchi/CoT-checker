import json

from scripts.instep_v1.fig_spectrum import figure, load


def test_load_log_and_json_later_wins(tmp_path):
    log = tmp_path / "a.log"
    log.write_text("[anatomy] prm_final      budget task 0.1 | spectrum 1:0.486 2:0.681 all:0.830 | reliance chars +0.07\n"
                   "[anatomy] sep budget task 0.1 | spectrum 1:0.5 all:0.6 | reliance x\n")
    js = tmp_path / "b.json"
    js.write_text(json.dumps({"keys": [{"key": "sep", "spectrum_auroc": {"1": 0.754, "all": 0.852}}]}))
    s = load([log, js])
    assert s["prm_final"] == {"1": 0.486, "2": 0.681, "all": 0.830}
    assert s["sep"] == {"1": 0.754, "all": 0.852}


def test_figure_puts_all_at_the_right_end():
    tex = figure({"sep": {"1": 0.754, "2": 0.846, "all": 0.852}})
    assert "(1,0.754) (2,0.846) (1024,0.852)" in tex and "step token" in tex and "text token" not in tex
