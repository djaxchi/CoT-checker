from scripts.instep_v1.showcase import classify

N = ["instep", "prm", "rp"]
THR = {"instep": 0.5, "prm": 0.5, "rp": 0.5}


def tr(labels, pi, pp, pr, s="math"):
    return {"set": s, "labels": labels, "p": {"instep": pi, "prm": pp, "rp": pr}}


def test_catch_needs_instep_flag_at_first_error_and_both_others_quiet():
    t = tr([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8], [0.1, 0.1, 0.3, 0.9], [0.2, 0.1, 0.2, 0.7])
    assert [(c, k) for c, _, k in classify(t, THR, N)] == [("catch", 2)]
    t["p"]["instep"][1] = 0.6  # an InStep false alarm before the error disqualifies it
    assert classify(t, THR, N) == []


def test_quiet_and_miss():
    q = tr([0, 0, 0], [0.1, 0.2, 0.3], [0.1, 0.8, 0.1], [0.9, 0.1, 0.1])
    assert [c for c, _, _ in classify(q, THR, N)] == ["quiet"]
    m = tr([0, 1], [0.1, 0.2], [0.1, 0.9], [0.1, 0.8])
    assert [c for c, _, _ in classify(m, THR, N)] == ["miss"]


def test_processbench_catch_compares_first_flags():
    t = tr([0, 0, 1, -1], [0.1, 0.2, 0.9, 0.9], [0.6, 0.1, 0.9, 0.9], [0.1, 0.1, 0.2, 0.9], s="pb_math")
    assert [(c, k) for c, _, k in classify(t, THR, N)] == [("catch", 2)]
