import json

from scripts.analysis.instep_lr_select import joint, pick, summarize


def _write(tmp_path, name, seed, cfgs, const=0.25, steps=100):
    f = tmp_path / f"{name}_seed{seed}.json"
    f.write_text(json.dumps({"seed": seed, "steps": steps, "const_loss": const, "configs": [
        {"tag": t, "lr": lr, "warmup": wu, "dead_at": dead, "val_ap": v, "val_auroc": v,
         "trace": [[s, late if s >= 25 else 0.9] for s in range(10, 101, 10)]}
        for t, lr, wu, v, late, dead in cfgs]}))
    return str(f)


def test_unstable_seed_disqualifies_a_config_and_ties_go_to_the_lower_lr(tmp_path):
    files = []
    for s in range(4):
        files.append(_write(tmp_path, "ds", s, [
            # the high lr is best on average but one seed's late loss beats the constant predictor
            ("hi", 3e-3, 0.05, 0.62, 0.30 if s == 1 else 0.15, None),
            ("mid", 2e-3, 0.05, 0.6105, 0.15, None),
            ("low", 1e-3, 0.2, 0.6090, 0.15, None),   # within 0.003 of mid -> tie, lower lr wins
            ("tiny", 5e-4, 0.2, 0.55, 0.15, None)]))
    t = summarize(files, "ap")
    assert not t["hi"]["stable"] and t["hi"]["n_unstable"] == 1
    assert t["mid"]["stable"] and t["low"]["stable"]
    assert pick(t) == "low"


def test_outlier_seed_and_dead_run_are_unstable(tmp_path):
    files = [_write(tmp_path, "ds", s, [("a", 1e-3, 0.05, 0.31 if s == 2 else 0.61, 0.1, None),
                                        ("b", 2e-3, 0.05, 0.60, 0.1, 40 if s == 0 else None)]) for s in range(3)]
    t = summarize(files, "ap")
    assert not t["a"]["stable"] and not t["b"]["stable"] and pick(t) is None


def test_joint_recipe_needs_stability_and_near_best_everywhere(tmp_path):
    ds = summarize([_write(tmp_path, "ds", s, [("x", 1e-3, 0.05, 0.60, 0.1, None),
                                                ("y", 2e-3, 0.05, 0.602, 0.1, None)]) for s in range(3)], "ap")
    prm = summarize([_write(tmp_path, "prm", s, [("x", 1e-3, 0.05, 0.90, 0.5, None),
                                                  ("y", 2e-3, 0.05, 0.89, 0.5, None)], const=0.69) for s in range(3)],
                    "auroc")
    picks = {"ds": pick(ds), "prm": pick(prm)}
    assert picks == {"ds": "x", "prm": "x"} and joint({"ds": ds, "prm": prm}, picks) == "x"
