"""Tests for step-level guided decoding.

The scientific claim rests on three things being true of the loop: guided picks
the candidate the checker likes best, random does not consult the checker at all,
and plain never branches. If any of those silently broke, the arms would stop
being a controlled comparison while still producing plausible-looking numbers,
so each is pinned here.
"""

from __future__ import annotations

import random
import types

import pytest

from scripts.onpolicy import online_bon


class FakeChecker:
    """Scores by table lookup and records that it was consulted."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def score_steps(self, problem, prior_steps, candidates):
        self.calls += 1
        return [self.table[c] for c in candidates]


def _args(**kw):
    base = dict(n_candidates=3, temperature=1.5, top_p=0.95, max_new_tokens=64,
                max_steps=4, device="cpu")
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture
def patched(monkeypatch):
    """sample_candidates returns a fixed pool, so only selection is under test."""
    pools = {}

    def fake_sample(backbone, tok, problem, prior_steps, n, *a, **k):
        depth = len(prior_steps)
        pool = pools.get(depth, ["good", "mid", "bad"])
        # 7 tokens per sampled candidate, discarded branches included
        return pool[:n], 7 * n

    monkeypatch.setattr(online_bon, "sample_candidates", fake_sample)
    # Mirror the real return shape: grade() gives a dict, not a bool. The stub
    # returning a bool is what let the truthiness bug through unnoticed.
    monkeypatch.setattr(online_bon, "grade",
                        lambda sol, gold: {"pred": gold, "gold_norm": gold,
                                           "correct": "good" in sol,
                                           "gradeable": True})
    return pools


def test_guided_picks_the_candidate_with_the_lowest_suspicion(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    r = online_bon.rollout("guided", "p", "g", None, None, ch, _args(max_steps=1),
                           random.Random(0))
    assert r["steps"] == ["good"]
    assert r["chosen_scores"] == [0.1]
    assert r["pool_scores"] == [[0.1, 0.5, 0.9]]
    assert ch.calls == 1


def test_guided_is_not_fooled_by_candidate_order(patched):
    # best candidate last: a loop that quietly kept the first would pass the
    # previous test and fail this one
    ch = FakeChecker({"good": 0.9, "mid": 0.5, "bad": 0.1})
    r = online_bon.rollout("guided", "p", "g", None, None, ch, _args(max_steps=1),
                           random.Random(0))
    assert r["steps"] == ["bad"]


def test_random_arm_never_consults_the_checker(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    r = online_bon.rollout("random", "p", "g", None, None, ch, _args(max_steps=1),
                           random.Random(0))
    assert ch.calls == 0, "the control arm must not use the checker"
    assert r["steps"][0] in {"good", "mid", "bad"}
    assert r["chosen_scores"] == []


def test_random_arm_actually_varies_with_the_seed(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    picks = {
        online_bon.rollout("random", "p", "g", None, None, ch, _args(max_steps=1),
                           random.Random(s))["steps"][0]
        for s in range(30)
    }
    assert len(picks) > 1, "a control that always picks the same candidate is not random"


def test_plain_arm_draws_a_single_candidate(patched, monkeypatch):
    seen = {}

    def fake_sample(backbone, tok, problem, prior_steps, n, *a, **k):
        seen["n"] = n
        return ["good"][:n], 7 * n

    monkeypatch.setattr(online_bon, "sample_candidates", fake_sample)
    ch = FakeChecker({"good": 0.1})
    online_bon.rollout("plain", "p", "g", None, None, ch, _args(max_steps=1),
                       random.Random(0))
    assert seen["n"] == 1, "the base policy must not branch"
    assert ch.calls == 0


def test_rollout_stops_at_a_boxed_answer(patched):
    patched[0] = ["step one", "step one", "step one"]
    patched[1] = ["the answer is \\boxed{7}"] * 3
    ch = FakeChecker({"step one": 0.2, "the answer is \\boxed{7}": 0.2})
    r = online_bon.rollout("guided", "p", "7", None, None, ch, _args(max_steps=8),
                           random.Random(0))
    assert r["n_steps"] == 2, "generation must stop once the answer is written"


def test_rollout_respects_the_step_budget(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    r = online_bon.rollout("guided", "p", "g", None, None, ch, _args(max_steps=3),
                           random.Random(0))
    assert r["n_steps"] == 3


def test_is_final_only_fires_on_a_boxed_answer():
    assert online_bon.is_final("so x = \\boxed{42}")
    assert not online_bon.is_final("next we compute x = 42")
    assert not online_bon.is_final("consider the box of weight 42")


# --------------------------------------------------------------------------
# cost accounting: branching is not free and the numbers must say so
# --------------------------------------------------------------------------

def test_branching_arms_are_charged_for_discarded_candidates(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    args = _args(max_steps=2, n_candidates=3)
    guided = online_bon.rollout("guided", "p", "g", None, None, ch, args,
                                random.Random(0))
    plain = online_bon.rollout("plain", "p", "g", None, None, ch, args,
                               random.Random(0))
    assert guided["gen_tokens"] == 3 * plain["gen_tokens"], (
        "a branching arm must be billed for every sampled candidate, not just "
        "the one it kept, or guided decoding looks cheaper than it is")


def test_checker_workload_is_recorded_for_the_guided_arm_only(patched):
    ch = FakeChecker({"good": 0.1, "mid": 0.5, "bad": 0.9})
    args = _args(max_steps=2, n_candidates=3)
    g = online_bon.rollout("guided", "p", "g", None, None, ch, args, random.Random(0))
    r = online_bon.rollout("random", "p", "g", None, None, ch, args, random.Random(0))
    assert g["checker_calls"] == 2 and g["scored_candidates"] == 6
    assert r["checker_calls"] == 0 and r["scored_candidates"] == 0


# --------------------------------------------------------------------------
# the report: the contrast that isolates the checker must be the one reported
# --------------------------------------------------------------------------

def test_mcnemar_counts_only_disagreements():
    from scripts.analysis.online_bon_report import mcnemar
    a = {f"p{i}": True for i in range(10)}
    b = dict(a)
    assert mcnemar(a, b)["n_discordant"] == 0

    b["p0"] = False          # a wins one
    b["p1"] = False          # a wins two
    a["p2"] = False          # b wins one
    m = mcnemar(a, b)
    assert m["a_wins"] == 2 and m["b_wins"] == 1 and m["n_discordant"] == 3


def test_paired_bootstrap_recovers_the_accuracy_difference():
    from scripts.analysis.online_bon_report import paired_bootstrap
    a = {f"p{i}": False for i in range(100)}
    b = {f"p{i}": i < 20 for i in range(100)}   # b right on 20 more
    d, lo, hi = paired_bootstrap(a, b, n_boot=400)
    assert d == pytest.approx(0.20, abs=1e-9)
    assert lo < 0.20 < hi


def test_rollout_reads_the_correct_field_not_the_grade_dict(monkeypatch):
    """grade() returns a dict, and bool(dict) is True for any non-empty dict.

    Taking its truthiness marked every rollout correct and produced 1.000
    accuracy for all three arms on a pool whose pass@1 is 0.366. The arms are
    only meaningful if a wrong answer can be scored wrong, so this pins it.
    """
    pools = {}

    def fake_sample(backbone, tok, problem, prior_steps, n, *a, **k):
        return ["the answer is \\boxed{3}"][:n] or ["x"], 7 * n

    monkeypatch.setattr(online_bon, "sample_candidates", fake_sample)
    monkeypatch.setattr(online_bon, "grade",
                        lambda sol, gold: {"pred": "3", "gold_norm": gold,
                                           "correct": False, "gradeable": True})
    ch = FakeChecker({"the answer is \\boxed{3}": 0.1})
    r = online_bon.rollout("guided", "p", "7", None, None, ch, _args(max_steps=1),
                           random.Random(0))
    assert r["correct"] is False, "a wrong answer must be scored wrong"

    monkeypatch.setattr(online_bon, "grade",
                        lambda sol, gold: {"pred": "7", "gold_norm": gold,
                                           "correct": True, "gradeable": True})
    r = online_bon.rollout("guided", "p", "7", None, None, ch, _args(max_steps=1),
                           random.Random(0))
    assert r["correct"] is True


def test_plain_arm_can_use_its_own_temperature(patched, monkeypatch):
    """The base policy's temperature is 1.0; the branching arms use 1.5.

    Running plain at the branching temperature turns the baseline into a
    strawman, so the flag has to actually reach the sampler.
    """
    seen = {}

    def fake_sample(backbone, tok, problem, prior_steps, n, temperature, *a, **k):
        seen[n] = temperature
        return ["good"][:n] or ["good"], 7 * n

    monkeypatch.setattr(online_bon, "sample_candidates", fake_sample)
    ch = FakeChecker({"good": 0.1})
    args = _args(max_steps=1, n_candidates=3, temperature=1.5, plain_temperature=1.0)
    online_bon.rollout("plain", "p", "g", None, None, ch, args, random.Random(0))
    online_bon.rollout("random", "p", "g", None, None, ch, args, random.Random(0))
    assert seen[1] == 1.0, "plain must sample at the policy's own temperature"
    assert seen[3] == 1.5, "branching arms keep the diversity temperature"


def test_sampler_passes_the_policys_top_k(monkeypatch):
    """The generator that produced every baseline samples with top_k=50.

    Omitting it made the plain arm a different policy, which is part of why it
    scored 0.193 against a true pass@1 of 0.366.
    """
    seen = {}

    class FakeTok:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, **kw):
            class E(dict):
                def to(self, d):
                    return self
            e = E(input_ids=__import__("torch").zeros((1, 3), dtype=__import__("torch").long))
            return e

        def decode(self, ids, **kw):
            return "a step"

    class FakeBackbone:
        def generate(self, **kw):
            seen.update(kw)
            import torch
            return torch.zeros((kw["num_return_sequences"], 5), dtype=torch.long)

    online_bon.sample_candidates(FakeBackbone(), FakeTok(), "p", [], 3,
                                 1.5, 0.95, 64, "cpu", top_k=50)
    assert seen["top_k"] == 50
    assert seen["temperature"] == 1.5 and seen["top_p"] == 0.95


# ---------------------------------------------------------------------------
# rejection arm
#
# The guided arm lost because branching at temperature 1.5 damaged the policy and
# because "safest of five" rewards steps that never commit (REPORT.md §20.9). The
# rejection arm removes both, so the properties worth pinning are the ones that
# would quietly put them back: that it draws one step at a time at the policy's
# own temperature, that it stops as soon as a step is not condemned, and that
# every discarded draw is still paid for.
# ---------------------------------------------------------------------------

def _reject_args(**kw):
    base = dict(n_candidates=1, temperature=1.5, plain_temperature=1.0,
                reject_temperature=1.0, top_p=0.95, top_k=50, max_new_tokens=64,
                max_steps=4, device="cpu", reject_tau=0.5, max_retries=2,
                blind_retry_rate=0.5)
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture
def draws(monkeypatch):
    """Each call to the sampler yields the next queued draw, so retries differ."""
    queue = []

    def fake_sample(backbone, tok, problem, prior_steps, n, temperature, *a, **k):
        assert n == 1, "the rejection arm must draw one step at a time"
        fake_sample.temperatures.append(temperature)
        text = queue.pop(0) if queue else "filler \\boxed{9}"
        return [text], 7

    fake_sample.temperatures = []
    monkeypatch.setattr(online_bon, "sample_candidates", fake_sample)
    monkeypatch.setattr(online_bon, "grade",
                        lambda sol, gold: {"pred": gold, "gold_norm": gold,
                                           "correct": "good" in sol,
                                           "gradeable": True})
    return queue, fake_sample


def test_a_step_the_checker_accepts_is_never_resampled(draws):
    queue, sampler = draws
    queue.extend(["good \\boxed{4}"])
    chk = FakeChecker({"good \\boxed{4}": 0.1})
    r = online_bon.rollout("reject", "p?", "4", None, None, chk, _reject_args(),
                           random.Random(0))
    assert r["attempts_per_step"] == [1]
    assert r["gen_tokens"] == 7        # one draw paid for, no retry
    assert chk.calls == 1


def test_a_condemned_step_is_resampled_and_the_discarded_draw_is_paid_for(draws):
    queue, _ = draws
    queue.extend(["bad", "good \\boxed{4}"])
    chk = FakeChecker({"bad": 0.9, "good \\boxed{4}": 0.1})
    r = online_bon.rollout("reject", "p?", "4", None, None, chk, _reject_args(),
                           random.Random(0))
    assert r["attempts_per_step"] == [2]
    assert r["steps"] == ["good \\boxed{4}"]
    assert r["gen_tokens"] == 14       # the rejected draw still cost tokens
    assert r["pool_scores"] == [[0.9, 0.1]]


def test_when_every_retry_is_condemned_the_least_bad_is_taken(draws):
    """Giving up and emitting nothing would be worse than emitting the best of a
    bad set, and would make the step budget the real policy."""
    queue, _ = draws
    queue.extend(["bad1", "bad2 \\boxed{4}", "bad3"])
    chk = FakeChecker({"bad1": 0.9, "bad2 \\boxed{4}": 0.7, "bad3": 0.8})
    r = online_bon.rollout("reject", "p?", "4", None, None, chk,
                           _reject_args(max_retries=2), random.Random(0))
    assert r["attempts_per_step"] == [3]
    assert r["steps"] == ["bad2 \\boxed{4}"]
    assert r["gen_tokens"] == 21


def test_the_rejection_arm_samples_at_the_policys_own_temperature(draws):
    """The entire -0.304 hole guided decoding had to climb out of came from
    sampling at 1.5 for candidate diversity. This arm must not reopen it."""
    queue, sampler = draws
    queue.extend(["good \\boxed{4}"])
    online_bon.rollout("reject", "p?", "4", None, None,
                       FakeChecker({"good \\boxed{4}": 0.1}),
                       _reject_args(temperature=1.5), random.Random(0))
    assert sampler.temperatures == [1.0]


def test_the_answer_step_is_resampled_like_any_other(draws):
    """It is the position the checker should be best at; exempting it would
    exempt the only step that commits to a number."""
    queue, _ = draws
    queue.extend(["think", "wrong \\boxed{7}", "good \\boxed{4}"])
    chk = FakeChecker({"think": 0.1, "wrong \\boxed{7}": 0.9,
                       "good \\boxed{4}": 0.1})
    r = online_bon.rollout("reject", "p?", "4", None, None, chk, _reject_args(),
                           random.Random(0))
    assert r["steps"] == ["think", "good \\boxed{4}"]


def test_the_blind_arm_never_consults_the_checker(draws):
    queue, _ = draws
    queue.extend(["a", "b \\boxed{4}"])
    chk = FakeChecker({})
    r = online_bon.rollout("reject_blind", "p?", "4", None, None, chk,
                           _reject_args(), random.Random(0))
    assert chk.calls == 0
    assert r["checker_calls"] == 0 and r["scored_candidates"] == 0


def test_the_blind_arm_never_retries_when_its_rate_is_zero(draws):
    """At rate 0 it is plain sampling exactly, which is what makes it a price tag
    for the retry loop rather than a second policy."""
    queue, _ = draws
    queue.extend(["only \\boxed{4}"])
    r = online_bon.rollout("reject_blind", "p?", "4", None, None, FakeChecker({}),
                           _reject_args(blind_retry_rate=0.0), random.Random(1))
    assert r["attempts_per_step"] == [1] and r["gen_tokens"] == 7


def test_the_step_budget_still_binds_the_rejection_arm(draws):
    queue, _ = draws
    chk = FakeChecker({})
    chk.score_steps = lambda p, prior, cands: [0.1] * len(cands)
    r = online_bon.rollout("reject", "p?", "4", None, None, chk,
                           _reject_args(max_steps=3), random.Random(0))
    assert r["n_steps"] <= 3


def test_the_threshold_is_calibrated_as_a_quantile_of_this_cells_own_scores(tmp_path):
    """A raw threshold means something different for every head; a quantile means
    "the worst q of steps this head has seen" whichever head it is."""
    import json as _json
    f = tmp_path / "scores.jsonl"
    f.write_text("\n".join(_json.dumps({"id": str(i), "scores": [i / 10]})
                           for i in range(11)))
    assert online_bon.calibrate_tau(f, 0.5) == pytest.approx(0.5)
    assert online_bon.calibrate_tau(f, 0.8) == pytest.approx(0.8)


def test_the_default_arms_are_the_three_the_gate_expects():
    """Job 462208 died in 33 seconds because the rejection arms were added to
    ARMS, which was also the default for --arms, so the verification gate
    silently enrolled in arms it never asked for and demanded a threshold it
    would never use."""
    assert online_bon.DEFAULT_ARMS == ("plain", "random", "guided")
    assert set(online_bon.DEFAULT_ARMS) < set(online_bon.ARMS)


def test_the_gate_is_checked_before_any_generation_argument():
    """The gate generates nothing, so nothing about generation may gate it."""
    src = (online_bon.__file__).replace(".pyc", ".py")
    body = open(src).read()
    verify_at = body.index("if a.verify_against:\n        sys.exit(verify(")
    reject_at = body.index('if any(arm.startswith("reject") for arm in a.arms):')
    assert verify_at < reject_at


def test_the_rejected_draft_text_is_kept_not_just_its_score(draws):
    """Without the text a reader can see that a draft was condemned at 0.94 but
    not what it said, which is the one thing needed to judge the checker."""
    queue, _ = draws
    queue.extend(["bad opening", "good \\boxed{4}"])
    chk = FakeChecker({"bad opening": 0.9, "good \\boxed{4}": 0.1})
    r = online_bon.rollout("reject", "p?", "4", None, None, chk, _reject_args(),
                           random.Random(0))
    assert r["rejected_drafts"] == [["bad opening"]]
    assert r["steps"] == ["good \\boxed{4}"]


def test_kept_attempt_indexes_the_score_of_the_winner(draws):
    """pool_scores holds every draft's score, so the winner has to be identified
    by index or the alignment is a guess."""
    queue, _ = draws
    queue.extend(["a", "b", "c \\boxed{4}"])
    chk = FakeChecker({"a": 0.9, "b": 0.8, "c \\boxed{4}": 0.1})
    r = online_bon.rollout("reject", "p?", "4", None, None, chk, _reject_args(),
                           random.Random(0))
    assert r["kept_attempt"] == [2]
    assert r["pool_scores"][0][r["kept_attempt"][0]] == 0.1
    assert r["rejected_drafts"] == [["a", "b"]]


def test_an_exhausted_step_records_the_drafts_it_beat(draws):
    """When retries run out the least condemned wins, and it is not the last one
    drawn, so the index must follow the winner rather than the order."""
    queue, _ = draws
    queue.extend(["worst", "best \\boxed{4}", "middling"])
    chk = FakeChecker({"worst": 0.9, "best \\boxed{4}": 0.7, "middling": 0.8})
    r = online_bon.rollout("reject", "p?", "4", None, None, chk,
                           _reject_args(max_retries=2), random.Random(0))
    assert r["kept_attempt"] == [1]
    assert r["rejected_drafts"] == [["worst", "middling"]]
    assert r["steps"] == ["best \\boxed{4}"]


def test_a_step_kept_first_try_records_no_rejected_draft(draws):
    queue, _ = draws
    queue.extend(["fine \\boxed{4}"])
    r = online_bon.rollout("reject", "p?", "4", None, None,
                           FakeChecker({"fine \\boxed{4}": 0.1}), _reject_args(),
                           random.Random(0))
    assert r["rejected_drafts"] == [[]] and r["kept_attempt"] == [0]


def test_the_blind_arm_also_records_what_its_coin_threw_away(draws):
    """Comparing what the checker rejects against what a coin rejects is the
    whole point of having the blind arm's text too."""
    queue, _ = draws
    queue.extend(["x", "y \\boxed{4}"])
    r = online_bon.rollout("reject_blind", "p?", "4", None, None, FakeChecker({}),
                           _reject_args(blind_retry_rate=1.0, max_retries=1),
                           random.Random(0))
    assert sum(len(v) for v in r["rejected_drafts"]) >= 1


def test_sample_candidates_defaults_to_the_zero_shot_context():
    """Every caller written before tts_roster_v1 must stay byte-identical."""
    import scripts.onpolicy.online_bon as ob
    from src.onpolicy.prompts import generation_prefix
    seen = {}

    class Tok:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, **kw):
            seen["ctx"] = text
            import torch

            class E(dict):
                def to(self, d):
                    return self
            return E(input_ids=torch.zeros((1, 3), dtype=torch.long))

        def decode(self, ids, **kw):
            return "a step"

    class Backbone:
        def generate(self, **kw):
            import torch
            return torch.zeros((kw["num_return_sequences"], 5), dtype=torch.long)

    ob.sample_candidates(Backbone(), Tok(), "P", ["s1"], 1, 1.0, 0.95, 32, "cpu")
    assert seen["ctx"] == generation_prefix("P", "s1")


def test_sample_candidates_builds_the_fewshot_context_when_asked():
    """A hardcoded prompt here fails silently: the pass runs and the scores
    describe a context the model never saw."""
    import scripts.onpolicy.online_bon as ob
    from src.onpolicy.fewshot import fewshot_prompt
    seen = {}

    class Tok:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, **kw):
            seen["ctx"] = text
            import torch

            class E(dict):
                def to(self, d):
                    return self
            return E(input_ids=torch.zeros((1, 3), dtype=torch.long))

        def decode(self, ids, **kw):
            return "a step"

    class Backbone:
        def generate(self, **kw):
            import torch
            return torch.zeros((kw["num_return_sequences"], 5), dtype=torch.long)

    ob.sample_candidates(Backbone(), Tok(), "P", ["s1"], 1, 1.0, 0.95, 32, "cpu",
                         50, "fewshot", "math", 4)
    assert seen["ctx"] == fewshot_prompt("P", "math") + "s1\n\n"


def test_score_plain_records_scores_without_changing_the_kept_step():
    """Calibrating a rejection threshold needs scores from the dataset it will
    run on: a quantile of PRM800K scores is a different rejection RATE on GSM8K,
    and unmatched rates compare compute rather than skill.

    The scoring must be inert. If it moved the plain arm's choice, plain would
    stop being the checker-blind control the whole design rests on.
    """
    import types
    import scripts.onpolicy.online_bon as ob

    calls = []

    class Checker:
        def score_steps(self, problem, prior, cands):
            calls.append(list(cands))
            return [0.9] * len(cands)

    def fake_sample(*a, **k):
        n = a[4] if len(a) > 4 else 1
        return (["a step" for _ in range(n)], 7)

    orig = ob.sample_candidates
    ob.sample_candidates = fake_sample
    try:
        args = types.SimpleNamespace(
            max_steps=2, temperature=1.0, plain_temperature=1.0, top_p=0.95,
            top_k=50, max_new_tokens=32, device="cpu", n_candidates=1,
            score_plain=True, prompt_style="zero", dataset="", n_shot=4)
        out = ob.rollout("plain", "P", "1", None, None, Checker(), args,
                         __import__("random").Random(0))
    finally:
        ob.sample_candidates = orig

    assert calls, "checker was never consulted"
    assert all(len(c) == 1 for c in calls), "plain must score one candidate, not a pool"
    assert all(s == "a step" for s in out["steps"]), "the kept step changed"


def test_score_plain_defaults_off_so_the_control_stays_free():
    import types
    import scripts.onpolicy.online_bon as ob

    class Checker:
        def score_steps(self, *a):
            raise AssertionError("plain must not score unless asked")

    def fake_sample(*a, **k):
        return (["a step"], 7)

    orig = ob.sample_candidates
    ob.sample_candidates = fake_sample
    try:
        args = types.SimpleNamespace(
            max_steps=1, temperature=1.0, plain_temperature=1.0, top_p=0.95,
            top_k=50, max_new_tokens=32, device="cpu", n_candidates=1,
            prompt_style="zero", dataset="", n_shot=4)
        ob.rollout("plain", "P", "1", None, None, Checker(), args,
                   __import__("random").Random(0))
    finally:
        ob.sample_candidates = orig


# ---- blind-arm retry rate --------------------------------------------------
#
# The blind arm prices the retry loop: same machinery, accept decided by a coin.
# It only prices anything if it draws as often as the real arm. §20.17 passed the
# reject arm's measured extra-draws-per-step straight in as the per-draw
# probability, so 0.6399 became p + p^2 = 1.05 extra draws against reject's 0.54.

def test_blind_probability_solves_the_geometric_sum():
    from scripts.onpolicy.online_bon import blind_probability
    p = blind_probability(0.54, 2)
    assert abs((p + p ** 2) - 0.54) < 1e-6
    assert abs(p - 0.3888) < 1e-3          # the value §20.17 said it should be


def test_blind_probability_reproduces_the_bug_it_replaces():
    """Passing the rate straight through, as the old launcher did, overshoots."""
    from scripts.onpolicy.online_bon import blind_probability
    naive = 0.6399
    assert naive + naive ** 2 > 1.0        # 1.05 extra draws
    fixed = blind_probability(0.54, 2)
    assert fixed + fixed ** 2 < naive + naive ** 2


def test_blind_probability_generalises_past_two_retries():
    from scripts.onpolicy.online_bon import blind_probability
    for k in (1, 3, 4):
        p = blind_probability(0.5, k)
        got = sum(p ** i for i in range(1, k + 1))
        assert abs(got - 0.5) < 1e-6, f"k={k} gave {got}"


def test_blind_probability_clamps_at_the_ends():
    from scripts.onpolicy.online_bon import blind_probability
    assert blind_probability(0.0, 2) == 0.0
    assert blind_probability(-1.0, 2) == 0.0
    assert blind_probability(5.0, 2) == 1.0      # unreachable target


def test_blind_probability_rejects_zero_retries():
    import pytest
    from scripts.onpolicy.online_bon import blind_probability
    with pytest.raises(ValueError):
        blind_probability(0.3, 0)
