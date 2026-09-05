#!/usr/bin/env python3
"""Why does a verifier that ranks well still lose to counting votes?

The measurement that prompted this: on-policy, the best verifier ranks solutions
to the *same* problem at AUROC 0.739, yet keeping its top pick scores 0.503
against 0.560 for plain majority voting, and weighting the vote by its score
lands on 0.561, which is the unweighted number. A verifier with real ranking
signal is adding nothing.

Two explanations, and they call for different work.

**Arity.** Best-of-N commits to one solution and throws away the other nine.
Majority voting reads all ten. A ranking good enough to beat a coin flip is not
automatically good enough to beat ten votes, so the comparison may be about how
much of the sample each rule consumes rather than about signal.

**Redundancy.** Agreement with the majority is itself a strong correctness
signal, and if the verifier is largely detecting the same thing, weighting a vote
by it cannot help. This is the one that would matter for the paper: it would say
internal-state verifiers and self-consistency are reading one signal, not two.

The script separates them. It scores agreement on its own, measures how much of
the verifier's ranking is explained by it, and then asks the practical question:
inside the majority bloc, where every solution already agrees, does the verifier
still pick the right one. If it does, the verifier carries information voting
does not, and the way to use it is to combine them rather than to replace one
with the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.analysis.onpolicy_downstream import (  # noqa: E402
    aggregate, auroc, cell_solutions, load_outcomes,
)
from src.eval.math_grade import normalize_answer  # noqa: E402


def agreement_fraction(sols: list[dict]) -> list[float]:
    """For each solution, the share of its problem's solutions giving its answer."""
    answers = [normalize_answer(s.get("pred")) for s in sols]
    counts = Counter(a for a in answers if a is not None)
    n = max(1, sum(counts.values()))
    return [counts.get(a, 0) / n if a is not None else 0.0 for a in answers]


def within(vals: list[tuple[np.ndarray, np.ndarray]]) -> float:
    out = [auroc(y, s) for y, s in vals if y.min() != y.max()]
    out = [v for v in out if not np.isnan(v)]
    return float(np.mean(out)) if out else float("nan")


def analyse(groups: dict[str, list[dict]], how: str = "worst_step") -> dict:
    agree_pairs, probe_pairs = [], []
    corr_num, corr_den_a, corr_den_p = 0.0, 0.0, 0.0
    bloc_hits, bloc_probe_hits, bloc_n = [], [], []
    for sols in groups.values():
        agree = np.array(agreement_fraction(sols))
        probe = np.array([aggregate(s["scores"], how) for s in sols])
        y = np.array([0 if s["correct"] else 1 for s in sols])
        # agreement is a correctness signal: higher share -> more likely right,
        # so it is negated to point the same way as the probe's suspicion score
        agree_pairs.append((y, -agree))
        probe_pairs.append((y, probe))
        if len(sols) > 1 and agree.std() > 0 and probe.std() > 0:
            a, p = agree - agree.mean(), probe - probe.mean()
            corr_num += float((a * p).sum())
            corr_den_a += float((a ** 2).sum())
            corr_den_p += float((p ** 2).sum())
        # inside the majority bloc every solution already agrees, so voting has
        # nothing left to say and only the verifier can break the tie
        top = agree.max()
        bloc = [i for i in range(len(sols)) if agree[i] == top]
        if len(bloc) > 1:
            bloc_n.append(len(bloc))
            bloc_hits.append(float(np.mean([sols[i]["correct"] for i in bloc])))
            pick = min(bloc, key=lambda i: (probe[i], i))
            bloc_probe_hits.append(float(sols[pick]["correct"]))
    r = (corr_num / np.sqrt(corr_den_a * corr_den_p)) if corr_den_a and corr_den_p else float("nan")
    return {
        "within_problem_auroc_agreement": within(agree_pairs),
        "within_problem_auroc_probe": within(probe_pairs),
        "pooled_corr_probe_vs_agreement": float(r),
        "n_problems_with_a_bloc": len(bloc_n),
        "mean_bloc_size": float(np.mean(bloc_n)) if bloc_n else float("nan"),
        "bloc_random_pick": float(np.mean(bloc_hits)) if bloc_hits else float("nan"),
        "bloc_probe_pick": float(np.mean(bloc_probe_hits)) if bloc_probe_hits else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid_root", required=True, type=Path)
    p.add_argument("--scores_name", default="onpolicy_verifier")
    p.add_argument("--outcomes", required=True, type=Path)
    p.add_argument("--cells", nargs="*", default=None,
                   help="Cell directory names; default is every cell present.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    outcomes = load_outcomes(args.outcomes)
    rows = []
    dirs = sorted(args.grid_root.iterdir())
    if args.cells:
        dirs = [d for d in dirs if d.name in set(args.cells)]
    for d in dirs:
        rj, sj = d / "results.json", d / f"pb_step_scores_{args.scores_name}.jsonl"
        if not (rj.exists() and sj.exists()):
            continue
        res = json.loads(rj.read_text())
        groups = cell_solutions(sj, outcomes)
        if not groups:
            continue
        rows.append({"cell": f"{res['rep']} x {res['learner']}", "seed": res["seed"],
                     **analyse(groups)})
    if not rows:
        raise SystemExit("no cells with scores")

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)

    a0 = rows[0]["within_problem_auroc_agreement"]
    print(f"agreement with the majority, on its own, ranks solutions inside a "
          f"problem at AUROC {a0:.3f}\n")
    print(f"{'cell':<44}{'probe':>8}{'agree':>8}{'corr':>8}"
          f"{'bloc rnd':>10}{'bloc probe':>12}")
    for cell, rs in sorted(by_cell.items(),
                           key=lambda kv: -np.mean([r["within_problem_auroc_probe"]
                                                    for r in kv[1]])):
        m = {k: float(np.mean([r[k] for r in rs])) for k in rs[0] if k != "cell"}
        print(f"{cell:<44}{m['within_problem_auroc_probe']:>8.3f}"
              f"{m['within_problem_auroc_agreement']:>8.3f}"
              f"{m['pooled_corr_probe_vs_agreement']:>8.3f}"
              f"{m['bloc_random_pick']:>10.3f}{m['bloc_probe_pick']:>12.3f}")
    print("\nprobe / agree: within-problem AUROC of each signal alone.")
    print("corr: correlation between the probe's suspicion and agreement, "
          "centred within each problem. Near zero means the two are reading "
          "different things and a combination should help.")
    print("bloc rnd / bloc probe: inside the majority bloc, where voting has "
          "nothing left to say, the accuracy of taking any member against taking "
          "the one the probe likes most.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"[agreement] wrote {args.out}")


if __name__ == "__main__":
    main()
