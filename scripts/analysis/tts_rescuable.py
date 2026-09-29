#!/usr/bin/env python3
"""Where does the gap to the oracle live, and can any scorer see it?

On each problem's full pool, separate strict minority failures from vote ties:

  unanimous_correct  every sample right
  majority_correct_mixed  the plurality is correct, with some wrong samples
  strict_minority_correct a correct answer is present but has fewer votes
  vote_tie_with_correct   the correct answer ties a wrong answer at the top
  no_correct_sample      no sample right

The two rescue cases contribute oracle minus expected majority credit. A vote
tie is not a full majority failure. For each scorer, on their union, report:

  top1        the highest-rated sample is correct (what best-of-N would get)
  group_win   expected correctness when selecting the highest mean-score group
              among all groups, with uniform tie handling
  auroc       within-problem AUROC of correct against incorrect samples,
              averaged over problems

and on majority_correct_mixed problems the damage an override would do:

  top1_wrong  the highest-rated sample is wrong
  group_loss  expected error of selecting the highest mean-score group

Suspicion scores (probes, PRM stored as 1 - P(correct)) use the worst step and
lower is better; confidence is higher-is-better.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.math_grade import answer_key  # noqa: E402
from src.analysis.downstream_audit import describe_pool, outcome  # noqa: E402


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", type=Path, required=True)
    p.add_argument("--scorers", nargs="+", required=True,
                   help="score-file cell names, or conf:<rule> for token confidence")
    p.add_argument("--stems", nargs="+", default=["tts_gsm8k", "tts_math500"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    out = {}
    for stem in a.stems:
        tr = [json.loads(l) for f in sorted(a.run_root.glob(f"{stem}.shard*_trajectories.jsonl"))
              for l in f.read_text().splitlines() if l.strip()]
        conf = {}
        for f in sorted(a.run_root.glob(f"{stem}.shard*_conf.jsonl")):
            for l in f.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); conf[r["traj_uid"]] = r["rules"]
        quality = {}
        for s in a.scorers:
            q = {}
            if s.startswith("conf:"):
                rule = s.split(":", 1)[1]
                q = {u: v[rule] for u, v in conf.items() if rule in v}
            else:
                for f in sorted((a.run_root / "scores").glob(f"{stem}__{s}.shard*.jsonl")):
                    for l in f.read_text().splitlines():
                        if l.strip():
                            r = json.loads(l); q[r["traj_uid"]] = -max(r["scores"])
            quality[s] = q                                   # higher is better throughout
        by = defaultdict(list)
        for r in tr:
            if r.get("gradeable"):
                by[r["fork_id"]].append(r)

        cases = Counter()
        majority_credit, oracle_credit = [], []
        minority_share = []
        stats = {s: defaultdict(list) for s in a.scorers}
        for pid, rs in by.items():
            keys = [r["answer_key"] if "answer_key" in r else answer_key(r.get("pred")) for r in rs]
            ok = [bool(r["correct"]) for r in rs]
            pool = describe_pool(keys, ok)
            case = pool["case"]
            rescuable = case in ("strict_minority_correct", "vote_tie_with_correct")
            if rescuable:
                minority_share.append(max(pool["counts"][pool["truth"].astype(bool)]) / len(rs))
            majority_credit.append(pool["majority"])
            oracle_credit.append(pool["oracle"])
            cases[case] += 1
            if not rescuable and case != "majority_correct_mixed":
                continue
            for s in a.scorers:
                q = np.array([quality[s].get(r["traj_uid"], np.nan) for r in rs], float)
                if not np.isfinite(q).all():
                    raise ValueError(f"{pid}: missing or non-finite scores for {s}")
                top1 = outcome(pool, -q, "rerank")
                group_win = outcome(pool, -q, "group_mean")
                st = stats[s]
                if rescuable:
                    st["top1"].append(top1)
                    st["group_win"].append(group_win)
                    st["auroc"].append(auroc(q[np.array(ok)], q[~np.array(ok)]))
                else:
                    st["top1_wrong"].append(1-top1)
                    st["group_loss"].append(1-group_win)
        n = len(by)
        res = {"n_problems": n, "cases": dict(cases),
               "majority_expected": float(np.mean(majority_credit)),
               "oracle": float(np.mean(oracle_credit)),
               "oracle_gap_problem_equivalents": float(np.sum(np.array(oracle_credit)-majority_credit)),
               "score_ties": "uniform expectation", "case_schema": "strict_minority_and_vote_ties_v2",
               "rescuable_share_of_correct_answer": {
                   "median": float(np.median(minority_share)) if minority_share else None,
                   "histogram": dict(Counter(round(x, 1) for x in minority_share))},
               "scorers": {s: {k: float(np.mean(v)) for k, v in st.items()} for s, st in stats.items()}}
        out[stem] = res
        print(f"== {stem}: {n} problems  " + "  ".join(f"{k} {v} ({100*v/n:.1f}%)" for k, v in sorted(cases.items())))
        print(f"   correct-answer vote share on rescuable problems: {res['rescuable_share_of_correct_answer']}")
        for s in a.scorers:
            v = res["scorers"][s]
            print(f"   {s:58s} rescuable: top1 {v.get('top1',0):.3f} group_win {v.get('group_win',0):.3f} "
                  f"auroc {v.get('auroc',0):.3f} | majority_correct: top1_wrong {v.get('top1_wrong',0):.3f} "
                  f"group_loss {v.get('group_loss',0):.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
