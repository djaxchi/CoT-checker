#!/usr/bin/env python3
"""Where does the gap to the oracle live, and can any scorer see it?

On each problem's full pool of ten samples, problems fall into four cases:

  unanimous_correct  every sample right
  majority_correct   the plurality answer is right (some samples wrong)
  rescuable          a right answer is present but not the plurality winner
                     (a lost tie counts as rescuable)
  unsolvable         no sample right

Only `rescuable` problems separate majority vote from the oracle. For each
scorer this reports, on rescuable problems:

  top1        the highest-rated sample is correct (what best-of-N would get)
  group_win   the correct answer group has a better mean score than the
              plurality group (what a group-level override would need)
  auroc       within-problem AUROC of correct against incorrect samples,
              averaged over problems

and on majority_correct problems the damage an override would do:

  top1_wrong  the highest-rated sample is wrong
  group_loss  some wrong group beats the correct plurality group on mean score

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
        minority_share = []
        stats = {s: defaultdict(list) for s in a.scorers}
        for pid, rs in by.items():
            keys = [r.get("answer_key") or answer_key(r.get("pred")) for r in rs]
            ok = [bool(r["correct"]) for r in rs]
            votes = Counter(keys)
            top = max(votes.values())
            winners = [k for k, v in votes.items() if v == top]
            good = {k for k, c in zip(keys, ok) if c}
            if all(ok):
                case = "unanimous_correct"
            elif not good:
                case = "unsolvable"
            elif len(winners) == 1 and winners[0] in good:
                case = "majority_correct"
            else:
                case = "rescuable"
                minority_share.append(max(votes[k] for k in good) / len(rs))
            cases[case] += 1
            if case not in ("rescuable", "majority_correct"):
                continue
            plural = max(votes, key=lambda k: (votes[k], k not in good))  # a tie counts as lost
            for s in a.scorers:
                q = np.array([quality[s].get(r["traj_uid"], np.nan) for r in rs], float)
                if np.isnan(q).all():
                    continue
                q = np.where(np.isnan(q), np.nanmin(q), q)
                best = int(np.argmax(q))
                gmean = {k: q[[i for i, kk in enumerate(keys) if kk == k]].mean() for k in votes}
                st = stats[s]
                if case == "rescuable":
                    st["top1"].append(ok[best])
                    best_good = max(gmean[k] for k in good)
                    st["group_win"].append(best_good > gmean[plural] if plural not in good else True)
                    st["auroc"].append(auroc(q[np.array(ok)], q[~np.array(ok)]))
                else:
                    st["top1_wrong"].append(not ok[best])
                    st["group_loss"].append(any(gmean[k] > gmean[plural] for k in votes if k not in good))
        n = len(by)
        res = {"n_problems": n, "cases": dict(cases),
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
