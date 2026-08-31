#!/usr/bin/env python3
"""Do the two labellers agree, and does the conclusion depend on which we used?

The on-policy arm ended up with two ways to mark the first wrong step, and they
were certified against humans in different ways because they are different kinds
of thing.

  judge     DeepSeek-R1 over the API, ReProbe's recipe, certified on
            human-labelled ProcessBench traces by F1_PB. Costs money, so it saw
            a budgeted sample.
  rollout   sample K continuations from each prefix and grade them; the first
            step the model cannot recover from is the first error. Free, covers
            everything, certified on PRM800K matched forks by whether the value
            drops at the step humans rated wrong.

They also mean different things. A judge marks a step that is *wrong*; a rollout
marks a step after which the model cannot *recover*. Agreement between them is
therefore not a correctness check on either, it is a measure of how much the two
readings of "first error" coincide on this distribution. What it is for is the
question one step further out: if the rank result comes out the same under both,
it does not depend on which reading we took, and if it does not, that is a
finding about the metric rather than about the representations.

Reports exact agreement, agreement within one step (the boundary between two
steps is not always sharp), how often one finds an error the other does not, and
where in the trace each of them points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.encode_prm800k_hidden_states import read_jsonl  # noqa: E402

NO_ERROR = -1


def load(path: Path) -> dict[str, dict]:
    out = {}
    for r in read_jsonl(path):
        uid = r.get("traj_uid") or r.get("id")
        if uid is None:
            continue
        out[uid] = r
    return out


def agreement(a: dict[str, dict], b: dict[str, dict]) -> dict:
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n_shared": 0}
    la = np.array([int(a[k]["first_error"]) for k in shared])
    lb = np.array([int(b[k]["first_error"]) for k in shared])
    ns = np.array([int(a[k].get("n_steps") or b[k].get("n_steps") or 1) for k in shared])
    both_err = (la != NO_ERROR) & (lb != NO_ERROR)
    return {
        "n_shared": len(shared),
        "exact": float((la == lb).mean()),
        # Both labellers place a boundary between two steps, and a step that goes
        # wrong at its last clause is a fair candidate for either side of it.
        "within_one": float((np.abs(la - lb) <= 1)[both_err].mean()) if both_err.any()
        else float("nan"),
        "exact_where_both_found_an_error": float((la == lb)[both_err].mean())
        if both_err.any() else float("nan"),
        "both_found_an_error": float(both_err.mean()),
        "only_a_found_one": float(((la != NO_ERROR) & (lb == NO_ERROR)).mean()),
        "only_b_found_one": float(((la == NO_ERROR) & (lb != NO_ERROR)).mean()),
        "neither": float(((la == NO_ERROR) & (lb == NO_ERROR)).mean()),
        "mean_relative_position_a": float(np.mean(la[both_err] / np.maximum(1, ns[both_err] - 1))),
        "mean_relative_position_b": float(np.mean(lb[both_err] / np.maximum(1, ns[both_err] - 1))),
        "shared_ids": shared,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, type=Path, help="first label file")
    p.add_argument("--b", required=True, type=Path, help="second label file")
    p.add_argument("--name_a", default="judge")
    p.add_argument("--name_b", default="rollout")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rep = agreement(load(args.a), load(args.b))
    if not rep["n_shared"]:
        raise SystemExit("no shared trajectories between the two label files")
    ids = rep.pop("shared_ids")
    print(f"{rep['n_shared']} trajectories labelled by both\n")
    print(f"  exact agreement                      {rep['exact']:.3f}")
    print(f"  within one step (both found an error){rep['within_one']:>8.3f}")
    print(f"  exact where both found an error      {rep['exact_where_both_found_an_error']:.3f}")
    print()
    print(f"  both found an error                  {rep['both_found_an_error']:.3f}")
    print(f"  only {args.name_a:<12} found one         {rep['only_a_found_one']:.3f}")
    print(f"  only {args.name_b:<12} found one         {rep['only_b_found_one']:.3f}")
    print(f"  neither                              {rep['neither']:.3f}")
    print()
    print(f"  {args.name_a} points {rep['mean_relative_position_a']:.2f} of the way "
          f"through, {args.name_b} {rep['mean_relative_position_b']:.2f}")
    print("\nThe two mark different things: a wrong step, and a step the model "
          "cannot recover from. This is how far apart those readings fall, not a "
          "score for either.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rep["name_a"], rep["name_b"] = args.name_a, args.name_b
        rep["a"], rep["b"] = str(args.a), str(args.b)
        rep["n_ids"] = len(ids)
        args.out.write_text(json.dumps(rep, indent=2))
        print(f"[agree] wrote {args.out}")


if __name__ == "__main__":
    main()
