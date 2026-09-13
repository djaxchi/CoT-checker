#!/usr/bin/env python3
"""Is the tie-break signal the hidden state, the confidence, or the boxed answer?

Phase 1's race (scripts/analysis/onpolicy_confidence_race.py) reported that a
token-confidence rule beat the verifier at the tie-break. This decomposes that
result and finds it is largely not about confidence. See
src/analysis/onpolicy_format.py for the mechanism.

Everything here is a re-reading of files already on disk: CPU only, no fitting,
no generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.onpolicy_confidence_race import load_confidence  # noqa: E402
from scripts.analysis.onpolicy_tiebreak_baselines import (  # noqa: E402
    fitted_questions, load_pool, read_rows,
)
from src.analysis.onpolicy_format import (  # noqa: E402
    bloc_purity, has_boxed, select_boxed_then, select_has_boxed,
)
from src.analysis.onpolicy_tiebreak import (  # noqa: E402
    answer_groups, cluster_bootstrap, top_bloc,
)
from src.eval.math_grade import normalize_answer  # noqa: E402


def build(onpolicy_root: Path, reprobe_root: Path, conf_files: list[Path],
          max_cells: int, seed: int, val_frac: float):
    pool, question = load_pool(onpolicy_root)
    fitted = fitted_questions(reprobe_root / "reprobe_train_judge_traces.jsonl",
                              sorted(reprobe_root.glob("labels.shard*.jsonl")),
                              seed, val_frac)
    clean = {pid for pid, q in question.items() if q not in fitted}
    conf, _ = load_confidence(conf_files)

    files = sorted((onpolicy_root / "grid").rglob("*scores*verifier*.jsonl"))[:max_cells]
    worst: dict[str, list[float]] = {}
    for f in files:
        for r in read_rows(f):
            worst.setdefault(r["id"], []).append(max(float(s) for s in r["scores"]))
    wmean = {k: float(np.mean(v)) for k, v in worst.items()}

    by_problem: dict[str, list[dict]] = {}
    for uid, o in pool["outcomes"].items():
        c = conf.get(uid, {})
        m = c.get("answer_token_margin", float("nan"))
        by_problem.setdefault(o["problem_id"], []).append({
            "uid": uid, "answer": normalize_answer(o.get("pred")),
            "correct": bool(o["correct"]), "margin": m, "boxed": has_boxed(m),
            "dc": c.get("bottom10_group_w32", float("nan")),
            "w": wmean.get(uid, float("nan"))})

    ties = []
    for pid, rows in by_problem.items():
        if pid not in clean:
            continue
        rows.sort(key=lambda r: r["uid"])
        for i, r in enumerate(rows):
            r["index"] = i
        bloc, answers = top_bloc(answer_groups(rows))
        if len(answers) > 1 and all(np.isfinite(rows[i]["w"]) for i in bloc):
            ties.append((pid, rows, bloc))
    return by_problem, ties, question, len(files)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onpolicy_root", type=Path,
                   default=ROOT / "cot-checker-results/onpolicy_v1")
    p.add_argument("--reprobe_root", type=Path,
                   default=ROOT / "cot-checker-results/reprobe_v1")
    p.add_argument("--confidence", type=Path, nargs="+", required=True)
    p.add_argument("--out_dir", type=Path,
                   default=ROOT / "results/onpolicy_format_confound")
    p.add_argument("--max_cells", type=int, default=24)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--split_val_frac", type=float, default=0.15)
    args = p.parse_args()

    by_problem, ties, question, n_cells = build(
        args.onpolicy_root, args.reprobe_root, args.confidence,
        args.max_cells, args.split_seed, args.split_val_frac)
    clusters = [question[p] for p, _, _ in ties]
    rnd = np.array([np.mean([rows[i]["correct"] for i in b]) for _, rows, b in ties])

    def ev(sel):
        return np.array([float(rows[sel(rows, b)]["correct"]) for _, rows, b in ties])

    def low(key):
        return lambda r, b: min(b, key=lambda i: (r[i][key], r[i]["index"]))

    def high(key):
        return lambda r, b: min(b, key=lambda i: (
            -r[i][key] if np.isfinite(r[i][key]) else np.inf, r[i]["index"]))

    rules = {
        "verifier_worst_step": low("w"),
        "deepconf_bottom10_w32": high("dc"),
        "has_boxed": select_has_boxed,
        "answer_margin": high("margin"),
        "has_boxed_then_verifier": lambda r, b: select_boxed_then(r, b, "w"),
    }
    acc = {n: ev(s) for n, s in rules.items()}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_ties": len(ties), "n_cells_averaged": n_cells,
        "random_tie_accuracy": float(rnd.mean()),
        "bloc_purity": {k: sum(1 for _, rows, b in ties if bloc_purity(rows, b) == k)
                        for k in ("all_boxed", "mixed", "none_boxed")},
        "vs_random": {n: {"tie_accuracy": float(a.mean()),
                          **cluster_bootstrap(a - rnd, clusters)}
                      for n, a in acc.items()},
        "paired": {},
    }
    for x, y in [("verifier_worst_step", "has_boxed"),
                 ("verifier_worst_step", "deepconf_bottom10_w32"),
                 ("has_boxed_then_verifier", "has_boxed"),
                 ("answer_margin", "has_boxed")]:
        report["paired"][f"{x} - {y}"] = cluster_bootstrap(acc[x] - acc[y], clusters)

    # Outcome asymmetry of the indicator itself, over the whole pool.
    allr = [r for rows in by_problem.values() for r in rows]
    bx = np.array([r["boxed"] for r in allr])
    cor = np.array([r["correct"] for r in allr])
    report["boxed_rate"] = {"overall": float(bx.mean()),
                            "given_correct": float(bx[cor].mean()),
                            "given_incorrect": float(bx[~cor].mean()),
                            "n": int(len(allr))}
    # Does the verifier read the same thing?
    w = np.array([r["w"] for r in allr if np.isfinite(r["w"])])
    wb = np.array([r["boxed"] for r in allr if np.isfinite(r["w"])])
    report["verifier_vs_boxed"] = {
        "mean_score_boxed": float(w[wb].mean()),
        "mean_score_unboxed": float(w[~wb].mean()),
        "point_biserial_r": float(np.corrcoef(w, wb.astype(float))[0, 1])}

    # Why the indicator works: it is largely a truncation detector. The pool was
    # generated with a 768-token cap and a trace that runs out never reaches its
    # box, so `has_boxed` is reading the generation budget as much as the
    # reasoning. Reported here because it decides what Phase 2 must change.
    conf_rows = [r for f in args.confidence for r in read_rows(f)]
    nt = np.array([r["n_gen_tokens"] for r in conf_rows], float)
    bx2 = np.array([has_boxed(r["rules"]["answer_token_margin"]) for r in conf_rows])
    cor2 = np.array([r["correct"] for r in conf_rows], bool)
    cap = float(np.percentile(nt, 99))
    trunc = nt >= cap - 2
    report["truncation"] = {
        "inferred_cap_tokens": cap,
        "share_at_cap": float(trunc.mean()),
        "p_unboxed_given_truncated": float((~bx2)[trunc].mean()),
        "p_unboxed_given_not_truncated": float((~bx2)[~trunc].mean()),
        "p_truncated_given_unboxed": float(trunc[~bx2].mean()),
        "accuracy_truncated": float(cor2[trunc].mean()),
        "accuracy_not_truncated": float(cor2[~trunc].mean()),
        "mean_tokens_boxed": float(nt[bx2].mean()),
        "mean_tokens_unboxed": float(nt[~bx2].mean()),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "format_confound.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "paired"}, indent=2))
    print("\npaired:")
    for k, v in report["paired"].items():
        print(f"  {k}: {v['delta']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] "
              f"crosses_zero={v['crosses_zero']}")
    print(f"\n[out] {args.out_dir/'format_confound.json'}")


if __name__ == "__main__":
    main()
