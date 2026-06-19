"""Evaluate a deployed dense probe on the PRM800K held-out TEST set.

Tests a trained DenseLinear probe (and its val-selected threshold) on the fresh,
problem-disjoint held-out PRM800K steps that it never saw. This is the clean
generalization number for the probe as a step-correctness classifier, distinct
from the ProcessBench F1_PB localization score.

Works for any model size: point --run_dir at that size's run (linear_probe.pt +
threshold.json) and --enc_dir/--stem at the held-out encoding produced with the
SAME backbone (the probe lives in its backbone's hidden space).

Reports AUC, balanced accuracy / F1 / accuracy at the deployed threshold, the
oracle-threshold ceiling on the test, and the val->test balanced-accuracy gap.

Outputs (results/prm800k_heldout_eval/):
  - <tag>.json         full metrics
  - heldout_eval.csv   one appended row per call (with --csv), for a cross-size table

Usage:
  python scripts/eval_prm800k_heldout_probe.py \
    --run_dir runs/s1_model_size_dense/qwen2_5_7b \
    --enc_dir runs/s1_model_size_dense/qwen2_5_7b/merged \
    --stem prm800k_heldout_test --tag 7B --csv
  # 7B from the 4D multitoken encoding (deployed readout = L28 / last):
  python scripts/eval_prm800k_heldout_probe.py --run_dir runs/.../qwen2_5_7b \
    --enc_dir runs/.../qwen2_5_7b/prm_multitoken --stem prm800k_heldout_test \
    --layer 28 --token last --tag 7B
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# make the project root importable when run as `python scripts/eval_...py`
# (Python puts scripts/ on sys.path, not the repo root, so `import src` fails)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.prm800k_val_data import load_prm800k_multitoken, load_prm800k_val
from src.data.processbench_probe_data import compute_scores, load_probe
from src.eval.probe_metrics import classification_metrics, oracle_threshold

ROOT = Path("results/prm800k_heldout_eval")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True,
                    help="size run dir with linear_probe.pt + threshold.json")
    ap.add_argument("--enc_dir", type=Path, required=True,
                    help="dir with <stem>_{h,y,meta} encoded with the SAME backbone")
    ap.add_argument("--stem", type=str, default="prm800k_heldout_test")
    ap.add_argument("--layer", type=int, default=None, help="4D multitoken: layer idx")
    ap.add_argument("--token", type=str, default="last", help="4D multitoken: first/last")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override; default = run_dir/threshold.json selected_threshold")
    ap.add_argument("--tag", type=str, default=None, help="row label, e.g. 7B")
    ap.add_argument("--out_dir", type=Path, default=ROOT)
    ap.add_argument("--csv", action="store_true", help="append a row to heldout_eval.csv")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.run_dir.name

    # ---- probe + deployed threshold -------------------------------------
    w, b = load_probe(args.run_dir)
    thr_path = args.run_dir / "threshold.json"
    val_balacc = None
    if args.threshold is not None:
        thr = args.threshold
    elif thr_path.exists():
        tj = json.loads(thr_path.read_text())
        thr = float(tj.get("selected_threshold", 0.5))
        val_balacc = tj.get("best_val_balanced_accuracy") or tj.get("val_balanced_accuracy")
    else:
        thr = 0.5

    # ---- held-out encoding (2D or a 4D multitoken slice) ----------------
    if args.layer is not None:
        d = load_prm800k_multitoken(args.enc_dir, args.stem, args.layer, args.token)
        src = f"{args.stem}[L{args.layer},{args.token}]"
    else:
        d = load_prm800k_val(args.enc_dir, stem=args.stem)
        src = args.stem
    if w.shape[0] != d.hidden.shape[1]:
        raise SystemExit(
            f"dim mismatch: probe {w.shape[0]} vs encoding {d.hidden.shape[1]} "
            f"({tag}). Encode the held-out set with this size's backbone.")

    score = compute_scores(d.hidden, w, b)
    y = d.label
    m = classification_metrics(score, y, thr)
    o_thr, o_bal = oracle_threshold(score, y, metric="balanced_accuracy")
    gap = (m["balanced_accuracy"] - float(val_balacc)) if val_balacc is not None else None

    out = {
        "tag": tag, "run_dir": str(args.run_dir), "encoding": src,
        "n": m["n"], "deployed_threshold": thr,
        "auc": round(m["auc"], 4),
        "test_balanced_accuracy": round(m["balanced_accuracy"], 4),
        "test_f1": round(m["f1"], 4),
        "test_accuracy": round(m["accuracy"], 4),
        "pred_pos_rate": round(m["pred_pos_rate"], 4),
        "mean_score_incorrect": round(m["mean_score_incorrect"], 4),
        "mean_score_correct": round(m["mean_score_correct"], 4),
        "oracle_threshold": round(o_thr, 3),
        "oracle_balanced_accuracy": round(o_bal, 4),
        "val_balanced_accuracy": None if val_balacc is None else round(float(val_balacc), 4),
        "val_to_test_gap": None if gap is None else round(gap, 4),
    }
    (args.out_dir / f"{tag}.json").write_text(json.dumps(out, indent=2))

    print(f"\n[eval] {tag} on {src}  (n={m['n']}, deployed thr={thr:g})")
    print(f"  AUC                 = {out['auc']:.3f}")
    print(f"  balanced acc @thr   = {out['test_balanced_accuracy']:.3f}"
          + (f"   (val {out['val_balanced_accuracy']:.3f}, "
             f"gap {out['val_to_test_gap']:+.3f})" if val_balacc is not None else ""))
    print(f"  F1 / accuracy       = {out['test_f1']:.3f} / {out['test_accuracy']:.3f}")
    print(f"  oracle bal-acc      = {out['oracle_balanced_accuracy']:.3f} @ thr={out['oracle_threshold']}")
    print(f"  mean score inc/cor  = {out['mean_score_incorrect']:.3f} / {out['mean_score_correct']:.3f}")

    if args.csv:
        csv_path = args.out_dir / "heldout_eval.csv"
        cols = ["tag", "n", "deployed_threshold", "auc", "test_balanced_accuracy",
                "test_f1", "test_accuracy", "oracle_balanced_accuracy",
                "val_balanced_accuracy", "val_to_test_gap", "encoding"]
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            wtr = csv.writer(f)
            if new:
                wtr.writerow(cols)
            wtr.writerow([out[c] for c in cols])
        print(f"  appended -> {csv_path}")


if __name__ == "__main__":
    main()
