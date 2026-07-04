"""parametric_retrieval_geometry_v0 stage 4: build the explorer payload.

Writes explorer_payload/ next to the run artifacts:

  manifest.json      experiment card: model, hs/block indices, positions per
                     mode, default view, colors/shapes, file map
  instances.json     per-instance static fields (question, answers, labels,
                     metadata) stored ONCE, column-array format
  points_hs{K:02d}_{mode}_{position}.json
                     per scatter view: instance index + PCA x/y (tiny files;
                     the explorer joins against instances.json)
  panels.json        class counts, between/within heatmap, CoT trajectory
                     summary, gbc-bin class distribution

  python scripts/parametric_retrieval/prg_build_explorer_payload.py \
      --out_dir runs/parametric_retrieval_geometry_v0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.parametric_retrieval import block_idx  # noqa: E402

# Categorical palette validated (dataviz six-checks) against the dark surface.
CLASS_COLORS = {
    "direct_retrieval": "#3987e5",
    "reasoning_unlocked": "#cc7a00",
    "unstable_retrieval": "#9467bd",
    "non_retrieved": "#e34948",
    "ctrl_retrieved": "#3987e5",
    "ctrl_unstable": "#9467bd",
    "ctrl_non_retrieved": "#e34948",
}
FAMILY_SHAPES = {
    "direct": "circle",
    "direct_natural": "square",
    "reverse": "triangle_up",
    "reverse_natural": "triangle_down",
    "completion": "diamond",
}
INSTANCE_FIELDS = ["question_id", "fact_id", "family", "is_control",
                   "question", "gold_answer", "direct_greedy_answer",
                   "cot_greedy_final_answer", "retrieval_class",
                   "direct_greedy_correct", "direct_pass_at_4",
                   "cot_greedy_correct", "cot_pass_at_4", "cot_marker_found",
                   "reasoning_unlocked_soft", "direct_greedy_status",
                   "cot_greedy_status", "category", "gbc", "gbc_bin",
                   "subject_type", "object_type", "page_title"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            text=True).strip()
    except Exception:
        return "unknown"


def jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("runs/parametric_retrieval_geometry_v0"))
    args = ap.parse_args()

    geo = args.out_dir / "geometry"
    pay = args.out_dir / "explorer_payload"
    pay.mkdir(exist_ok=True)

    meta = pd.read_parquet(args.out_dir / "metadata.parquet")
    grading = pd.DataFrame([json.loads(ln) for ln in
                            (args.out_dir / "grading.jsonl")
                            .read_text().splitlines() if ln.strip()])
    pca = pd.read_parquet(geo / "pca_coordinates.parquet")

    inst = grading.merge(
        meta[["question_id", "question", "gbc", "page_title",
              "subject_type", "object_type"]],
        on="question_id", how="left", validate="one_to_one")
    inst = inst.sort_values("question_id").reset_index(drop=True)
    inst_index = {q: i for i, q in enumerate(inst.question_id)}
    instances = {c: [jsonable(v) for v in inst[c]] for c in INSTANCE_FIELDS}
    (pay / "instances.json").write_text(json.dumps(
        {"n": len(inst), "fields": instances}, ensure_ascii=False))

    files = {}
    for (k, mode, pos), g in pca.groupby(["hs_idx", "prompt_mode", "position"],
                                         observed=True):
        name = f"points_hs{k:02d}_{mode}_{pos}.json"
        payload = {
            "hs_idx": int(k), "block_idx": block_idx(int(k)),
            "prompt_mode": mode, "position": pos,
            "evr": [float(g.evr1.iloc[0]), float(g.evr2.iloc[0])],
            "i": [inst_index[q] for q in g.question_id],
            "x": [round(float(v), 4) for v in g.x],
            "y": [round(float(v), 4) for v in g.y],
        }
        (pay / name).write_text(json.dumps(payload))
        files.setdefault(f"hs{k:02d}", {}).setdefault(mode, {})[pos] = name

    def read_panel(name: str) -> list[dict]:
        p = geo / name
        if not p.exists():
            return []
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return []
        return [{k: jsonable(v) for k, v in r.items()}
                for r in df.to_dict(orient="records")]

    panels = {
        "class_counts": read_panel("class_counts.csv"),
        "between_within": read_panel("between_within_ratio.csv"),
        "trajectory_summary": read_panel("trajectory_summary.csv"),
        "centroid_distances": read_panel("centroid_distances.csv"),
    }
    (pay / "panels.json").write_text(json.dumps(panels))

    hs_indices = sorted(pca.hs_idx.unique().tolist())
    manifest = {
        "id": "parametric_retrieval_geometry_v0",
        "title": "Parametric Retrieval Geometry v0 — Qwen2.5-7B",
        "description": ("WikiProfile closed-book retrieval-regime geometry "
                        "using Qwen2.5-7B-Instruct hidden states."),
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "google/WikiProfile",
        "hs_indices": [int(k) for k in hs_indices],
        "block_indices": [block_idx(int(k)) for k in hs_indices],
        "positions": {m: sorted(pca[pca.prompt_mode == m].position.unique())
                      for m in ["direct", "cot"]},
        "default_view": {"hs_idx": 20, "block_idx": 19,
                         "position": "final_prompt_token",
                         "prompt_mode": "direct",
                         "color_by": "retrieval_class",
                         "projection": "PCA-2D"},
        "class_colors": CLASS_COLORS,
        "family_shapes": FAMILY_SHAPES,
        "instances_file": "instances.json",
        "panels_file": "panels.json",
        "points_files": files,
        "n_instances": len(inst),
        "n_qa_instances": int((~inst.is_control).sum()),
    }
    (pay / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_pts = sum(len(v2) for v in files.values() for v2 in v.values())
    print(f"[payload] wrote manifest + instances ({len(inst)}) + "
          f"{n_pts} point files + panels to {pay}", flush=True)


if __name__ == "__main__":
    main()
