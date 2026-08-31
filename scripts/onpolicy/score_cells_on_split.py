#!/usr/bin/env python3
"""Score already-trained grid cells on a new evaluation split.

The T1 question is whether the leaderboard's *ordering* survives when the text
was written by the model whose states we read. Answering it needs no retraining:
the nineteen dense cells are already trained, their weights are on disk next to
their results.json, and `evaluate_processbench` takes any store split whose meta
carries id / step_idx / label / n_steps. So this rebuilds each cell exactly as it
was left and points it at the on-policy split, which keeps the verifiers
identical between the two arms and leaves the text distribution as the only thing
that changed.

**The one piece a cell does not save is its rescaling statistics.** They are fit
on the training split at train time and used only inside the collate function, so
scoring elsewhere has to reproduce them. Refitting is exact rather than
approximate: `rescale.fit` draws a fixed 200,000-row sample under seed 0 from a
deterministic ordering, and the sequence path takes a fixed stride over the first
20,000 spans, so the same store gives the same numbers. What makes that safe is
the fingerprint: the cell recorded a digest of the training split it read, and
this script refuses to run unless the store it is about to refit from digests to
the same string. Refitting from a rebuilt or different store would otherwise pass
silently and rescale the on-policy scores by statistics the cell never saw.

Statistics are cached per (representation, fingerprint), because all three
learners and all three seeds of a representation share them.

Writes, per cell, `pb_step_scores_<split>.jsonl` in the exact layout
scripts/analysis/pb_threshold_calibration.py already consumes, so calib-20 on the
on-policy arm is the same offline computation it is on the off-policy one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.harness import rescale as rs  # noqa: E402
from src.harness.learners import build_learner, is_sequence  # noqa: E402
from src.harness.spanloader import SpanLoader  # noqa: E402
from src.repstore import split_fingerprint  # noqa: E402
from src.repstore.store import ShardedRepSplit  # noqa: E402
from train_easy_probe_method import (  # noqa: E402
    auroc_numpy, evaluate_processbench, resolve_threshold_grid,
)
from train_rep_learner_cell import (  # noqa: E402
    REP_READOUT, REP_SPARSE, REP_SPARSE_SEQ, build_handles, load_vectors, score_all,
)
from derive_delta_from_token_store import derive_split  # noqa: E402


def eval_plan(n: int, batch: int) -> list[np.ndarray]:
    return [np.arange(i, min(i + batch, n), dtype=np.int64) for i in range(0, n, batch)]


def stats_cache_path(cache_dir: Path | None, rep: str, mode: str, fp: str) -> Path | None:
    return None if cache_dir is None else cache_dir / f"{rep}__{mode}__{fp}.npz"


def save_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in stats.items()})


def load_stats(path: Path) -> dict:
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    out["center"] = bool(np.asarray(out["center"]).reshape(-1)[0])
    out["rows"] = int(np.asarray(out["rows"]).reshape(-1)[0])
    if "kind" in out:
        out["kind"] = str(np.asarray(out["kind"]).reshape(-1)[0])
    return out


def fit_vector_stats(prm_store: Path, train_stem: str, rep: str, mode: str,
                     vec_cache: Path | None, fp: str) -> dict:
    """Reproduce train_rep_learner_cell's vector-path fit, line for line."""
    Xtr, _, _ = load_vectors(prm_store, train_stem, rep, vec_cache, sort=False,
                             fingerprint=fp)
    return rs.fit(Xtr) if mode == "zscore" else rs.fit_whiten(Xtr)


def fit_sequence_stats(prm_store: Path, train_stem: str, mode: str) -> dict:
    """Reproduce the sequence-path fit: every 4th span of the first 20,000."""
    if mode != "zscore":
        raise SystemExit("the sequence path only ever fit zscore statistics")
    handles, _ = build_handles(ShardedRepSplit(prm_store / train_stem))
    probe = SpanLoader(handles, 512, torch.device("cpu"), preload=False)
    sample = np.concatenate([
        probe._rows(np.arange(int(probe.starts[k]),
                              int(probe.starts[k] + probe.lengths[k])))
        for k in range(0, min(len(handles), 20000), 4)])
    return rs.fit(sample)


def train_stem_of(res: dict, override: str | None) -> str:
    """Which split the cell trained on. results.json records the fingerprints of
    every split it read, keyed `prm/<stem>`, but not which one was the training
    one, so pick the single `train` stem among them and make the ambiguity
    explicit rather than guessing."""
    if override:
        return override
    stems = [k.split("/", 1)[1] for k in res["inputs"] if k.startswith("prm/")]
    train = [s for s in stems if "train" in s]
    if len(train) != 1:
        raise SystemExit(f"cannot tell which of {stems} is the training split; "
                         f"pass --train_stem")
    return train[0]


def cell_stats(res: dict, prm_store: Path, vec_cache: Path | None,
               cache_dir: Path | None, cache: dict,
               train_stem_override: str | None = None) -> dict | None:
    """Rescaling statistics for one cell, refit from the store it trained on."""
    mode = res["protocol"]["rescale"]
    if mode == "none":
        return None
    rep = res["rep"]
    train_stem = train_stem_of(res, train_stem_override)
    recorded = res["inputs"][f"prm/{train_stem}"]
    seq = is_sequence(res["learner"])
    key = (rep, mode, recorded, seq)
    if key in cache:
        return cache[key]
    disk = stats_cache_path(cache_dir, rep, mode, recorded)
    if disk is not None and disk.exists():
        cache[key] = load_stats(disk)
        return cache[key]

    seen = split_fingerprint(prm_store / train_stem)
    if seen != recorded:
        raise SystemExit(
            f"[score] {prm_store / train_stem} fingerprints {seen} but the cell "
            f"trained on {recorded}. Refitting rescaling statistics from a "
            f"different store would rescale these scores by numbers the cell "
            f"never saw. Point --prm_store at the store the cell read.")
    t0 = time.perf_counter()
    stats = (fit_sequence_stats(prm_store, train_stem, mode) if seq else
             fit_vector_stats(prm_store, train_stem, rep, mode, vec_cache, recorded))
    print(f"[stats] {rep} ({mode}, {'seq' if seq else 'vec'}) refit from "
          f"{stats['rows']:,} rows in {time.perf_counter()-t0:.0f}s", flush=True)
    if disk is not None:
        save_stats(disk, stats)
    cache[key] = stats
    return stats


def score_cell(cell_dir: Path, res: dict, split_dir: Path, stats: dict | None,
               device, batch: int, t_max: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(scores, per-step y, meta) for one cell on one split, in global order."""
    rep, learner = res["rep"], res["learner"]
    model = build_learner(learner, int(res["dim"]), t_max=t_max,
                          dropout=res["protocol"]["dropout"]).to(device)
    state = torch.load(cell_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    if is_sequence(learner):
        view = ShardedRepSplit(split_dir)
        handles, meta = build_handles(view)
        loader = SpanLoader(handles, t_max, device, preload=True, stats=stats)
        y = np.array([h[4] for h in handles], dtype=np.int8)
        scores = score_all(model, len(handles), loader.collate,
                           eval_plan(len(handles), batch))
    else:
        X, y, meta = derive_split(split_dir, REP_READOUT[rep], sort=True)
        tfm = None if stats is None else rs.to_torch(stats, device)

        def collate(idx):
            xb = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(device)
            if tfm is not None:
                xb = rs.apply_torch(xb, tfm)
            return xb, None, torch.zeros(len(idx), device=device)

        scores = score_all(model, X.shape[0], collate, eval_plan(X.shape[0], batch))
    return scores, np.asarray(y), meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", required=True, type=Path,
                   help="Cell directories (each holding results.json and model.pt), "
                        "or parents to search with --glob.")
    p.add_argument("--glob", default=None,
                   help="If given, expand each --cells entry with this pattern, "
                        "e.g. '*__seed4*'.")
    p.add_argument("--split_dir", required=True, type=Path,
                   help="Store split to score, ProcessBench meta layout.")
    p.add_argument("--split_name", required=True,
                   help="Name for the output file: pb_step_scores_<name>.jsonl.")
    p.add_argument("--prm_store", type=Path, default=None,
                   help="Training store, for refitting rescaling statistics. "
                        "Defaults to the path each cell recorded.")
    p.add_argument("--vec_cache_dir", type=Path, default=None)
    p.add_argument("--train_stem", default=None,
                   help="Override the training split name; by default it is the "
                        "single 'train' stem among the cell's recorded inputs.")
    p.add_argument("--stats_cache_dir", type=Path, default=None,
                   help="Reuse refit statistics across the learners and seeds of a "
                        "representation.")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Write beside each cell (default) or under this root.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--t_max", type=int, default=512)
    p.add_argument("--threshold_grid", default="0.01")
    p.add_argument("--summary", type=Path, default=None)
    args = p.parse_args()

    cells: list[Path] = []
    for c in args.cells:
        cells.extend(sorted(c.glob(args.glob)) if args.glob else [c])
    cells = [c for c in cells if (c / "results.json").exists()]
    if not cells:
        raise SystemExit("[score] no cells with a results.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = resolve_threshold_grid(args.threshold_grid)
    split_fp = split_fingerprint(args.split_dir)
    print(f"[score] {len(cells)} cells on {args.split_dir} ({split_fp})", flush=True)

    stats_cache: dict = {}
    rows = []
    for cell in cells:
        res = json.loads((cell / "results.json").read_text())
        rep = res["rep"]
        if rep in REP_SPARSE or rep == REP_SPARSE_SEQ:
            print(f"[score] skip {cell.name}: {rep} needs its SAE codes derived for "
                  f"this split; the on-policy arm is dense only", flush=True)
            continue
        if not (cell / "model.pt").exists():
            print(f"[score] skip {cell.name}: no model.pt", flush=True)
            continue
        prm_store = args.prm_store or Path(res["prm_store"])
        stats = cell_stats(res, prm_store, args.vec_cache_dir, args.stats_cache_dir,
                           stats_cache, args.train_stem)
        t0 = time.perf_counter()
        scores, y, meta = score_cell(cell, res, args.split_dir, stats, device,
                                     args.batch_size, args.t_max)
        t_val = float(res["in_domain"]["val_threshold"])
        pb_rows, m_val = evaluate_processbench(scores, meta, t_val)
        best_f1, best_t = -1.0, grid[0]
        for t in grid:
            _, mt = evaluate_processbench(scores, meta, t)
            if mt["F1_PB"] > best_f1:
                best_f1, best_t = mt["F1_PB"], t
        out_dir = cell if args.out_dir is None else args.out_dir / cell.name
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"pb_step_scores_{args.split_name}.jsonl").open("w") as f:
            for r in pb_rows:
                f.write(json.dumps(r) + "\n")
        row = {
            "cell": cell.name, "rep": rep, "learner": res["learner"],
            "seed": res["seed"], "split": args.split_name,
            "split_fingerprint": split_fp, "n_steps": int(len(scores)),
            "rescale": res["protocol"]["rescale"],
            "val_threshold": t_val,
            "step_auroc": float(auroc_numpy(y, scores)),
            "F1_PB_at_val_threshold": float(m_val["F1_PB"]),
            "Acc_error": float(m_val["Acc_error"]),
            "Acc_correct": float(m_val["Acc_correct"]),
            "oracle_F1_PB": float(best_f1), "oracle_threshold": float(best_t),
            "n_traces": int(m_val["n_traces"]),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        rows.append(row)
        print(f"[score] {cell.name:<46} step AUROC {row['step_auroc']:.4f}  "
              f"F1_PB@val {row['F1_PB_at_val_threshold']:.4f}  "
              f"oracle {row['oracle_F1_PB']:.4f}", flush=True)

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(rows, indent=2))
        print(f"[score] wrote {args.summary}")
    print(f"[score] scored {len(rows)} cells. F1 is NOT comparable to the off-policy "
          f"arm (different prevalence); the rank is.")


if __name__ == "__main__":
    main()
