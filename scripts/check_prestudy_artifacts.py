"""
Sanity-check all prestudy_v1 artifacts.

Returns non-zero exit code on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"[FAIL] {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_file_exists(path: Path, label: str) -> bool:
    if path.exists():
        ok(f"{label} exists")
        return True
    fail(f"{label} missing: {path}")
    return False


# ---------------------------------------------------------------------------
# JSONL checks
# ---------------------------------------------------------------------------

def check_jsonl_count(path: Path, label: str, expected: int) -> list[dict] | None:
    if not path.exists():
        fail(f"{label}: file missing")
        return None
    rows = read_jsonl(path)
    if len(rows) == expected:
        ok(f"{label}: row count = {expected}")
    else:
        fail(f"{label}: row count = {len(rows)}, expected {expected}")
    return rows


def check_label_distribution(rows: list[dict], label: str, expected: dict[int, int]) -> None:
    """Verify exact label counts for ALL entries in expected, including zeros."""
    counts = Counter(r["label"] for r in rows)
    for lbl, exp_count in expected.items():
        actual = counts.get(lbl, 0)
        if actual == exp_count:
            ok(f"{label}: label={lbl} count = {exp_count}")
        else:
            fail(f"{label}: label={lbl} count = {actual}, expected {exp_count}")
    # Also check that no unexpected labels exist
    unexpected = {lbl: c for lbl, c in counts.items() if lbl not in expected}
    if unexpected:
        fail(f"{label}: unexpected labels found: {unexpected}")


def check_uid_uniqueness(rows: list[dict], label: str) -> None:
    uids = [r.get("uid") for r in rows]
    if len(uids) == len(set(uids)):
        ok(f"{label}: all UIDs unique ({len(uids)})")
    else:
        from collections import Counter
        dups = {u: c for u, c in Counter(uids).items() if c > 1}
        fail(f"{label}: {len(dups)} duplicate UIDs found: {list(dups.items())[:5]}")


def check_no_rating_zero(rows: list[dict], label: str) -> None:
    bad = [r for r in rows if r.get("rating") == 0]
    if bad:
        fail(f"{label}: {len(bad)} rows have rating=0")
    else:
        ok(f"{label}: no rating=0")


def check_no_empty_candidate(rows: list[dict], label: str) -> None:
    bad = [r for r in rows if not r.get("candidate_step", "").strip()]
    if bad:
        fail(f"{label}: {len(bad)} rows have empty candidate_step")
    else:
        ok(f"{label}: no empty candidate_step")


# ---------------------------------------------------------------------------
# Cache checks
# ---------------------------------------------------------------------------

def check_npy_shape(npy_path: Path, label: str, expected_rows: int) -> np.ndarray | None:
    if not npy_path.exists():
        fail(f"{label} npy missing: {npy_path}")
        return None
    arr = np.load(npy_path)
    if arr.shape[0] == expected_rows:
        ok(f"{label} npy: shape {arr.shape}")
    else:
        fail(f"{label} npy: row count = {arr.shape[0]}, expected {expected_rows}")
    return arr


def check_no_nan_inf(arr: np.ndarray, label: str) -> None:
    # Cast to float32 for reliable NaN/Inf detection on float16 arrays
    arr_f32 = arr.astype(np.float32)
    if np.any(np.isnan(arr_f32)):
        fail(f"{label}: contains NaN")
    elif np.any(np.isinf(arr_f32)):
        fail(f"{label}: contains Inf")
    else:
        ok(f"{label}: no NaN/Inf")


def check_meta(
    meta_path: Path,
    label: str,
    expected_rows: int,
    max_seq_len: int,
) -> list[dict] | None:
    if not meta_path.exists():
        fail(f"{label} meta missing: {meta_path}")
        return None
    rows = read_jsonl(meta_path)
    if len(rows) != expected_rows:
        fail(f"{label} meta: row count = {len(rows)}, expected {expected_rows}")
        return rows
    ok(f"{label} meta: row count = {expected_rows}")

    truncated = [r for r in rows if r.get("was_truncated") is True]
    if truncated:
        fail(f"{label} meta: {len(truncated)} rows have was_truncated=true")
    else:
        ok(f"{label} meta: all was_truncated=false")

    overlength = [
        r for r in rows
        if isinstance(r.get("n_tokens"), int)
        and r["n_tokens"] > max_seq_len
    ]
    if overlength:
        fail(f"{label} meta: {len(overlength)} rows have n_tokens > {max_seq_len}")
    else:
        ok(f"{label} meta: all n_tokens <= {max_seq_len}")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Check prestudy_v1 artifacts.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    args = parser.parse_args()

    data_dir = args.data_dir
    cache_dir = args.cache_dir
    max_seq_len = args.max_seq_len

    print("=" * 60)
    print("Checking prestudy_v1 artifacts")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Required JSONL files — existence and exact row counts
    # -----------------------------------------------------------------------
    jsonl_specs: dict[str, int] = {
        "prm800k_pos_base_20k.jsonl": 20000,
        "prm800k_neg_base_20k.jsonl": 20000,
        "prm800k_probe_train_40k.jsonl": 40000,
        "prm800k_mixed_train_40k.jsonl": 40000,
        "prm800k_val_1k.jsonl": 1000,
        "prm800k_contrastive_forks_20.jsonl": 20,
        "prm800k_contrastive_forks_20_flat.jsonl": 40,
    }

    jsonl_rows: dict[str, list[dict]] = {}
    for fname, expected in jsonl_specs.items():
        rows = check_jsonl_count(data_dir / fname, fname, expected)
        if rows is not None:
            jsonl_rows[fname] = rows

    # -----------------------------------------------------------------------
    # 2. UID uniqueness within each file
    # -----------------------------------------------------------------------
    for fname in [
        "prm800k_pos_base_20k.jsonl",
        "prm800k_neg_base_20k.jsonl",
        "prm800k_probe_train_40k.jsonl",
        "prm800k_mixed_train_40k.jsonl",
        "prm800k_val_1k.jsonl",
        "prm800k_contrastive_forks_20_flat.jsonl",
    ]:
        if fname in jsonl_rows:
            check_uid_uniqueness(jsonl_rows[fname], fname)

    # -----------------------------------------------------------------------
    # 3. Exact label distributions (including zero-expected classes)
    # -----------------------------------------------------------------------
    label_specs: dict[str, dict[int, int]] = {
        "prm800k_pos_base_20k.jsonl":              {0: 20000, 1: 0},
        "prm800k_neg_base_20k.jsonl":              {0: 0,     1: 20000},
        "prm800k_probe_train_40k.jsonl":           {0: 20000, 1: 20000},
        "prm800k_mixed_train_40k.jsonl":           {0: 20000, 1: 20000},
        "prm800k_val_1k.jsonl":                    {0: 500,   1: 500},
        "prm800k_contrastive_forks_20_flat.jsonl": {0: 20,    1: 20},
    }
    for fname, dist in label_specs.items():
        if fname in jsonl_rows:
            check_label_distribution(jsonl_rows[fname], fname, dist)

    # -----------------------------------------------------------------------
    # 4. No rating=0, no empty candidate_step
    # -----------------------------------------------------------------------
    for fname in [
        "prm800k_pos_base_20k.jsonl",
        "prm800k_neg_base_20k.jsonl",
        "prm800k_probe_train_40k.jsonl",
        "prm800k_mixed_train_40k.jsonl",
        "prm800k_val_1k.jsonl",
        "prm800k_contrastive_forks_20_flat.jsonl",
    ]:
        if fname in jsonl_rows:
            rows = jsonl_rows[fname]
            check_no_rating_zero(rows, fname)
            check_no_empty_candidate(rows, fname)

    # -----------------------------------------------------------------------
    # 5. Train/val UID and problem_id disjointness
    # -----------------------------------------------------------------------
    train_rows = jsonl_rows.get("prm800k_probe_train_40k.jsonl", [])
    val_rows = jsonl_rows.get("prm800k_val_1k.jsonl", [])

    train_uids = {r["uid"] for r in train_rows}
    val_uids = {r["uid"] for r in val_rows}
    uid_overlap = train_uids & val_uids
    if uid_overlap:
        fail(f"train/val UID overlap: {len(uid_overlap)} shared UIDs")
    else:
        ok("train/val: no UID overlap")

    manifest_path = data_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        pid_overlap_count = manifest.get("overlap_checks", {}).get("train_val_problem_id_overlap")
        split_policy = manifest.get("split_policy", "")
        allow_pid_overlap = "overlap allowed" in split_policy
        if pid_overlap_count is not None:
            if pid_overlap_count == 0:
                ok("train/val: problem_id overlap = 0 (per manifest)")
            elif allow_pid_overlap:
                ok(f"train/val: problem_id overlap = {pid_overlap_count} (allowed by manifest policy)")
            else:
                fail(f"train/val: problem_id overlap = {pid_overlap_count} but policy requires disjoint")
    else:
        train_pids = {r["problem_id"] for r in train_rows}
        val_pids = {r["problem_id"] for r in val_rows}
        pid_overlap = train_pids & val_pids
        if pid_overlap:
            fail(f"train/val problem_id overlap: {len(pid_overlap)} (no manifest to check policy)")
        else:
            ok("train/val: no problem_id overlap")

    # -----------------------------------------------------------------------
    # 6. Contrastive fork structure
    # -----------------------------------------------------------------------
    fork_rows = jsonl_rows.get("prm800k_contrastive_forks_20.jsonl", [])
    fork_failures = 0
    for fork in fork_rows:
        pos = fork.get("positive_completion", {})
        neg = fork.get("negative_completion", {})
        if pos.get("rating") != 1 or pos.get("label") != 0:
            fail(f"Fork {fork.get('fork_id')}: positive_completion has wrong rating/label")
            fork_failures += 1
        if neg.get("rating") != -1 or neg.get("label") != 1:
            fail(f"Fork {fork.get('fork_id')}: negative_completion has wrong rating/label")
            fork_failures += 1
        if "prefix" not in fork:
            fail(f"Fork {fork.get('fork_id')}: missing prefix field")
            fork_failures += 1

    if fork_rows and fork_failures == 0:
        ok(f"contrastive_forks_20: all {len(fork_rows)} forks have valid pos/neg structure")

    # Check flat fork file: each fork_id should appear exactly twice (one pos, one neg)
    flat_rows = jsonl_rows.get("prm800k_contrastive_forks_20_flat.jsonl", [])
    if flat_rows:
        from collections import defaultdict
        by_fork: dict[str, list[dict]] = defaultdict(list)
        for r in flat_rows:
            by_fork[r.get("fork_id", "")].append(r)
        flat_failures = 0
        for fid, members in by_fork.items():
            roles = [m.get("pair_role") for m in members]
            if sorted(roles) != ["negative", "positive"]:
                fail(f"flat fork {fid}: expected one positive + one negative, got {roles}")
                flat_failures += 1
            # Verify shared prefix
            prefixes = {m.get("prefix") for m in members}
            if len(prefixes) > 1:
                fail(f"flat fork {fid}: positive and negative have different prefixes")
                flat_failures += 1
        if flat_failures == 0:
            ok(f"contrastive_forks_20_flat: all {len(by_fork)} pairs have valid structure and matching prefixes")

    # -----------------------------------------------------------------------
    # 7. Cache files — existence, shape, NaN/Inf, metadata alignment
    # -----------------------------------------------------------------------
    cache_specs = [
        ("pos_base_20k", 20000),
        ("neg_base_20k", 20000),
        ("probe_train_40k", 40000),
        ("mixed_train_40k", 40000),
        ("val_1k", 1000),
        ("contrastive_forks_20_flat", 40),
    ]

    for stem, n in cache_specs:
        h_path = cache_dir / f"{stem}_h.npy"
        y_path = cache_dir / f"{stem}_y.npy"
        m_path = cache_dir / f"{stem}_meta.jsonl"

        check_file_exists(h_path, f"cache/{stem}_h.npy")
        check_file_exists(y_path, f"cache/{stem}_y.npy")
        check_file_exists(m_path, f"cache/{stem}_meta.jsonl")

        h_arr = check_npy_shape(h_path, f"cache/{stem}_h", n)
        if h_arr is not None:
            check_no_nan_inf(h_arr, f"cache/{stem}_h")

        check_npy_shape(y_path, f"cache/{stem}_y", n)
        check_meta(m_path, f"cache/{stem}_meta", n, max_seq_len)

    # -----------------------------------------------------------------------
    # 8. Manifest files — existence and required keys
    # -----------------------------------------------------------------------
    required_manifest_keys = [
        "run_name", "seed", "source_dataset", "label_mapping",
        "discarded_ratings", "length_policy", "max_seq_len",
        "split_policy", "counts", "overlap_checks", "raw_counts",
        "files", "created_at",
    ]
    if check_file_exists(manifest_path, "manifest.json"):
        missing = [k for k in required_manifest_keys if k not in manifest]
        if missing:
            fail(f"manifest.json missing keys: {missing}")
        else:
            ok("manifest.json: all required keys present")

    enc_manifest_path = cache_dir / "encoding_manifest.json"
    required_enc_keys = [
        "run_name", "model", "offline", "local_files_only", "layer",
        "token_position", "length_policy", "max_seq_len", "hidden_dim",
        "model_dtype", "saved_dtype", "files", "timing", "hardware",
        "created_at",
    ]
    if check_file_exists(enc_manifest_path, "encoding_manifest.json"):
        with open(enc_manifest_path) as f:
            em = json.load(f)
        missing = [k for k in required_enc_keys if k not in em]
        if missing:
            fail(f"encoding_manifest.json missing keys: {missing}")
        else:
            ok("encoding_manifest.json: all required keys present")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) failed.")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
