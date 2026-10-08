"""Builder: v1 split inheritance, PB overlap exclusion, PB sets unchanged (v2 plan 4, 9)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.prm_trajectories import problem_key

ROOT = Path(__file__).resolve().parents[2]
PB = ("pb_gsm8k", "pb_math", "pb_olympiadbench", "pb_omnimath")


def wjl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def fixture(tmp_path):
    from transformers import AutoTokenizer
    try:
        AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3-8B tokenizer not cached")
    v1 = tmp_path / "v1"
    probs = {"train": "Train problem A", "dev": "Dev problem B", "calib": "Calib problem C", "test": "Test problem D"}
    for sp, q in probs.items():
        wjl(v1 / "meta" / f"{sp}.jsonl", [{"trace_id": f"v1_{sp}", "problem_key": problem_key(q), "split": sp,
                                           "y": [0, 1], "label_mask": [True, True], "first_error": 1}])
        wjl(v1 / "inputs" / f"{sp}.jsonl", [{"trace_id": f"v1_{sp}", "problem": q, "steps": ["a", "b"]}])
    enc = [{"trace_id": f"v1_{sp}", "split": sp, "n_tokens": 10, "n_steps": 2, "step_starts": [5, 8],
            "step_ends": [7, 10], "prefix_len": 5, "ids_sha1": "x", "shard": 0} for sp in probs]
    for sp in PB:
        q = "PB shared problem" if sp == "pb_math" else f"PB {sp}"
        wjl(v1 / "inputs" / f"{sp}.jsonl", [{"trace_id": f"{sp}_0", "problem": q, "steps": ["s"]}])
        wjl(v1 / "meta" / f"{sp}.jsonl", [{"trace_id": f"{sp}_0", "problem_key": problem_key(q), "y": [1],
                                           "label_mask": [True], "first_error": 0, "pb_label": 0,
                                           "pb_subset": sp[3:], "split": sp}])
        enc.append({"trace_id": f"{sp}_0", "split": sp, "n_tokens": 9, "n_steps": 1, "step_starts": [7],
                    "step_ends": [9], "prefix_len": 7, "ids_sha1": "y", "shard": 0})
    wjl(v1 / "encode_manifest.jsonl", enc)
    steps = ["- Step 1: Do x.", "- Step 2: Do y."]
    rows = []
    for q in list(probs.values()) + ["  pb SHARED   problem ", "Brand new problem E"]:
        rows.append({"question": q, "answer": "gt", "reply": "\n".join(steps) + "\n",
                     "claims": [{"sentence": s, "aligned_token_ids": [1, 2]} for s in steps],
                     "verified": [0.0, 1.0]})
    pq = tmp_path / "ds.parquet"
    pd.DataFrame(rows).to_parquet(pq)
    out = tmp_path / "v2"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build_reprobe_trajectory_dataset.py"),
                        "--ds_parquet", str(pq), "--v1_manifest", str(v1), "--out_dir", str(out),
                        "--tokenizer", "Qwen/Qwen3-8B", "--local_files_only", "--num_shards", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    return v1, out, probs


def test_split_inheritance_overlap_exclusion_and_pb_unchanged(fixture):
    v1, out, probs = fixture
    for sp, q in probs.items():
        metas = [json.loads(l) for l in open(out / "meta" / f"{sp}.jsonl")]
        assert any(m["problem_key"] == problem_key(q) for m in metas), sp
    every = [json.loads(l) for sp in ("train", "dev", "calib", "test") for l in open(out / "meta" / f"{sp}.jsonl")]
    assert all(m["problem_key"] != problem_key("PB shared problem") for m in every)
    audit = json.loads((out / "data_audit.json").read_text())
    assert audit["split_inheritance"]["pb_overlap_traces_excluded"] == 1
    assert sum(audit["split_inheritance"]["new_problems_assigned"].values()) == 1
    for sp in PB:
        a = [json.loads(l) for l in open(v1 / "inputs" / f"{sp}.jsonl")]
        b = [json.loads(l) for l in open(out / "inputs" / f"{sp}.jsonl")]
        assert a == b
    hum = [json.loads(l) for l in open(out / "meta" / "prm_human_test.jsonl")]
    assert hum[0]["trace_id"] == "v1_test" and hum[0]["split"] == "prm_human_test"
    enc = [json.loads(l) for l in open(out / "encode_manifest.jsonl")]
    assert {e["split"] for e in enc} >= set(PB) | {"prm_human_test", "train"}
