# Sprint 2 — Fork dataset & objective design note

Scope: how PRM800K forks become training signal for representation shaping, the
ranking and triplet formulations, the latent construction, the recommended
dataset size, and the exact build / encode / train / eval commands.

This sprint shapes the *representation* only. The ProcessBench evaluation path is
unchanged: train the representation, freeze it, train a fresh linear probe on
`probe_train_40k`, select a threshold on `val_1k`, score ProcessBench. The
ranking/triplet heads are auxiliary and discarded; the saved `representation.pt`
is byte-format identical to the S1 SAE artifact.

---

## 1. Fork structure

A fork is a single reasoning state shared by sibling candidate next steps:

```
key = problem_id :: solution_id :: step_idx
```

All siblings share the same problem and the same reasoning prefix; they differ
only in the proposed next step. Each sibling carries a PRM800K rating, mapped as
in S1:

| rating | meaning              | label |
|--------|----------------------|-------|
| +1     | correct / preferred  | 0     |
| -1     | error / rejected     | 1     |
|  0     | neutral (dropped)    | —     |

A fork is *valid* for contrast iff it has at least one positive and at least one
negative sibling. The prestudy manifest
(`/scratch/d/dchikhi/cot_mech/prestudy_v1/data/manifest.json`) reports
`valid_forks_found = 56342` under the current filtering (rating in {-1,+1},
non-flagged, non-empty, within `max_seq_len`). That is enough for the target
split below with surplus held out.

The exact per-fork sibling distribution (mean/max positives and negatives, total
all-pairs count) is computed by the builder and written to
`forks_manifest.json` -> `sibling_distribution`. Read those numbers before
committing to a pair mode; do not assume the distribution.

---

## 2. One pair per fork vs all positive×negative pairs

Two ways to turn a fork into training pairs:

- **one** — sample a single (positive, negative) pair per fork.
  - Pairs ≈ number of valid forks (≈ one per fork).
  - Every fork contributes equally; high-degree forks (many siblings) do not
    dominate the gradient.
  - Encoding cost ≈ 3 items/fork (anchor + 1 pos + 1 neg).
  - Matches the balanced 40k design used in S1, so the comparison to the S1
    dense/SAE-mixed baselines is clean (same effective dataset scale).

- **all** — every positive against every negative (Cartesian product per fork).
  - Pairs = Σ_forks (n_pos × n_neg); see `total_pairs_if_all_over_valid_forks`.
  - Gives more, harder contrasts but **upweights forks with many siblings**,
    which biases the representation toward heavily-annotated problems.
  - Encoding cost grows with the number of distinct siblings referenced.

**Recommendation: `one` for the headline 2×2 matrix.** It keeps the dataset
scale comparable to S1 (clean A/B against the existing baselines), avoids
high-degree-fork bias, and bounds encoding to ~3×forks items. Use `all` only as
a follow-up ablation once a `one`-mode cell shows a positive signal — and report
it separately, since it changes the effective example distribution. The builder
emits the exact pair counts for both modes in the manifest so the choice is
grounded, not assumed.

---

## 3. Latent construction

Everything lives in one activation space: the Qwen2.5-1.5B last-layer hidden
state, taken at a single token. The fork encoder (`encode_prm800k_forks.py`)
produces three item types per fork, all in that same space:

| role     | text encoded                                             | token taken |
|----------|----------------------------------------------------------|-------------|
| anchor   | `Problem… Previous reasoning:{prefix} Current step:\n`    | last token  |
| positive | anchor text + preferred continuation                     | last candidate token |
| negative | anchor text + rejected continuation                      | last candidate token |

The anchor is the model state **right before** the next step is produced — the
"reasoning prefix representation." Because the anchor prompt text is byte-identical
to the prefix embedded in the positive/negative items, anchor and continuations
are directly comparable. The SAE/AE encoder maps each item h → z = ReLU(W_enc h + b);
objectives operate on z.

Why this anchor (and the alternative): the prefix last token is the cleanest
in-pipeline anchor — it reuses the existing tokenization and needs no new
model surgery. An alternative anchor (mean over prefix step tokens) would change
the encoder contract and is left as an ablation if triplet underperforms.

Ranking needs no anchor (it scores continuations); triplet uses all three.

---

## 4. Ranking formulation

A scalar head s(z) = w·z + b scores each continuation latent. Within a pair we
require the preferred sibling to outscore the rejected one, d = s(z_pos) − s(z_neg):

- logistic (RankNet, default): L_rank = softplus(−d) = −log σ(d)
- margin (hinge): L_rank = relu(margin − d)

Total representation loss per cell:

```
L = MSE_recon  +  λ_L1 · mean|z|        # λ_L1 = 0 for ae_rank, args.l1_weight for sae_rank
      +  obj_weight · L_rank
```

Reconstruction is averaged over the anchor, positive, and negative items in the
batch so the representation still covers prefixes, not just continuations.
Gradients flow through the encoder; the head is discarded after training.

## 5. Triplet formulation

In latent space, pull the anchor toward the positive continuation and push it
away from the negative:

- d_pos = dist(z_anchor, z_pos), d_neg = dist(z_anchor, z_neg)
- L_trip = relu(d_pos − d_neg + margin)
- dist = squared Euclidean (`l2`, default) or 1 − cosine (`cosine`)

```
L = MSE_recon  +  λ_L1 · mean|z|        # λ_L1 = 0 for ae_triplet, args.l1_weight for sae_triplet
      +  obj_weight · L_trip
```

Same recon averaging and head-discard policy.

---

## 6. Final dataset size

Target (matches S1 scale, surplus held out):

- train forks: 40,000
- val forks: 5,000
- pair mode: `one` (≈ one pair per fork)
- items encoded ≈ 3 × forks (anchor + pos + neg), `one` mode

Confirm the realized counts in `forks_manifest.json` -> `counts` after building.

---

## 7. Commands

Paths are placeholders; substitute the TamIA scratch locations. All jobs run
offline (`--local_files_only`); pre-cache `Qwen/Qwen2.5-1.5B`.

```bash
RAW=/scratch/d/dchikhi/cot_mech/prestudy_v1/raw          # PRM800K raw jsonl dir
FORKS=/scratch/d/dchikhi/cot_mech/s2_forks/data
ENC=/scratch/d/dchikhi/cot_mech/s2_forks/encoded
CACHE=/scratch/d/dchikhi/cot_mech/prestudy_v1/encoded    # existing probe_train_40k / val_1k / mixed_train_40k
PB=/scratch/d/dchikhi/cot_mech/processbench_full         # per-subset pb_step_{h.npy,meta.jsonl}
MODEL=Qwen/Qwen2.5-1.5B
```

### 7.1 Build forks
```bash
python scripts/build_prm800k_forks.py \
  --raw_dir $RAW --out_dir $FORKS \
  --tokenizer_name_or_path $MODEL --local_files_only \
  --run_name s2_forks --n_train_forks 40000 --n_val_forks 5000 --pair_mode one
```

### 7.2 Encode fork items (anchor + positive + negative)
```bash
python scripts/encode_prm800k_forks.py \
  --items $FORKS/forks_train_items.jsonl --out_dir $ENC --stem forks_train_items \
  --model_name_or_path $MODEL --local_files_only --run_name s2_enc_train

python scripts/encode_prm800k_forks.py \
  --items $FORKS/forks_val_items.jsonl --out_dir $ENC --stem forks_val_items \
  --model_name_or_path $MODEL --local_files_only --run_name s2_enc_val
```

### 7.3 Train the 2×2 matrix (each cell also runs the ProcessBench eval)
```bash
PB_SPECS="gsm8k:$PB/gsm8k/pb_step_h.npy:$PB/gsm8k/pb_step_meta.jsonl \
          math:$PB/math/pb_step_h.npy:$PB/math/pb_step_meta.jsonl \
          olympiadbench:$PB/olympiadbench/pb_step_h.npy:$PB/olympiadbench/pb_step_meta.jsonl \
          omnimath:$PB/omnimath/pb_step_h.npy:$PB/omnimath/pb_step_meta.jsonl"

FORK_ARGS="--fork_items_h $ENC/forks_train_items_h.npy \
           --fork_items_meta $ENC/forks_train_items_meta.jsonl \
           --fork_pairs $FORKS/forks_train_pairs.jsonl"

for M in ae_rank sae_rank ae_triplet sae_triplet; do
  EXTRA=""
  case $M in
    *_rank)    EXTRA="--rank_kind logistic --obj_weight 1.0" ;;
    *_triplet) EXTRA="--triplet_metric l2 --triplet_margin 1.0 --obj_weight 1.0" ;;
  esac
  python scripts/train_easy_probe_method.py \
    --method $M --cache_dir $CACHE --out_dir runs/s2/$M \
    $FORK_ARGS $EXTRA \
    --pb_specs $PB_SPECS --threshold_grid 0.005
done
```

Baseline (dense AE, recon only — no forks needed):
```bash
python scripts/train_easy_probe_method.py \
  --method ae --cache_dir $CACHE --out_dir runs/s2/ae \
  --pb_specs $PB_SPECS --threshold_grid 0.005
```

### 7.4 Re-evaluate a saved representation+probe (optional)
Each training run already writes `eval_summary.json` / `eval_metrics_*`. To
re-score a saved `representation.pt` + `linear_probe.pt` without retraining:
```bash
python scripts/evaluate_saved_probe_on_processbench.py --help
```

### 7.5 Leaderboard merge
Fold the new cells into the leaderboard with the existing merge step:
```bash
python scripts/merge_easy_probe_leaderboard.py --help
```

---

## 8. Pipeline

All jobs live in `scripts/tamia/jobs/s2/`. Dataset prep is CPU/IO only and runs
directly on the login node (no Slurm); GPU work is submitted to Slurm. Each
training stage is **one** Slurm job whose `JOBS` list is distributed round-robin
across the 4 GPUs (concurrent; one job pinned per `CUDA_VISIBLE_DEVICES`;
non-zero exit if any job fails):

```
build_forks.sh               # login node (no GPU): build forks  -> $SCRATCH/cot_mech/s2_forks/data
stats_forks.sh               # login node (no GPU): print manifest statistics
02_encode_forks.sbatch       # 1 GPU: encode train+val items      -> $SCRATCH/cot_mech/s2_forks/encoded
03_train_matrix_sanity.sbatch  # 4 GPU: controls+4 cells, 1 epoch, 512 pairs, gsm8k-only (smoke)
04_train_matrix_full.sbatch    # 4 GPU: controls + 4 cells @ obj_weight=1 (matched comparison)
05_train_sweep.sbatch          # 4 GPU: controls + 4 cells x obj_weight{1,10,100} (14 jobs)
_matrix_common.sh            # shared JOBS runner (sourced by 03/04/05)
```

**Recon-only controls** (`ae_recon`, `sae_recon`) train on the *same* fork items
with the objective term OFF, so any F1 delta vs `*_rank` / `*_triplet` is
attributable to the objective. **Diagnostics** are written per run:
`train_metrics.json` carries `final_objective_loss`, `final_pair_accuracy`,
`final_margin_satisfaction`; `representation_history.json` carries the per-epoch
recon / l1 / objective / pair_accuracy / margin_satisfaction curves.

`JOBS` entries are `<method>` for the recon controls or `<method>:<obj_weight>`
for the objective cells. Output/log tag is the method name (controls) or
`<method>_w<weight>` (objective cells).

Dependency chain:
```
build_forks.sh                                  # login node, CPU/IO
    ↓
stats_forks.sh                                  # login node, CPU/IO (inspect manifest)
    ↓
sbatch 02_encode_forks.sbatch                   # GPU
    ↓
sbatch 03_train_matrix_sanity.sbatch            # GPU (verify), then:
    ↓
sbatch 04_train_matrix_full.sbatch              # GPU (matched comparison @ w=1)
    ↓
sbatch 05_train_sweep.sbatch                    # GPU (obj_weight sweep)
```

Run, with the full paths:
```bash
bash scripts/tamia/jobs/s2/build_forks.sh
bash scripts/tamia/jobs/s2/stats_forks.sh
sbatch scripts/tamia/jobs/s2/02_encode_forks.sbatch
sbatch scripts/tamia/jobs/s2/03_train_matrix_sanity.sbatch
sbatch scripts/tamia/jobs/s2/04_train_matrix_full.sbatch
sbatch scripts/tamia/jobs/s2/05_train_sweep.sbatch
```

### Output / checkpoint layout

```
$SCRATCH/cot_mech/s2_forks/
├── data/                                  # build_forks.sh
│   ├── forks_full.jsonl
│   ├── forks_train_items.jsonl
│   ├── forks_val_items.jsonl
│   ├── forks_train_pairs.jsonl
│   ├── forks_val_pairs.jsonl
│   └── forks_manifest.json
└── encoded/                               # 02_encode_forks
    ├── forks_train_items_h.npy
    ├── forks_train_items_meta.jsonl
    ├── forks_train_items_encoding_manifest.json
    ├── forks_val_items_h.npy
    ├── forks_val_items_meta.jsonl
    └── forks_val_items_encoding_manifest.json

$HOME/CoT-checker/runs/s2/<full|sanity>/   # 03/04 matrix
├── logs/
│   ├── ae_rank.log
│   ├── sae_rank.log
│   ├── ae_triplet.log
│   └── sae_triplet.log
└── <method>/                              # one dir per cell
    ├── representation.pt                  # frozen encoder (the checkpoint)
    ├── linear_probe.pt                    # fresh probe trained on top
    ├── config.yaml
    ├── threshold.json
    ├── val_scores.npy
    ├── train_metrics.json
    ├── eval_summary.json                  # all (subset x threshold) runs
    ├── eval_metrics_<subset>_{fixed_t0.5,val_selected,oracle}.json
    ├── pb_step_scores_<subset>.jsonl
    └── pb_predictions_<subset>.jsonl

$HOME/CoT-checker/results/logs/            # Slurm stdout/err: %x-%j.out / .err
```

`representation.pt` is the saved checkpoint per cell; it is format-identical to
the S1 SAE artifact, so existing eval/merge tooling consumes it unchanged.
