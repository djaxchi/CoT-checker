# Public-SAE audit: offline download + repo inventory

Goal: a cheap external-SAE sanity check for step-correctness separability, run
**fully offline** on TamIA. This file records (a) the decision logic for *which*
SAE/model to use, (b) the exact pre-fetch commands to run on an internet-enabled
node, and (c) the inventory + layer mapping discovered by inspecting the repo.

---

## 1. Decision logic (resolved)

> Q: is there an *exact* public SAE for `Qwen/Qwen2.5-7B` **base**, residual
> stream, at/near L20/L28?

**No.** As of June 2026 there is no public residual-stream SAE for Qwen2.5-7B
*base*. The Qwen-Scope releases (`Qwen/SAE-Res-*`) cover only Qwen3 / Qwen3.5
(8B/9B/27B), not Qwen2.5-7B. The only matching public SAE family is:

* `andyrdt/saes-qwen2.5-7b-instruct` — residual-stream **BatchTopK** SAEs for
  `Qwen/Qwen2.5-7B-Instruct`.

Therefore we are in the **Instruct-matched** case:

* Backbone for this audit = `Qwen/Qwen2.5-7B-Instruct` (NOT our base 7B).
* Dense baseline = Instruct residuals from the **same** backbone.
* **Hard rule:** never encode base-7B activations with the Instruct SAE, and
  never compare Instruct-SAE features against base-model dense activations as if
  they were the same representation. Our existing
  `runs/s1_model_size_dense/qwen2_5_7b/merged/prm800k_heldout_test_*` caches are
  **base** model and are NOT reused here except for the *examples/labels* (the
  text JSONL), which we re-encode with the Instruct backbone.

---

## 2. Pre-fetch (run on a login / internet node, BEFORE sbatch)

```bash
export HF_HOME=/scratch/d/dchikhi/hf
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME"

# Backbone for the Instruct-matched audit.
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir "$HF_HOME/models/Qwen/Qwen2.5-7B-Instruct"

# External SAE repo. The full repo is ~100 GB (28 ae.pt of 3.76 GB each).
# For the PILOT we only need 2 layers x 1 trainer = 2 files. Fetch just those:
huggingface-cli download andyrdt/saes-qwen2.5-7b-instruct \
  --include "resid_post_layer_19/trainer_1/*" "resid_post_layer_27/trainer_1/*" \
  --local-dir "$HF_HOME/models/andyrdt/saes-qwen2.5-7b-instruct"

# (Optional, only if the pilot is promising) add k=128 = trainer_2:
#   --include "resid_post_layer_19/trainer_2/*" "resid_post_layer_27/trainer_2/*"
# (Optional, full repo)  drop --include entirely.
```

Sanity-inspect what landed:

```bash
SAE_ROOT="$HF_HOME/models/andyrdt/saes-qwen2.5-7b-instruct"
find "$SAE_ROOT" -maxdepth 3 -type f | sort
find "$SAE_ROOT" -maxdepth 2 -type d | sort
python -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])),indent=2))" \
  "$SAE_ROOT/resid_post_layer_19/trainer_1/config.json"
```

Inside Slurm jobs, force offline:

```bash
export HF_HOME=/scratch/d/dchikhi/hf
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

---

## 3. Repo inventory (from HF API, June 2026)

```
andyrdt/saes-qwen2.5-7b-instruct/
  README.md
  resid_post_layer_{3,7,11,15,19,23,27}/
    trainer_{0,1,2,3}/
      ae.pt            (3,758,637,401 bytes ~= 3.76 GB, float32 state_dict)
      config.json      (~0.9 KB)
      eval_results.json
```

* Architecture: **BatchTopK** SAE (`dict_class: BatchTopKSAE`,
  `trainer_class: BatchTopKTrainer`; ref arXiv:2412.06410).
* `activation_dim = 3584` (Qwen2.5-7B hidden size), `dict_size = 131072` (128K,
  ~36.6x expansion).
* `trainer_0..3` -> `k = 32, 64, 128, 256` respectively. (trainer_1 confirmed
  `k=64` from its config.json.)
* Training data: blend of chat (`lmsys/lmsys-chat-1m`) + pretraining
  (`monology/pile-uncopyrighted`) + emergent-misalignment data; first 8 tokens of
  each sample dropped; activations with norm > 10x batch-median filtered.
  ctx_len 1024, hook `io: "out"` of `resid_post_layer_N`.

### ae.pt state_dict keys (what `encode_public_sae.py` loads directly)

```
encoder.weight  (dict_size, activation_dim) = (131072, 3584)
encoder.bias    (dict_size,)
decoder.weight  (activation_dim, dict_size) = (3584, 131072)   # unit-norm cols
b_dec           (activation_dim,)                              # pre-encode subtract + post-decode add
k               scalar int buffer
threshold       scalar float buffer  (learned JumpReLU-style inference threshold)
```

Inference (matches dictionary_learning `BatchTopKSAE.encode(use_threshold=True)`):

```
pre = relu( encoder(h - b_dec) )
z   = pre * (pre > threshold)          # variable active count, not exactly k
h_hat = decoder(z) + b_dec
r   = h - h_hat
```

We reimplement this in `encode_public_sae.py` (~30 lines) and load `ae.pt`
directly, so the job has **no dependency** on the `andyrdt/dictionary_learning`
package being installed on the offline node.

---

## 4. Layer mapping: SAE `resid_post_layer_N` <-> our `hidden_states` index

Qwen2.5-7B has 28 decoder blocks (indices 0..27). HF `output_hidden_states`
returns 29 tensors: `hidden_states[0]` = embeddings, `hidden_states[i]` = output
of block `i-1`. The SAE `resid_post_layer_N` hooks the **output of block N**
(0-indexed). Therefore:

```
output of block N  ==  hidden_states[N+1]
```

Our S3 readouts ("L20", "L28") are `hidden_states[20]` and `hidden_states[28]`:

| our readout      | hidden_states idx | = output of block | SAE folder              |
|------------------|-------------------|-------------------|-------------------------|
| **L20**          | 20                | block 19          | `resid_post_layer_19`   |
| **L28** (final)  | 28                | block 27          | `resid_post_layer_27`   |

So L20 and L28 map to **exact** available SAE layers (19 and 27). No nearest-layer
fallback is needed.

**This mapping is verified empirically, not just asserted:** `encode_public_sae.py`
computes reconstruction FVU = `||h - h_hat||^2 / ||h - mean(h)||^2`. A correct
layer pairing reconstructs well (BatchTopK SAEs typically FVU < ~0.25 on
in-distribution activations); a layer/indexing mismatch yields FVU near or above
1.0 and the script flags it loudly. (Our prompt format + last-token-of-step
readout is mildly OOD vs the SAE's chat/pile training, so a somewhat elevated FVU
is expected; a *catastrophic* FVU means the pairing or hook is wrong.)
