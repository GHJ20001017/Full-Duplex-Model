# Jev soft-target distillation: V7-B training

This is this project's training implementation, **not Jev's official training source**.
The model is Qwen3-0.6B, used for full-duplex text turn decisions (`continue/yield/wait`).
This repository contains code only: no model weights, private datasets, reference caches,
API keys, or training logs. It does not call the Jev API.

## Algorithm

- Read A/B/C logits at the last **non-padding input token**, without appending an answer.
- Sample four zero-mean-projected Gaussian perturbations of these logits.
- Reward: target-weighted log score plus 0.75 times spherical score.
- REINFORCE with leave-one-out reward baseline (RLOO), **without batch standard-deviation normalization**.
- Sigma decreases from 0.4 to 0.1 over optimizer updates, including a one-epoch run.
- Loss: RL + soft cross-entropy (weight 1) + masked `2 * KL(reference || student)`.
- Reference preservation applies only to original examples, not augmented echo examples.
- FP32 trainable parameters and AdamW state, BF16 autocast, gradient checkpointing.

This is **not pure RL**. Raw RLOO also changes gradient scale relative to the older
normalized estimator. Diagnostic norms measure gradients at logits, not full model
parameter gradients. Single-seed validation does not establish algorithm superiority.

## Files

| File | Role |
|---|---|
| `train_rlcd.py` | Model loading, training loop, reward, checkpoint export |
| `rlcd_algorithm.py` | Sigma schedule, RLOO/legacy advantages, gradient diagnostics |
| `preservation.py` | Cache fingerprint validation and masked reference KL |
| `build_reference.py` | Portable replacement for the original server-specific cache builder |
| `rlcd_taxonomies.py` | Exact training-local prompt snapshot; saved in each checkpoint |
| `test_algorithm.py`, `test_preservation.py` | CPU-only algorithm and cache tests |
| `train.sh` | Recorded V7-B settings; refuses existing output directories |

The parent inference modules are not overwritten. Inference must load `rlcd_config.json`
from the trained checkpoint rather than use a different hardcoded prompt.

## Data contract

JSONL, one object per line:

```json
{"assistant_text":"您的快递预计明天下午送达。","user_asr":"您的快递预计","target":{"continue":0.9,"yield":0.05,"wait":0.05}}
```

`assistant_text` and `user_asr` are strings. `target` must contain all three labels,
with finite nonnegative probabilities summing to one. Source labels and provenance
can be retained as additional fields. Do not call generated labels human gold.

Historical V7-B input order: 21,154 original rows followed by 8,000 augmentation rows.
Only the first 21,154 received reference preservation. Original rows comprised 8,340
Jev-derived soft targets and 12,814 smoothed synthetic hard targets. Augmentation used
2,000 assistant contexts, each producing exact prefix echo, noisy echo, explicit
correction, and unfinished-user contrasts. No augmentation generator or dataset is
bundled here: supply your approved JSONL and preserve split/context isolation.

## Run

CUDA is required for reference inference and training; unit tests run on CPU.
Install the dependencies in an isolated environment. The torch/transformers pins record
the original research runtime, not a guarantee of availability on every package index.

Set absolute paths (weights and data must be obtained separately):

```sh
export MODEL_ID=/absolute/path/to/v4/checkpoint
export DATA=/absolute/path/to/train.jsonl
export REFERENCE_FILE=/absolute/path/to/reference.json
export CKPT_OUT=/absolute/path/to/new-checkpoint
export CUDA_VISIBLE_DEVICES=0

python3 build_reference.py --model "$MODEL_ID" --data "$DATA" \
  --output "$REFERENCE_FILE" --preserve-first 21154
sh train.sh
```

Adjust `--preserve-first` for your dataset. Reference rows use probability vectors;
augmentation rows use `null`. The cache stores the exact dataset-byte SHA256 and
training refuses a mismatch. Do not reuse a cache after reordering/editing the data.
The recorded continuation starts from **V4**, not the untrained Qwen base model.
V4 weights are not distributed in this code submission, so this is not a self-contained
from-scratch reproduction of historical metrics.

```sh
python3 -m unittest discover -s . -p 'test_*.py'
python3 -m compileall -q .
```

## Evaluation boundaries

Historical V7-B validation: 97.90% accuracy on 2,146 examples, 94.99% WAIT recall.
Synthetic echo/contrast diagnostics (1,200 rows) and a 72-row synthetic probe scored
100%. Those are selection/diagnostic sets, **not independent real-audio test results**.
Compared with V4's 98.56% / 96.80%, the full preservation gate was not met.
Evaluate soft-distribution fit separately from source-label accuracy; the targets can
conflict. Keep an untouched test set and never add validation failures to training.

Publishing adaptations remove server-specific paths and historical comments; the
V7-B reward, gradient estimator, schedule, loss weights and option readout are retained.
Do not include credentials, proprietary prompts/data, or model weights in future commits.
