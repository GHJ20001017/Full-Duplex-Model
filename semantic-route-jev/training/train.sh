#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${MODEL_ID:?Set MODEL_ID to the local V4 checkpoint}"
: "${DATA:?Set DATA to an absolute JSONL path}"
: "${REFERENCE_FILE:?Build the frozen reference cache first}"
: "${CKPT_OUT:?Set CKPT_OUT to a new absolute output directory}"
[ ! -e "$CKPT_OUT" ] || { echo 'Refusing to overwrite CKPT_OUT' >&2; exit 1; }
mkdir -p "$CKPT_OUT"
export MODEL_ID DATA REFERENCE_FILE CKPT_OUT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TOKENIZERS_PARALLELISM=false
export BATCH=2 GRAD_ACCUM=16 GROUP_SIZE=4 SEED=250925 EPOCHS=1
export TAXONOMY=continue APPEND_A=0 TAKE_REAL_LEN=1 CLASS_BALANCE=0 TARGET_PRIOR=''
export LR_BACKBONE=1e-6 LR_HEAD=1e-6 REFERENCE_KL_WEIGHT=2.0 ADVANTAGE_MODE=rloo
export DIAGNOSTICS_OUT="$CKPT_OUT/gradient_diagnostics.jsonl" HIST_OUT="$CKPT_OUT/train_history.json"
exec python3 -u train_rlcd.py
