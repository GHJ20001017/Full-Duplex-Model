#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR/.."

# 可在启动命令中通过 CKPT=/path/to/checkpoint 覆盖模型目录。
export CKPT="${CKPT:-$PWD/checkpoint}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PORT="${PORT:-8792}"
export HOST="${HOST:-0.0.0.0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

exec "${PYTHON_BIN:-python3}" -u api_server.py
