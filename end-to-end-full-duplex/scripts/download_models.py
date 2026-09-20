#!/usr/bin/env python3
"""Download Qwen3-1.7B and the official Mimi codec, without Moshi LM weights."""

import argparse
import sys
from pathlib import Path

QWEN_REPO = "Qwen/Qwen3-1.7B"
MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "checkpoints" / "base"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Download root (default: module checkpoints/base).",
    )
    parser.add_argument("--qwen-revision", default="main", help="Qwen branch, tag or commit SHA.")
    parser.add_argument("--mimi-revision", default="main", help="Mimi branch, tag or commit SHA.")
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    qwen_dir = output / "qwen3-1.7b"
    mimi_dir = output / "mimi"
    mimi_file = mimi_dir / MIMI_FILE
    skip_qwen = qwen_dir.exists()
    skip_mimi = mimi_file.exists()

    if not (skip_qwen and skip_mimi):
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            print("Missing dependency. Run: python3 -m pip install -U huggingface_hub", file=sys.stderr)
            return 1

    try:
        output.mkdir(parents=True, exist_ok=True)
        if skip_qwen:
            print(f"Skipping Qwen3: target already exists: {qwen_dir}", flush=True)
        else:
            print(f"Downloading {QWEN_REPO} to {qwen_dir}", flush=True)
            snapshot_download(
                repo_id=QWEN_REPO,
                revision=args.qwen_revision,
                local_dir=str(qwen_dir),
            )
        if skip_mimi:
            print(f"Skipping Mimi: target already exists: {mimi_file}", flush=True)
        else:
            print(f"Downloading Mimi codec to {mimi_dir}", flush=True)
            hf_hub_download(
                repo_id=MIMI_REPO,
                filename=MIMI_FILE,
                revision=args.mimi_revision,
                local_dir=str(mimi_dir),
            )
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print("Fix connectivity, access or disk space, then rerun the same command.", file=sys.stderr)
        return 1

    print(f"Downloads complete.\nQwen3: {qwen_dir}\nMimi: {mimi_dir / MIMI_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
