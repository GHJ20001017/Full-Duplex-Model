#!/usr/bin/env python3
"""Download Qwen3, Mimi, Qwen3-ASR and ForcedAligner from ModelScope.

Skip model directories containing files; this is not an integrity check.
"""

import argparse
import sys
from pathlib import Path

QWEN_REPO = "Qwen/Qwen3-1.7B"
ASR_REPO = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "checkpoints" / "base"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Download root (default: module checkpoints/base).",
    )
    parser.add_argument("--qwen-revision", default="master", help="ModelScope branch, tag or commit SHA.")
    parser.add_argument("--mimi-revision", default="master", help="ModelScope branch, tag or commit SHA.")
    parser.add_argument("--asr-revision", default="master", help="ModelScope branch, tag or commit SHA.")
    parser.add_argument("--aligner-revision", default="master", help="ModelScope branch, tag or commit SHA.")
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    models = [
        (QWEN_REPO, output / "qwen3-1.7b", args.qwen_revision),
        (MIMI_REPO, output / "mimi", args.mimi_revision),
        (ASR_REPO, output / "qwen3-asr-1.7b", args.asr_revision),
        (ALIGNER_REPO, output / "qwen3-forcedaligner-0.6b", args.aligner_revision),
    ]

    try:
        pending = []
        for repo, directory, revision in models:
            if directory.exists() and not directory.is_dir():
                raise NotADirectoryError(f"Model target is not a directory: {directory}")
            if any(path.is_file() for path in directory.rglob("*")):
                print(
                    f"Skipping {repo}: target contains files: {directory} "
                    "(completeness not verified)", flush=True,
                )
            else:
                pending.append((repo, directory, revision))

        if pending:
            try:
                from modelscope.hub.file_download import model_file_download
                from modelscope.hub.snapshot_download import snapshot_download
            except ImportError:
                print("Missing dependency. Run: python3 -m pip install -U modelscope", file=sys.stderr)
                return 1

        for repo, directory, revision in pending:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {repo} to {directory}", flush=True)
            if repo == MIMI_REPO:
                model_file_download(
                    model_id=repo,
                    file_path=MIMI_FILE,
                    revision=revision,
                    local_dir=str(directory),
                )
            else:
                snapshot_download(
                    model_id=repo,
                    revision=revision,
                    local_dir=str(directory),
                )
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "Fix connectivity, access or disk space. Before retrying, move incomplete "
            "model files aside: nonempty targets will be skipped.", file=sys.stderr,
        )
        return 1

    print("Download pass complete (existing files were not validated).")
    for repo, directory, _ in models:
        print(f"{repo}: {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
