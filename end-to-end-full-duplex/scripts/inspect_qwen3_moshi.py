#!/usr/bin/env python3
"""Load a local Qwen3 checkpoint and inspect the Qwen3-Moshi shell."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qwen_model", type=Path, help="Local Qwen3 model directory.")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: pip install torch transformers"
        ) from exc

    from models.qwen3_moshi import load_qwen3_moshi

    model = AutoModelForCausalLM.from_pretrained(
        str(args.qwen_model.expanduser().resolve()),
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    wrapped = load_qwen3_moshi(
        model,
        mimi_codebook_sizes=tuple([args.codebook_size] * args.num_codebooks),
        text_pad_token_id=model.config.eos_token_id or 0,
    )
    trainable = sum(parameter.numel() for parameter in wrapped.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in wrapped.parameters())
    print(f"Qwen3 hidden size: {wrapped.config.hidden_size}")
    print(f"Qwen3 vocabulary: {wrapped.config.vocab_size}")
    print(f"Mimi codebooks: {wrapped.config.num_codebooks}")
    print(f"Mimi cardinalities: {wrapped.config.codebook_sizes}")
    print(f"Trainable parameters: {trainable:,} / total: {total:,}")
    print("Temporal: Qwen3; Depth: randomly initialized Moshi-style transformer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
