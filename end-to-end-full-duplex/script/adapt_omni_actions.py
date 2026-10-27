"""Save an offline Qwen2.5-Omni checkpoint with native Thinker action tokens.

Run from end-to-end-full-duplex with PYTHONPATH=src. This initializes new rows;
no training or learned full-duplex behavior is implied. Never overwrites a model.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thinker-only", action="store_true", help="Save only Thinker, explicitly without speech output")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cpu", help="A single device, e.g. cpu or cuda:0")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("--model-dir must be an existing local checkpoint directory")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("--output-dir must be absent or empty")
    if not args.output_dir.parent.is_dir():
        parser.error("Parent of --output-dir must already exist")

    import json
    import torch
    from transformers import (
        Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )
    from qwen_duplex import extend_action_tokens, save_duplex_checkpoint

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model_class = Qwen2_5OmniThinkerForConditionalGeneration if args.thinker_only else Qwen2_5OmniForConditionalGeneration
    model = model_class.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
    ).to(args.device).eval()
    actions = extend_action_tokens(model, processor.tokenizer)
    save_duplex_checkpoint(model, processor, args.output_dir)
    print(json.dumps({"checkpoint": str(args.output_dir), "actions": actions,
                      "trained_action_policy": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
