"""python -m trainer.train_stage1 --help (all model assets must exist locally)."""
import os
# Set before importing transformers, datasets or any hub-dependent runtime.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import argparse
from pathlib import Path

from .engine import TrainConfig, prepare_output, seed_everything, train


def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=(
        'Single-process Stage 1 only. No FSDP, pure-text preservation optimizer, '
        'or alignment jitter. num_workers=0. Resume only your own trusted checkpoint. '
        'Keep max-steps fixed on resume; stop-after-steps allows a planned early pause.'))
    for name in ('manifest', 'qwen-model', 'mimi-checkpoint', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--text-pad-token-id', type=int, required=True,
                   help='Explicit existing Qwen vocabulary ID; no token is added or guessed')
    p.add_argument('--text-condition-dropout', type=float, default=.3,
                   help='Probability of dropping all text conditioning for a training utterance (targets unchanged)')
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--stop-after-steps', type=int)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation-steps', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--temporal-lr', type=float, default=3e-5)
    p.add_argument('--depth-lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--validation-interval', type=int, default=10, help='0 explicitly disables validation')
    p.add_argument('--validation-fraction', type=float, default=.05)
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--resume', action='store_true', help='Trust and restore output-dir/checkpoint.pt')
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--max-samples', type=int)
    p.add_argument('--min-duration', type=float, default=0.)
    p.add_argument('--max-duration', type=float)
    p.add_argument('--depth-hidden-size', type=int, default=1024)
    p.add_argument('--depth-layers', type=int, default=6)
    p.add_argument('--depth-heads', type=int, default=16)
    p.add_argument('--depth-ffn-size', type=int, default=4096)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = TrainConfig(**{name: getattr(args, name) for name in TrainConfig.__dataclass_fields__})
    config.validate()
    for name in ('manifest', 'mimi_checkpoint'):
        path = getattr(args, name).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f'{name} must be a local file')
        setattr(args, name, path)
    args.qwen_model = args.qwen_model.expanduser().resolve(strict=True)
    if not args.qwen_model.is_dir():
        raise ValueError('qwen-model must be a local directory')
    # Refuse overwrites before loading potentially expensive models.
    prepare_output(args.output_dir, args.resume)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from dataset import EmiliaDataset, collate_emilia
    from models.qwen3_moshi import load_qwen3_moshi
    from .codec import FrozenMimiCodec

    seed_everything(args.seed)
    qwen_config = AutoConfig.from_pretrained(str(args.qwen_model), local_files_only=True, trust_remote_code=False)
    if qwen_config.model_type != 'qwen3':
        raise ValueError('Stage 1 requires Qwen3')
    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_model), local_files_only=True,
                                              use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer with offsets is required')
    tokenizer('offset check', return_offsets_mapping=True)
    if not 0 <= args.text_pad_token_id < qwen_config.vocab_size or args.text_pad_token_id not in tokenizer.get_vocab().values():
        raise ValueError('text-pad-token-id must already exist in tokenizer and Qwen model vocabulary')
    codec = FrozenMimiCodec(args.mimi_checkpoint, device=args.device)
    options = dict(validation_fraction=args.validation_fraction, seed=args.seed,
                   cache_dir=args.cache_dir, data_root=args.data_root, max_samples=args.max_samples,
                   min_duration=args.min_duration, max_duration=args.max_duration)
    training = EmiliaDataset(args.manifest, tokenizer, codec, args.text_pad_token_id, split='train', **options)
    validation = (EmiliaDataset(args.manifest, tokenizer, codec, args.text_pad_token_id,
                               split='val', **options) if args.validation_interval else None)
    if not len(training) or (validation is not None and not len(validation)):
        raise ValueError('Empty train/validation split; adjust split or disable validation explicitly')
    qwen = AutoModelForCausalLM.from_pretrained(str(args.qwen_model), local_files_only=True,
                                               trust_remote_code=False, torch_dtype=torch.float32)
    qwen.config.use_cache = False
    if args.gradient_checkpointing:
        qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = load_qwen3_moshi(qwen, mimi_codebook_sizes=codec.codebook_sizes,
                            text_pad_token_id=args.text_pad_token_id, num_audio_streams=1,
                            acoustic_delay=2, depth_hidden_size=args.depth_hidden_size,
                            depth_layers=args.depth_layers, depth_heads=args.depth_heads,
                            depth_ffn_size=args.depth_ffn_size)
    # Collation owns absent-frame sentinels; the model masks physical padding.
    collate = collate_emilia
    result = train(model, training, collate, args.output_dir, config, validation_data=validation,
                   resume=args.resume, stop_after_steps=args.stop_after_steps,
                   run_metadata={'qwen_model': str(args.qwen_model), 'codec': codec.cache_identity,
                                 'gradient_checkpointing': args.gradient_checkpointing})
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
