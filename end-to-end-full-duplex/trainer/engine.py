"""Injectable Stage-1 engine; checkpoints are written only at update boundaries."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch


@dataclass
class TrainConfig:
    max_steps: int = 100
    batch_size: int = 1
    accumulation_steps: int = 1
    seed: int = 0
    temporal_lr: float = 3e-5
    depth_lr: float = 2e-4
    weight_decay: float = 0.1
    warmup_steps: int = 0
    grad_clip: float = 1.0
    validation_interval: int = 10
    device: str = 'cpu'
    precision: str = 'fp32'
    text_condition_dropout: float = 0.3

    def validate(self):
        if not math.isfinite(self.text_condition_dropout) or not 0 <= self.text_condition_dropout <= 1:
            raise ValueError('text_condition_dropout must be finite and in [0, 1]')
        if min(self.max_steps, self.batch_size, self.accumulation_steps) <= 0:
            raise ValueError('max_steps, batch_size and accumulation_steps must be positive')
        if not 0 <= self.warmup_steps < self.max_steps or self.validation_interval < 0:
            raise ValueError('Invalid warmup or validation interval')
        if not all(math.isfinite(v) and v > 0 for v in (self.temporal_lr, self.depth_lr, self.grad_clip)):
            raise ValueError('Learning rates and gradient clipping must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid weight decay')
        device = torch.device(self.device)
        if device.type not in ('cpu', 'cuda') or self.precision not in ('fp32', 'bf16'):
            raise ValueError('Only CPU/fp32 or CUDA/fp32,bf16 are supported')
        if device.type == 'cpu' and self.precision != 'fp32':
            raise ValueError('CPU training requires fp32')
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA is not available')
            if self.precision == 'bf16' and not torch.cuda.is_bf16_supported():
                raise ValueError('CUDA device does not support bf16')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare_output(output_dir, resume=False):
    path = Path(output_dir).expanduser().resolve()
    if resume:
        if not (path / 'checkpoint.pt').is_file():
            raise ValueError('Resume requires this run\'s output-dir/checkpoint.pt')
    else:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError('Output directory is nonempty; use --resume for a trusted own checkpoint')
        path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint(path):
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = True
    else:
        raise RuntimeError('Upgrade PyTorch: safe weights_only checkpoint loading is required')
    state = torch.load(path, **kwargs)
    if state.get('format') != 'qwen3-moshi-stage1-v1':
        raise ValueError('Not a trainer-owned Stage-1 checkpoint')
    return state


def save_checkpoint(path, state):
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('wb') as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _batch(dataset, indices, collate, device):
    # Deliberately synchronous: equivalent to num_workers=0; codec may live on GPU.
    batch = collate([dataset[i] for i in indices])
    for key in ('audio_tokens', 'text_tokens', 'frame_mask'):
        batch[key] = batch[key].to(device)
    mask = batch['frame_mask']
    if mask.dtype != torch.bool or mask.ndim != 2 or not mask.any(1).all():
        raise ValueError('Each example must have a nonempty boolean frame mask')
    if torch.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError('Frame mask must be right padded')
    durations = [float(x) for x in batch['durations']]
    if len(durations) != len(indices) or not all(math.isfinite(x) and x > 0 for x in durations):
        raise ValueError('Expected positive real audio durations')
    return batch


def _loss(model, batch, config):
    conditioning = None
    if model.training and config.text_condition_dropout:
        # One Bernoulli decision per utterance, shared by all its frames. Text
        # targets are untouched; only the model's conditioning path is masked.
        keep = torch.rand(batch['text_tokens'].shape[0], 1, device=batch['text_tokens'].device)
        conditioning = (keep >= config.text_condition_dropout).expand_as(batch['text_tokens'])
    context = torch.autocast('cuda', dtype=torch.bfloat16) if config.precision == 'bf16' else nullcontext()
    with context:
        losses = model.loss(model.forward_single_stream(batch['audio_tokens'], batch['text_tokens'],
                                                        frame_mask=batch['frame_mask'],
                                                        text_condition_mask=conditioning))
    for name in ('loss', 'text_loss', 'audio_loss', 'audio_codebook_loss'):
        if not torch.isfinite(losses[name]).all():
            raise FloatingPointError(f'Nonfinite {name}; optimizer update aborted')
    if losses['audio_codebook_loss'].shape != (8,):
        raise ValueError('Expected eight codebook losses')
    return losses


def _metrics(losses):
    result = {name: float(losses[name].detach()) for name in ('loss', 'text_loss', 'audio_loss')}
    result.update({f'audio_codebook_{i}_loss': float(v.detach())
                   for i, v in enumerate(losses['audio_codebook_loss'])})
    return result


def evaluate(model, dataset, collate, config):
    if len(dataset) == 0:
        raise ValueError('Validation dataset is empty')
    was_training, rng = model.training, rng_state()
    totals, examples, seconds, valid, slots, text_pads = {}, 0, 0., 0, 0, 0
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(dataset), config.batch_size):
                indices = list(range(start, min(start + config.batch_size, len(dataset))))
                batch = _batch(dataset, indices, collate, config.device)
                values = _metrics(_loss(model, batch, config))
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.) + value * len(indices)
                examples += len(indices)
                seconds += sum(float(x) for x in batch['durations'])
                valid += int(batch['frame_mask'].sum())
                slots += batch['frame_mask'].numel()
                text_pads += int(((batch['text_tokens'] == model.config.text_pad_token_id)
                                  & batch['frame_mask']).sum())
        return {**{k: v / examples for k, v in totals.items()}, 'audio_seconds': seconds,
                'pad_ratio': 1 - valid / slots, 'valid_frames': valid,
                'text_pad_ratio': text_pads / valid, 'examples': examples}
    finally:
        model.train(was_training)
        restore_rng(rng)


def train(model, train_data, collate, output_dir, config, *, validation_data=None,
          resume=False, stop_after_steps=None, run_metadata=None):
    """Train to max_steps; stop_after_steps simulates a clean interrupted run.

    Resume requires unchanged configuration (including scheduler horizon). To pause
    a long planned run use stop_after_steps/--stop-after-steps, not a new max_steps.
    Only consume checkpoints you created and trust. No arbitrary checkpoint path.
    """
    config.validate()
    if len(train_data) == 0:
        raise ValueError('Training dataset is empty')
    if validation_data is not None and len(validation_data) == 0:
        raise ValueError('Validation dataset is empty; disable validation explicitly')
    if config.validation_interval and validation_data is None:
        raise ValueError('Validation enabled but no validation dataset supplied')
    if stop_after_steps is not None and stop_after_steps <= 0:
        raise ValueError('stop_after_steps must be positive')
    if getattr(model.config, 'num_audio_streams', None) != 1:
        raise ValueError('Stage 1 requires num_audio_streams=1')
    output = prepare_output(output_dir, resume)
    model.to(config.device)
    model_config = asdict(model.config) if is_dataclass(model.config) else model.config.to_dict()
    run = {'training': asdict(config), 'model': model_config, 'metadata': run_metadata or {}}
    data = {'train': train_data.fingerprint,
            'validation': validation_data.fingerprint if validation_data is not None else None}
    run_id, data_id = fingerprint(run), fingerprint(data)
    temporal, depth = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (temporal if name.startswith(('qwen_model.', 'assistant_audio_embeddings.')) else depth).append(parameter)
    optimizer = torch.optim.AdamW([{'params': temporal, 'lr': config.temporal_lr},
                                   {'params': depth, 'lr': config.depth_lr}], betas=(0.9, 0.95),
                                  weight_decay=config.weight_decay)
    def lr_factor(step):
        if config.warmup_steps and step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1., max(0., progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    step, epoch, position, total_seconds = 0, 0, 0, 0.
    if resume:
        saved = load_checkpoint(output / 'checkpoint.pt')
        if saved['run_fingerprint'] != run_id or saved['data_fingerprint'] != data_id:
            raise ValueError('Resume configuration or data fingerprint mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        step, epoch, position = saved['global_step'], saved['epoch'], saved['batch_position']
        total_seconds = saved['total_audio_seconds']
        restore_rng(saved['rng'])
    else:
        seed_everything(config.seed)
    # Remove metrics beyond the last committed update after a crash, and append.
    log_path = output / 'metrics.jsonl'
    if resume and log_path.exists():
        retained = []
        lines = log_path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A crash may leave the last append incomplete, including a UTF-8
                # character. Never conceal corruption in a completed/earlier row.
                if index == len(lines) - 1 and not line.endswith(b'\n'):
                    break
                raise
            if item['global_step'] <= step:
                retained.append(json.dumps(item, allow_nan=False))
        repaired = log_path.with_suffix('.jsonl.tmp')
        repaired.write_text(''.join(line + '\n' for line in retained))
        os.replace(repaired, log_path)
    target = min(config.max_steps, step + stop_after_steps) if stop_after_steps is not None else config.max_steps
    model.train()
    order, order_epoch = None, None
    with log_path.open('a', encoding='utf-8') as log:
        while step < target:
            optimizer.zero_grad(set_to_none=True)
            totals, seconds, valid, slots, text_pads = {}, 0., 0, 0, 0
            for _ in range(config.accumulation_steps):
                if position >= len(train_data):
                    epoch, position = epoch + 1, 0
                if order_epoch != epoch:
                    generator = torch.Generator().manual_seed(config.seed + epoch)
                    # Keep the permutation tensor for this epoch; converting the
                    # entire Emilia ordering to Python integers wastes memory.
                    order = torch.randperm(len(train_data), generator=generator)
                    order_epoch = epoch
                indices = order[position:position + config.batch_size].tolist()
                batch = _batch(train_data, indices, collate, config.device)
                losses = _loss(model, batch, config)
                (losses['loss'] / config.accumulation_steps).backward()
                for key, value in _metrics(losses).items():
                    totals[key] = totals.get(key, 0.) + value / config.accumulation_steps
                position += len(indices)
                seconds += sum(float(x) for x in batch['durations'])
                valid += int(batch['frame_mask'].sum())
                slots += batch['frame_mask'].numel()
                text_pads += int(((batch['text_tokens'] == model.config.text_pad_token_id)
                                  & batch['frame_mask']).sum())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            used_lrs = [group['lr'] for group in optimizer.param_groups]
            optimizer.step()
            scheduler.step()
            step += 1
            total_seconds += seconds
            metrics = dict(kind='train', global_step=step, optimizer_updates=step, epoch=epoch,
                           batch_position=position, audio_seconds=seconds, total_audio_seconds=total_seconds,
                           pad_ratio=1 - valid / slots, valid_frames=valid,
                           text_pad_ratio=text_pads / valid, grad_norm=float(norm),
                           temporal_lr=used_lrs[0], depth_lr=used_lrs[1], **totals)
            rows = [metrics]
            if validation_data is not None and config.validation_interval and step % config.validation_interval == 0:
                rows.append(dict(kind='validation', global_step=step,
                                 **evaluate(model, validation_data, collate, config)))
            save_checkpoint(output / 'checkpoint.pt', dict(format='qwen3-moshi-stage1-v1',
                model=model.state_dict(), config=run, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rng=rng_state(), global_step=step, epoch=epoch, batch_position=position,
                run_fingerprint=run_id, data_fingerprint=data_id, data=data, total_audio_seconds=total_seconds))
            for row in rows:
                log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
    return dict(global_step=step, epoch=epoch, batch_position=position, total_audio_seconds=total_seconds)
