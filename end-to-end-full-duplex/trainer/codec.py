"""Frozen local Mimi adapter. No guessed vocabulary sizes or hub resolution."""
import hashlib
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FrozenMimiCodec:
    def __init__(self, checkpoint, device='cpu', *, loader=None):
        checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        if not checkpoint.is_file() or checkpoint.suffix != '.safetensors':
            raise ValueError('Mimi must be a local .safetensors checkpoint')
        if loader is None:
            from moshi.models.loaders import get_mimi
            loader = get_mimi
        self.device = torch.device(device)
        self.model = loader(checkpoint, device=str(self.device), num_codebooks=8)
        self.model.eval().requires_grad_(False)
        # These are public runtime properties of Mimi/CompressionModel. Unsupported
        # implementations fail here rather than inheriting guessed constants.
        self.sample_rate = getattr(self.model, 'sample_rate', None)
        self.frame_rate = getattr(self.model, 'frame_rate', None)
        cardinality = getattr(self.model, 'cardinality', None)
        count = getattr(self.model, 'num_codebooks', None)
        if self.sample_rate != 24000 or self.frame_rate != 12.5 or count != 8:
            raise ValueError('Unsupported Mimi: require runtime 24000 Hz, 12.5 fps, eight codebooks')
        if not isinstance(cardinality, int) or isinstance(cardinality, bool) or cardinality <= 0:
            raise ValueError('Mimi must expose a positive integer runtime cardinality')
        self.codebook_sizes = (cardinality,) * count
        self.cache_identity = 'mimi-sha256:' + sha256_file(checkpoint) + ':8:24000:12.5'

    def __call__(self, waveform):
        if (waveform.device.type != 'cpu' or waveform.ndim != 1
                or waveform.numel() == 0
                or not waveform.is_floating_point() or not torch.isfinite(waveform).all()):
            raise ValueError('Expected finite CPU float waveform [N] at 24000 Hz')
        self.model.eval()
        with torch.inference_mode():
            codes = self.model.encode(waveform.to(self.device)[None, None, :])
            if codes.ndim != 3 or codes.shape[:2] != (1, 8) or codes.shape[-1] == 0:
                raise ValueError('Mimi returned invalid code shape')
            if codes.dtype != torch.long or torch.any(codes < 0) or torch.any(codes >= self.codebook_sizes[0]):
                raise ValueError('Mimi returned out-of-vocabulary codes')
            codes = codes[0].cpu()
        # Ordinary tensors, not inference tensors: downstream embeddings save indices.
        return codes.clone()
