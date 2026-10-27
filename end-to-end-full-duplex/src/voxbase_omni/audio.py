"""Audio representations used by the VoxBase-omni input model."""

import torch
from torch import Tensor, nn


def require_integer(value: Tensor, name: str) -> None:
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must contain integer indices")


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden_size),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.net(hidden)


class MimiCodeEmbedding(nn.Module):
    """[B, 8, T] Mimi indices -> [B, T, D], without waveform decoding."""

    num_codebooks = 8

    def __init__(self, hidden_size: int, codebook_size: int = 2048, adapter_dim: int = 256):
        super().__init__()
        if min(hidden_size, codebook_size, adapter_dim) <= 0:
            raise ValueError("Embedding and adapter dimensions must be positive")
        self.codebook_size = codebook_size
        self.semantic_embedding = nn.Embedding(codebook_size, hidden_size)
        self.acoustic_embedding = nn.Embedding(codebook_size, hidden_size)
        self.acoustic_adapters = nn.ModuleList(
            ResidualAdapter(hidden_size, adapter_dim) for _ in range(7)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, codes: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        require_integer(codes, "codes")
        if codes.ndim != 3 or codes.shape[1] != self.num_codebooks:
            raise ValueError("codes must have shape [batch, 8, frames]")
        batch, _, frames = codes.shape
        if frame_mask is None:
            frame_mask = torch.ones((batch, frames), dtype=torch.bool, device=codes.device)
        if frame_mask.shape != (batch, frames) or frame_mask.dtype != torch.bool:
            raise ValueError("frame_mask must be bool [batch, frames]")
        # Padded codes may use -1; never let them reach an embedding lookup.
        valid_codes = codes.transpose(1, 2)[frame_mask]
        if ((valid_codes < 0) | (valid_codes >= self.codebook_size)).any():
            raise ValueError(f"Valid Mimi indices must be in [0, {self.codebook_size})")
        codes = codes.masked_fill(~frame_mask[:, None, :], 0).long()
        semantic = self.semantic_embedding(codes[:, 0])
        acoustic = torch.stack(
            [adapter(self.acoustic_embedding(codes[:, i + 1]))
             for i, adapter in enumerate(self.acoustic_adapters)],
            dim=0,
        ).mean(dim=0)
        return self.norm(semantic + acoustic).masked_fill(~frame_mask[..., None], 0)


class AudioProjector(nn.Module):
    """Project the AuT output width into the language backbone's embedding width."""

    def __init__(self, audio_hidden_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(audio_hidden_size),
            nn.Linear(audio_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class GatedAudioFusion(nn.Module):
    """Frame-local residual fusion; unavailable frames bypass this module entirely."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.continuous_norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, discrete: Tensor, continuous: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((discrete, continuous), dim=-1)))
        return discrete + gate * self.continuous_norm(continuous)


def align_continuous(
    features: Tensor,
    frame_indices: Tensor,
    discrete_mask: Tensor,
    feature_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Gather explicitly aligned AuT frames; -1 means unavailable, not zero audio.

    The producer must construct indices from real timestamps/availability. Equal
    tensor lengths do not prove alignment, and interpolation could leak future audio.
    """
    require_integer(frame_indices, "continuous_frame_indices")
    if features.ndim != 3 or not features.is_floating_point():
        raise ValueError("continuous_features must be floating point [batch, time, width]")
    if frame_indices.shape != discrete_mask.shape or features.shape[0] != discrete_mask.shape[0]:
        raise ValueError("continuous_frame_indices must match the discrete frame shape")
    batch, time, width = features.shape
    if feature_mask is None:
        feature_mask = torch.ones((batch, time), dtype=torch.bool, device=features.device)
    if feature_mask.shape != (batch, time) or feature_mask.dtype != torch.bool:
        raise ValueError("continuous_feature_mask must be bool [batch, continuous_time]")
    valid_indices = frame_indices[discrete_mask]
    if ((valid_indices < -1) | (valid_indices >= time)).any():
        raise ValueError("Continuous frame index is out of range (use -1 for missing)")
    available = discrete_mask & (frame_indices >= 0)
    if time == 0:
        return features.new_zeros((*discrete_mask.shape, width)), available
    safe_indices = frame_indices.masked_fill(~available, 0).long()
    if (available & ~feature_mask.gather(1, safe_indices)).any():
        raise ValueError("An alignment index refers to a padded continuous frame")
    aligned = features.gather(1, safe_indices[..., None].expand(-1, -1, width))
    return aligned, available
