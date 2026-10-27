"""Optional Qwen3-ASR AuT integration (no ASR text decoder in the forward path)."""

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence


class Qwen3ASRAudioEncoder(nn.Module):
    """Adapt the upstream packed AuT output into padded continuous features.

    This is a differentiable offline/chunk encoder, not a stateful streaming API.
    Upstream attention is bidirectional within chunks. Only provide received
    audio and track feature availability externally for streaming use.
    """

    def __init__(self, audio_tower: nn.Module):
        super().__init__()
        self.audio_tower = audio_tower
        self.output_dim = audio_tower.config.output_dim

    @classmethod
    def from_pretrained(
        cls, name_or_path: str = "Qwen/Qwen3-ASR-0.6B", **kwargs
    ) -> "Qwen3ASRAudioEncoder":
        try:
            from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
        except ImportError as exc:
            raise ImportError('Install the continuous encoder extra: pip install -e ".[continuous]"') from exc
        # Load via the official checkpoint class so its key mapping stays authoritative.
        # The temporary ASR text decoder is released; only the AuT is retained.
        asr = Qwen3ASRForConditionalGeneration.from_pretrained(name_or_path, **kwargs)
        return cls(asr.thinker.audio_tower)

    @staticmethod
    def load_feature_extractor(name_or_path: str = "Qwen/Qwen3-ASR-0.6B", **kwargs):
        from transformers import WhisperFeatureExtractor

        return WhisperFeatureExtractor.from_pretrained(name_or_path, **kwargs)

    def forward(self, input_features: Tensor, feature_attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        if input_features.ndim != 3 or not input_features.is_floating_point():
            raise ValueError("input_features must be floating point [batch, mel_bins, mel_frames]")
        batch, bins, time = input_features.shape
        if batch == 0 or bins != self.audio_tower.config.num_mel_bins:
            raise ValueError("AuT input batch/mel bin count is invalid")
        if feature_attention_mask.shape != (batch, time):
            raise ValueError("feature_attention_mask must have shape [batch, mel_frames]")
        if not ((feature_attention_mask == 0) | (feature_attention_mask == 1)).all():
            raise ValueError("feature_attention_mask must be binary")
        mask = feature_attention_mask.bool()
        lengths = mask.long().sum(-1)
        expected = torch.arange(time, device=mask.device)[None, :] < lengths[:, None]
        if not torch.equal(mask, expected) or (lengths == 0).any():
            raise ValueError("AuT requires nonempty, right-padded mel features")
        parameter = next(self.audio_tower.parameters())
        outputs = []
        # Match upstream get_audio_features: encode each unpadded audio independently.
        for mel, length in zip(input_features, lengths.tolist()):
            output = self.audio_tower(
                mel[:, :length].to(device=parameter.device, dtype=parameter.dtype),
                feature_lens=torch.tensor([length], device=parameter.device, dtype=torch.long),
            ).last_hidden_state
            outputs.append(output)
        features = pad_sequence(outputs, batch_first=True)
        output_lengths = torch.tensor([value.shape[0] for value in outputs], device=features.device)
        output_mask = torch.arange(features.shape[1], device=features.device)[None, :] < output_lengths[:, None]
        return features, output_mask
