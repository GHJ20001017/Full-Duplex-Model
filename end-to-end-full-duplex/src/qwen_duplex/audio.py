"""Arrived-only native audio encoding; this is prefix recomputation, not encoder KV caching."""

import math

import numpy as np
import torch


class PrefixAudioEncoder:
    """Freeze submitted positions while recomputing the arrived waveform prefix.

    Inputs must already be mono at the processor's sample rate. No resampling,
    endpoint oracle, future waveform, or offline/full-utterance features are used.
    Frozen edge representations need not equal later offline representations.
    Prefix cost grows with session duration; max_audio_seconds is a hard bound,
    not a sliding-window implementation or a real-time throughput guarantee.
    """

    def __init__(self, model, processor, *, max_audio_seconds=30.0):
        self.thinker = getattr(model, "thinker", model)
        self.extractor = processor.feature_extractor
        self.sample_rate = self.extractor.sampling_rate
        if not math.isfinite(max_audio_seconds) or max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be finite and positive")
        self.max_samples = int(max_audio_seconds * self.sample_rate)
        self.reset()

    def reset(self):
        self.waveform = np.empty(0, dtype=np.float32)
        self.committed = 0

    @torch.inference_mode()
    def push(self, pcm, *, sampling_rate):
        if self.thinker.training:
            raise ValueError("Use eval() for incremental audio encoding")
        waveform = np.asarray(pcm)
        if sampling_rate != self.sample_rate:
            raise ValueError(f"Resample audio to {self.sample_rate} Hz before push()")
        if waveform.ndim != 1 or not waveform.size or not np.issubdtype(waveform.dtype, np.floating):
            raise ValueError("pcm must be a nonempty mono floating-point waveform")
        if not np.isfinite(waveform).all():
            raise ValueError("pcm contains non-finite samples")
        arrived = np.concatenate((self.waveform, waveform.astype(np.float32)))
        if len(arrived) > self.max_samples:
            raise ValueError("Audio prefix budget exceeded; reset/rebuild the session explicitly")
        # Match the native processor's zero-padding and mask semantics, without truncation.
        padded_length = max(self.extractor.n_samples, math.ceil(len(arrived) / self.extractor.hop_length)
                            * self.extractor.hop_length + self.extractor.hop_length)
        batch = self.extractor(
            arrived, sampling_rate=self.sample_rate, padding="max_length", max_length=padded_length,
            truncation=False, return_attention_mask=True, return_tensors="pt",
        )
        weight = self.thinker.get_input_embeddings().weight
        hidden = self.thinker.get_audio_features(
            batch["input_features"].to(device=weight.device, dtype=weight.dtype),
            feature_attention_mask=batch["attention_mask"].to(weight.device),
        )
        # Transformers versions return either a Tensor or a model-output container.
        if not isinstance(hidden, torch.Tensor):
            hidden = hidden.last_hidden_state
        if hidden.ndim != 2 or hidden.shape[1] != weight.shape[1] or not torch.isfinite(hidden).all():
            raise RuntimeError("Native audio encoder returned invalid embeddings")
        if len(hidden) < self.committed:
            raise RuntimeError("Native audio positions moved backwards")
        new_hidden = hidden[self.committed:].detach().clone()
        self.waveform = arrived
        self.committed = len(hidden)
        return new_hidden
