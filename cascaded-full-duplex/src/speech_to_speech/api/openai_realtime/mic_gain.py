"""Microphone level helpers shared by the local client's capture path.

Both wake-word gating and the upload auto-gain need a level estimate for 64 ms
capture blocks, so both use :func:`short_term_level` instead of a whole-block
RMS: a block that contains a syllable plus silence averages out well below any
usable threshold, which is what made gating fail on low-gain microphones.
"""

from __future__ import annotations

import math
import struct

DEFAULT_SAMPLE_RATE = 16000
SHORT_FRAME_MS = 10


def _pcm16_samples(pcm: bytes) -> tuple[int, ...]:
    count = len(pcm) // 2
    if count == 0:
        return ()
    return struct.unpack(f"<{count}h", pcm[: count * 2])


def short_term_level(pcm: bytes, *, sample_rate: int = DEFAULT_SAMPLE_RATE, frame_ms: int = SHORT_FRAME_MS) -> float:
    """Return the loudest ``frame_ms`` RMS inside ``pcm`` (0.0 when empty)."""

    step = max(2, int(sample_rate * 2 * frame_ms / 1000))
    step -= step % 2
    usable = len(pcm) - (len(pcm) % 2)
    level = 0.0
    for start in range(0, usable, step):
        samples = _pcm16_samples(pcm[start : start + step])
        if not samples:
            continue
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        if rms > level:
            level = rms
    return level


def apply_gain(pcm: bytes, gain: float) -> bytes:
    """Scale 16-bit PCM by ``gain`` with clipping; ``gain == 1`` is a no-op."""

    if gain == 1.0:
        return pcm
    samples = _pcm16_samples(pcm)
    if not samples:
        return pcm
    scaled = [max(-32768, min(32767, int(sample * gain))) for sample in samples]
    return struct.pack(f"<{len(scaled)}h", *scaled) + pcm[len(samples) * 2 :]


class AdaptiveGain:
    """Slow, bounded microphone auto-gain for the upload path.

    The gain only adapts to speech-level input, so a quiet room is never
    amplified into something the server's voice detection could mistake for
    speech, and it saturates at ``max_gain`` while an already-healthy microphone
    keeps ``gain == min_gain`` (no behaviour change for normal setups).
    """

    def __init__(
        self,
        *,
        target_level: float = 2000.0,
        min_gain: float = 1.0,
        max_gain: float = 16.0,
        speech_level: float = 80.0,
        smoothing: float = 0.15,
    ) -> None:
        if target_level <= 0:
            raise ValueError("target_level must be positive")
        if min_gain <= 0 or max_gain < min_gain:
            raise ValueError("min_gain must be positive and no larger than max_gain")
        if speech_level < 0:
            raise ValueError("speech_level must not be negative")
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        self.target_level = target_level
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.speech_level = speech_level
        self.smoothing = smoothing
        self.gain = min_gain

    def process(self, pcm: bytes) -> bytes:
        """Return ``pcm`` amplified, adapting the gain on speech-level blocks."""

        level = short_term_level(pcm)
        if level >= self.speech_level:
            desired = min(self.max_gain, max(self.min_gain, self.target_level / level))
            self.gain += self.smoothing * (desired - self.gain)
            self.gain = min(self.max_gain, max(self.min_gain, self.gain))
        return apply_gain(pcm, self.gain)
