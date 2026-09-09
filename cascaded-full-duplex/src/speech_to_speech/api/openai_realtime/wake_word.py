"""Client-side wake-word gating for the native microphone client."""

from __future__ import annotations

import math
import struct
import time
from collections import deque
from typing import Any


class OpenWakeWordDetector:
    """Run openWakeWord locally and recognize a bundled wake-word model."""

    def __init__(self, keyword: str, sample_rate: int, threshold: float = 0.5) -> None:
        try:
            import numpy as np
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError("Wake-word support requires openwakeword; install speech-to-speech[wake-word].") from exc
        if sample_rate != 16000:
            raise ValueError("openWakeWord detection currently requires a 16000 Hz microphone")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("wake-word threshold must be between 0 and 1")
        self._np = np
        self._model = Model(
            wakeword_models=[keyword.strip().lower().replace(" ", "_")],
            inference_framework="onnx",
        )
        self._threshold = threshold
        self.frame_length = 1280
        self._buffer = bytearray()

    def process(self, pcm: bytes) -> bool:
        self._buffer.extend(pcm)
        frame_bytes = self.frame_length * 2
        while len(self._buffer) >= frame_bytes:
            frame = self._buffer[:frame_bytes]
            del self._buffer[:frame_bytes]
            samples = self._np.frombuffer(frame, dtype=self._np.int16)
            scores = self._model.predict(samples)
            if any(float(score) >= self._threshold for score in scores.values()):
                self._buffer.clear()
                return True
        return False

    def close(self) -> None:
        self._model.reset()


class WakeWordGate:
    """Discard microphone audio until a wake word is followed by speech."""

    def __init__(
        self,
        detector: Any,
        *,
        timeout_s: float = 300.0,
        sample_rate: int = 16000,
        speech_rms_threshold: float = 500.0,
        preroll_ms: int = 400,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.detector = detector
        self.timeout_s = timeout_s
        self.sample_rate = sample_rate
        self.speech_rms_threshold = speech_rms_threshold
        self._armed_until = 0.0
        self._active = False
        self._wake_detected = False
        self._last_speech_at = 0.0
        self._preroll: deque[bytes] = deque()
        self._preroll_bytes = max(0, int(sample_rate * 2 * preroll_ms / 1000))
        self._preroll_size = 0

    @property
    def armed(self) -> bool:
        return self._armed_until > time.monotonic()

    def process(self, chunk: bytes) -> list[bytes]:
        now = time.monotonic()
        if not self._active and not self.armed and self.detector.process(chunk):
            self._armed_until = now + self.timeout_s
            self._wake_detected = True
            self._preroll.clear()
            self._preroll_size = 0
            print(f"WAKE WORD detected; speak within {self.timeout_s:g}s", flush=True)
            return []

        if self._active:
            if self._has_speech(chunk):
                self._last_speech_at = now
            elif now - self._last_speech_at >= self.timeout_s:
                self._active = False
                print("Wake window expired; waiting for wake word", flush=True)
                return []
            return [chunk]

        if now >= self._armed_until:
            self._armed_until = 0.0
            self._preroll.clear()
            self._preroll_size = 0
            return []

        self._append_preroll(chunk)
        if self._has_speech(chunk):
            self._active = True
            self._last_speech_at = now
            output = list(self._preroll)
            self._preroll.clear()
            self._preroll_size = 0
            print("Speech detected; audio pipeline enabled", flush=True)
            return output
        return []

    def close(self) -> None:
        close = getattr(self.detector, "close", None)
        if callable(close):
            close()

    def consume_wake(self) -> bool:
        """Return and clear the pending wake-word notification."""
        detected = self._wake_detected
        self._wake_detected = False
        return detected

    def _append_preroll(self, chunk: bytes) -> None:
        if self._preroll_bytes == 0:
            return
        self._preroll.append(chunk)
        self._preroll_size += len(chunk)
        while self._preroll and self._preroll_size > self._preroll_bytes:
            self._preroll_size -= len(self._preroll.popleft())

    def _has_speech(self, chunk: bytes) -> bool:
        usable = chunk[: len(chunk) - len(chunk) % 2]
        if not usable:
            return False
        samples = struct.unpack(f"<{len(usable) // 2}h", usable)
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        return rms >= self.speech_rms_threshold
