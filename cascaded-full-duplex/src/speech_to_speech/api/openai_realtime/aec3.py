"""Small ctypes wrapper around the native WebRTC EchoCanceller3 adapter."""

from __future__ import annotations

import ctypes
import os
import platform
from collections import deque
from pathlib import Path


class AEC3Error(RuntimeError):
    """Raised when the native AEC3 backend cannot be loaded or used."""


def _default_library_names() -> list[str]:
    if platform.system() == "Darwin":
        return ["libs2s_aec3.dylib", "s2s_aec3.dylib"]
    if platform.system() == "Windows":
        return ["s2s_aec3.dll"]
    return ["libs2s_aec3.so", "s2s_aec3.so"]


def _find_library() -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("S2S_AEC3_LIBRARY")
    if configured:
        candidates.append(Path(configured).expanduser())
    project_root = Path(__file__).resolve().parents[4]
    for name in _default_library_names():
        candidates.extend(
            [
                project_root / "native" / "aec3" / "build" / name,
                project_root / "native" / "aec3" / "install" / "lib" / name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AEC3Error(
        "WebRTC AEC3 native library not found. Build it with "
        "native/aec3/build_macos.sh or set S2S_AEC3_LIBRARY."
    )


class AEC3Processor:
    """Process arbitrary PCM chunks through real WebRTC EchoCanceller3 frames."""

    def __init__(self, sample_rate: int = 16000) -> None:
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("AEC3 requires an 8/16/32/48 kHz sample rate")
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate // 100
        self.frame_bytes = self.frame_samples * 2
        self._library = ctypes.CDLL(str(_find_library()))
        self._library.s2s_aec3_create.argtypes = [ctypes.c_int]
        self._library.s2s_aec3_create.restype = ctypes.c_void_p
        self._library.s2s_aec3_process_render.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int]
        self._library.s2s_aec3_process_render.restype = ctypes.c_int
        self._library.s2s_aec3_process_capture.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
        ]
        self._library.s2s_aec3_process_capture.restype = ctypes.c_int
        self._library.s2s_aec3_destroy.argtypes = [ctypes.c_void_p]
        self._handle = self._library.s2s_aec3_create(sample_rate)
        if not self._handle:
            raise AEC3Error("WebRTC AEC3 initialization failed")
        self._render_pending = bytearray()
        self._capture_pending = bytearray()
        self._capture_output: deque[bytes] = deque()

    def process_render(self, pcm: bytes) -> None:
        """Feed the exact PCM block written to the speaker into AEC3."""
        self._render_pending.extend(pcm)
        while len(self._render_pending) >= self.frame_bytes:
            frame = self._render_pending[: self.frame_bytes]
            del self._render_pending[: self.frame_bytes]
            samples = (ctypes.c_int16 * self.frame_samples).from_buffer_copy(frame)
            result = self._library.s2s_aec3_process_render(self._handle, samples, self.frame_samples)
            if result != 0:
                raise AEC3Error(f"AEC3 render processing failed: {result}")

    def process_capture(self, pcm: bytes) -> bytes:
        """Return a same-sized capture block after AEC3 processing."""
        requested = len(pcm)
        self._capture_pending.extend(pcm)
        while len(self._capture_pending) >= self.frame_bytes:
            frame = self._capture_pending[: self.frame_bytes]
            del self._capture_pending[: self.frame_bytes]
            input_samples = (ctypes.c_int16 * self.frame_samples).from_buffer_copy(frame)
            output_samples = (ctypes.c_int16 * self.frame_samples)()
            result = self._library.s2s_aec3_process_capture(
                self._handle, input_samples, output_samples, self.frame_samples
            )
            if result != 0:
                raise AEC3Error(f"AEC3 capture processing failed: {result}")
            self._capture_output.append(bytes(output_samples))

        output = bytearray()
        while self._capture_output and len(output) < requested:
            output.extend(self._capture_output.popleft())
        if len(output) < requested:
            output.extend(b"\x00" * (requested - len(output)))
        elif len(output) > requested:
            extra = bytes(output[requested:])
            self._capture_output.appendleft(extra)
            del output[requested:]
        return bytes(output)

    def close(self) -> None:
        if self._handle:
            self._library.s2s_aec3_destroy(self._handle)
            self._handle = None
