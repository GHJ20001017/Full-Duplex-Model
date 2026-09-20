#!/usr/bin/env python3
"""Visualize Mimi codebooks and measure waveform reconstruction quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24_000
FRAME_RATE = 12.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path, help="Input WAV/FLAC files.")
    parser.add_argument(
        "--mimi-checkpoint", type=Path, required=True,
        help="Local Mimi checkpoint (.safetensors) from ModelScope.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/step2-codebooks"),
        help="Directory for plots, reconstructed audio, codes and metrics.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--num-codebooks", type=int, default=8, help="Number of Mimi codebooks.")
    parser.add_argument(
        "--max-seconds", type=float, default=10.0,
        help="Trim each case to this duration; use 0 to disable trimming.",
    )
    return parser.parse_args()


def load_audio(path: Path, max_seconds: float):
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Run: pip install soundfile") from exc

    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if max_seconds > 0:
        audio = audio[: int(max_seconds * sample_rate)]
    if audio.size == 0:
        raise ValueError(f"Audio is empty: {path}")

    import torch
    import torch.nn.functional as F

    waveform = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
    if sample_rate != SAMPLE_RATE:
        target_length = round(waveform.shape[-1] * SAMPLE_RATE / sample_rate)
        waveform = F.interpolate(waveform, size=target_length, mode="linear", align_corners=False)
    return waveform, SAMPLE_RATE


def load_mimi(checkpoint: Path, device: str, num_codebooks: int):
    try:
        import torch
        from moshi.models import loaders
    except ImportError as exc:
        raise RuntimeError(
            "Missing Mimi runtime. Install Moshi and its dependencies before running "
            "this diagnostic, for example: pip install moshi"
        ) from exc

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Mimi checkpoint not found: {checkpoint}")
    mimi = loaders.get_mimi(checkpoint, device=device, num_codebooks=num_codebooks)
    mimi.eval()
    return mimi, torch


def align_reconstruction(original, reconstructed):
    length = min(original.shape[-1], reconstructed.shape[-1])
    return original[..., :length], reconstructed[..., :length]


def calculate_metrics(original, reconstructed) -> dict[str, float | int]:
    original, reconstructed = align_reconstruction(original, reconstructed)
    error = reconstructed - original
    signal_power = float(np.mean(original**2))
    error_power = float(np.mean(error**2))
    return {
        "num_samples": int(original.size),
        "duration_seconds": float(original.size / SAMPLE_RATE),
        "rmse": float(np.sqrt(error_power)),
        "max_absolute_error": float(np.max(np.abs(error))),
        "snr_db": float(10 * np.log10(max(signal_power, 1e-12) / max(error_power, 1e-12))),
        "original_peak": float(np.max(np.abs(original))),
        "reconstructed_peak": float(np.max(np.abs(reconstructed))),
    }


def save_waveform(path: Path, audio: np.ndarray) -> None:
    import soundfile as sf

    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")


def save_plot(path: Path, original: np.ndarray, reconstructed: np.ndarray, codes: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Run: pip install matplotlib") from exc

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    original, reconstructed = align_reconstruction(original, reconstructed)
    duration = original.size / SAMPLE_RATE
    time = np.arange(original.size) / SAMPLE_RATE
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), constrained_layout=True)
    axes[0].plot(time, original, color="#147d92", linewidth=0.55, label="Original")
    axes[0].plot(time, reconstructed, color="#d97732", linewidth=0.45, alpha=0.8, label="Mimi reconstruction")
    axes[0].set_title(f"Waveform comparison - {duration:.2f} s")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    image = axes[1].imshow(
        codes,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[0, codes.shape[1] / FRAME_RATE, 0.5, codes.shape[0] + 0.5],
        cmap="viridis",
    )
    axes[1].set_title("Mimi codebook tokens (one row per codebook; color is token ID)")
    axes[1].set_xlabel("Time (s, 12.5 frames/s)")
    axes[1].set_ylabel("Codebook index")
    axes[1].set_yticks(np.arange(1, codes.shape[0] + 1))
    fig.colorbar(image, ax=axes[1], label="token ID")

    residual = reconstructed - original
    axes[2].plot(time, residual, color="#7c3aed", linewidth=0.5)
    axes[2].set_title("Reconstruction residual (Mimi - original)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Error")
    axes[2].grid(alpha=0.2)
    fig.suptitle("Step2 - Mimi codec and codebook visualization", fontsize=16)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_case(path: Path, mimi, torch, output_dir: Path, max_seconds: float) -> dict:
    waveform, _ = load_audio(path, max_seconds)
    waveform = waveform.to(next(mimi.parameters()).device)
    with torch.inference_mode():
        codes = mimi.encode(waveform)
        reconstruction = mimi.decode(codes)

    codes_np = codes.detach().cpu().numpy()
    if codes_np.ndim != 3:
        raise RuntimeError(f"Expected code shape [B, codebooks, frames], got {codes_np.shape}")
    codes_np = codes_np[0]
    original_np = waveform[0, 0].detach().cpu().numpy()
    reconstruction_np = reconstruction[0, 0].detach().float().cpu().numpy()
    original_np, reconstruction_np = align_reconstruction(original_np, reconstruction_np)

    stem = path.stem
    np.save(output_dir / f"{stem}.codebooks.npy", codes_np)
    save_waveform(output_dir / f"{stem}.reconstructed.wav", reconstruction_np)
    metrics = calculate_metrics(original_np, reconstruction_np)
    metrics.update({"input": str(path), "code_shape": list(codes_np.shape), "sample_rate": SAMPLE_RATE})
    save_plot(output_dir / f"{stem}.diagnostic.png", original_np, reconstruction_np, codes_np)
    return metrics


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mimi, torch = load_mimi(args.mimi_checkpoint.expanduser().resolve(), args.device, args.num_codebooks)
    results = []
    for audio_path in args.audio:
        try:
            results.append(run_case(audio_path.expanduser().resolve(), mimi, torch, args.output_dir, args.max_seconds))
        except Exception as exc:
            print(f"Failed: {audio_path}: {exc}", file=sys.stderr)
            return 1
    (args.output_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for result in results:
        print(f"{result['input']}: shape={result['code_shape']}, SNR={result['snr_db']:.2f} dB")
    print(f"Saved diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
