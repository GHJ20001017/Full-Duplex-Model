#!/usr/bin/env python3
"""Offline Emilia ASR verification and forced alignment; never edits source data.

Requires qwen-asr and its audio dependencies. Defaults use local model weights.
All returned timestamps are journaled without filtering, clipping or repair.
Quality assessment is deferred: generated does not mean accepted for training.
Timestamps are character/word positions in seconds, not tokenizer boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import unicodedata
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LANGUAGES = {"zh": "Chinese", "en": "English"}


def normalize_text(text, language):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return "".join(text.split()) if language == "Chinese" else " ".join(text.split())


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def compare(reference, hypothesis, language):
    ref, hyp = (normalize_text(t, language) for t in (reference, hypothesis))
    left, right = (ref, hyp) if language == "Chinese" else (ref.split(), hyp.split())
    rate = edit_distance(left, right) / max(1, len(left))
    return {"normalized_reference": ref, "normalized_asr": hyp,
            "metric": "cer" if language == "Chinese" else "wer", "error_rate": rate}


def resolve_audio(record, metadata):
    # Emilia extraction flattens the archive's original wav path.
    candidates = [metadata.with_suffix(ext) for ext in (".mp3", ".wav", ".flac")]
    if record.get("wav"):
        candidates.append(metadata.parent / record["wav"])
    for path in candidates:
        if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".flac"}:
            return path.resolve()
    raise FileNotFoundError(f"No audio beside {metadata}")


def serialize_timestamps(items):
    # Preserve every returned interval, including zero duration and out-of-range
    # values. Non-finite numbers are strings so the journal remains valid JSON.
    def number(value):
        value = float(value)
        return value if math.isfinite(value) else str(value)

    return [{"text": item.text, "start": number(item.start_time),
             "end": number(item.end_time)} for item in items]


def read_journal(path, config):
    completed = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid journal line {number}; preserve/repair it before resuming") from exc
                if row.get("config") != config:
                    raise ValueError("Resume configuration differs; choose a new output file")
                completed[row["metadata"]] = row
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/gpu3/guhj/data"))
    parser.add_argument("--dataset", type=Path, nargs="+", help="Override default bilingual dataset roots")
    parser.add_argument("--output", type=Path, required=True, help="Append-only JSONL outcomes")
    parser.add_argument("--asr-model", type=Path, default=PROJECT / "models/qwen3-asr-1.7b")
    parser.add_argument("--aligner-model", type=Path, default=PROJECT / "models/qwen3-forcedaligner-0.6b")
    parser.add_argument("--max-cer", type=float, default=0.20, help="Recorded for later evaluation only; never filters generation")
    parser.add_argument("--max-wer", type=float, default=0.20, help="Recorded for later evaluation only; never filters generation")
    parser.add_argument("--max-samples", type=int, help="New samples PER dataset; default all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="With --resume, append new attempts for error rows")
    parser.add_argument("--dry-run", action="store_true", help="Check metadata/audio paths without loading models or writing output")
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or v < 0 for v in (args.max_cer, args.max_wer)):
        parser.error("Thresholds must be finite and nonnegative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    roots = [p.resolve() for p in (args.dataset or [args.data_root / "emilia-zh-200h", args.data_root / "emilia-en-100h"])]
    for root in roots:
        if not (root / "extracted").is_dir():
            parser.error(f"Missing extracted directory: {root}")
    config = {"schema": 2, "asr_model": str(args.asr_model.resolve()),
              "aligner_model": str(args.aligner_model.resolve()),
              "max_cer": args.max_cer, "max_wer": args.max_wer, "dtype": args.dtype,
              "datasets": [str(p) for p in roots]}
    if args.output.exists() and not args.resume and not args.dry_run:
        parser.error("Output exists; use --resume or a new output (never overwritten)")
    completed = read_journal(args.output, config) if args.resume else {}
    if not args.dry_run:
        for model in (args.asr_model, args.aligner_model):
            if not (model / "config.json").is_file():
                parser.error(f"Missing local model config: {model}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
            from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
            from qwen_asr.inference.utils import normalize_audios
        except ImportError as exc:
            parser.error(f"Install qwen-asr in your inference environment: {exc}")
        kwargs = {"dtype": getattr(torch, args.dtype), "device_map": args.device,
                  "local_files_only": True}
        asr = Qwen3ASRModel.from_pretrained(str(args.asr_model.resolve()), **kwargs)
        aligner = Qwen3ForcedAligner.from_pretrained(str(args.aligner_model.resolve()), **kwargs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"generated": 0, "error": 0, "checked": 0, "skipped": 0}
    handle = None
    try:
        if not args.dry_run:
            handle = args.output.open("a" if args.resume else "x", encoding="utf-8")
            # A valid last row without newline must not be concatenated with the next.
            if args.resume and args.output.stat().st_size:
                with args.output.open("rb") as existing:
                    existing.seek(-1, 2)
                    if existing.read(1) != b"\n":
                        handle.write("\n")
        for root in roots:
            processed = 0
            for metadata in sorted((root / "extracted").rglob("*.json")):
                if args.max_samples is not None and processed >= args.max_samples:
                    break
                key = str(metadata)
                raw = metadata.read_bytes()
                fingerprint = hashlib.sha256(raw).hexdigest()
                prior = completed.get(key)
                if prior and prior["metadata_sha256"] == fingerprint and not (args.retry_errors and prior["status"] == "error"):
                    counts["skipped"] += 1
                    continue
                processed += 1
                row = {"metadata": key, "metadata_sha256": fingerprint, "config": config,
                       "status": "error", "quality_status": "not_evaluated"}
                try:
                    record = json.loads(raw)
                    text = record["text"]
                    language = LANGUAGES[record["language"]]
                    if not isinstance(text, str) or not normalize_text(text, language):
                        raise ValueError("Empty reference text")
                    audio = resolve_audio(record, metadata)
                    row.update({"id": record["id"], "text": text, "language": language,
                                "speaker": record.get("speaker"), "audio": str(audio),
                                "source_duration": record.get("duration")})
                    if args.dry_run:
                        counts["checked"] += 1
                        continue
                    # Decode once through the same 16 kHz mono utility used by Qwen.
                    waveform = normalize_audios(str(audio))[0]
                    duration = len(waveform) / 16000
                    if not 0 < duration <= 180:
                        raise ValueError(f"Audio must be in (0, 180] seconds for this pipeline, got {duration}")
                    row["duration"] = duration
                    errors = {}
                    try:
                        prediction = asr.transcribe(audio=(waveform, 16000), language=language)[0]
                        row["asr_text"] = prediction.text
                        row.update(compare(text, prediction.text, language))
                    except Exception as exc:
                        errors["asr"] = f"{type(exc).__name__}: {exc}"
                    # Alignment of the reference does not depend on ASR success
                    # or its match score. Quality assessment is a later stage.
                    try:
                        aligned = aligner.align(audio=(waveform, 16000), text=text, language=language)[0]
                        row["timestamps"] = serialize_timestamps(aligned)
                        row["timestamp_unit"] = "seconds"
                    except Exception as exc:
                        errors["alignment"] = f"{type(exc).__name__}: {exc}"
                    if errors:
                        row["errors"] = errors
                    else:
                        row["status"] = "generated"
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    print(f"{metadata.name}: {row['error']}", file=sys.stderr)
                counts[row["status"]] += 1
                if handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                print(json.dumps({"sample": metadata.stem, **counts}), flush=True)
    finally:
        if handle:
            handle.close()
    print(json.dumps(counts))
    return 2 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
