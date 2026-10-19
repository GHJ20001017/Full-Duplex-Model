# STT Summary

This document summarizes the Speech-to-Text (STT) implementations in the `STT/` folder, including language support, language abbreviations, and usage in `s2s_pipeline.py`.

## Available STT Modes (`--stt`)

- `whisper` → `STT/whisper_stt_handler.py`
- `whisper-mlx` → `STT/lightning_whisper_mlx_handler.py`
- `mlx-audio-whisper` → `STT/mlx_audio_whisper_handler.py`
- `faster-whisper` → `STT/faster_whisper_handler.py`
- `parakeet-tdt` → `STT/parakeet_tdt_handler.py`
- `paraformer` → `STT/paraformer_handler.py`
- `fun-asr-nano` → `STT/fun_asr_nano_handler.py`
- `openai` → `STT/openai_compatible_handler.py`

## Language Support by Handler

### 1) Whisper (`--stt whisper`)

- Handler: `WhisperSTTHandler`
- Language input flag: `--language` (from shared Whisper args)
- Supports fixed language (e.g. `en`) or `auto`
- Internal supported language list:
  - `en`, `fr`, `es`, `zh`, `ja`, `ko`, `hi`, `de`, `pt`, `pl`, `it`, `nl`
- Behavior:
  - Detects language from token output
  - If detected language is outside the supported list, it falls back to the previous language

### 2) Lightning Whisper MLX (`--stt whisper-mlx`)

- Handler: `LightningWhisperSTTHandler`
- Uses same shared `--language` argument as Whisper
- Internal supported language list:
  - `en`, `fr`, `es`, `zh`, `ja`, `ko`, `hi`, `de`, `pt`, `pl`, `it`, `nl`
- Behavior:
  - If `--language auto`, model auto-detects each utterance
  - If detected language is unsupported, falls back to last supported language

### 3) MLX Audio Whisper (`--stt mlx-audio-whisper`)

- Handler: `MLXAudioWhisperSTTHandler`
- Model flag: `--mlx_audio_whisper_model_name`
- Language still comes from shared `--language` flag (wired in pipeline)
- Internal supported language list:
  - `en`, `fr`, `es`, `zh`, `ja`, `ko`, `hi`, `de`, `pt`, `pl`, `it`, `nl`
- Behavior:
  - Uses fixed language unless `--language auto`
  - Falls back to last known supported language when needed

### 4) Faster-Whisper (`--stt faster-whisper`)

- Handler: `FasterWhisperSTTHandler`
- Language flag: `--faster_whisper_stt_gen_language`
- Default language: `en`
- Note:
  - This handler passes generation kwargs directly to `faster_whisper.WhisperModel.transcribe(...)`
  - Effective language coverage depends on selected Faster-Whisper/OpenAI Whisper model

### 5) Parakeet TDT (`--stt parakeet-tdt`)

- Handler: `ParakeetTDTSTTHandler`
- Language flag: `--parakeet_tdt_language` (optional)
- Supports auto language detection when language not specified
- Declared supported language list (25 European languages):
  - `en`, `de`, `fr`, `es`, `it`, `pt`, `nl`, `pl`, `ru`, `uk`, `cs`, `sk`, `hu`, `ro`, `bg`, `hr`, `sl`, `sr`, `da`, `no`, `sv`, `fi`, `et`, `lv`, `lt`
- Backend behavior:
  - On macOS/MPS: MLX (`mlx-community/parakeet-tdt-0.6b-v3`)
  - On CUDA/CPU: nano-parakeet (`nvidia/parakeet-tdt-0.6b-v3`)

### 6) Paraformer (`--stt paraformer`)

- Handler: `ParaformerSTTHandler`
- Model flag: `--paraformer_stt_model_name`
- Default model: `paraformer-zh-streaming`
- No dedicated language flag in current args class
- Practical support:
  - Depends on selected FunASR model checkpoint
  - Default setup is Chinese-oriented (`zh`)

### 7) OpenAI-compatible endpoint (`--stt openai`)

- Handler: `OpenAICompatibleSTTHandler`
- Endpoint: `POST /v1/audio/transcriptions`
- Upload: mono PCM16 WAV at 16 kHz
- Supports JSON (`{"text": "..."}`) and plain-text responses
- Keeps at most one best-effort progressive request in flight per pipeline while
  final requests are submitted independently; stale-turn filtering still applies
- See [`docs/openai-compatible-stt.md`](../../../docs/openai-compatible-stt.md)

### 8) Fun-ASR-Nano (`--stt fun-asr-nano`)

- Handler: `FunASRNanoSTTHandler`; FunASR `AutoModel` + PyTorch, no vLLM.
- Included in the [root requirements](../../../../requirements.txt). After activating
  `speech_to_speech_system`, install from `cascaded-full-duplex` with
  `(cd .. && python -m pip install -r requirements.txt)`.
- Default model: `FunAudioLLM/Fun-ASR-Nano-2512`; full local paths are also accepted.
- Required: `--fun_asr_nano_stt_hotwords_file /absolute/path/hotwords.txt`.
  UTF-8 (BOM allowed), one phrase per line; blank lines and `#` comment lines
  are ignored, duplicates removed, internal spaces preserved. Missing/unreadable/
  empty word lists fail before model loading. The list is read once at startup
  and passed as `hotwords` on every inference; restart to reload it.
- Language: `--fun_asr_nano_stt_language 中文|英文|日文` (default `中文`),
  reported as `zh|en|ja`. ITN defaults to true; disable with
  `--fun_asr_nano_stt_itn false`.
- Hub: `--fun_asr_nano_stt_hub hf|ms`; default `hf`. Official model loading
  uses `trust_remote_code=True`; choose trusted model sources only.
- With `--enable_live_transcription`, re-decodes cumulative VAD snapshots
  independently and emits full, revisable partial text; final input is the
  complete utterance. No cross-revision cache/prefix or delta-text concatenation.
  This is progressive re-inference, not native cached streaming. Latency and
  concurrent TTS resource use require real-device measurements.
- Without live transcription, only the final utterance is decoded.

## Language Abbreviations (ISO-style codes seen in STT handlers)

| Code | Language |
|---|---|
| `en` | English |
| `fr` | French |
| `es` | Spanish |
| `zh` | Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `hi` | Hindi |
| `de` | German |
| `pt` | Portuguese |
| `pl` | Polish |
| `it` | Italian |
| `nl` | Dutch |
| `ru` | Russian |
| `uk` | Ukrainian |
| `cs` | Czech |
| `sk` | Slovak |
| `hu` | Hungarian |
| `ro` | Romanian |
| `bg` | Bulgarian |
| `hr` | Croatian |
| `sl` | Slovenian |
| `sr` | Serbian |
| `da` | Danish |
| `no` | Norwegian |
| `sv` | Swedish |
| `fi` | Finnish |
| `et` | Estonian |
| `lv` | Latvian |
| `lt` | Lithuanian |
| `auto` | Per-utterance automatic language detection |

## Usage Examples

Use the main Conda environment. For `whisper-mlx`, first uncomment
`lightning-whisper-mlx` in the root requirements; for `faster-whisper`, uncomment
`faster-whisper`. Reinstall from `cascaded-full-duplex` with
`(cd .. && python -m pip install -r requirements.txt)`; if starting in this
`STT` directory, use `(cd ../../../.. && python -m pip install -r requirements.txt)`.
Reading the unchanged manifest does not install these commented optional backends.
For these examples on non-macOS, append `--qwen3_tts_backend torch` to use
Qwen3 without installing the optional GGML wheel.

### Whisper (Transformers)

```bash
speech-to-speech serve --stt whisper --language en
speech-to-speech serve --stt whisper --language auto
```

### Whisper MLX (LightningWhisperMLX)

```bash
speech-to-speech serve --stt whisper-mlx --language auto --device mps
```

### MLX Audio Whisper

```bash
speech-to-speech serve --stt mlx-audio-whisper \
  --mlx_audio_whisper_model_name mlx-community/whisper-large-v3-turbo \
  --language auto
```

### Faster-Whisper

```bash
speech-to-speech serve --stt faster-whisper \
  --faster_whisper_stt_model_name large-v3 \
  --faster_whisper_stt_gen_language en
```

### Parakeet TDT

```bash
speech-to-speech serve --stt parakeet-tdt --parakeet_tdt_device auto
speech-to-speech serve --stt parakeet-tdt --parakeet_tdt_language de
```

With live transcription (MLX or CUDA/nano-parakeet backend):

```bash
speech-to-speech serve --stt parakeet-tdt \
  --enable_live_transcription \
  --live_transcription_update_interval 0.25
```

### Paraformer

```bash
speech-to-speech serve --stt paraformer --paraformer_stt_model_name paraformer-zh-streaming
```
